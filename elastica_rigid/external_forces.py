import numpy as np
from numpy.typing import NDArray

from numba import njit
from elastica.external_forces import NoForces

# -----------------------------------------------------------------------------
# Numba free-functions (used by class methods below)
# -----------------------------------------------------------------------------


@njit(cache=True)  # type: ignore
def compute_wheel_forces_to_external(
    direction: NDArray[np.float64],
    left_wheel_force: NDArray[np.float64],
    right_wheel_force: NDArray[np.float64],
    track_width: np.float64,
    external_forces: NDArray[np.float64],
    external_torques: NDArray[np.float64],
) -> None:
    """
    Add wheel forces (body frame) to system external forces and torques.

    Builds the body director from direction (d1) and its 90° rotation (d2),
    then: external_forces += director @ (left + right);
          external_torques += (track_width/2) * (-left + right).
    """
    R = np.array(
        [
            [np.cos(np.pi / 2), -np.sin(np.pi / 2)],
            [np.sin(np.pi / 2), np.cos(np.pi / 2)],
        ]
    )
    d1 = direction
    d2 = R @ d1
    director = np.empty((2, 2))
    director[:, 0] = d1
    director[:, 1] = d2

    external_forces += director @ (left_wheel_force + right_wheel_force)
    external_torques[0] += (track_width / 2) * (
        -left_wheel_force[0] + right_wheel_force[0]
    )


@njit(cache=True)  # type: ignore
def compute_potential_field_wheel_forces(
    x: NDArray[np.float64],
    d1: NDArray[np.float64],
    K: np.float64,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """u = -K * (x · d1); return left_wheel_force, right_wheel_force both [u, 0]."""
    u = -K * (x[0] * d1[0] + x[1] * d1[1])
    left = np.array([u, np.float64(0.0)])
    right = np.array([u, np.float64(0.0)])
    return left, right


@njit(cache=True)  # type: ignore
def interp_piecewise_linear(
    t: np.float64,
    times: NDArray[np.float64],
    values: NDArray[np.float64],
) -> np.float64:
    """Piecewise-linear interpolation; extrapolate with first/last value."""
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    i = int(np.searchsorted(times, t, side="right") - 1)
    j = i + 1
    t0, t1 = times[i], times[j]
    v0, v1 = values[i], values[j]
    if t1 == t0:
        return v1
    s = (t - t0) / (t1 - t0)
    return v0 + (v1 - v0) * s


@njit(cache=True)  # type: ignore
def closest_point_on_aabb(
    p: NDArray[np.float64],
    mins: NDArray[np.float64],
    maxs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Closest point on axis-aligned box to p."""
    return np.minimum(np.maximum(p, mins), maxs)


@njit(cache=True)  # type: ignore
def contact_force_circle_vs_aabb(
    center: NDArray[np.float64],
    radius: np.float64,
    mins: NDArray[np.float64],
    maxs: NDArray[np.float64],
    stiffness: np.float64,
    eps: np.float64,
) -> NDArray[np.float64]:
    """
    Penalty contact force between a circle and an axis-aligned rectangle.
    Returns a force (2,) pushing the circle center away from the rectangle.
    """
    closest = closest_point_on_aabb(center, mins, maxs)
    d = center - closest
    dist_sq = d[0] * d[0] + d[1] * d[1]
    dist = np.sqrt(dist_sq)

    if dist >= radius:
        return np.zeros(2, dtype=np.float64)

    if dist > eps:
        n = d / dist
        penetration = radius - dist
        return stiffness * penetration * n

    # Center inside rectangle: push out via nearest face
    dx_left = center[0] - mins[0]
    dx_right = maxs[0] - center[0]
    dy_bottom = center[1] - mins[1]
    dy_top = maxs[1] - center[1]
    dists = np.array([dx_left, dx_right, dy_bottom, dy_top], dtype=np.float64)
    idx = np.argmin(dists)
    min_dist = dists[idx]

    n = np.zeros(2, dtype=np.float64)
    if idx == 0:
        n[0], n[1] = -1.0, 0.0
    elif idx == 1:
        n[0], n[1] = 1.0, 0.0
    elif idx == 2:
        n[0], n[1] = 0.0, -1.0
    else:
        n[0], n[1] = 0.0, 1.0

    penetration = radius + min_dist
    return stiffness * penetration * n


@njit(cache=True)  # type: ignore
def wheel_velocity_2d(
    v: NDArray[np.float64],
    omega: np.float64,
    d1: NDArray[np.float64],
    half_width: np.float64,
    sign_d1: np.float64,
) -> NDArray[np.float64]:
    """Velocity at wheel contact: v + omega * sign_d1 * d1 * half_width."""
    out = np.empty(2, dtype=np.float64)
    out[0] = v[0] + omega * sign_d1 * d1[0] * half_width
    out[1] = v[1] + omega * sign_d1 * d1[1] * half_width
    return out


@njit(cache=True)  # type: ignore
def point_in_friction_region(
    x: NDArray[np.float64],
    fric_a: np.float64,
    fric_b: np.float64,
) -> bool:
    """True if y >= a*x + b (shaded region)."""
    return x[1] >= fric_a * x[0] + fric_b


@njit(cache=True)  # type: ignore
def compute_friction_force_mag_dir(
    v: NDArray[np.float64],
    speed: np.float64,
    mu_f: np.float64,
    mass: np.float64,
    fric_eps: np.float64,
) -> tuple[np.float64, NDArray[np.float64]]:
    """Momentum-based friction: f_mag = mu_f * mass * speed, f_dir = -v/|v|."""
    if speed <= 0.0:
        return np.float64(0.0), np.array([0.0, 0.0], dtype=np.float64)
    f_mag = mu_f * mass * speed
    f_dir = -v / (speed + fric_eps)
    return f_mag, f_dir


@njit(cache=True)  # type: ignore
def compute_single_wheel_friction(
    x_wheel: NDArray[np.float64],
    v: NDArray[np.float64],
    omega: np.float64,
    d1: NDArray[np.float64],
    half_width: np.float64,
    sign_d1: np.float64,
    fric_a: np.float64,
    fric_b: np.float64,
    mu_f: np.float64,
    mass: np.float64,
    fric_eps: np.float64,
) -> tuple[np.float64, NDArray[np.float64]]:
    """Friction (f_mag, f_dir) for one wheel; (0, zeros) if not in shaded region."""
    if not point_in_friction_region(x_wheel, fric_a, fric_b):
        return np.float64(0.0), np.array([0.0, 0.0], dtype=np.float64)
    v_wheel = wheel_velocity_2d(v, omega, d1, half_width, sign_d1)
    speed = np.sqrt(v_wheel[0] * v_wheel[0] + v_wheel[1] * v_wheel[1])
    return compute_friction_force_mag_dir(v_wheel, speed, mu_f, mass, fric_eps)


@njit(cache=True)  # type: ignore
def torque_z_from_force_2d(
    lever_arm: NDArray[np.float64],
    f_dir: NDArray[np.float64],
    f_mag: np.float64,
) -> np.float64:
    """Z-component of torque from force f_mag * f_dir at lever_arm (2D)."""
    return f_mag * (lever_arm[0] * f_dir[1] - lever_arm[1] * f_dir[0])


@njit(cache=True)  # type: ignore
def boundary_penetration_forces(
    center: NDArray[np.float64],
    radius: np.float64,
    xmin: np.float64,
    xmax: np.float64,
    ymin: np.float64,
    ymax: np.float64,
    k: np.float64,
) -> NDArray[np.float64]:
    """Penalty force (2,) to keep circle inside [xmin,xmax]x[ymin,ymax]."""
    out = np.zeros(2, dtype=np.float64)
    pen_left = (xmin + radius) - center[0]
    if pen_left > 0.0:
        out[0] += k * pen_left
    pen_right = center[0] - (xmax - radius)
    if pen_right > 0.0:
        out[0] -= k * pen_right
    pen_bottom = (ymin + radius) - center[1]
    if pen_bottom > 0.0:
        out[1] += k * pen_bottom
    pen_top = center[1] - (ymax - radius)
    if pen_top > 0.0:
        out[1] -= k * pen_top
    return out


# -----------------------------------------------------------------------------
# Force classes (delegate to numba free-functions above)
# -----------------------------------------------------------------------------


class ConstantForce(NoForces):
    """
    This class applies a constant thrust to the robot.
    """

    def __init__(
        self,
        left_wheel_force: NDArray[np.float64],
        right_wheel_force: NDArray[np.float64],
        duration: float,
    ) -> None:
        """

        Parameters
        ----------
        left_wheel_force: numpy.ndarray
            1D (dim) array containing data with 'float' type. Left wheel force vector.
        right_wheel_force: numpy.ndarray
            1D (dim) array containing data with 'float' type. Right wheel force vector.
        duration: float
            Duration of the force application in seconds.
        """
        super().__init__()
        self._left_wheel_force = left_wheel_force
        self._right_wheel_force = right_wheel_force
        self._duration = np.float64(duration)

    def apply_forces(self, system, time: np.float64 = np.float64(0.0)) -> None:
        if time < self._duration:
            compute_wheel_forces_to_external(
                system.direction[:, 0],
                self._left_wheel_force,
                self._right_wheel_force,
                system.width[0],
                system.external_forces[:, 0],
                system.external_torques,
            )


class OpenLoopForce(NoForces):
    """
    This class applies a sequence of thrust on the robot
    """

    def __init__(
        self,
        time_intervals: NDArray[np.float64],
        left_wheel_forces: NDArray[np.float64],
        right_wheel_forces: NDArray[np.float64],
    ) -> None:
        """
        Note, the integrity check for the time interval should be done outside.
        In other word, it is assumed here that time intervals are in-order, and they
        do not overlap.

        Parameters
        ----------
        time_intervals : NDArray[np.float64]
            (N, 2)
        left_wheel_forces : NDArray[np.float64]
            (N, 2)
        right_wheel_forces : NDArray[np.float64]
            (N, 2)
        """
        super().__init__()
        self._time_intervals = np.asarray(time_intervals, dtype=np.float64)
        self._left_wheel_forces = np.asarray(left_wheel_forces, dtype=np.float64)
        self._right_wheel_forces = np.asarray(right_wheel_forces, dtype=np.float64)

    def apply_forces(self, system, time: np.float64 = np.float64(0.0)) -> None:
        left_force, right_force = self._get_thrust(
            time,
            self._time_intervals,
            self._left_wheel_forces,
            self._right_wheel_forces,
        )
        compute_wheel_forces_to_external(
            system.direction[:, 0],
            left_force,
            right_force,
            system.width[0],
            system.external_forces[:, 0],
            system.external_torques,
        )

    @staticmethod
    @njit
    def _get_thrust(
        time: np.float64, time_intervals, left_wheel_forces, right_wheel_forces
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return wheel thrust vectors active at `time`, else zero vectors."""
        starts = time_intervals[:, 0]
        ends = time_intervals[:, 1]
        active = np.nonzero((starts <= time) & (time < ends))[0]
        if active.size == 0:
            return np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)
        idx = int(active[0])
        return left_wheel_forces[idx], right_wheel_forces[idx]


class PotentialFieldForce(NoForces):
    """
    Applies potential field forces via the control law:
    u_t^(l) = u_t^(r) = -K x_t ⋅ d_1,t

    Both wheels receive the same force in the forward (d_1) direction,
    pulling the robot toward the origin when K > 0.

    Attributes
    ----------
    K : float
        Stiffness parameter (N/m). Defaults to 0.5 if not provided.
    """

    def __init__(
        self,
        K: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        K: float
            Stiffness parameter (N/m). Defaults to 0.5.
        """
        super().__init__()
        self._K = np.float64(K)

    def apply_forces(self, system, time: np.float64 = np.float64(0.0)) -> None:
        """
        Apply potential field forces: u = -K x_t ⋅ d_1,t, then apply
        left_wheel_force = right_wheel_force = [u, 0] in body frame.
        """
        left_wheel_force, right_wheel_force = compute_potential_field_wheel_forces(
            system.position[:, 0], system.direction[:, 0], self._K
        )
        compute_wheel_forces_to_external(
            system.direction[:, 0],
            left_wheel_force,
            right_wheel_force,
            system.width[0],
            system.external_forces[:, 0],
            system.external_torques,
        )


class WheelForceSequence(NoForces):
    """
    Apply a time-varying open-loop wheel force command sequence.

    The left/right wheel forces are specified in the body frame as scalars along d1:
      left_wheel_force  = [u_l(t), 0]
      right_wheel_force = [u_r(t), 0]

    A piecewise-linear interpolation is used between knot points, and both
    wheel commands are set to 0 for time >= stop_time (default: 4s for Task 3).
    """

    def __init__(
        self,
        times: NDArray[np.float64],
        left_u: NDArray[np.float64],
        right_u: NDArray[np.float64],
        *,
        stop_time: float | None = None,
    ) -> None:
        super().__init__()
        self._times = np.asarray(times, dtype=np.float64)
        self._left_u = np.asarray(left_u, dtype=np.float64)
        self._right_u = np.asarray(right_u, dtype=np.float64)
        self._stop_time = np.float64(stop_time) if stop_time is not None else np.inf

    def apply_forces(self, system, time: np.float64 = np.float64(0.0)) -> None:
        if time >= self._stop_time:
            return

        ul = interp_piecewise_linear(np.float64(time), self._times, self._left_u)
        ur = interp_piecewise_linear(np.float64(time), self._times, self._right_u)

        left_wheel_force = np.array([ul, 0.0], dtype=np.float64)
        right_wheel_force = np.array([ur, 0.0], dtype=np.float64)
        compute_wheel_forces_to_external(
            system.direction[:, 0],
            left_wheel_force,
            right_wheel_force,
            system.width[0],
            system.external_forces[:, 0],
            system.external_torques,
        )


class EnvironmentForces2D(NoForces):
    """
    Environment forcing for Task 3:
    - Position-dependent kinetic (ground) friction in a shaded region (mu_f).
    - Penalty contact forces against boundary walls and rectangular obstacles (mu_c).
    """

    def __init__(
        self,
        mu_f: float = 0.5,
        mu_c: float = 1000.0,
        *,
        g: float = 9.81,
        bounds: tuple[float, float, float, float] = (0.0, 3.0, 0.0, 4.0),
        obstacles: tuple[tuple[float, float, float, float], ...] = (
            (0.0, 0.6, 1.0, 3.0),  # left intrusion
            (2.4, 3.0, 1.0, 3.0),  # right intrusion
        ),
        friction_line: tuple[float, float] = (-4.0 / 3.0, 4.0),  # y = a*x + b
        friction_eps: float = 1e-12,
        callback_params: dict = {},
    ) -> None:
        super().__init__()
        self._mu_f = np.float64(mu_f)
        self._mu_c = np.float64(mu_c)
        self._g = np.float64(g)
        self._bounds = tuple(np.float64(v) for v in bounds)
        self._obstacles = obstacles
        self._fric_a = np.float64(friction_line[0])
        self._fric_b = np.float64(friction_line[1])
        self._fric_eps = float(friction_eps)

        self.callback_params = callback_params

    def apply_forces(self, system, time: np.float64 = np.float64(0.0)) -> None:
        x = system.position[:, 0]
        d1 = system.direction[:, 0]
        R = np.array(
            [
                [np.cos(np.pi / 2), -np.sin(np.pi / 2)],
                [np.sin(np.pi / 2), np.cos(np.pi / 2)],
            ]
        )
        d2 = R @ d1
        half_width = np.float64(system.width[0] / 2.0)
        x_left = x + d2 * half_width
        x_right = x - d2 * half_width
        v = system.velocity[:, 0]
        omega = system.omega[0]

        fric_eps = np.float64(self._fric_eps)

        # Right wheel friction (sign_d1 = +1: v_right = v + omega * d1 * half_width)
        f_mag_r, f_dir_r = compute_single_wheel_friction(
            x_right,
            v,
            omega,
            d1,
            half_width,
            np.float64(1.0),
            self._fric_a,
            self._fric_b,
            self._mu_f,
            np.float64(system.mass[0]),
            fric_eps,
        )
        system.external_forces[:, 0] += f_mag_r * f_dir_r
        system.external_torques[0] += torque_z_from_force_2d(
            -d2 * half_width, f_dir_r, f_mag_r
        )
        self.callback_params["right_friction_force_mag"].append(float(f_mag_r))
        self.callback_params["right_friction_force_dir"].append(f_dir_r.copy())

        # Left wheel friction (sign_d1 = -1)
        f_mag_l, f_dir_l = compute_single_wheel_friction(
            x_left,
            v,
            omega,
            d1,
            half_width,
            np.float64(-1.0),
            self._fric_a,
            self._fric_b,
            self._mu_f,
            np.float64(system.mass[0]),
            fric_eps,
        )
        system.external_forces[:, 0] += f_mag_l * f_dir_l
        system.external_torques[0] += torque_z_from_force_2d(
            d2 * half_width, f_dir_l, f_mag_l
        )
        self.callback_params["left_friction_force_mag"].append(float(f_mag_l))
        self.callback_params["left_friction_force_dir"].append(f_dir_l.copy())

        # Boundary penalty
        xmin, xmax, ymin, ymax = self._bounds
        r = np.float64(system.radius[0])
        k = self._mu_c
        system.external_forces[:, 0] += boundary_penetration_forces(
            x, r, xmin, xmax, ymin, ymax, k
        )

        # Obstacle penalty
        for oxmin, oxmax, oymin, oymax in self._obstacles:
            mins = np.array([oxmin, oymin], dtype=np.float64)
            maxs = np.array([oxmax, oymax], dtype=np.float64)
            system.external_forces[:, 0] += contact_force_circle_vs_aabb(
                x, r, mins, maxs, k, fric_eps
            )
