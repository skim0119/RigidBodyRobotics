import numpy as np
from numpy.typing import NDArray

from numba import njit
from elastica.external_forces import NoForces


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
    external_torques += (track_width / 2) * (-left_wheel_force + right_wheel_force)


class ConstantForce(NoForces):
    """
    This class applies a constant gravitational force to the entire rod.
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
                system.direction,
                self._left_wheel_force,
                self._right_wheel_force,
                system.width,
                system.external_forces,
                system.external_torques,
            )


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
        x = system.position
        d1 = system.direction
        u = -self._K * np.dot(x, d1)
        left_wheel_force = np.array([u, 0.0], dtype=np.float64)
        right_wheel_force = np.array([u, 0.0], dtype=np.float64)
        compute_wheel_forces_to_external(
            system.direction,
            left_wheel_force,
            right_wheel_force,
            system.width,
            system.external_forces,
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
        stop_time: float = 4.0,
    ) -> None:
        super().__init__()
        self._times = np.asarray(times, dtype=np.float64)
        self._left_u = np.asarray(left_u, dtype=np.float64)
        self._right_u = np.asarray(right_u, dtype=np.float64)
        self._stop_time = np.float64(stop_time)

        if self._times.ndim != 1:
            raise ValueError("times must be a 1D array")
        if self._left_u.shape != self._times.shape or self._right_u.shape != self._times.shape:
            raise ValueError("left_u/right_u must have the same shape as times")
        if self._times.size < 2:
            raise ValueError("times must have at least two knot points")
        if not np.all(self._times[1:] >= self._times[:-1]):
            raise ValueError("times must be non-decreasing")

    @staticmethod
    def _interp_piecewise_linear(
        t: np.float64, times: NDArray[np.float64], values: NDArray[np.float64]
    ) -> np.float64:
        if t <= times[0]:
            return np.float64(values[0])
        if t >= times[-1]:
            return np.float64(values[-1])
        # Right-continuous selection of the current knot
        i = int(np.searchsorted(times, t, side="right") - 1)
        j = i + 1
        t0 = times[i]
        t1 = times[j]
        v0 = values[i]
        v1 = values[j]
        if t1 == t0:
            # Degenerate interval; take right value
            return np.float64(v1)
        s = (t - t0) / (t1 - t0)
        return np.float64(v0 + (v1 - v0) * s)

    def apply_forces(self, system, time: np.float64 = np.float64(0.0)) -> None:
        if time >= self._stop_time:
            return

        ul = self._interp_piecewise_linear(np.float64(time), self._times, self._left_u)
        ur = self._interp_piecewise_linear(np.float64(time), self._times, self._right_u)

        left_wheel_force = np.array([ul, 0.0], dtype=np.float64)
        right_wheel_force = np.array([ur, 0.0], dtype=np.float64)
        compute_wheel_forces_to_external(
            system.direction,
            left_wheel_force,
            right_wheel_force,
            system.width,
            system.external_forces,
            system.external_torques,
        )


def _closest_point_on_aabb(
    p: NDArray[np.float64],
    mins: NDArray[np.float64],
    maxs: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.minimum(np.maximum(p, mins), maxs)


def _contact_force_circle_vs_aabb(
    center: NDArray[np.float64],
    radius: np.float64,
    mins: NDArray[np.float64],
    maxs: NDArray[np.float64],
    stiffness: np.float64,
    *,
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """
    Penalty contact force between a circle and an axis-aligned rectangle.

    Returns a force that pushes the circle center away from the rectangle if overlapping.
    """
    closest = _closest_point_on_aabb(center, mins, maxs)
    d = center - closest
    dist = float(np.sqrt(d[0] * d[0] + d[1] * d[1]))

    if dist >= float(radius):
        return np.zeros((2,), dtype=np.float64)

    if dist > eps:
        n = d / dist
        penetration = float(radius) - dist
        return stiffness * penetration * n

    # Center is inside rectangle (or extremely close). Push out via nearest face.
    dx_left = float(center[0] - mins[0])
    dx_right = float(maxs[0] - center[0])
    dy_bottom = float(center[1] - mins[1])
    dy_top = float(maxs[1] - center[1])
    min_dist = min(dx_left, dx_right, dy_bottom, dy_top)

    if min_dist == dx_left:
        n = np.array([-1.0, 0.0], dtype=np.float64)
    elif min_dist == dx_right:
        n = np.array([1.0, 0.0], dtype=np.float64)
    elif min_dist == dy_bottom:
        n = np.array([0.0, -1.0], dtype=np.float64)
    else:
        n = np.array([0.0, 1.0], dtype=np.float64)

    penetration = float(radius) + min_dist
    return stiffness * penetration * n


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
            (0.0, 0.6, 1.0, 3.0),   # left intrusion
            (2.4, 3.0, 1.0, 3.0),   # right intrusion
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
        # --- Kinetic friction (shaded region) ---
        x = system.position
        d1 = system.direction
        R = np.array([[np.cos(np.pi / 2), -np.sin(np.pi / 2)], [np.sin(np.pi / 2), np.cos(np.pi / 2)]])
        d2 = R @ d1
        x_left = x + d2 * system.width / 2.0
        x_right = x - d2 * system.width / 2.0
        v = system.velocity
        omega = system.omega[0]

        # Right wheel friction
        # shaded if y >= a*x + b (Fig. 3)
        f_mag = 0.0
        f_dir = np.zeros((2,), dtype=np.float64)
        if float(x_right[1]) >= float(self._fric_a * x_right[0] + self._fric_b):
            v_right = v + omega * d1 * system.width / 2.0
            speed = float(np.linalg.norm(v_right))
            if speed > 0.0:
                # f_mag = float(self._mu_f * system.mass * self._g)  # weight based friction
                f_mag = float(self._mu_f * system.mass * speed)  # momentum based friction (dissipation)
                f_dir = -v_right / (speed + self._fric_eps)
                system.external_forces += f_mag * f_dir
                system.external_torques += f_mag * np.cross(system.width / 2.0 * (-d2), f_dir)
        self.callback_params["right_friction_force_mag"].append(f_mag)
        self.callback_params["right_friction_force_dir"].append(f_dir)
        
        # Left wheel friction
        # shaded if y >= a*x + b (Fig. 3)
        f_mag = 0.0
        f_dir = np.zeros((2,), dtype=np.float64)
        if float(x_left[1]) >= float(self._fric_a * x_left[0] + self._fric_b):
            v_left = v + omega * (-d1) * system.width / 2.0
            speed = float(np.linalg.norm(v_left))
            if speed > 0.0:
                # f_mag = float(self._mu_f * system.mass * self._g)  # weight based friction
                f_mag = float(self._mu_f * system.mass * speed)  # momentum based friction (dissipation)
                f_dir = -v_left / (speed + self._fric_eps)
                system.external_forces += f_mag * f_dir
                system.external_torques += f_mag * np.cross(system.width / 2.0 * d2, f_dir)
        self.callback_params["left_friction_force_mag"].append(f_mag)
        self.callback_params["left_friction_force_dir"].append(f_dir)
        

        # --- Boundary collision: keep robot inside [xmin,xmax]x[ymin,ymax] ---
        xmin, xmax, ymin, ymax = self._bounds
        r = np.float64(system.radius)
        k = self._mu_c

        # Left wall
        pen = float((xmin + r) - x[0])
        if pen > 0.0:
            system.external_forces += k * pen * np.array([1.0, 0.0], dtype=np.float64)
        # Right wall
        pen = float(x[0] - (xmax - r))
        if pen > 0.0:
            system.external_forces += k * pen * np.array([-1.0, 0.0], dtype=np.float64)
        # Bottom wall
        pen = float((ymin + r) - x[1])
        if pen > 0.0:
            system.external_forces += k * pen * np.array([0.0, 1.0], dtype=np.float64)
        # Top wall
        pen = float(x[1] - (ymax - r))
        if pen > 0.0:
            system.external_forces += k * pen * np.array([0.0, -1.0], dtype=np.float64)

        # --- Obstacle collision: keep robot outside rectangles ---
        for (oxmin, oxmax, oymin, oymax) in self._obstacles:
            mins = np.array([oxmin, oymin], dtype=np.float64)
            maxs = np.array([oxmax, oymax], dtype=np.float64)
            system.external_forces += _contact_force_circle_vs_aabb(
                x, r, mins, maxs, k, eps=self._fric_eps
            )
