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
