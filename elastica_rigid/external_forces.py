import numpy as np
from numpy.typing import NDArray

from numba import njit
from elastica.external_forces import NoForces


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
            self._compute_forces(
                system.direction,
                self._left_wheel_force,
                self._right_wheel_force,
                system.width,
                system.external_forces,
                system.external_torques,
            )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _compute_forces(
        direction: NDArray[np.float64],
        left_wheel_force: NDArray[np.float64],
        right_wheel_force: NDArray[np.float64],
        track_width: np.float64,
        external_forces: NDArray[np.float64],
        external_torques: NDArray[np.float64],
    ) -> None:
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


class PotentialFieldForce(NoForces):
    """
    Abstract class for applying potential field forces to a robot.
    This class provides the structure for implementing control laws like:
    u_t^(l) = u_t^(r) = -Kx_t ⋅ d_1,t

    The implementation details will be added later.

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
        self.K = np.float64(K)

    def apply_forces(self, system, time: np.float64 = np.float64(0.0)) -> None:
        """
        Apply potential field forces to the system.

        This is an abstract implementation - details to be filled in later.
        The control law u_t^(l) = u_t^(r) = -Kx_t ⋅ d_1,t will be implemented here.

        Parameters
        ----------
        system: RodType | RigidBodyType
            The system to apply forces to.
        time: float
            Current simulation time.
        """
        # Abstract implementation - to be filled in later
        # Will compute: u_t^(l) = u_t^(r) = -Kx_t ⋅ d_1,t
        # and apply forces to left and right wheels
        pass
