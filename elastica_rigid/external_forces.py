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
        force: NDArray[np.float64],
        duration: float,
    ) -> None:
        """

        Parameters
        ----------
        force: numpy.ndarray
            1D (dim) array containing data with 'float' type. Gravitational acceleration vector.
            Defaults to [0.0, -9.80665, 0.0] if not provided.

        """
        super().__init__()
        self._force = force
        self._duration = np.float64(duration)

    def apply_forces(self, system, time: np.float64 = np.float64(0.0)) -> None:
        if time < self._duration:
            self._compute_forces(self._force, system.mass, system.external_forces)

    @staticmethod
    @njit(cache=True)  # type: ignore
    def _compute_forces(
        force: NDArray[np.float64],
        mass: NDArray[np.float64],
        external_forces: NDArray[np.float64],
    ) -> None:
        external_forces += force


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
