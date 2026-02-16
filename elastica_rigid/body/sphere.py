import numpy as np
from numpy.typing import NDArray
from numba import njit

import elastica as ea
from elastica._rotations import _get_rotation_matrix
from elastica._linalg import _batch_matmul


class Sphere(ea.Sphere):
    def update_kinematics(
        self,
        time: np.float64,
        prefac: np.float64,
    ) -> None:
        """
        Update kinematic state.

        Typically called after velocity and omega (angular velocity) have been updated.

        Parameters
        ----------
        time : float
            Current time.
        prefac : float
            Integration prefactor.
        """
        overload_operator_kinematic_numba(
            prefac,
            self.position_collection,
            self.director_collection,
            self.velocity_collection,
            self.omega_collection,
        )

    def update_dynamics(
        self,
        time: np.float64,
        prefac: np.float64,
    ) -> None:
        """
        Update dynamic state.

        Typically called after acceleration and alpha (angular acceleration) have been updated.

        Parameters
        ----------
        time : float
            Current time.
        prefac : float
            Integration prefactor.
        """
        overload_operator_dynamic_numba(
            prefac,
            self.velocity_collection,
            self.omega_collection,
            self.acceleration_collection,
            self.alpha_collection,
        )


"""
Symplectic stepper operation
"""


@njit(cache=True)  # type: ignore
def overload_operator_kinematic_numba(
    prefac: np.float64,
    position_collection: NDArray[np.float64],
    director_collection: NDArray[np.float64],
    velocity_collection: NDArray[np.float64],
    omega_collection: NDArray[np.float64],
) -> None:
    """Performs in-place update of kinematic states (position and director) using Numba.

    This operator updates the position and director collections of a rod based on
    its velocity and angular velocity. The director update uses Rodrigues' rotation
    formula.

    Parameters
    ----------
    prefac : numpy.float64
        Pre-factor (e.g., time step `dt`) to scale the velocity and angular velocity.
    position_collection : numpy.ndarray
        Position of the rod nodes. Modified in-place.
    director_collection : numpy.ndarray
        Director (orientation) of the rod elements. Modified in-place.
    velocity_collection : numpy.ndarray
        Linear velocity of the rod nodes.
    omega_collection : numpy.ndarray
        Angular velocity of the rod elements.
    """
    # x += v*dt
    blocksize = position_collection.shape[1]
    for i in range(3):
        for k in range(blocksize):
            position_collection[i, k] += prefac * velocity_collection[i, k]
    rotation_matrix = _get_rotation_matrix(prefac, omega_collection)
    director_collection[:] = _batch_matmul(rotation_matrix, director_collection)


@njit(cache=True)  # type: ignore
def overload_operator_dynamic_numba(
    prefac: np.float64,
    velocity_collection: NDArray[np.float64],
    omega_collection: NDArray[np.float64],
    acceleration_collection: NDArray[np.float64],
    alpha_collection: NDArray[np.float64],
) -> None:
    """Performs in-place update of dynamic states (linear and angular velocities) using Numba.

    This operator updates the rate collection (which stores linear and angular velocities)
    of a rod based on the second derivative array (linear and angular accelerations).

    Parameters
    ----------
    prefac : numpy.float64
        Pre-factor (e.g., time step `dt`) to scale the second derivative terms.
    """
    # Always goes in LHS : that means the update is on the rates alone
    # (v,ω) += dt * (dv/dt, dω/dt)
    # rate_collection[..., : n_kinematic_rates] += second_deriv_aray
    blocksize = velocity_collection.shape[1]
    for i in range(3):
        for k in range(blocksize):
            velocity_collection[i, k] += prefac * acceleration_collection[i, k]
    blocksize = omega_collection.shape[1]
    for i in range(3):
        for k in range(blocksize):
            omega_collection[i, k] += prefac * alpha_collection[i, k]

# Variations of the Sphere class implementations
from .mixin.exact_angular_momentum_formula import WithExactAngMomentum

class SphereExact(WithExactAngMomentum, Sphere):
    pass

from .mixin.midpoint_angular_momentum import WithMidpointAngMomentum

class SphereImplicit(WithMidpointAngMomentum, Sphere):
    pass