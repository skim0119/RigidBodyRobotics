from typing import Any, Optional, Type
from numpy.typing import NDArray

import numpy as np
import numba


@numba.njit(cache=True)  # type: ignore
def _update_rigid_SO2_kinematic_state(
    prefac: np.float64,
    position: NDArray[np.float64],
    velocity: NDArray[np.float64],
    direction: NDArray[np.float64],
    omega: NDArray[np.float64],
) -> None:
    position += prefac * velocity

    # Supports both single-body (shape (2,1), omega shape (1,))
    # and memory-block arrays (shape (2,N), omega shape (N,)).
    n = direction.shape[1]
    for k in range(n):
        theta = prefac * omega[k]
        c = np.cos(theta)
        s = np.sin(theta)
        x = direction[0, k]
        y = direction[1, k]
        direction[0, k] = c * x - s * y
        direction[1, k] = s * x + c * y


@numba.njit(cache=True)  # type: ignore
def _update_rigid_SO2_dynamic_state(
    prefac: np.float64,
    velocity: NDArray[np.float64],
    acceleration: NDArray[np.float64],
    omega: NDArray[np.float64],
    alpha: NDArray[np.float64],
):
    velocity += prefac * acceleration
    omega += prefac * alpha


@numba.njit(cache=True)  # type: ignore
def _update_accelerations(
    acceleration: NDArray[np.float64],
    alpha: NDArray[np.float64],
    mass: NDArray[np.float64],
    inertia: NDArray[np.float64],
    external_forces: NDArray[np.float64],
    external_torques: NDArray[np.float64],
) -> None:
    """
    Update <acceleration and angular acceleration> given <internal force/torque and external force/torque>.
    """

    acceleration[:] = external_forces / mass[0]
    alpha[:] = external_torques / inertia[0]


@numba.njit(cache=True)  # type: ignore
def _zeroed_out_external_forces_and_torques(
    external_forces: NDArray[np.float64], external_torques: NDArray[np.float64]
) -> None:
    """
    This function is to zeroed out external forces and torques.
    """
    external_forces[:] = 0.0
    external_torques[:] = 0.0
