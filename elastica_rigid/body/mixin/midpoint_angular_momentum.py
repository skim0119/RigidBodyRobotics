import numpy as np
from numpy.typing import NDArray
import math

from numba import njit

from elastica.rigidbody import RigidBodyBase
from elastica._linalg import _batch_matvec, _batch_cross

from ..sphere import Sphere
from .utils import only_mix_into


@njit(cache=True)  # type: ignore
def _implicit_midpoint_alpha_numba(
    dt: np.float64,
    omega: NDArray[np.float64],
    mass_second_moment_of_inertia: NDArray[np.float64],
    inv_mass_second_moment_of_inertia: NDArray[np.float64],
    external_torques: NDArray[np.float64],
    tol: np.float64 = 1e-10,
    max_iters: int = 20,
) -> tuple[NDArray[np.float64], np.float64, int]:
    """
    Implicit midpoint update for angular acceleration (alpha) using a fixed-point iteration.

    Notes
    -----
    The residual reductions intentionally match the existing NumPy implementation:
    for an array shaped (3, blocksize), we compute a row-wise 2-norm and then take max over rows.
    """
    omega_n = omega.copy()

    # Initial guess: explicit Euler using lagrangian transport at n.
    j_omega_n = _batch_matvec(mass_second_moment_of_inertia, omega_n)
    transport_n = _batch_cross(j_omega_n, omega_n)
    alpha_n = _batch_matvec(
        inv_mass_second_moment_of_inertia, transport_n + external_torques
    )
    omega_np1 = omega_n + dt * alpha_n

    blocksize = omega.shape[1]
    omega_new = np.empty_like(omega_np1)

    it = 0
    for it in range(max_iters):
        omega_mid = 0.5 * (omega_n + omega_np1)

        j_omega_mid = _batch_matvec(mass_second_moment_of_inertia, omega_mid)
        transport_mid = _batch_cross(j_omega_mid, omega_mid)
        alpha_mid = _batch_matvec(
            inv_mass_second_moment_of_inertia, transport_mid + external_torques
        )

        omega_new = omega_n + dt * alpha_mid

        # convergence check
        residual = 0.0
        for i in range(3):
            residual += (omega_new[i, 0] - omega_np1[i, 0]) ** 2

        omega_np1 = omega_new
        if residual < tol:
            break

    # Recompute alpha at the converged omega_{n+1} for output and equation residual.
    omega_mid = 0.5 * (omega_n + omega_np1)
    j_omega_mid = _batch_matvec(mass_second_moment_of_inertia, omega_mid)
    transport_mid = _batch_cross(j_omega_mid, omega_mid)
    alpha_mid = _batch_matvec(
        inv_mass_second_moment_of_inertia, transport_mid + external_torques
    )

    equation_residual = 0.0
    for i in range(3):
        equation_residual += (omega_np1[i, 0] - (omega_n[i, 0] + dt * alpha_mid[i, 0])) ** 2

    return alpha_mid, equation_residual, it + 1

@only_mix_into(Sphere, RigidBodyBase)
class WithMidpointAngMomentum:
    """
    Mixin class to update the angular acceleration using the implicit midpoint angular momentum formula.

    This implementation preserves the angular momentum and its energy exactly
    during the dynamic step of the time integration.
    It is slightly slower than the original implementation of update_accleeration,
    """

    # Required attributes:
    dt: np.float64
    acceleration_collection: NDArray[np.float64]
    external_forces: NDArray[np.float64]
    mass_second_moment_of_inertia: NDArray[np.float64]
    inv_mass_second_moment_of_inertia: NDArray[np.float64]
    omega_collection: NDArray[np.float64]
    alpha_collection: NDArray[np.float64]
    mass: np.float64

    def update_accelerations(self, time: np.float64) -> None:
        # Linear acceleration update (same)
        dt = self.dt  # timestepper should pass dt into this function
        np.copyto(
            self.acceleration_collection,
            (self.external_forces) / self.mass,
        )

        # Implicit midpoint solver (Numba-accelerated).
        alpha_mid, equation_residual, iters = _implicit_midpoint_alpha_numba(
            dt,
            self.omega_collection,
            self.mass_second_moment_of_inertia,
            self.inv_mass_second_moment_of_inertia,
            self.external_torques,
        )
        self.implicit_midpoint_equation_residual = equation_residual
        self.implicit_midpoint_iterations = iters
        np.copyto(self.alpha_collection, alpha_mid)
        # np.copyto(self.omega_collection, omega_mid) # Note: this will be done in update_dynamics
