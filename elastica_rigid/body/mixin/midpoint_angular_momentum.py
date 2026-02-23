import numpy as np
from numpy.typing import NDArray

from elastica.rigidbody import RigidBodyBase
from elastica._linalg import _batch_matvec, _batch_cross

from ..sphere import Sphere
from .utils import only_mix_into

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

        # Implicit mid-omega solver
        omega_n = self.omega_collection.copy()
        omega_np1 = self.omega_collection.copy() + dt * _batch_matvec(
            self.inv_mass_second_moment_of_inertia,
            (
                _batch_cross(
                    _batch_matvec(self.mass_second_moment_of_inertia, omega_n), omega_n
                )
                + self.external_torques
            ),
        )

        tol = 1e-10
        max_iters = 20

        for it in range(max_iters):
            # midpoint omega
            omega_mid = 0.5 * (omega_n + omega_np1)

            j_omega_mid = _batch_matvec(self.mass_second_moment_of_inertia, omega_mid)
            lagrangian_transport_mid = _batch_cross(j_omega_mid, omega_mid)
            alpha_mid = _batch_matvec(
                self.inv_mass_second_moment_of_inertia,
                (lagrangian_transport_mid + self.external_torques),
            )

            omega_new = omega_n + dt * alpha_mid

            # convergence check
            # NOTE: this is a fixed-point update residual, not the equation residual.
            residual = np.max(np.linalg.norm(omega_new - omega_np1, axis=-1))
            omega_np1 = omega_new
            if residual < tol:
                break

        # Compute the true implicit midpoint equation residual at the converged omega_{n+1}:
        # F(omega_{n+1}) = omega_{n+1} - omega_n - dt * alpha( (omega_n + omega_{n+1})/2 ) = 0
        omega_mid = 0.5 * (omega_n + omega_np1)
        j_omega_mid = _batch_matvec(self.mass_second_moment_of_inertia, omega_mid)
        lagrangian_transport_mid = _batch_cross(j_omega_mid, omega_mid)
        alpha_mid = _batch_matvec(
            self.inv_mass_second_moment_of_inertia,
            (lagrangian_transport_mid + self.external_torques),
        )
        equation_residual_vec = omega_np1 - (omega_n + dt * alpha_mid)
        self.implicit_midpoint_equation_residual = np.max(
            np.linalg.norm(equation_residual_vec, axis=-1)
        )
        self.implicit_midpoint_iterations = it + 1

        np.copyto(self.alpha_collection, alpha_mid)
        # np.copyto(self.omega_collection, omega_mid) # Note: this will be done in update_dynamics
