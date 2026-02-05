import numpy as np
from numpy.typing import NDArray
from numba import njit

import elastica as ea
from elastica._linalg import _batch_matvec, _batch_cross

from .sphere import Sphere


class SphereImplicit(Sphere):
    dt: np.float64

    def update_accelerations(self, time: np.float64) -> None:
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
            residual = np.max(np.linalg.norm(omega_new - omega_np1, axis=-1))
            omega_np1 = omega_new
            if residual < tol:
                break
        np.copyto(self.alpha_collection, alpha_mid)
        # np.copyto(self.omega_collection, omega_mid) # Note: this will be done in update_dynamics
