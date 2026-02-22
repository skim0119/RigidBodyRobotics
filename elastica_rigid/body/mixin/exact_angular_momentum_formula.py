from numpy.typing import NDArray

import numpy as np

from elastica.rigidbody import RigidBodyBase
from elastica._linalg import _batch_matvec
from elastica_rigid._rotations import _rotate_vector

from ..sphere import Sphere
from .utils import only_mix_into


@only_mix_into(Sphere, RigidBodyBase)
class WithExactAngMomentum:
    """
    Mixin class to update the angular acceleration using the exact angular momentum formula.

    Mathematically, this implementation should be exactly same as the original
    implementation of update_accleeration using Langrange Transport term.
    The implementation here is done to validate the mathematical derivative in SE(3) terms.
    """

    # Required attributes:
    acceleration_collection: NDArray[np.float64]
    external_forces: NDArray[np.float64]
    mass_second_moment_of_inertia: NDArray[np.float64]
    inv_mass_second_moment_of_inertia: NDArray[np.float64]
    omega_collection: NDArray[np.float64]
    alpha_collection: NDArray[np.float64]
    mass: np.float64

    def update_accelerations(self, time: np.float64, dt: np.float64) -> None:
        # Linear acceleration update
        np.copyto(
            self.acceleration_collection,
            (self.external_forces) / self.mass,
        )

        # I apply common sub expression elimination here, as J w
        current_angular_momentum = _batch_matvec(
            self.mass_second_moment_of_inertia, self.omega_collection
        )
        co_adjointed_angular_momentum = _rotate_vector(
            current_angular_momentum,
            scale=1.0,
            axis_collection=self.omega_collection * dt,
        )
        momentum_change = (
            _batch_matvec(
                self.inv_mass_second_moment_of_inertia,
                co_adjointed_angular_momentum,
            )
            - self.omega_collection
        )
        self.alpha_collection[:] = momentum_change / dt
