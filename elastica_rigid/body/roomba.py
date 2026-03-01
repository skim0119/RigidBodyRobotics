from typing import Optional, Type, Union

import numpy as np
from numpy.typing import NDArray

from elastica.systems.protocol import SystemProtocol

from ._se2_equations import (
    _update_accelerations,
    _update_rigid_SO2_dynamic_state,
    _update_rigid_SO2_kinematic_state,
    _zeroed_out_external_forces_and_torques,
)


class SE2RigidBody(SystemProtocol):
    """Planar rigid body in SE(2) with force/torque actuation."""

    REQUISITE_MODULES: list[Type] = []

    def __init__(
        self,
        position: NDArray[np.float64],
        direction: NDArray[np.float64],
        mass: float,
        inertia: float,
        *,
        initial_velocity: Optional[NDArray[np.float64]] = None,
        initial_acceleration: Optional[NDArray[np.float64]] = None,
        initial_omega: Optional[Union[float, NDArray[np.float64]]] = None,
        initial_alpha: Optional[Union[float, NDArray[np.float64]]] = None,
    ) -> None:
        """
        Initialize an SE(2) rigid body.

        Parameters
        ----------
        position : np.ndarray
            Initial position in world frame.
        direction : np.ndarray
            Initial heading vector in world frame.
        mass : float
            Body mass.
        inertia : float
            Planar rotational inertia.
        """
        self.mass = np.array([float(mass)], dtype=np.float64)
        self.inertia = np.array([float(inertia)], dtype=np.float64)

        self.position = np.zeros((2, 1), dtype=np.float64)
        self.position[:, 0] = _assert_vector2(position, "position")
        self.direction = np.zeros((2, 1), dtype=np.float64)
        self.direction[:, 0] = _assert_vector2(direction, "direction")
        direction_norm = float(np.linalg.norm(self.direction[:, 0]))
        if direction_norm > 1e-12:
            self.direction[:, 0] /= direction_norm
        self.velocity = np.zeros((2, 1), dtype=np.float64)
        if initial_velocity is not None:
            self.velocity[:, 0] = _assert_vector2(initial_velocity, "initial_velocity")
        self.acceleration = np.zeros((2, 1), dtype=np.float64)
        if initial_acceleration is not None:
            self.acceleration[:, 0] = _assert_vector2(
                initial_acceleration, "initial_acceleration"
            )
        self.omega = np.zeros((1,), dtype=np.float64)
        if initial_omega is not None:
            self.omega[0] = _assert_scalar1(initial_omega, "initial_omega")
        self.alpha = np.zeros((1,), dtype=np.float64)
        if initial_alpha is not None:
            self.alpha[0] = _assert_scalar1(initial_alpha, "initial_alpha")

        self.external_forces = np.zeros((2, 1), dtype=np.float64)
        self.external_torques = np.zeros((1,), dtype=np.float64)

    @classmethod
    def create_body(
        cls,
        initial_position: NDArray[np.float64],
        initial_direction: NDArray[np.float64],
        mass: float,
        inertia: float,
    ) -> "SE2RigidBody":
        """Create a stationary SE(2) rigid body."""
        return cls(
            position=initial_position,
            direction=initial_direction,
            mass=mass,
            inertia=inertia,
        )

    def zeroed_out_external_forces_and_torques(self, time: np.float64) -> None:
        _zeroed_out_external_forces_and_torques(
            self.external_forces,
            self.external_torques,
        )

    def compute_internal_forces_and_torques(self, time: np.float64) -> None:
        pass

    def update_accelerations(self, time: np.float64, dt: np.float64) -> None:
        _update_accelerations(
            self.acceleration,
            self.alpha,
            self.mass,
            self.inertia,
            self.external_forces,
            self.external_torques,
        )

    def update_dynamics(self, time: np.float64, dt: np.float64) -> None:
        _update_rigid_SO2_dynamic_state(
            np.float64(dt),
            self.velocity,
            self.acceleration,
            self.omega,
            self.alpha,
        )

    def update_kinematics(self, time: np.float64, dt: np.float64) -> None:
        _update_rigid_SO2_kinematic_state(
            np.float64(dt),
            self.position,
            self.velocity,
            self.direction,
            self.omega,
        )

    def compute_translational_kinetic_energy(self) -> float:
        return 0.5 * self.mass[0] * np.dot(self.velocity[:, 0], self.velocity[:, 0])

    def compute_rotational_kinetic_energy(self) -> float:
        return 0.5 * self.inertia[0] * self.omega[0] ** 2

    def compute_kinetic_energy(self) -> float:
        return (
            self.compute_translational_kinetic_energy()
            + self.compute_rotational_kinetic_energy()
        )


class Roomba(SE2RigidBody):
    """
    Roomba-like SE(2) rigid body.

    Extends `SE2RigidBody` with wheel geometry:
    - `radius`: wheel radius
    - `width`: track width (distance between wheel contact lines)
    """

    def __init__(
        self,
        position: NDArray[np.float64],
        direction: NDArray[np.float64],
        mass: float,
        inertia: float,
        radius: float,
        width: float,
        *,
        initial_velocity: Optional[NDArray[np.float64]] = None,
        initial_acceleration: Optional[NDArray[np.float64]] = None,
        initial_omega: Optional[Union[float, NDArray[np.float64]]] = None,
        initial_alpha: Optional[Union[float, NDArray[np.float64]]] = None,
    ) -> None:
        super().__init__(
            position=position,
            direction=direction,
            mass=mass,
            inertia=inertia,
            initial_velocity=initial_velocity,
            initial_acceleration=initial_acceleration,
            initial_omega=initial_omega,
            initial_alpha=initial_alpha,
        )
        self.radius = np.array([float(radius)], dtype=np.float64)
        self.width = np.array([float(width)], dtype=np.float64)

    @classmethod
    def create_robot(
        cls,
        initial_position: NDArray[np.float64],
        initial_direction: NDArray[np.float64],
        mass: float,
        inertia: float,
        radius: float,
        width: float,
    ) -> "Roomba":
        """Create a stationary Roomba."""
        return cls(
            position=initial_position,
            direction=initial_direction,
            mass=mass,
            inertia=inertia,
            radius=radius,
            width=width,
        )

    create_roomba = create_robot


def _assert_vector2(value: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Validate and return a float64 vector with shape (2,)."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {arr.shape}")
    return arr


def _assert_scalar1(
    value: Union[float, NDArray[np.float64]],
    name: str,
) -> NDArray[np.float64]:
    """Validate and return a float64 scalar packed as shape (1,)."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.array([float(arr)], dtype=np.float64)
    if arr.shape == (1,):
        return arr
    raise ValueError(f"{name} must be a scalar or shape (1,), got {arr.shape}")
