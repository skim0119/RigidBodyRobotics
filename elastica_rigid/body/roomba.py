from typing import Optional, Type, Union

from numpy.typing import NDArray

import numpy as np
from elastica.systems.protocol import SystemProtocol

from ._se2_equations import (
    _update_rigid_SO2_dynamic_state,
    _update_rigid_SO2_kinematic_state,
    _update_accelerations,
    _zeroed_out_external_forces_and_torques,
)


class Roomba(SystemProtocol):
    """
    A 2D differential-drive mobile robot (Roomba) model.

    This class represents a planar rigid body robot that can move in 2D space.
    The robot's pose consists of a 2D position and a 2D rotation (SO(2)).
    """

    REQUISITE_MODULES: list[Type] = []

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
    ):
        """
        Initialize a Roomba robot.

        Parameters
        ----------
        position : np.ndarray
            2D initial position [x, y] in meters
        direction : np.ndarray
            2D initial heading direction vector [d1x, d1y] (will be normalized)
        mass : float
            Mass of the robot in kg
        inertia : float
            Moment of inertia about the vertical axis in kg*m^2²
        radius : float
            Wheel radius in meters
        width : float
            Distance between wheels (track width) in meters
        initial_velocity : np.ndarray, optional
            2D initial linear velocity [vx, vy] in m/s. If None, defaults to zero.
        initial_acceleration : np.ndarray, optional
            2D initial linear acceleration [ax, ay] in m/s^2. If None, defaults to zero.
        initial_omega : float or np.ndarray, optional
            Initial angular velocity in rad/s (scalar or shape (1,)). If None, defaults to zero.
        initial_alpha : float or np.ndarray, optional
            Initial angular acceleration in rad/s² (scalar or shape (1,)). If None, defaults to zero.
        """

        # Store physical parameters
        self.mass = np.array([mass], dtype=np.float64)
        self.inertia = np.array([inertia], dtype=np.float64)
        self.radius = np.array([radius], dtype=np.float64)
        self.width = np.array([width], dtype=np.float64)

        self.position[:, 0] = position
        self.direction[:, 0] = direction
        dnorm = np.linalg.norm(self.direction[:, 0])
        if dnorm > 1e-12:
            self.direction[:, 0] /= dnorm

        # Initialize velocities and accelerations
        if initial_velocity is not None:
            self.velocity[:, 0] = initial_velocity
        else:
            self.velocity = np.zeros((2, 1), dtype=np.float64)

        if initial_acceleration is not None:
            self.acceleration[:, 0] = initial_acceleration
        else:
            self.acceleration = np.zeros((2, 1), dtype=np.float64)

        if initial_omega is not None:
            self.omega = np.atleast_1d(np.asarray(initial_omega, dtype=np.float64))
        else:
            self.omega = np.zeros((1,), dtype=np.float64)

        if initial_alpha is not None:
            self.alpha = np.atleast_1d(np.asarray(initial_alpha, dtype=np.float64))
        else:
            self.alpha = np.zeros((1,), dtype=np.float64)

        # External
        self.external_forces = np.zeros((2, 1), dtype=np.float64)
        self.external_torques = np.zeros((1,), dtype=np.float64)

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
        """
        Create a stationary Roomba robot instance.

        This is a convenience class method that creates and returns a Roomba instance.

        Parameters
        ----------
        initial_position : np.ndarray
            2D initial position [x, y] in meters
        initial_direction : np.ndarray
            2D initial heading direction vector [d1x, d1y] (will be normalized)
        mass : float
            Mass of the robot in kg
        inertia : float
            Moment of inertia about the vertical axis in kg·m²
        radius : float
            Wheel radius in meters
        width : float
            Distance between wheels (track width) in meters

        Returns
        -------
        Roomba
            A new Roomba instance
        """

        return cls(
            position=initial_position,
            direction=initial_direction,
            mass=mass,
            inertia=inertia,
            radius=radius,
            width=width,
        )

    def zeroed_out_external_forces_and_torques(self, time: np.float64) -> None:
        _zeroed_out_external_forces_and_torques(
            self.external_forces,
            self.external_torques,
        )

    def compute_internal_forces_and_torques(self, time: np.float64) -> None:
        pass

    # Interface to time-stepper mixins (Symplectic, Explicit), which calls this method
    def update_accelerations(self, time: np.float64, dt: np.float64) -> None:
        _update_accelerations(
            self.acceleration,
            self.alpha,
            self.mass,
            self.inertia,
            self.external_forces,
            self.external_torques,
        )

    def update_dynamics(self, time, dt):
        prefac = np.float64(dt)
        _update_rigid_SO2_dynamic_state(
            prefac,
            self.velocity,
            self.acceleration,
            self.omega,
            self.alpha,
        )

    def update_kinematics(self, time, dt):
        prefac = np.float64(dt)
        _update_rigid_SO2_kinematic_state(
            prefac,
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
