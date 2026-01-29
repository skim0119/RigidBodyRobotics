from typing import Optional, Type, Union

from numpy.typing import NDArray

import numpy as np
from elastica.systems.protocol import SystemProtocol

from .equations import (
    _update_rigid_SO2_dynamic_state,
    _update_rigid_SO2_kinematic_state,
    _update_accelerations,
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
            Moment of inertia about the vertical axis in kg·m²
        radius : float
            Wheel radius in meters
        width : float
            Distance between wheels (track width) in meters
        initial_velocity : np.ndarray, optional
            2D initial linear velocity [vx, vy] in m/s. If None, defaults to zero.
        initial_acceleration : np.ndarray, optional
            2D initial linear acceleration [ax, ay] in m/s². If None, defaults to zero.
        initial_omega : float or np.ndarray, optional
            Initial angular velocity in rad/s (scalar or shape (1,)). If None, defaults to zero.
        initial_alpha : float or np.ndarray, optional
            Initial angular acceleration in rad/s² (scalar or shape (1,)). If None, defaults to zero.
        """

        # Store physical parameters
        self.mass = np.float64(mass)
        self.inertia = np.float64(inertia)
        self.radius = np.float64(radius)
        self.width = np.float64(width)

        # Convert 2D position to 3D [x, y, 0]
        self.position = np.array(position, dtype=float)
        self.direction = np.array(direction, dtype=float)

        # Initialize velocities and accelerations
        if initial_velocity is not None:
            self.velocity = np.asarray(initial_velocity, dtype=np.float64)
        else:
            self.velocity = np.zeros((2,), dtype=np.float64)

        if initial_acceleration is not None:
            self.acceleration = np.asarray(initial_acceleration, dtype=np.float64)
        else:
            self.acceleration = np.zeros((2,), dtype=np.float64)

        if initial_omega is not None:
            self.omega = np.atleast_1d(np.asarray(initial_omega, dtype=np.float64))
        else:
            self.omega = np.zeros((1,), dtype=np.float64)

        if initial_alpha is not None:
            self.alpha = np.atleast_1d(np.asarray(initial_alpha, dtype=np.float64))
        else:
            self.alpha = np.zeros((1,), dtype=np.float64)

        # External
        self.external_forces = np.zeros((3,), dtype=np.float64)
        self.external_torques = np.zeros((3,), dtype=np.float64)

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
        self.external_forces[:] = 0.0
        self.external_torques[:] = 0.0

    def compute_internal_forces_and_torques(self, time: np.float64) -> None:
        pass

    # Interface to time-stepper mixins (Symplectic, Explicit), which calls this method
    def update_accelerations(self, time: np.float64) -> None:
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
