"""Explicit time steppers for integrating kinematic and dynamic equations of rigid bodies."""

from typing import Protocol

from elastica.systems.protocol import SymplecticSystemProtocol
from elastica.typing import SystemCollectionType

import numpy as np


class RigidSystem(SymplecticSystemProtocol, Protocol):
    """Protocol for rigid body systems used by time steppers."""

    def emplace_back_state(self) -> None:
        """
        Store current state.
        (For later, advanced explicit methods like RK schemes.)
        """
        raise NotImplementedError


class ExplicitEulerForward:
    """
    Explicit Euler forward stepper for differential-algebraic equations.
    """

    def step(
        self,
        SystemCollection: SystemCollectionType,
        time: np.float64,
        dt: np.float64,
    ) -> np.float64:
        """
        Perform one explicit Euler forward step over the systems.

        Returns
        -------
        float
            The time after the integration step.
        """
        simulation_time = np.float64(time)
        simulation_dt = np.float64(dt)

        # Compute external forces and couples
        SystemCollection.synchronize(simulation_time)

        # Step (dynamics first so kinematics uses updated velocity)
        for system in SystemCollection.final_systems():
            system.update_accelerations(time)
            system.update_dynamics(time, dt)
            system.update_kinematics(time, dt)

        SystemCollection.constrain_values(simulation_time)
        SystemCollection.constrain_rates(simulation_time)
        SystemCollection.apply_callbacks(
            simulation_time, round(simulation_time / simulation_dt)
        )

        # Zero out the external forces and torques
        for system in SystemCollection.final_systems():
            system.zeroed_out_external_forces_and_torques(simulation_time)

        return time + dt

    def step_single_instance(
        self,
        system: RigidSystem,
        time: np.float64,
        dt: np.float64,
    ) -> np.float64:
        """
        Perform one step for a single system instance (mainly for testing).
        """
        system.update_accelerations(time)
        system.update_dynamics(time, dt)
        system.update_kinematics(time, dt)
        return time + dt
