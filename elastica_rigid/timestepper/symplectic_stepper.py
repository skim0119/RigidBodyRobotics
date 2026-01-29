"""Symplectic time steppers for integrating kinematic and dynamic equations of rigid bodies."""

from elastica.typing import (
    SystemCollectionType,
)

import numpy as np

from .explicit_stepper import RigidSystem


class SymplecticEulerForward:
    """
    Symplectic Euler forward stepper for differential-algebraic equations.
    Updates dynamics before kinematics (momentum-first) for better energy conservation.
    """

    def step(
        self,
        SystemCollection: SystemCollectionType,
        time: np.float64,
        dt: np.float64,
    ) -> np.float64:
        """
        Perform one symplectic Euler forward step over the systems.

        Returns
        -------
        float
            The time after the integration step.
        """
        simulation_time = np.float64(time)
        simulation_dt = np.float64(dt)

        # Compute external forces and couples
        SystemCollection.synchronize(simulation_time)

        # Step: dynamics first, then kinematics (symplectic order)
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
