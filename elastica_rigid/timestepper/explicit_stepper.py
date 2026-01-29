__doc__ = """Symplectic time steppers and concepts for integrating the kinematic and dynamic equations of rod-like objects.  """

from typing import Protocol, TYPE_CHECKING, Any, Callable

from itertools import zip_longest

from elastica.systems.protocol import SystemProtocol
from elastica.typing import (
    SystemCollectionType,
    SystemType,
    StepType,
    SteppersOperatorsType,
)

import numpy as np


class RigidSystem(SystemProtocol, Protocol):
    """
    """

    def update_state(self, time, dt): 
        """
        Update state
        """

    def emplace_back_state(self):
        """
        Store current state in 
        (For later, advanced explicit methods like RK schemes.)
        """
        raise NotImplementedError


class ExplicitEulerForward(:
    """
    Explicit Euler forward stepper for differential-algebraic equations.
    """

    def __init__(self):
        # placeholder for stepping information
        self.info: dict[str, Any] = {}

    def step(
        self,
        SystemCollection: SystemCollectionType,
        time: np.float64,
        dt: np.float64,
    ) -> np.float64:
        """
        Function for doing symplectic stepper over the user defined rods (system).

        Returns
        -------
        time: float
            The time after the integration step.

        """
        simulation_time = np.float64(time)
        simulation_dt = np.float64(dt)

        # Compute external forces and couples
        SystemCollection.synchronize(simulation_time)

        # Step
        for system in SystemCollection.final_systems():
            system.update_state(time, dt)

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
        (The function is used for single system instance, mainly for testing purposes.)
        """

        system.update_state(time, dt)
        return time + dt
