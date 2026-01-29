"""
Task 1
------

Task 1: Open-Loop Dynamics Validation

This task validates the accuracy of the rigid body dynamics model and the numerical integration engine under a constant input.

**Objective**:
Simulate the robot on an unbounded plane, applying a constant force of 0.1 N to the left wheel (`u_l(t) ≡ 0.1 N`) for a continuous duration of 10 seconds.

**Requirements**:
- Generate time-series plots for:
    - Linear speed, \( w_t = \frac{||\eta_t||}{m} \)
    - Angular velocity, \( \omega_t = \frac{l_t}{I} \)
- Compare the numerically integrated results with analytical solutions derived from the equations of motion.

**Success Criteria**:
Numerical and analytical results should closely align, verifying the correctness of the fundamental model implementation.
"""

import numpy as np
from collections import defaultdict

import elastica as ea
import elastica_rigid as er

from postprocessing import *


class Simulator(
    ea.BaseSystemCollection,
    ea.Forcing,
    ea.CallBacks,
):
    pass


sim = Simulator()
sim.append_allowed_types(er.Roomba)

# Simulation parameters
simulation_time = 10.0  # (sec)
dt = 0.01  # (sec)

# Robot parameters
robot = er.Roomba.create_robot(
    initial_position=np.array([0.0, 0.0]),
    initial_director=np.array([1.0, 0.0]),  # x-axis
    mass=0.1,  # (kg)
    inertia=0.05,  # (kg m^2)
    radius=0.2,  # (m)
    width=0.15,  # (m)
)
sim.append(robot)

# Forces added to the robot
force_left = np.array([0.1, 0.0])  # (N)
duration = 10.0  # (sec)
sim.add_forcing_to(robot).using(er.ConstantForce, force_left, duration)


# Add call backs
class CallBack(ea.CallBackBaseClass):
    """
    Tracks the velocity norms of the rod
    """

    def __init__(self, step_skip: int, callback_params: dict):
        super().__init__()
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            # Collect x
            self.callback_params["velocity_norms"].append(
                np.linalg.norm(system.velocity_collection.copy())
            )
            return


recorded_history = defaultdict(list)
sim.collect_diagnostics(robot).using(
    CallBack, step_skip=500, callback_params=recorded_history
)

sim.finalize()

# timestepper = er.ExplicitEulerForward()
timestepper = er.SymplecticEulerForward()
total_steps = int(simulation_time / dt)
print("Total steps", total_steps)

time = 0.0
for i in range(total_steps):
    time = timestepper.step(sim, time, dt)

# plot_timoshenko(shearable_rod, end_force, False, True)
