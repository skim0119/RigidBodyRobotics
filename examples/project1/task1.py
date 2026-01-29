"""
Task 1: Open-Loop Dynamics Validation
-------------------------------------

This task validates the accuracy of the rigid body dynamics model and the numerical integration engine under a constant input.

**Objective**:
Simulate the robot on an unbounded plane, applying a constant force of 0.1 N to the left wheel (`u_l(t) ≡ 0.1 N`) for a continuous duration of 10 seconds.

**Requirements**:
- Generate time-series plots for:
    - Linear speed, w_t = ||eta_t|| / m
    - Angular velocity, omega_t = l_t / I
- Compare the numerically integrated results with analytical solutions derived from the equations of motion.

**Success Criteria**:
Numerical and analytical results should closely align, verifying the correctness of the fundamental model implementation.
"""

import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt

import elastica as ea
import elastica_rigid as er


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
    initial_direction=np.array([1.0, 0.0]),  # x-axis
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
    Tracks linear speed w_t = ||η_t||/m = ||v_t|| and angular velocity ω_t = l_t/I.
    """

    def __init__(self, step_skip: int, callback_params: dict):
        super().__init__()
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            # Linear speed w_t = ||v_t|| (same as ||η_t||/m since η = m*v)
            self.callback_params["linear_speed"].append(
                np.linalg.norm(system.velocity.copy())
            )
            # Angular velocity ω_t (rad/s)
            self.callback_params["angular_velocity"].append(
                float(system.omega.flat[0])
            )
            return


recorded_history = defaultdict(list)
sim.collect_diagnostics(robot).using(
    CallBack, step_skip=1, callback_params=recorded_history
)

sim.finalize()

# timestepper = er.ExplicitEulerForward()
timestepper = er.SymplecticEulerForward()
total_steps = int(simulation_time / dt)
print("Total steps", total_steps)

time = 0.0
for i in range(total_steps):
    time = timestepper.step(sim, time, dt)

# --- Post-processing: time-series plots and comparison with analytical solutions ---
# Constant force F = [0.1, 0] N, no torque → a = F/m, α = 0
# Analytical: v(t) = v_0 + (F/m)*t = (F/m)*t, ω(t) = ω_0 = 0
# Linear speed w(t) = ||v(t)|| = (F/m)*t (force along x, v_0=0)
F_mag = 0.1  # N
m = robot.mass
I = robot.inertia

t_num = np.array(recorded_history["time"])
linear_speed_num = np.array(recorded_history["linear_speed"])
angular_velocity_num = np.array(recorded_history["angular_velocity"])

# Analytical solutions
linear_speed_analytical = (F_mag / float(m)) * t_num
angular_velocity_analytical = np.zeros_like(t_num)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

ax1.plot(t_num, linear_speed_num, "b-", label="Numerical", lw=1.5)
ax1.plot(t_num, linear_speed_analytical, "r--", label="Analytical", lw=1.5)
ax1.set_ylabel(r"Linear speed $w_t = \|\eta_t\|/m$ (m/s)")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

ax2.plot(t_num, angular_velocity_num, "b-", label="Numerical", lw=1.5)
ax2.plot(t_num, angular_velocity_analytical, "r--", label="Analytical", lw=1.5)
ax2.set_ylabel(r"Angular velocity $\omega_t = l_t/I$ (rad/s)")
ax2.set_xlabel("Time (s)")
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.3)

plt.suptitle("Task 1: Open-Loop Dynamics — Numerical vs Analytical")
plt.tight_layout()
plt.savefig("task1_linear_and_angular_velocity.pdf")
plt.savefig("task1_linear_and_angular_velocity.png", dpi=150)
plt.show()
