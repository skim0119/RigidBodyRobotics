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
from scipy.special import fresnel

import elastica as ea
import elastica_rigid as er


class Simulator(
    ea.BaseSystemCollection,
    ea.Forcing,
    ea.CallBacks,
):
    pass


sim = Simulator()
sim.append_allowed_types(er.Roomba)  # Cool things!

# Simulation parameters
F_l_forward = 0.1  # N (left wheel forward component)
F_r_forward = 0.0  # N
simulation_time = 10.0  # (sec)
dt = 0.01  # (sec)

# Robot parameters
robot = er.Roomba.create_robot(
    initial_position=np.array([0.0, 0.0]),
    initial_direction=np.array([1.0, 0.0]),  # x-axis
    mass=2,  # (kg)
    inertia=0.05,  # (kg m^2)
    radius=0.2,  # (m)
    width=0.15,  # (m)
)
sim.append(robot)

# Forces added to the robot
force_left = np.array([F_l_forward, 0.0])  # (N)  d1, d2
force_right = np.array([F_r_forward, 0.0])  # (N)  d1, d2
duration = 10.0  # (sec)
sim.add_forcing_to(robot).using(er.ConstantForce, force_left, force_right, duration)


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
            self.callback_params["position"].append(system.position[:, 0].copy())
            self.callback_params["direction"].append(system.direction[:, 0].copy())
            # Linear speed w_t = ||v_t|| (same as ||η_t||/m since η = m*v)
            self.callback_params["linear_speed"].append(
                np.linalg.norm(system.velocity[:, 0].copy())
            )
            # Angular velocity ω_t (rad/s)
            self.callback_params["angular_velocity"].append(float(system.omega[0]))
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
# Force on left wheel only: F_l = [0.1, 0] N (forward), F_r = [0, 0] N.
# Net force in inertial: F_net = (F_l + F_r) @ director = 0.1 * d1(t) → a(t) = (0.1/m) * d1(t).
# Torque from wheel forces: τ = (track_width/2) * (F_l - F_r)_forward → α = τ/I = const.
# So ω(t) = ω_0 + α*t = α*t, θ(t) = (1/2)*α*t²; v(t) = ∫_0^t a(s) ds involves Fresnel integrals.
m = float(robot.mass[0])
inertia = float(robot.inertia[0])
track_width = float(robot.width[0])

t_num = np.array(recorded_history["time"])
linear_speed_num = np.array(recorded_history["linear_speed"])
angular_velocity_num = np.array(recorded_history["angular_velocity"])

if F_l_forward == F_r_forward:
    # Drive forward
    linear_speed_analytical = t_num * 2 * F_l_forward / m
    angular_velocity_analytical = t_num * 0.0
elif F_l_forward == -F_r_forward:
    # Rotation
    linear_speed_analytical = t_num * 0
    angular_velocity_analytical = t_num * F_r_forward * track_width / inertia
else:
    # Angular acceleration and angular velocity (one wheel drives → moment)
    torque = (track_width / 2) * (-F_l_forward + F_r_forward)  # N·m
    alpha_rad = torque / inertia  # rad/s²
    angular_velocity_analytical = alpha_rad * t_num

    # Linear speed
    total_force = F_r_forward + F_l_forward
    z = np.sqrt(-alpha_rad / np.pi) * t_num
    S_z, C_z = fresnel(z)
    v_x_analytical = total_force * np.sqrt(np.pi / -alpha_rad) * C_z / m
    v_y_analytical = total_force * np.sqrt(np.pi / -alpha_rad) * S_z / m
    linear_speed_analytical = np.sqrt(v_x_analytical**2 + v_y_analytical**2)

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
plt.savefig("task1_linear_and_angular_velocity.png", dpi=300)

# --- Trajectory and facing direction ---
positions = np.array(recorded_history["position"])  # (n, 2)
directions = np.array(recorded_history["direction"])  # (n, 2)
x, y = positions[:, 0], positions[:, 1]
d1x, d1y = directions[:, 0], directions[:, 1]

fig2, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, "b-", lw=1.5, label="Trajectory")

# Subsample arrows so they don't overlap (e.g. ~20 along path)
arrow_skip = max(1, len(x) // 20)
scale = 0.15 * np.max(np.ptp(positions, axis=0))  # arrow length ~15% of range
ax.quiver(
    x[::arrow_skip],
    y[::arrow_skip],
    d1x[::arrow_skip],
    d1y[::arrow_skip],
    scale=1.0 / scale if scale > 0 else 1.0,
    scale_units="xy",
    color="C1",
    alpha=0.8,
    label="Facing direction",
)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Task 1: Robot trajectory and facing direction")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("task1_trajectory.png", dpi=150)
