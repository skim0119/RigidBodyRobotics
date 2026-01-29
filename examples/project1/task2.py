"""
Task 2
------

Task 2: Energy Conservation Analysis

This experiment aims to highlight the stability characteristics of different numerical schemes by simulating a robot and analyzing its energy conservation over time.

**Objective**:
Simulate the robot on an unbounded plane, subjecting it to a potential field via a control law. Analyze the evolution of combined energy over time.

**Setup**:
- Robot initialized at `x0 = [1, 0]^T`
- Applied force/control law: `u_t^(l) = u_t^(r) = -Kx_t ⋅ d_1,t` where `K` is the stiffness
- Simulation horizon: 10 seconds

**Requirements**:
- Analyze the evolution of the combined energy:
    - Combined Energy: `E_t = H_t + (K ||x_t||^2) / 2`
    - Kinetic Energy: `H_t = (||η_t||^2) / (2m) + (l_t^2) / (2I)`
- Generate comparative plots illustrating energy drift
- Compare results from explicit Euler method and symplectic integrator

**Success Criteria**:
The comparison should demonstrate the superior long-term conservation properties of geometric methods (symplectic integrator) compared to explicit Euler.
"""

import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
from tqdm import tqdm

import elastica as ea
import elastica_rigid as er


class Simulator(
    ea.BaseSystemCollection,
    ea.Forcing,
    ea.CallBacks,
):
    pass


class EnergyCallBack(ea.CallBackBaseClass):
    """
    Tracks kinetic energy (H_t) and combined energy (E_t)
    H_t = (||η_t||^2) / (2m) + (l_t^2) / (2I)
    E_t = H_t + (K ||x_t||^2) / 2
    """

    def __init__(self, step_skip: int, callback_params: dict, K: float):
        super().__init__()
        self.every = step_skip
        self.callback_params = callback_params
        self.K = K

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["position"].append(system.position.copy())
            self.callback_params["direction"].append(system.direction.copy())
            # η_t = m * v_t (linear momentum), l_t = I * ω_t (angular momentum)
            H_t = system.compute_kinetic_energy()
            V_t = self.K * np.dot(system.position, system.position)
            E_t = H_t + V_t
            self.callback_params["kinetic_energy"].append(H_t)
            self.callback_params["potential_energy"].append(V_t)
            self.callback_params["combined_energy"].append(E_t)


def run_simulation(timestepper, simulation_time: float, dt: float, K: float):
    """Build sim with robot, potential field, callback; run and return history."""
    sim = Simulator()
    sim.append_allowed_types(er.Roomba)

    robot = er.Roomba.create_robot(
        initial_position=np.array([1.0, 0.0]),  # x0 = [1, 0]^T
        initial_direction=np.array([1.0, 0.0]),  # x-axis
        mass=0.1,
        inertia=0.05,
        radius=0.2,
        width=0.15,
    )
    sim.append(robot)

    sim.add_forcing_to(robot).using(er.PotentialFieldForce, K)

    recorded = defaultdict(list)
    sim.collect_diagnostics(robot).using(
        EnergyCallBack, step_skip=1, callback_params=recorded, K=K
    )

    sim.finalize()

    total_steps = int(simulation_time / dt)
    time = 0.0
    for _ in tqdm(range(total_steps)):
        time = timestepper.step(sim, time, dt)

    return recorded


# Parameters
simulation_time = 10.0  # (s)
dt = 0.001  # (s)
K = 0.5  # (N/m)

# Run with Explicit Euler
timestepper_euler = er.ExplicitEulerForward()
history_euler = run_simulation(timestepper_euler, simulation_time, dt, K)

# Run with Symplectic Euler
# timestepper_symplectic = er.SymplecticEulerForward()
timestepper_symplectic = ea.PositionVerlet()  # !!
history_symplectic = run_simulation(timestepper_symplectic, simulation_time, dt, K)

# Plot energy conservation comparison
t_euler = np.array(history_euler["time"])
E_euler = np.array(history_euler["combined_energy"])
t_symp = np.array(history_symplectic["time"])
E_symp = np.array(history_symplectic["combined_energy"])
# V_symp = np.array(history_symplectic["potential_energy"])
# H_symp = np.array(history_symplectic["kinetic_energy"])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t_euler, E_euler, "b-", label="Explicit Euler", lw=1.5)
ax.plot(t_symp, E_symp, "r-", label="Symplectic Euler", lw=1.5)
# ax.plot(t_symp, V_symp, "g-", label="Potential energy", lw=1.5)
# ax.plot(t_symp, H_symp, "b-", label="Kinetic energy", lw=1.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel(r"Combined energy $E_t = H_t + \frac{K \|x_t\|^2}{2}$ (J)")
ax.set_title("Task 2: Energy conservation — Explicit Euler vs Symplectic Euler")
ax.set_ylim(0.4, 0.6)
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("task2_energy_conservation.png", dpi=300)

# --- Trajectory and facing direction (Symplectic only) ---
positions_symp = np.array(history_symplectic["position"])   # (n, 2)
directions_symp = np.array(history_symplectic["direction"])  # (n, 2)
x_symp, y_symp = positions_symp[:, 0], positions_symp[:, 1]
d1x_symp, d1y_symp = directions_symp[:, 0], directions_symp[:, 1]

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(x_symp, y_symp, "b-", lw=1.5, label="Trajectory (Symplectic)")

# Subsample arrows so they don't overlap (e.g. ~20 along path)
arrow_skip = max(1, len(x_symp) // 20)
scale = 0.15 * np.max(np.ptp(positions_symp, axis=0))  # arrow length ~15% of range
ax2.quiver(
    x_symp[::arrow_skip],
    y_symp[::arrow_skip],
    d1x_symp[::arrow_skip],
    d1y_symp[::arrow_skip],
    scale=1.0 / scale if scale > 0 else 1.0,
    scale_units="xy",
    color="C1",
    alpha=0.8,
    label="Facing direction",
)
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")
ax2.set_title("Task 2: Symplectic robot trajectory and facing direction")
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.3)
ax2.set_aspect("equal")
plt.tight_layout()
plt.savefig("task2_trajectory.png", dpi=150)
