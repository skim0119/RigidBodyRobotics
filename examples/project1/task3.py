"""
Task 3: Integrated Environmental Navigation
-------------------------------------------

This capstone task validates the complete rigid-body physics engine by combining:
- open-loop wheel force commands (Fig. 4)
- wall collision (penalty contact) using mu_c
- position-dependent kinetic friction in the shaded region using mu_f

Deliverables:
- Trajectory plot overlaid on the workspace map (Fig. 2 / Fig. 3)
- Time-series of momentum and total kinetic energy H_t
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import elastica as ea
import elastica_rigid as er


class Simulator(
    ea.BaseSystemCollection,
    ea.Forcing,
    ea.CallBacks,
):
    pass


class Task3CallBack(ea.CallBackBaseClass):
    """
    Records state and derived quantities:
    - linear momentum magnitude ||p|| where p = m v
    - angular momentum l = I omega
    - kinetic energy H = 1/2 m ||v||^2 + 1/2 I omega^2
    """

    def __init__(self, step_skip: int, callback_params: dict):
        super().__init__()
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every != 0:
            return

        m = float(system.mass)
        I = float(system.inertia)
        v = system.velocity.copy()
        omega = float(system.omega[0])

        speed = float(np.linalg.norm(v))
        p_norm = m * speed
        l = I * omega
        H = 0.5 * m * (speed**2) + 0.5 * I * (omega**2)

        self.callback_params["time"].append(float(time))
        self.callback_params["position"].append(system.position.copy())
        self.callback_params["direction"].append(system.direction.copy())
        self.callback_params["external_force"].append(system.external_forces.copy())
        self.callback_params["p_norm"].append(p_norm)
        self.callback_params["l"].append(l)
        self.callback_params["H"].append(H)


def plot_environment(ax):
    """Draw Fig. 2 workspace outline + obstacles and Fig. 3 friction region."""
    # Workspace bounds: [0,3]x[0,4]
    xmin, xmax, ymin, ymax = 0.0, 3.0, 0.0, 4.0

    # Obstacles (intrusions): [0,0.6]x[1,3] and [2.4,3]x[1,3]
    obstacles = [
        (0.0, 0.6, 1.0, 3.0),
        (2.4, 3.0, 1.0, 3.0),
    ]

    # Outer boundary rectangle
    ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], "k-", lw=2)

    # Obstacles
    for oxmin, oxmax, oymin, oymax in obstacles:
        ax.fill(
            [oxmin, oxmax, oxmax, oxmin, oxmin],
            [oymin, oymin, oymax, oymax, oymin],
            color="white",
            edgecolor="black",
            linewidth=2,
            zorder=2,
        )

    # Friction region (Fig. 3): shaded half-plane y >= (-4/3)x + 4, clipped to bounds
    # The line crosses the bounds at (0,4) and (3,0)
    tri = np.array([[0.0, 4.0], [3.0, 0.0], [3.0, 4.0]], dtype=float)
    ax.fill(
        tri[:, 0],
        tri[:, 1],
        color="0.85",
        edgecolor="none",
        zorder=0,
        label=r"$\mu_f$ region",
    )

    ax.set_xlim(xmin - 0.1, xmax + 0.1)
    ax.set_ylim(ymin - 0.1, ymax + 0.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def main():
    # Simulation parameters
    simulation_time = 20.0  # s
    dt = 0.001  # s (smaller dt for stiff contact)

    # Environment parameters
    mu_f = 0.05
    mu_c = 1000.0  # N/m (penalty stiffness)

    # Robot parameters
    robot = er.Roomba.create_robot(
        initial_position=np.array([0.3, 0.7]),  # Fig. 3
        initial_direction=np.array([1.0, 0.0]),
        mass=2.0,  # (kg)
        inertia=0.05,  # (kg m^2)
        radius=0.2,  # (m)
        width=0.15,  # (m)
    )

    sim = Simulator()
    sim.append_allowed_types(er.Roomba)
    sim.append(robot)

    recorded = defaultdict(list)
    # Wheel force commands (Fig. 4), nonzero for t in [0,4], then 0
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    u_left = np.array([0.0, 0.5, -0.5, 0.0, 1.5], dtype=float)
    u_right = np.array([0.0, 0.5, 0.25, 0.0, 0.5], dtype=float)
    sim.add_forcing_to(robot).using(
        er.WheelForceSequence,
        times,
        u_left,
        u_right,
    )
    workspace_bounds = (0.0, 3.0, 0.0, 4.0)
    obstacles = [
        (0.0, 0.6, 1.0, 3.0),
        (2.4, 3.0, 1.0, 3.0),
    ]
    sim.add_forcing_to(robot).using(
        er.EnvironmentForces2D,
        mu_f,
        mu_c,
        bounds=workspace_bounds,
        obstacles=obstacles,
        callback_params=recorded,
    )
    sim.collect_diagnostics(robot).using(
        Task3CallBack, step_skip=10, callback_params=recorded
    )

    sim.finalize()

    # timestepper = er.SymplecticEulerForward()
    timestepper = ea.PositionVerlet()
    total_steps = int(simulation_time / dt)
    time = 0.0
    for _ in range(total_steps):
        time = timestepper.step(sim, time, dt)

    # --- Quick qualitative checks (prints) ---
    pos = np.array(recorded["position"])
    if pos.size:
        mins = pos.min(axis=0)
        maxs = pos.max(axis=0)
        in_fric = pos[:, 1] >= (-4.0 / 3.0) * pos[:, 0] + 4.0
        print(
            "Task3 summary:",
            f"pos_min={mins}, pos_max={maxs}, fric_fraction={float(in_fric.mean()):.3f}",
        )

    # --- Plot 1: Trajectory over environment map ---
    dirs = np.array(recorded["direction"])

    # Extract friction data
    left_fric_mag = np.array(recorded["left_friction_force_mag"])
    left_fric_dir = np.array(recorded["left_friction_force_dir"])
    right_fric_mag = np.array(recorded["right_friction_force_mag"])
    right_fric_dir = np.array(recorded["right_friction_force_dir"])

    # Ensure friction arrays match position array length (take minimum to be safe)
    n_pos = len(pos)
    n_fric = min(
        len(left_fric_mag),
        len(left_fric_dir),
        len(right_fric_mag),
        len(right_fric_dir),
        n_pos,
    )
    pos = pos[:n_fric]
    dirs = dirs[:n_fric]
    left_fric_mag = left_fric_mag[:n_fric]
    left_fric_dir = left_fric_dir[:n_fric]
    right_fric_mag = right_fric_mag[:n_fric]
    right_fric_dir = right_fric_dir[:n_fric]

    # Compute wheel positions: x_left = x + d2 * width/2, x_right = x - d2 * width/2
    # d2 is perpendicular to d1 (heading direction), rotated 90° counterclockwise
    R_90 = np.array([[0, -1], [1, 0]])  # 90° rotation matrix
    d2 = (R_90 @ dirs.T).T  # Perpendicular direction
    width = float(robot.width)
    pos_left = pos + d2 * (width / 2.0)
    pos_right = pos - d2 * (width / 2.0)

    fig1, ax1 = plt.subplots(figsize=(7, 8))
    plot_environment(ax1)
    ax1.plot(pos[:, 0], pos[:, 1], "C0-", lw=1.5, label="Trajectory", zorder=3)

    # Robot body (circle of radius r) along the trajectory
    r = float(robot.radius)
    skip = max(1, len(pos) // 25)
    pos_skip = pos[::skip]
    for i, p in enumerate(pos_skip):
        circle = mpatches.Circle(
            p,
            r,
            facecolor="C0",
            edgecolor="C0",
            alpha=0.25,
            linewidth=1.5,
            zorder=3.5,
            label=(rf"Robot ($r$ = {r} m)" if i == 0 else None),
        )
        ax1.add_patch(circle)

    # Friction force vectors (subsample)
    # Scale friction vectors for visualization
    max_fric_mag = (
        max(left_fric_mag.max(), right_fric_mag.max())
        if len(left_fric_mag) > 0
        else 1.0
    )
    fric_scale = 0.1 / (max_fric_mag + 1e-6)  # Scale so max arrow length is ~0.1 m

    left_fric_vec = left_fric_mag[:, None] * left_fric_dir * fric_scale
    right_fric_vec = right_fric_mag[:, None] * right_fric_dir * fric_scale

    ax1.quiver(
        pos_left[::skip, 0],
        pos_left[::skip, 1],
        left_fric_vec[::skip, 0],
        left_fric_vec[::skip, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.003,
        color="C2",
        alpha=0.7,
        label="Left friction",
        zorder=4,
    )
    ax1.quiver(
        pos_right[::skip, 0],
        pos_right[::skip, 1],
        right_fric_vec[::skip, 0],
        right_fric_vec[::skip, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.003,
        color="C3",
        alpha=0.7,
        label="Right friction",
        zorder=4,
    )

    # Heading arrows (subsample)
    ax1.quiver(
        pos[::skip, 0],
        pos[::skip, 1],
        dirs[::skip, 0],
        dirs[::skip, 1],
        angles="xy",
        scale_units="xy",
        scale=8.0,
        width=0.004,
        color="C1",
        alpha=0.8,
        label="Heading",
        zorder=4,
    )
    ax1.set_title("Task 3: Trajectory over environment")
    ax1.legend()
    ax1.grid(True, alpha=0.25)
    fig1.tight_layout()
    fig1.savefig("task3_trajectory.png", dpi=150)

    # --- Plot 2: Momentum and energy time-series ---
    t = np.array(recorded["time"])
    p_norm = np.array(recorded["p_norm"])
    l = np.array(recorded["l"])
    H = np.array(recorded["H"])

    fig2, (ax2, ax3) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    ax2.plot(t, p_norm, "C0-", lw=1.2, label=r"$\|p_t\| = m\|v_t\|$")
    ax2.plot(t, l, "C1-", lw=1.2, label=r"$l_t = I\omega_t$")
    ax2.set_ylabel("Momentum")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right")

    ax3.plot(t, H, "C2-", lw=1.2, label=r"$H_t$ (kinetic)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Energy (J)")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper right")

    fig2.suptitle("Task 3: Momentum and total kinetic energy")
    fig2.tight_layout()
    fig2.savefig("task3_momentum_energy.png", dpi=150)

    plt.close("all")

    # Animation
    # Animate and save the video with the environment and trajectory
    viz = er.Visualize(
        np.array(recorded["time"]),
        np.array(recorded["position"]),
        np.array(recorded["direction"]),
        np.array(recorded["external_force"]),
        robot_radius=0.2,
        show_trajectory=True,
        show_external_forces=True,
        fps=30,
    )
    viz.stamp_environment_on_figure(
        bounds=workspace_bounds,
        obstacles=obstacles,
    )
    anim = viz.animate()
    anim.save("task3_video.mp4", writer="ffmpeg", dpi=150)


if __name__ == "__main__":
    main()
