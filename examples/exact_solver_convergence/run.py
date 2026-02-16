from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

import elastica as ea
from elastica._linalg import _batch_matvec, _batch_cross

import elastica_rigid as er
from elastica_rigid._rotations import _rotate_vector



def predict_deltaE(sphere, dt: float) -> float:
    """
    For rigid-body explicit Euler on omega:

      omega_{n+1} = omega_n + dt * alpha_n
      I * alpha_n = c_n,  where c_n = (I*omega_n) x omega_n + tau_n

    With E(omega) = 1/2 omega^T I omega and (for torque-free) omega^T c = 0,
    the per-step energy increase is:

      deltaE = (dt^2 / 2) * c^T I^{-1} c   >= 0

    This function computes the RHS using the *current* state on the sphere.
    """
    # Shapes: (3, 1)
    omega = sphere.omega_collection
    inertia = sphere.mass_second_moment_of_inertia
    inv_inertia = sphere.inv_mass_second_moment_of_inertia
    tau = getattr(sphere, "external_torques", np.zeros_like(omega))

    Iomega = _batch_matvec(inertia, omega)
    c = _batch_cross(Iomega, omega) + tau
    invI_c = _batch_matvec(inv_inertia, c)
    # scalar: sum over components, single element
    cT_invI_c = float(np.sum(c * invI_c))
    return 0.5 * (dt**2) * cT_invI_c


def run(
    T: float = 10.0,
    dt: float = 1e-5,
    fps: float = 60,
):
    stepper = ea.PositionVerlet()
    sphere = er.SphereExact(center=np.zeros(3), base_radius=1, density=1e3)
    sphere.dt = dt

    sphere.omega_collection[:] = np.array([[0.01], [15.0], [0.01]])
    sphere.mass_second_moment_of_inertia[1, 1, 0] *= 2
    sphere.inv_mass_second_moment_of_inertia[1, 1, 0] /= 2
    sphere.mass_second_moment_of_inertia[2, 2, 0] *= 4
    sphere.inv_mass_second_moment_of_inertia[2, 2, 0] /= 4

    recorded_history = defaultdict(list)
    step_skip = int(max(1, 1.0 / fps / dt))

    time = 0.0
    total_steps = int(T / dt)
    for tidx in tqdm(range(total_steps)):
        time = stepper.step_single_instance(sphere, time, dt)
        # Predict ΔE using current state (before stepping)
        if tidx % step_skip == 0:
            predicted_deltaE = predict_deltaE(sphere, dt)
            recorded_history["deltaE_pred_step"].append(predicted_deltaE)
            recorded_history["energy_step"].append(sphere.compute_rotational_energy())

            recorded_history["time"].append(time)
            recorded_history["position"].append(sphere.position_collection.copy())
            recorded_history["director"].append(sphere.director_collection.copy())
            iomega = sphere.mass_second_moment_of_inertia[:, :, 0] @ sphere.omega_collection[
                :, 0
            ]
            recorded_history["Iomega"].append(iomega)
            recorded_history["omega"].append(sphere.omega_collection.copy())
            recorded_history["alpha"].append(sphere.alpha_collection.copy())

    return recorded_history

if __name__ == "__main__":
    T_FINAL = 10.0
    DT = 1e-5
    fps = 60
    results = run(
        T=T_FINAL,
        dt=DT,
        fps=fps,
    )

    # Post-processing
    # From list to array
    time_step = np.array(results["time"])
    omega = np.array(results["omega"]).squeeze()
    iomega = np.array(results["Iomega"]).squeeze()
    alpha = np.array(results["alpha"]).squeeze()
    director = np.array(results["director"]).squeeze()

    energy_step = np.array(results["energy_step"])
    deltaE_pred_step = np.array(results["deltaE_pred_step"])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(time_step, np.einsum("ikj,ik->ij", director, omega))
    ax.set_xlabel("time")
    ax.set_ylabel("angular momentum (Qomega)")
    plt.savefig("omega.png")
    plt.close("all")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(time_step, np.einsum("ikj,ik->ij", director, iomega))
    ax.set_xlabel("time")
    ax.set_ylabel("angular momentum (QJomega)")
    plt.savefig("iomega.png")
    plt.close("all")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(time_step, energy_step, label="E(t)")
    ax.set_xlabel("time")
    ax.set_ylabel("rotational energy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.savefig("energy.png")
    plt.close("all")

    # Overlay: E(t) with per-step ΔE on a twin axis
    fig = plt.figure(figsize=(10, 6))
    axE = fig.add_subplot(111)
    axE.plot(time_step, energy_step, color="C0", label="E(t)")
    axE.set_xlabel("time")
    axE.set_ylabel("rotational energy", color="C0")
    axE.tick_params(axis="y", labelcolor="C0")
    axE.grid(True, alpha=0.3)

    axD = axE.twinx()
    t_mid = time_step[:-1]  # ΔE_n aligned to [t_n, t_{n+1})
    axD.plot(time_step, deltaE_pred_step, color="C1", linestyle="--", label="ΔE_pred = dt^2/2 * c^T I^{-1} c")
    axD.set_ylabel("per-step energy increment", color="C1")
    axD.tick_params(axis="y", labelcolor="C1")

    lines1, labels1 = axE.get_legend_handles_labels()
    lines2, labels2 = axD.get_legend_handles_labels()
    axE.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.tight_layout()
    plt.savefig("energy_with_deltaE_overlay.png")
    plt.close("all")

    # Animation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.cla()

        current_director = director[frame]
        rotation_axis_local = omega[frame] / np.linalg.norm(omega[frame])
        rotation_axis_global = current_director.T @ rotation_axis_local
        angular_momentum_axis_local = iomega[frame] / np.linalg.norm(iomega[frame])
        angular_momentum_axis_global = current_director.T @ angular_momentum_axis_local

        # Plot the basis vectors (rows of the director matrix)
        colors = ["r", "g", "b"]
        labels = ["x", "y", "z"]
        for i in range(3):
            ax.quiver(
                0,
                0,
                0,
                current_director[i, 0],
                current_director[i, 1],
                current_director[i, 2],
                color=colors[i],
                label=f"{labels[i]}-axis",
                length=1.0,
                normalize=True,
            )

        # Plot the rotation axis
        ax.quiver(
            0,
            0,
            0,
            rotation_axis_global[0],
            rotation_axis_global[1],
            rotation_axis_global[2],
            color="k",
            label="Rotation Axis",
            length=1.0,
            normalize=True,
        )
        ax.quiver(
            0,
            0,
            0,
            angular_momentum_axis_global[0],
            angular_momentum_axis_global[1],
            angular_momentum_axis_global[2],
            color="Orange",
            label="angular momentum",
            length=1.0,
            normalize=True,
        )

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Director Orientation at Time = {time_step[frame]:.2f}s")
        ax.legend()
        ax.grid(True)


    ani = animation.FuncAnimation(fig, update, frames=len(time_step), interval=50, blit=False)

    # Save the animation as an MP4 file
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=fps, metadata=dict(artist="Me"), bitrate=1800)
    ani.save("director_animation.mp4", writer=writer)
    # plt.show() # Comment out plt.show() if you only want to save the video
