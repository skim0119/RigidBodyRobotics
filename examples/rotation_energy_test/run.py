from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import elastica as ea
import elastica_rigid as er


def run(stepper, T=10.0, dt=0.01):
    sphere = er.Sphere(center=np.zeros(3), base_radius=1, density=1e3)
    sphere.omega_collection[:] = np.array(
        [[1.0], [1.0], [0.5]]
    )  # Initial angular velocity
    # To see the precession::
    sphere.mass_second_moment_of_inertia[2, 2, 0] *= 1
    sphere.inv_mass_second_moment_of_inertia[2, 2, 0] /= 1

    # Recording history
    recorded_history = defaultdict(list)
    step_skip = int(max(1, 1.0 / 30 / dt))

    time = 0.0
    total_steps = int(T / dt)
    for tidx in tqdm(range(total_steps)):
        time = stepper.step_single_instance(sphere, time, dt)

        if tidx % step_skip == 0:
            recorded_history["time"].append(time)
            recorded_history["position"].append(sphere.position_collection.copy())
            recorded_history["director"].append(sphere.director_collection.copy())
            recorded_history["omega"].append(sphere.omega_collection.copy())
            recorded_history["alpha"].append(sphere.alpha_collection.copy())
            recorded_history["Tt"].append(sphere.compute_translational_energy())
            recorded_history["Tr"].append(sphere.compute_rotational_energy())
            recorded_history["energy"].append(sphere.compute_rotational_energy())

    return recorded_history


timesteppers = {
    # "ExplicitEulerForward": er.ExplicitEulerForward(),
    # "SymplecticEulerForward": er.SymplecticEulerForward(),
    "PositionVerlet": ea.PositionVerlet(),
}
for name, stepper in timesteppers.items():
    results = run(stepper)
    break

import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Post-processing
# From list to array
time = np.array(results["time"])
omega = np.array(results["omega"]).squeeze()
alpha = np.array(results["alpha"]).squeeze()
director = np.array(results["director"]).squeeze()  # Shape (T, 3, 3)
energy = np.array(results["energy"]).squeeze()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(time, omega)
plt.savefig("omega.png")
plt.close("all")

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(time, energy)
plt.savefig("energy.png")
plt.close("all")

# Animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")


def update(frame):
    ax.cla()

    current_director = director[frame]
    rotation_axis_local = omega[frame] / np.linalg.norm(omega[frame])
    rotation_axis_global = current_director.T @ rotation_axis_local

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

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Director Orientation at Time = {time[frame]:.2f}s")
    ax.legend()
    ax.grid(True)


ani = animation.FuncAnimation(fig, update, frames=len(time), interval=50, blit=False)

# Save the animation as an MP4 file
Writer = animation.writers["ffmpeg"]
writer = Writer(fps=15, metadata=dict(artist="Me"), bitrate=1800)
ani.save("director_animation.mp4", writer=writer)
# plt.show() # Comment out plt.show() if you only want to save the video
