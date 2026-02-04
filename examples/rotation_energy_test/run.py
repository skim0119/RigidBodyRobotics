from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

import elastica as ea
import elastica_rigid as er


class Simulator(
    ea.BaseSystemCollection,
    ea.Forcing,
    ea.CallBacks,
):
    pass


class CallBack(ea.CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        super().__init__()
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["director"].append(system.director_collection.copy())
            self.callback_params["omega"].append(system.omega_collection.copy())
            self.callback_params["alpha"].append(system.alpha_collection.copy())
            self.callback_params["Tt"].append(system.compute_translational_energy())
            self.callback_params["Tr"].append(system.compute_rotational_energy())
            return


def run(stepper, T=10.0, dt=0.01):
    sim = Simulator()
    sphere = ea.Sphere(center=np.zeros(3), base_radius=1, density=1e3)
    sphere.omega_collection[:] = 1.0
    # To see the precession::
    sphere.mass_second_moment_of_inertia[2,2,0] *= 1
    sphere.inv_mass_second_moment_of_inertia[2,2,0] /= 1
    sim.append(sphere)

    # Simulation parameters
    simulation_time = 10.0  # (sec)
    dt = 0.01  # (sec)

    recorded_history = defaultdict(list)
    sim.collect_diagnostics(sphere).using(
        CallBack, step_skip=10, callback_params=recorded_history
    )

    sim.finalize()
    total_steps = int(T / dt)

    time = 0.0
    for i in range(total_steps):
        time = stepper.step(sim, time, dt)

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

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(time, omega)
plt.savefig("omega.png")
plt.close("all")

# Animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")


def update(frame):
    ax.cla()

    current_director = director[frame]

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
