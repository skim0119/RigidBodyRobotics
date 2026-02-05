from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import elastica as ea
import elastica_rigid as er


def run(stepper, T=100.0, dt=0.01):
    sphere = er.Sphere(center=np.zeros(3), base_radius=1, density=1e3)
    sphere.omega_collection[:] = np.array(
        [[1.0], [1.0], [0.5]]
    )  # Initial angular velocity
    sphere.mass_second_moment_of_inertia[2, 2, 0] *= 10
    sphere.inv_mass_second_moment_of_inertia[2, 2, 0] /= 10

    isphere = er.SphereImplicit(center=np.zeros(3), base_radius=1, density=1e3)
    isphere.dt = dt  # FIXME: This is quick fix to put dt in update_accelerations.
    isphere.omega_collection[:] = np.array(
        [[1.0], [1.0], [0.5]]
    )  # Initial angular velocity
    isphere.mass_second_moment_of_inertia[2, 2, 0] *= 10
    isphere.inv_mass_second_moment_of_inertia[2, 2, 0] /= 10

    # Recording history
    recorded_history = defaultdict(list)
    step_skip = int(max(1, 1.0 / 30 / dt))

    time = 0.0
    total_steps = int(T / dt)
    for tidx in tqdm(range(total_steps)):
        sphere.external_torques[:] = np.sin(np.pi * tidx * dt)
        isphere.external_torques[:] = np.sin(np.pi * tidx * dt)

        stepper.step_single_instance(sphere, time, dt)
        time = stepper.step_single_instance(isphere, time, dt)

        if tidx % step_skip == 0:
            recorded_history["time"].append(time)
            recorded_history["energy_sphere"].append(sphere.compute_rotational_energy())
            recorded_history["energy_sphere_implicit"].append(
                isphere.compute_rotational_energy()
            )

    return recorded_history


timesteppers = {
    # "ExplicitEulerForward": er.ExplicitEulerForward(),
    # "SymplecticEulerForward": er.SymplecticEulerForward(),
    "PositionVerlet": ea.PositionVerlet(),
}
for name, stepper in timesteppers.items():
    results = run(stepper)
    break

# Post-processing
# From list to array
time = np.array(results["time"])
energy = np.array(results["energy_sphere"]).squeeze()
ienergy = np.array(results["energy_sphere_implicit"]).squeeze()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
# ax.plot(time, energy, label="original explicit lagrange transport")
ax.plot(time, ienergy, label="implicit mid-omega for lagrange transport")
ax.legend()
plt.savefig("energy.png")
plt.close("all")
