from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import elastica as ea
import elastica_rigid as er


def run(sphere_cls, T=10.0, dt=0.00001, fps=60):
    stepper = ea.PositionVerlet()
    sphere = sphere_cls(center=np.zeros(3), base_radius=1, density=1e3)
    sphere.dt = dt  # FIXME: Temporary

    sphere.omega_collection[:] = np.array([[0.01], [15.0], [0.01]])  # initial omega
    sphere.mass_second_moment_of_inertia[1, 1, 0] *= 2
    sphere.inv_mass_second_moment_of_inertia[1, 1, 0] /= 2
    sphere.mass_second_moment_of_inertia[2, 2, 0] *= 4
    sphere.inv_mass_second_moment_of_inertia[2, 2, 0] /= 4

    # Recording history
    recorded_history = defaultdict(list)
    step_skip = int(max(1, 1.0 / fps / dt))

    time = 0.0
    total_steps = int(T / dt)
    for tidx in tqdm(range(total_steps)):
        time = stepper.step_single_instance(sphere, time, dt)

        if tidx % step_skip == 0:
            recorded_history["time"].append(time)
            recorded_history["position"].append(sphere.position_collection.copy())
            recorded_history["director"].append(sphere.director_collection.copy())
            iomega = (
                sphere.mass_second_moment_of_inertia[:, :, 0]
                @ sphere.omega_collection[:, 0]
            )
            recorded_history["Iomega"].append(iomega)
            recorded_history["omega"].append(sphere.omega_collection.copy())
            recorded_history["alpha"].append(sphere.alpha_collection.copy())
            recorded_history["Tt"].append(sphere.compute_translational_energy())
            recorded_history["Tr"].append(sphere.compute_rotational_energy())
            recorded_history["energy"].append(sphere.compute_rotational_energy())
            # Implicit midpoint diagnostics (from SphereImplicit.update_accelerations)
            recorded_history["implicit_equation_residual"].append(
                float(getattr(sphere, "implicit_midpoint_equation_residual", np.nan))
            )
            recorded_history["implicit_iterations"].append(
                int(getattr(sphere, "implicit_midpoint_iterations", -1))
            )

    return recorded_history


FPS = 20
T_FINAL = 5.0

DT_SWEEP = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]  # edit as you like. 1e-8 gives very similar results.

all_errors = []
for dt in DT_SWEEP:
    res_exact = run(
        sphere_cls=er.SphereExact, T=T_FINAL, dt=dt, fps=FPS
    )
    res_impl = run(sphere_cls=er.SphereImplicit, T=T_FINAL, dt=dt, fps=FPS)

    t_exact = np.array(res_exact["time"], dtype=float)
    t_impl = np.array(res_impl["time"], dtype=float)

    omega_exact_body = np.array(res_exact["omega"]).squeeze()
    omega_impl_body = np.array(res_impl["omega"]).squeeze()
    director_exact = np.array(res_exact["director"]).squeeze()
    director_impl = np.array(res_impl["director"]).squeeze()

    # transform omega to the same plotted frame as other scripts in this repo
    omega_exact_plot = np.einsum("ikj,ik->ij", director_exact, omega_exact_body)
    omega_impl_plot = np.einsum("ikj,ik->ij", director_impl, omega_impl_body)

    err = omega_impl_plot - omega_exact_plot
    err_norm = np.linalg.norm(err, axis=1)
    all_errors.append(
        {
            "dt": float(dt),
            "time": t_impl,
            "err_norm": err_norm,
            "omega_exact_plot": omega_exact_plot,
            "omega_impl_plot": omega_impl_plot,
            "omega_exact_body": omega_exact_body,
            "omega_impl_body": omega_impl_body,
        }
    )

# Post-processing
# 1) overlay omega(t) for all dt (stacked vertically; one subplot per dt)
n_cases = len(all_errors)
fig_h = max(6.0, 2.4 * n_cases)
fig, axs = plt.subplots(n_cases, 1, figsize=(10, fig_h), sharex=True, sharey=False)
if n_cases == 1:
    axs = [axs]

period_markers = [0.722, 2.167, 3.611]
component_labels = ["x", "y", "z"]

for i, item in enumerate(all_errors):
    ax = axs[i]
    t = item["time"]
    w_exact = item["omega_exact_plot"]
    w_impl = item["omega_impl_plot"]

    for j, lbl in enumerate(component_labels):
        # Same color for exact vs implicit, different linestyle
        ax.plot(t, w_exact[:, j], "-", alpha=0.9, label=f"exact ω{lbl}")
        ax.plot(t, w_impl[:, j], "--", alpha=0.9, label=f"implicit ω{lbl}")

    for x in period_markers:
        ax.axvline(x, color="red", linestyle="--", alpha=0.35, label="expected-period")

    ax.set_title(f"ω(t) implicit midpoint vs exact (dt={item['dt']:.1e})")
    ax.set_ylabel(r"$\omega$ (plotted frame)")
    ax.grid(True, alpha=0.3)

    # Avoid repeating a huge legend on every subplot
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        # Deduplicate labels while preserving order
        seen = set()
        uniq = [
            (h, lbl)
            for h, lbl in zip(handles, labels)
            if not (lbl in seen or seen.add(lbl))
        ]
        ax.legend([h for h, _ in uniq], [lbl for _, lbl in uniq], loc="best", ncol=3)

axs[-1].set_xlabel("time")
plt.tight_layout()
plt.savefig("omega_overlay_dt_sweep.png", dpi=200)
plt.close("all")

# 2) error vs time for each dt
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
for item in all_errors:
    ax.semilogy(item["time"], item["err_norm"] + 1e-20, label=f"dt={item['dt']:.2e}")
ax.set_xlabel("time")
ax.set_ylabel(r"$||\omega_{\mathrm{impl}}-\omega_{\mathrm{exact}}||_2$")
ax.set_xlim(0, 2)
ax.set_title("Omega difference vs time (smaller dt -> smaller error)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig("omega_error_vs_time_dt_sweep.png")
plt.close("all")

# 3) summary error vs dt (no warmup)
# L1: max error over time, L2: mean error over time
dts = np.array([item["dt"] for item in all_errors], dtype=float)
errs_L1 = []
errs_L2 = []
for item in all_errors:
    err = item["err_norm"]
    errs_L1.append(float(np.max(err)))
    errs_L2.append(float(np.mean(err)))
errs_L1 = np.array(errs_L1, dtype=float)
errs_L2 = np.array(errs_L2, dtype=float)

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111)
ax.loglog(dts, errs_L1, "o-", label="L1: max_t")
ax.loglog(dts, errs_L2, "s-", label="L2: mean_t")
ax.set_xlabel("dt")
ax.set_ylabel("error")
ax.set_title("Convergence: omega difference decreases as dt decreases")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig("omega_error_vs_dt.png")
plt.close("all")
