"""
Sweep dt, using exact explicit solver, for the rigid-body example and produce:

1) An overlay plot of energy drift ΔE(t) = E(t) - E(0) for different dt.
2) A convergence plot showing energy drift -> 0 as dt -> 0.
"""

import numpy as np
import matplotlib.pyplot as plt

from run import run


def run_and_extract_energy(T: float, dt: float, fps: float) -> tuple[np.ndarray, np.ndarray]:
    res = run(T=T, dt=dt, fps=fps)
    t = np.asarray(res["time"])
    omega = np.asarray(res["omega"]).squeeze()
    E = np.asarray(res["energy_step"]).squeeze()
    return t, omega, E


if __name__ == "__main__":
    # Parameters
    FPS = 30
    T_FINAL = 5.0
    DT_SWEEP = [1e-3, 1e-4, 1e-5, 1e-6]

    # Run sweep
    sweep = []
    for dt in DT_SWEEP:
        t, omega, E = run_and_extract_energy(T=T_FINAL, dt=float(dt), fps=FPS)
        dE = E - E[0]
        sweep.append((float(dt), t, omega, E, dE))

    # --- Plot 1: ΔE(t) overlay for different dt ---
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    for dt, t, omega, E, dE in sweep:
        y = np.abs(dE)
        ax.plot(t, y, label=f"dt={dt:g}")
    ax.set_xlabel("time")
    ax.set_ylabel("|ΔE(t)| = |E(t) - E(0)|")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig("energy_drift_overlay_dt.png", dpi=200)
    plt.close(fig)

    # --- Plot 2: Convergence of energy drift to zero as dt -> 0 ---
    dt_vals = np.array([dt for dt, *_ in sweep], dtype=float)
    mean_drift = np.array([float(np.mean(np.abs(dE))) for _, _, _, _, dE in sweep], dtype=float)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.loglog(dt_vals, mean_drift, "s-", label="max_t |E(t)-E(0)|")

    # Reference slopes, anchored at the smallest-dt point (if nonzero)
    anchor_idx = int(np.argmin(dt_vals))
    anchor_val = max(mean_drift[anchor_idx], 1e-300)
    ref_dt1 = anchor_val * (dt_vals / dt_vals[anchor_idx]) ** 1
    ref_dt2 = anchor_val * (dt_vals / dt_vals[anchor_idx]) ** 2
    ax.loglog(dt_vals, ref_dt1, "--", color="0.55", label="reference slope: O(dt)")
    ax.loglog(dt_vals, ref_dt2, "--", color="0.35", label="reference slope: O(dt^2)")

    ax.set_xlabel("dt")
    ax.set_ylabel("energy drift metric")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig("energy_drift_convergence.png", dpi=200)
    plt.close(fig)

    # --- Plot 3: omega(t) for all dt (stacked vertically; one subplot per dt) ---
    n_cases = len(sweep)
    fig_h = max(6.0, 2.4 * n_cases)
    fig, axs = plt.subplots(n_cases, 1, figsize=(10, fig_h), sharex=True, sharey=False)
    for i, (dt, t, omega, _, _) in enumerate(sweep):
        ax = axs[i]
        for j, lbl in enumerate(["x", "y", "z"]):
            ax.plot(t, omega[:, j], label=lbl)
        ax.set_title(f"angular velocity, dt={dt:g}")
        ax.set_ylabel("angular velocity")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axs[-1].set_xlabel("time")

    plt.tight_layout()
    plt.savefig("angular_velocity_dt_sweep.png", dpi=200)
    plt.close(fig)

