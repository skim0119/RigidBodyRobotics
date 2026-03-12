"""Task 2: Estimation performance with low-pass and complementary filters."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import elastica as ea

from common import DEFAULT_DT, SIM_TIME, make_simulator, project2_results_dir, rmse
from estimation import ComplementaryHeadingFilter, GyroIntegrator, LowPassPositionFilter
from sensing import DEFAULT_GPS_COMPASS_NOISE, GPSCompassSensor, GyroscopeSensor


SEED = 20260306
ALPHA_X = 0.12
ALPHA_THETA = 0.985


def run_task2(seed: int = SEED) -> dict:
    sim, robot, true_log = make_simulator()

    gps_log = defaultdict(list)
    gyro_log = defaultdict(list)
    sim.collect_diagnostics(robot).using(
        GPSCompassSensor,
        step_skip=1,
        callback_params=gps_log,
        seed=seed,
        config=DEFAULT_GPS_COMPASS_NOISE,
    )
    sim.collect_diagnostics(robot).using(
        GyroscopeSensor,
        step_skip=1,
        callback_params=gyro_log,
        seed=seed + 1,
    )

    sim.finalize()

    total_steps = int(SIM_TIME / DEFAULT_DT)
    stepper = ea.PositionVerlet()
    time = 0.0
    for _ in range(total_steps):
        time = stepper.step(sim, time, DEFAULT_DT)

    t = np.asarray(true_log["time"], dtype=np.float64)
    x_true = np.asarray(true_log["position"], dtype=np.float64)
    d1_true = np.asarray(true_log["direction"], dtype=np.float64)
    theta_true = np.arctan2(d1_true[:, 1], d1_true[:, 0])

    y_gps = np.asarray(gps_log["y_gps"], dtype=np.float64)
    y_compass = np.asarray(gps_log["y_compass"], dtype=np.float64)
    y_gyro = np.asarray(gyro_log["y_gyro"], dtype=np.float64)

    pos_filter = LowPassPositionFilter(alpha_x=ALPHA_X, xhat=y_gps[0])
    heading_filter = ComplementaryHeadingFilter(
        alpha_theta=ALPHA_THETA,
        dt=DEFAULT_DT,
        theta_hat=y_compass[0],
        estimate_bias=False,
    )
    gyro_integrator = GyroIntegrator(dt=DEFAULT_DT, theta=float(theta_true[0]))

    xhat = np.zeros_like(y_gps)
    thetahat = np.zeros_like(theta_true)
    theta_gyro_int = np.zeros_like(theta_true)

    for i in range(t.size):
        xhat[i] = pos_filter.update(y_gps[i])
        thetahat[i] = heading_filter.update(float(y_compass[i]), float(y_gyro[i]))
        theta_gyro_int[i] = gyro_integrator.update(float(y_gyro[i]))

    return {
        "time": t,
        "x_true": x_true,
        "theta_true": theta_true,
        "y_gps": y_gps,
        "y_compass": y_compass,
        "y_gyro": y_gyro,
        "xhat": xhat,
        "thetahat": thetahat,
        "theta_gyro_int": theta_gyro_int,
        "alpha_x": ALPHA_X,
        "alpha_theta": ALPHA_THETA,
        "seed": seed,
    }


def save_figures(data: dict, out_dir: Path) -> None:
    t = data["time"]
    x_true = data["x_true"]
    y_gps = data["y_gps"]
    xhat = data["xhat"]

    theta_true = data["theta_true"]
    y_compass = data["y_compass"]
    theta_gyro_int = data["theta_gyro_int"]
    thetahat = data["thetahat"]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    axes[0].plot(t, x_true[:, 0], label="x_true", lw=1.5)
    axes[0].plot(t, y_gps[:, 0], label="gps_x", lw=0.9)
    axes[0].plot(t, xhat[:, 0], label="xhat_x", lw=1.2)
    axes[0].set_ylabel("x (m)")
    axes[0].set_title("Task 2: Position estimation on x-axis")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, x_true[:, 1], label="y_true", lw=1.5)
    axes[1].plot(t, y_gps[:, 1], label="gps_y", lw=0.9)
    axes[1].plot(t, xhat[:, 1], label="xhat_y", lw=1.2)
    axes[1].set_ylabel("y (m)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Task 2: Position estimation on y-axis")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "task2_position_estimation.png", dpi=180)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(t, theta_true, label="theta_true", lw=1.6)
    ax2.plot(t, y_compass, label="compass", lw=0.9)
    ax2.plot(t, theta_gyro_int, label="integrated_gyro", lw=1.1)
    ax2.plot(t, thetahat, label="thetahat (complementary)", lw=1.4)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Heading (rad)")
    ax2.set_title("Task 2: Heading estimation")
    ax2.grid(alpha=0.3)
    ax2.legend(ncol=2)
    fig2.tight_layout()
    fig2.savefig(out_dir / "task2_heading_estimation.png", dpi=180)
    plt.close(fig2)


def summarize_metrics(data: dict, out_dir: Path) -> dict[str, float]:
    x_true = data["x_true"]
    y_gps = data["y_gps"]
    xhat = data["xhat"]

    theta_true = data["theta_true"]
    y_compass = data["y_compass"]
    theta_gyro_int = data["theta_gyro_int"]
    thetahat = data["thetahat"]

    gps_rmse = rmse(y_gps, x_true)
    xhat_rmse = rmse(xhat, x_true)

    compass_rmse = rmse(y_compass, theta_true, angle=True)
    gyro_int_rmse = rmse(theta_gyro_int, theta_true, angle=True)
    thetahat_rmse = rmse(thetahat, theta_true, angle=True)

    time = data["time"]
    stationary = (time >= 0.5) & (time <= 2.0)
    xhat_stationary = xhat[stationary]

    empirical_var_x = float(np.var(xhat_stationary[:, 0]))
    empirical_var_y = float(np.var(xhat_stationary[:, 1]))

    theory_var = LowPassPositionFilter.steady_state_variance(
        data["alpha_x"], DEFAULT_GPS_COMPASS_NOISE.sigma_gps2
    )

    metrics = {
        "gps_pos_rmse": gps_rmse,
        "filtered_pos_rmse": xhat_rmse,
        "compass_heading_rmse": compass_rmse,
        "gyro_int_heading_rmse": gyro_int_rmse,
        "filtered_heading_rmse": thetahat_rmse,
        "empirical_var_x_stationary": empirical_var_x,
        "empirical_var_y_stationary": empirical_var_y,
        "theory_var_stationary": theory_var,
        "var_ratio_x": empirical_var_x / theory_var,
        "var_ratio_y": empirical_var_y / theory_var,
    }

    lines = [
        "Task 2 Estimation Summary",
        f"seed={int(data['seed'])}",
        f"alpha_x={float(data['alpha_x']):.4f}",
        f"alpha_theta={float(data['alpha_theta']):.4f}",
        "",
        "RMSE table:",
        f"raw GPS position vs truth: {metrics['gps_pos_rmse']:.6f}",
        f"filtered position vs truth: {metrics['filtered_pos_rmse']:.6f}",
        f"compass heading vs truth: {metrics['compass_heading_rmse']:.6f}",
        f"integrated gyro heading vs truth: {metrics['gyro_int_heading_rmse']:.6f}",
        f"filtered heading vs truth: {metrics['filtered_heading_rmse']:.6f}",
        "",
        "Stationary variance check (0.5s-2.0s):",
        f"theory_var = alpha/(2-alpha)*sigma_gps^2 = {metrics['theory_var_stationary']:.6f}",
        f"empirical_var_x = {metrics['empirical_var_x_stationary']:.6f} (ratio {metrics['var_ratio_x']:.3f})",
        f"empirical_var_y = {metrics['empirical_var_y_stationary']:.6f} (ratio {metrics['var_ratio_y']:.3f})",
    ]
    (out_dir / "task2_summary.txt").write_text("\n".join(lines) + "\n")
    return metrics


def main() -> None:
    out_dir = project2_results_dir()
    data = run_task2()

    save_figures(data, out_dir)
    metrics = summarize_metrics(data, out_dir)

    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
