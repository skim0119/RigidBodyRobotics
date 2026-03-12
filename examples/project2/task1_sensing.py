"""Task 1: Sensing model validation on open-loop motion."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import elastica as ea

from common import (
    DEFAULT_DT,
    SIM_TIME,
    make_simulator,
    project2_results_dir,
    wrap_angle_array,
)
from estimation import GyroIntegrator
from sensing import (
    DEFAULT_GPS_COMPASS_NOISE,
    DEFAULT_GYROSCOPE_NOISE,
    GPSCompassSensor,
    GyroscopeSensor,
)


def run_task1() -> dict:
    sim, robot, true_log = make_simulator()

    gps_log = defaultdict(list)
    gyro_log = defaultdict(list)

    sim.collect_diagnostics(robot).using(
        GPSCompassSensor,
        step_skip=1,
        callback_params=gps_log,
        seed=10,
    )
    sim.collect_diagnostics(robot).using(
        GyroscopeSensor,
        step_skip=1,
        callback_params=gyro_log,
        seed=20,
    )

    sim.finalize()

    total_steps = int(SIM_TIME / DEFAULT_DT)
    stepper = ea.PositionVerlet()
    time = 0.0
    for _ in range(total_steps):
        time = stepper.step(sim, time, DEFAULT_DT)

    t_true = np.asarray(true_log["time"], dtype=np.float64)
    pos_true = np.asarray(true_log["position"], dtype=np.float64)
    dir_true = np.asarray(true_log["direction"], dtype=np.float64)
    omega_true = np.asarray(true_log["angular_velocity"], dtype=np.float64)

    y_gps = np.asarray(gps_log["y_gps"], dtype=np.float64)
    y_compass = np.asarray(gps_log["y_compass"], dtype=np.float64)
    y_gyro = np.asarray(gyro_log["y_gyro"], dtype=np.float64)
    theta_true = np.arctan2(dir_true[:, 1], dir_true[:, 0])

    theta_gyro = GyroIntegrator(dt=DEFAULT_DT, theta=float(theta_true[0]))
    theta_gyro_int = np.zeros_like(y_gyro)
    for i in range(theta_gyro_int.size):
        theta_gyro_int[i] = theta_gyro.update(y_gyro[i])

    gps_residual = y_gps - pos_true
    compass_residual = wrap_angle_array(y_compass - theta_true)
    gyro_residual = y_gyro - (omega_true + DEFAULT_GYROSCOPE_NOISE.mu_beta)

    return {
        "time": t_true,
        "x_true": pos_true,
        "theta_true": theta_true,
        "omega_true": omega_true,
        "y_gps": y_gps,
        "y_compass": y_compass,
        "y_gyro": y_gyro,
        "theta_gyro_int": theta_gyro_int,
        "gps_residual": gps_residual,
        "compass_residual": compass_residual,
        "gyro_residual": gyro_residual,
        "gps_compass_cov": np.diag(
            [
                DEFAULT_GPS_COMPASS_NOISE.sigma_gps2,
                DEFAULT_GPS_COMPASS_NOISE.sigma_gps2,
                DEFAULT_GPS_COMPASS_NOISE.sigma_compass2,
            ]
        ).astype(np.float64),
    }


def save_figures(data: dict, out_dir: Path) -> None:
    t = data["time"]
    x_true = data["x_true"]
    theta_true = data["theta_true"]
    omega_true = data["omega_true"]

    y_gps = data["y_gps"]
    y_compass = data["y_compass"]
    y_gyro = data["y_gyro"]
    theta_gyro_int = data["theta_gyro_int"]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9))

    axes[0].plot(t, y_gps[:, 0], label="gps_x", lw=1.0)
    axes[0].plot(t, y_gps[:, 1], label="gps_y", lw=1.0)
    axes[0].plot(t, x_true[:, 0], label="x_true", lw=1.5)
    axes[0].plot(t, x_true[:, 1], label="y_true", lw=1.5)
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title("Task 1: GPS vs true position")
    axes[0].grid(alpha=0.3)
    axes[0].legend(ncol=2, fontsize=8)

    axes[1].plot(t, y_compass, label="compass", lw=1.0)
    axes[1].plot(t, theta_true, label="theta_true", lw=1.6)
    axes[1].set_ylabel("Heading (rad)")
    axes[1].set_title("Task 1: Compass vs true heading")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(t, y_gyro, label="gyro", lw=1.0)
    axes[2].plot(t, omega_true, label="omega_true", lw=1.6)
    axes[2].set_ylabel("Angular rate (rad/s)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Task 1: Gyro vs true angular rate")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "task1_sensor_timeseries.png", dpi=180)
    plt.close(fig)

    gps_res = data["gps_residual"]
    compass_res = data["compass_residual"]
    gyro_res = data["gyro_residual"]

    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
    axes2 = axes2.ravel()
    axes2[0].hist(gps_res[:, 0], bins=40, alpha=0.75, color="C0")
    axes2[0].set_title("GPS residual x")
    axes2[1].hist(gps_res[:, 1], bins=40, alpha=0.75, color="C1")
    axes2[1].set_title("GPS residual y")
    axes2[2].hist(compass_res, bins=40, alpha=0.75, color="C2")
    axes2[2].set_title("Compass residual")
    axes2[3].hist(gyro_res, bins=40, alpha=0.75, color="C3")
    axes2[3].set_title("Gyro residual")
    for ax in axes2:
        ax.grid(alpha=0.25)
    fig2.suptitle("Task 1: Sensor residual histograms")
    fig2.tight_layout()
    fig2.savefig(out_dir / "task1_residual_histograms.png", dpi=180)
    plt.close(fig2)

    heading_drift = wrap_angle_array(theta_gyro_int - theta_true)
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(t, theta_true, label="theta_true", lw=1.5)
    ax3.plot(t, theta_gyro_int, label="integrated_gyro_heading", lw=1.2)
    ax3.plot(t, heading_drift, label="drift = wrap(theta_gyro - theta_true)", lw=1.2)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Angle (rad)")
    ax3.set_title("Task 1: Gyro integrated heading drift")
    ax3.grid(alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(out_dir / "task1_gyro_heading_drift.png", dpi=180)
    plt.close(fig3)


def summarize_statistics(data: dict, out_dir: Path) -> dict[str, float]:
    gps_res = data["gps_residual"]
    compass_res = data["compass_residual"]
    gyro_res = data["gyro_residual"]

    stats = {
        "gps_x_mean": np.mean(gps_res[:, 0]),
        "gps_x_var": np.var(gps_res[:, 0]),
        "gps_y_mean": np.mean(gps_res[:, 1]),
        "gps_y_var": np.var(gps_res[:, 1]),
        "compass_mean": np.mean(compass_res),
        "compass_var": np.var(compass_res),
        "gyro_mean": np.mean(gyro_res),
        "gyro_var": np.var(gyro_res),
    }

    target_gps_compass = DEFAULT_GPS_COMPASS_NOISE
    target_gyro = DEFAULT_GYROSCOPE_NOISE
    summary_lines = [
        "Task 1 Sensing Summary",
        "Empirical residual mean/variance:",
        f"gps_x: mean={stats['gps_x_mean']:.6f}, var={stats['gps_x_var']:.6f}, target_var={target_gps_compass.sigma_gps2:.6f}",
        f"gps_y: mean={stats['gps_y_mean']:.6f}, var={stats['gps_y_var']:.6f}, target_var={target_gps_compass.sigma_gps2:.6f}",
        f"compass: mean={stats['compass_mean']:.6f}, var={stats['compass_var']:.6f}, target_var={target_gps_compass.sigma_compass2:.6f}",
        f"gyro: mean={stats['gyro_mean']:.6f}, var={stats['gyro_var']:.6f}, target_var={target_gyro.sigma_gyro2:.6f}",
    ]

    (out_dir / "task1_summary.txt").write_text("\n".join(summary_lines) + "\n")
    return stats


def main() -> None:
    out_dir = project2_results_dir()
    data = run_task1()

    save_figures(data, out_dir)  # (1), (2), (3)
    stats = summarize_statistics(data, out_dir)

    for key, value in stats.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
