from __future__ import annotations

import numpy as np
from matplotlib.patches import Rectangle


BALL_LABELS = ("white", "red", "yellow")
BALL_COLORS = {"white": "#9CA3AF", "red": "#D7263D", "yellow": "#F4D35E"}
BALL_EDGE_COLORS = {"white": "#374151", "red": "#7f1423", "yellow": "#8b7200"}


def plot_full_xy_trajectories(
    ax,
    traj: np.ndarray,
    *,
    show_points: bool = True,
    show_start_end: bool = True,
) -> dict[str, int]:
    """Plot full XY trajectories for all balls on ax. traj shape: (T,3,2)."""
    plotted_points: dict[str, int] = {}
    for b, label in enumerate(BALL_LABELS):
        x = traj[:, b, 0]
        y = traj[:, b, 1]
        # Dark outline so light trajectories remain visible.
        ax.plot(x, y, color="#1f2937", linewidth=3.2, alpha=0.9, zorder=1)
        ax.plot(x, y, color=BALL_COLORS[label], linewidth=1.8, label=label, zorder=2)
        if show_points:
            ax.scatter(
                x, y, color=BALL_COLORS[label], edgecolors="none", s=6, alpha=0.35, zorder=2
            )
        if show_start_end:
            ax.scatter(
                [x[0], x[-1]],
                [y[0], y[-1]],
                color=BALL_COLORS[label],
                edgecolors=BALL_EDGE_COLORS[label],
                s=28,
                zorder=3,
            )
        plotted_points[label] = int(x.shape[0])
    return plotted_points


def add_table_bounds(ax, table_width_m: float, table_height_m: float) -> None:
    rect = Rectangle(
        (0.0, 0.0),
        table_width_m,
        table_height_m,
        fill=False,
        linewidth=1.6,
        linestyle="--",
        edgecolor="#264653",
        label="table bounds",
    )
    ax.add_patch(rect)


def set_xy_limits(
    ax,
    traj: np.ndarray,
    table_width_m: float,
    table_height_m: float,
    *,
    bounds_only: bool = False,
) -> None:
    if bounds_only:
        ax.set_xlim(0.0, table_width_m)
        ax.set_ylim(0.0, table_height_m)
        return
    xmin = float(np.min(traj[:, :, 0]))
    xmax = float(np.max(traj[:, :, 0]))
    ymin = float(np.min(traj[:, :, 1]))
    ymax = float(np.max(traj[:, :, 1]))
    dx = max(1e-6, xmax - xmin)
    dy = max(1e-6, ymax - ymin)
    padx = 0.08 * dx
    pady = 0.08 * dy
    ax.set_xlim(min(xmin, 0.0) - padx, max(xmax, table_width_m) + padx)
    ax.set_ylim(min(ymin, 0.0) - pady, max(ymax, table_height_m) + pady)


def style_xy_axis(ax, *, title: str = "Extracted Ball Trajectories (Flattened XY)") -> None:
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")


def plot_position_vs_time(ax_x, ax_y, t: np.ndarray, traj: np.ndarray) -> None:
    """Plot x(t), y(t) per ball. traj shape: (T,3,2)."""
    for b, label in enumerate(BALL_LABELS):
        ax_x.plot(t, traj[:, b, 0], color=BALL_COLORS[label], linewidth=1.8, label=label)
        ax_y.plot(t, traj[:, b, 1], color=BALL_COLORS[label], linewidth=1.8, label=label)
    ax_x.set_title("Ball Position vs Time")
    ax_x.set_ylabel("x [m]")
    ax_y.set_ylabel("y [m]")
    ax_y.set_xlabel("time [s]")
    for ax in (ax_x, ax_y):
        ax.grid(alpha=0.3)
        ax.legend(loc="best")


def compute_speed_from_traj(t: np.ndarray, traj: np.ndarray) -> np.ndarray:
    """Return speed array of shape (T,3) from traj shape (T,3,2)."""
    dt = np.diff(t)
    valid_dt = dt > 1e-9
    speed = np.zeros((traj.shape[0], traj.shape[1]), dtype=np.float64)
    for b in range(traj.shape[1]):
        dxy = np.linalg.norm(np.diff(traj[:, b, :], axis=0), axis=1)
        speed[1:, b] = np.where(valid_dt, dxy / np.maximum(dt, 1e-9), 0.0)
    return speed


def plot_speed_vs_time(ax, t: np.ndarray, speed: np.ndarray) -> None:
    """Plot speed(t) per ball. speed shape: (T,3)."""
    for b, label in enumerate(BALL_LABELS):
        ax.plot(t, speed[:, b], color=BALL_COLORS[label], linewidth=1.8, label=label)
    ax.set_title("Ball Speed vs Time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("speed [m/s]")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
