"""Animation of robot on a 2D field with optional environment stamping."""

from __future__ import annotations

from typing import Any, Optional

from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class Visualize:
    """
    Generate animations of robot motion from time-series data.

    Takes time, position, direction, external force, and wheel force arrays,
    and supports drawing a static environment (walls, obstacles, friction region)
    plus animated robot, trajectory, and force arrows.
    """

    def __init__(
        self,
        time: NDArray[np.float64],
        position: NDArray[np.float64],
        direction: NDArray[np.float64],
        external_force: NDArray[np.float64],
        *,
        robot_radius: float = 0.2,
        fps: int = 30,
        step_skip: int = 1,
        show_trajectory: bool = True,
        show_external_forces: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        time : np.ndarray
            Shape (n,) time points.
        position : np.ndarray
            Shape (n, 2) robot positions [x, y].
        direction : np.ndarray
            Shape (n, 2) unit heading vectors.
        external_force : np.ndarray
            Shape (n, 2) external forces at each time step.
        robot_radius : float
            Robot disk radius for drawing (default 0.2).
        fps : int
            Frames per second for animation (default 30).
        step_skip : int
            Use every step_skip-th frame to shorten animation (default 1).
        show_trajectory : bool
            Draw full path trail (default True).
        show_external_forces : bool
            Draw external force arrows (default True).
        """
        n = len(time)
        if position.shape != (n, 2) or direction.shape != (n, 2):
            raise ValueError(
                "position and direction must have shape (n, 2) with n = len(time)"
            )
        if external_force.shape != (n, 2):
            raise ValueError("external_force must have shape (n, 2)")

        self._time = np.asarray(time, dtype=np.float64)
        self._position = np.asarray(position, dtype=np.float64)
        self._direction = np.asarray(direction, dtype=np.float64)
        self._external_force = np.asarray(external_force, dtype=np.float64)
        self._robot_radius = float(robot_radius)
        self._fps = int(fps)
        self._step_skip = max(1, int(step_skip))
        self._show_trajectory = bool(show_trajectory)
        self._show_external_forces = bool(show_external_forces)

        # Indices used in animation (after step_skip)
        self._frame_indices = np.arange(0, n, self._step_skip)
        if self._frame_indices.size == 0:
            self._frame_indices = np.array([0])

        # Environment stamping: set by stamp_environment(), used in _setup_axes
        self._bounds: Optional[tuple[float, float, float, float]] = None
        self._obstacles: Optional[list[tuple[float, float, float, float]]] = None

        self._fig: Optional[Figure] = None
        self._ax: Optional[Axes] = None
        self._anim: Optional[FuncAnimation] = None
        self._traj_line: Any = None
        self._robot_circle: Any = None
        self._dir_arrow: Any = None
        self._ext_force_arrow: Any = None

    @staticmethod
    def stamp_environment(
        ax: Axes,
        bounds: tuple[float, float, float, float],
        obstacles: Optional[list[tuple[float, float, float, float]]] = None,
    ) -> None:
        """
        Draw static environment on the given axes: walls and obstacles.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to draw on.
        bounds : tuple of float
            (xmin, xmax, ymin, ymax) workspace bounds.
        obstacles : list of tuple, optional
            Each (oxmin, oxmax, oymin, oymax) for a rectangular obstacle.
        """
        xmin, xmax, ymin, ymax = bounds
        ax.plot(
            [xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            "k-",
            lw=2,
            zorder=1,
        )
        if obstacles:
            for oxmin, oxmax, oymin, oymax in obstacles:
                ax.fill(
                    [oxmin, oxmax, oxmax, oxmin, oxmin],
                    [oymin, oymin, oymax, oymax, oymin],
                    color="white",
                    edgecolor="black",
                    linewidth=2,
                    zorder=2,
                )
        ax.set_xlim(xmin - 0.1, xmax + 0.1)
        ax.set_ylim(ymin - 0.1, ymax + 0.1)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.25)

    def _stamp_environment_if_set(self) -> None:
        if self._ax is None:
            return
        if self._bounds is not None:
            self.stamp_environment(
                self._ax,
                self._bounds,
                obstacles=self._obstacles,
            )

    def stamp_environment_on_figure(
        self,
        bounds: tuple[float, float, float, float],
        obstacles: Optional[list[tuple[float, float, float, float]]] = None,
    ) -> None:
        """
        Set environment parameters to be drawn when building the animation figure.
        Call before animate(), save_gif(), or save_mp4().
        """
        self._bounds = bounds
        self._obstacles = obstacles

    def _setup_figure(self) -> None:
        self._fig, self._ax = plt.subplots(figsize=(7, 8))
        self._stamp_environment_if_set()
        # If no bounds were set, set lims from data
        if self._bounds is None and self._ax is not None:
            margin = 0.5
            x = self._position[:, 0]
            y = self._position[:, 1]
            self._ax.set_xlim(x.min() - margin, x.max() + margin)
            self._ax.set_ylim(y.min() - margin, y.max() + margin)
            self._ax.set_aspect("equal")
            self._ax.set_xlabel("x (m)")
            self._ax.set_ylabel("y (m)")
            self._ax.grid(True, alpha=0.25)

    def _init_artists(self) -> None:
        """Create trajectory line, robot circle, and arrow artists (stored on self)."""
        ax = self._ax
        if ax is None:
            return
        self._traj_line = ax.plot([], [], "C0-", lw=1.5, label="Trajectory", zorder=3)[
            0
        ]
        self._robot_circle = plt.Circle(
            (0.0, 0.0),
            self._robot_radius,
            color="C0",
            fill=True,
            ec="k",
            lw=1,
            zorder=5,
        )
        ax.add_patch(self._robot_circle)
        # Create invisible proxy lines for legend (only for legend display)
        self._dir_arrow_proxy = ax.plot(
            [], [], color="C1", linewidth=1.5, label="Direction", visible=False
        )[0]
        self._ext_force_arrow_proxy = ax.plot(
            [], [], color="red", linewidth=1.5, label="External Force", visible=False
        )[0]
        # Arrows created in first frame
        self._dir_arrow = None
        self._ext_force_arrow = None

    def _update_frame(self, frame_idx: int) -> None:
        i = self._frame_indices[frame_idx]
        pos = self._position[i]
        d = self._direction[i]
        ef = self._external_force[i]
        ax = self._ax
        if ax is None:
            return

        # Trajectory up to current time
        if self._show_trajectory:
            self._traj_line.set_data(
                self._position[: i + 1, 0], self._position[: i + 1, 1]
            )
        else:
            self._traj_line.set_data([], [])

        # Robot circle at current position
        self._robot_circle.center = (float(pos[0]), float(pos[1]))

        # Direction arrow (from center, along direction)
        arrow_len = self._robot_radius * 1.5
        dx, dy = float(d[0]) * arrow_len, float(d[1]) * arrow_len
        if self._dir_arrow is not None:
            self._dir_arrow.remove()
        self._dir_arrow = ax.arrow(
            pos[0],
            pos[1],
            dx,
            dy,
            head_width=0.05,
            head_length=0.03,
            fc="C1",
            ec="C1",
            zorder=6,
        )

        # Force arrows: normalize magnitude for visibility
        scale = 0.15
        ef_norm = float(np.linalg.norm(ef))
        if ef_norm > 1e-12 and self._show_external_forces:
            ex = ef[0] * scale / ef_norm
            ey = ef[1] * scale / ef_norm
        else:
            ex, ey = 0.0, 0.0
        if self._ext_force_arrow is not None:
            self._ext_force_arrow.remove()
        self._ext_force_arrow = ax.arrow(
            pos[0],
            pos[1],
            ex,
            ey,
            head_width=0.02,
            head_length=0.02,
            fc="red",
            ec="red",
            zorder=6,
        )

    def animate(self) -> FuncAnimation:
        """
        Build figure and animation and display it (plt.show() is left to the user).
        Returns the FuncAnimation instance.
        """
        self._setup_figure()
        if self._ax is None:
            raise RuntimeError("figure setup failed")
        self._init_artists()

        pbar = tqdm(total=len(self._frame_indices), desc="Animating")

        def update(frame_idx: int) -> None:
            pbar.update(1)
            self._update_frame(frame_idx)

        self._anim = FuncAnimation(
            self._fig,
            update,
            frames=len(self._frame_indices),
            interval=1000 / self._fps,
            blit=False,
        )
        self._ax.legend(loc="upper right")
        return self._anim

    def save_mp4(self, filename: str, fps: Optional[int] = None) -> None:
        """Save animation as MP4. Requires ffmpeg."""
        if self._fig is None or self._anim is None:
            self.animate()
        if self._anim is None:
            raise RuntimeError("animation not created")
        try:
            writer = plt.rcParams.get("animation.ffmpeg_path", "ffmpeg")
            from matplotlib.animation import FFMpegWriter

            w = FFMpegWriter(fps=fps or self._fps)
            self._anim.save(filename, writer=w)
        except Exception as e:
            raise RuntimeError("MP4 save failed (is ffmpeg installed?): ") from e
