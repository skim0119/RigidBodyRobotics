"""Roomba feedback path-following on an infinity-shaped trajectory.

Run:
    python3 examples/roomba_path_following/run.py

Controls:
- Press r: reset robot and diagnostics.
- Press h: toggle HUD.
- Press p: toggle diagnostic plots.
- Press Esc: quit.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import tkinter as tk

import numpy as np

import elastica_rigid as er
from elastica_rigid import DEFAULT_UI_CONFIG

from controller import Controller
from model import (
    AgentConfig,
    InfinityPath,
    InfinityPathParams,
    RoombaPathFollowingModel,
    SimulationParams,
)
from policy import FeedbackControlParams, FeedbackPoseTracking


@dataclass(frozen=True)
class SimulationSetup:
    fps: int = 60
    sim: SimulationParams = field(
        default_factory=lambda: SimulationParams(
            dt=0.01,
            substeps_per_frame=2,
            robot_draw_radius=14.0,
            heading_length=34.0,
            trail_max_len=1400,
            diag_window=280,
        )
    )


@dataclass(frozen=True)
class ControlSetup:
    mild: FeedbackControlParams = field(
        default_factory=lambda: FeedbackControlParams(
            kp_position=0.30,
            kd_position=1.10,
            kp_heading=10.5,
            kd_heading=2.20,
            max_force=230.0,
            max_torque=180.0,
        )
    )
    over_dissipation_scale: float = 3.0


@dataclass(frozen=True)
class RobotSetup:
    mass: float = 1.5
    inertia: float = 8.0
    wheel_radius: float = 0.06
    width: float = 0.25


def create_roomba(x: float, y: float, robot_setup: RobotSetup) -> er.Roomba:
    return er.Roomba.create_robot(
        initial_position=np.array([x, y], dtype=float),
        initial_direction=np.array([1.0, 0.0], dtype=float),
        mass=robot_setup.mass,
        inertia=robot_setup.inertia,
        radius=robot_setup.wheel_radius,
        width=robot_setup.width,
    )


def main() -> None:
    # Main knobs for easy experimentation.
    sim_setup = SimulationSetup()
    control_setup = ControlSetup()
    robot_setup = RobotSetup()

    root = tk.Tk()
    root.title("Roomba Infinity Path Following (3 Controllers)")

    initial_x = DEFAULT_UI_CONFIG.left_panel_width * 0.5
    initial_y = DEFAULT_UI_CONFIG.window_height * 0.5

    mild_params = control_setup.mild
    over_diss_params = replace(
        mild_params,
        kd_position=mild_params.kd_position * control_setup.over_dissipation_scale,
        kd_heading=mild_params.kd_heading * control_setup.over_dissipation_scale,
    )
    no_diss_params = replace(mild_params, kd_position=0.0, kd_heading=0.0)

    agents = [
        AgentConfig(
            name="Mild",
            roomba=create_roomba(initial_x - 20.0, initial_y - 10.0, robot_setup),
            policy=FeedbackPoseTracking(mild_params),
            body_color="#4cc9f0",
            heading_color="#f9844a",
            trail_color="#4cc9f0",
        ),
        AgentConfig(
            name="Over-dissipation",
            roomba=create_roomba(initial_x + 20.0, initial_y - 10.0, robot_setup),
            policy=FeedbackPoseTracking(over_diss_params),
            body_color="#ff922b",
            heading_color="#ffd8a8",
            trail_color="#ff922b",
        ),
        AgentConfig(
            name="No dissipation",
            roomba=create_roomba(initial_x, initial_y + 20.0, robot_setup),
            policy=FeedbackPoseTracking(no_diss_params),
            body_color="#69db7c",
            heading_color="#b2f2bb",
            trail_color="#69db7c",
        ),
    ]

    stepper = er.SymplecticEulerForward()
    path = InfinityPath(
        InfinityPathParams(
            center_x=initial_x,
            center_y=initial_y,
            size=270.0,
            angular_speed=0.50,
        )
    )
    model = RoombaPathFollowingModel(
        agents=agents,
        stepper=stepper,
        path=path,
        params=sim_setup.sim,
    )

    view = er.TkView2D(root)
    controller = Controller(root, model, view, fps=sim_setup.fps)
    controller.run()

    root.mainloop()


if __name__ == "__main__":
    main()
