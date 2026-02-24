"""Simulation model and view query API for the SE(2) Roomba sandbox."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

import elastica as ea
from elastica_rigid import Roomba
from elastica_rigid.visualize.tk_app.protocol import (
    ObjectPose2D,
    PlotPanel,
    PlotSeries,
    TargetPose2D,
    Trail2D,
)

from .telemetry import Telemetry


@dataclass(frozen=True)
class SimConfig:
    dt: float = 1.0 / 60.0

    # Passive dissipation time constants
    tau_v_damp: float = 0.8
    tau_omega_damp: float = 0.45


DEFAULT_SIM_CONFIG = SimConfig()


class PolicyProtocol(Protocol):
    """
    Duck-typed policy interface consumed by SimulationModel.

    compute_action should return the thurst on left and right wheels.
    """

    def compute_action(
        self,
        *,
        roomba: Roomba,
        target_x: float,
        target_y: float,
        target_theta: float,
        prev_u_left: float,
        prev_u_right: float,
    ) -> tuple[float, float]: ...


class SimulationModel:
    def __init__(
        self,
        *,
        roomba: Roomba,
        stepper: ea.StepperProtocol,
        policy: PolicyProtocol,
        initial_x: float,
        initial_y: float,
        config: SimConfig = DEFAULT_SIM_CONFIG,
        telemetry: Telemetry | None = None,
    ) -> None:
        self.roomba = roomba
        self.stepper = stepper
        self.config = config
        self.policy = policy
        self.initial_x = initial_x
        self.initial_y = initial_y

        self.target_x = initial_x
        self.target_y = initial_y
        self.target_theta = 0.0
        self.time = 0.0

        self.u_left = 0.0
        self.u_right = 0.0

        # I keep the default behavior like this for now.
        # Maybe in the future, using dependency injection might be better.
        # I think it is okay, history size and trail size is somewhat irrelavent,
        # and I this is demo case for model.
        self.telemetry = telemetry
        if self.telemetry is None:
            self.telemetry = Telemetry(
                history_size=300,
                trail_size=220,
            )

    def compute_current_theta(self) -> float:
        return math.atan2(
            float(self.roomba.direction[1]), float(self.roomba.direction[0])
        )

    # View query API
    def get_object_poses(self) -> list[ObjectPose2D]:
        return [
            ObjectPose2D(
                x=float(self.roomba.position[0]),
                y=float(self.roomba.position[1]),
                dir_x=float(self.roomba.direction[0]),
                dir_y=float(self.roomba.direction[1]),
                radius=28.0,
                heading_length=42.0,
            )
        ]

    def get_target_pose(self) -> TargetPose2D:
        return TargetPose2D(
            x=self.target_x,
            y=self.target_y,
            theta=self.target_theta,
            marker_radius=7.0,
            heading_length=42.0,
        )

    def get_trails(self) -> list[Trail2D]:
        return [Trail2D(points=self.telemetry.trail, width=2)]

    def get_plotting_data(self) -> list[PlotPanel]:
        pos_x = list(self.telemetry.robot_x_hist) + list(self.telemetry.target_x_hist)
        pos_y = list(self.telemetry.robot_y_hist) + list(self.telemetry.target_y_hist)
        pos_x += [float(self.roomba.position[0]), self.target_x]
        pos_y += [float(self.roomba.position[1]), self.target_y]

        max_pos = (
            max(pos_x + pos_y)
            if (pos_x or pos_y)
            else max(self.initial_x, self.initial_y)
        )
        max_range = float(max(1.0, max_pos + 5.0))
        return [
            PlotPanel(
                title="Wheel Power (W): left vs right",
                series=[
                    PlotSeries(
                        color="#ff6b6b",
                        values=self.telemetry.left_power_hist,
                        label="left power",
                    ),
                    PlotSeries(
                        color="#4dabf7",
                        values=self.telemetry.right_power_hist,
                        label="right power",
                    ),
                ],
            ),
            PlotPanel(
                title="Orientation (deg): robot vs cursor target",
                series=[
                    PlotSeries(
                        color="#f9844a",
                        values=self.telemetry.robot_theta_hist,
                        label="robot theta",
                    ),
                    PlotSeries(
                        color="#90be6d",
                        values=self.telemetry.target_theta_hist,
                        label="cursor theta target",
                    ),
                ],
                fixed_range=(-180.0, 180.0),
            ),
            PlotPanel(
                title="Position (px): x robot/target, y robot/target",
                series=[
                    PlotSeries(
                        color="#ffd166",
                        values=self.telemetry.robot_x_hist,
                        label="robot x",
                    ),
                    PlotSeries(
                        color="#06d6a0",
                        values=self.telemetry.target_x_hist,
                        label="target x",
                    ),
                    PlotSeries(
                        color="#ef476f",
                        values=self.telemetry.robot_y_hist,
                        label="robot y",
                    ),
                    PlotSeries(
                        color="#118ab2",
                        values=self.telemetry.target_y_hist,
                        label="target y",
                    ),
                ],
                fixed_range=(0.0, max_range),
            ),
        ]

    def get_hud_text(self) -> str:
        theta = self.compute_current_theta()
        angle_deg = (math.degrees(theta) + 360.0) % 360.0
        target_deg = (math.degrees(self.target_theta) + 360.0) % 360.0
        return (
            "Optimal wheel control: move cursor for position target, two-finger scroll for heading target, press r to reset\n"
            f"x={self.roomba.position[0]:6.1f}, y={self.roomba.position[1]:6.1f}, "
            f"theta={angle_deg:6.1f} deg, theta*={target_deg:6.1f} deg\n"
            f"u_l={self.u_left:7.2f} N, u_r={self.u_right:7.2f} N, "
        )

    def set_target_position(self, x: float, y: float) -> None:
        self.target_x = x
        self.target_y = y

    def adjust_target_theta(self, delta: float) -> None:
        self.target_theta += delta

    def reset(self) -> None:
        self.roomba.position[:] = [self.initial_x, self.initial_y]
        self.target_x = float(self.roomba.position[0])
        self.target_y = float(self.roomba.position[1])
        self.target_theta = 0.0

        self.roomba.direction[:] = [1.0, 0.0]
        self.roomba.velocity[:] = [0.0, 0.0]
        self.roomba.omega[:] = [0.0]
        self.roomba.acceleration[:] = [0.0, 0.0]
        self.roomba.alpha[:] = [0.0]
        self.roomba.external_forces[:] = [0.0, 0.0]
        self.roomba.external_torques[:] = [0.0]

        self.u_left = 0.0
        self.u_right = 0.0
        self.time = 0.0
        self.telemetry.clear()

    def step(self) -> None:
        action = self.policy.compute_action(
            roomba=self.roomba,
            target_x=self.target_x,
            target_y=self.target_y,
            target_theta=self.target_theta,
            prev_u_left=self.u_left,
            prev_u_right=self.u_right,
        )
        self.u_left, self.u_right = action

        heading = self.roomba.direction.copy()
        force_ctrl = heading * (self.u_left + self.u_right)
        torque_ctrl = 0.5 * self.roomba.width * (-self.u_left + self.u_right)

        m = float(self.roomba.mass)
        inertia = float(self.roomba.inertia)
        force_damp = -(m / self.config.tau_v_damp) * self.roomba.velocity
        torque_damp = -(inertia / self.config.tau_omega_damp) * float(
            self.roomba.omega[0]
        )

        self.roomba.external_forces[:] = force_ctrl + force_damp
        self.roomba.external_torques[:] = [torque_ctrl + torque_damp]
        self.time = self.stepper.step_single_instance(
            self.roomba, self.time, self.config.dt
        )

        x = float(self.roomba.position[0])
        y = float(self.roomba.position[1])
        self.telemetry.push_pose(x, y)

        d1 = self.roomba.direction
        v_forward = float(np.dot(self.roomba.velocity, d1))
        half_width = float(self.roomba.width) * 0.5
        omega = float(self.roomba.omega[0])
        v_left_forward = v_forward + omega * half_width
        v_right_forward = v_forward - omega * half_width

        self.telemetry.record(
            left_power=self.u_left * v_left_forward,
            right_power=self.u_right * v_right_forward,
            robot_theta=self.compute_current_theta(),
            target_theta=self.target_theta,
            robot_x=x,
            target_x=self.target_x,
            robot_y=y,
            target_y=self.target_y,
        )
