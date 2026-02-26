from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from elastica_rigid.body.roomba import Roomba
from elastica_rigid.timestepper.symplectic_stepper import SymplecticEulerForward
from elastica_rigid.visualize.tk_app.protocol import (
    ObjectPose2D,
    PlotPanel,
    PlotSeries,
    TargetPose2D,
    Trail2D,
)

from policy import FeedbackPoseTracking, ReferenceState, wrap_angle


@dataclass(frozen=True)
class SimulationParams:
    dt: float = 0.01
    substeps_per_frame: int = 2
    robot_draw_radius: float = 14.0
    heading_length: float = 34.0
    trail_max_len: int = 1200
    diag_window: int = 300


@dataclass(frozen=True)
class InfinityPathParams:
    center_x: float
    center_y: float
    size: float = 170.0
    angular_speed: float = 0.45


@dataclass(frozen=True)
class Diagnostics:
    position_error: float
    heading_error: float
    speed: float
    ref_speed: float
    force_norm: float
    torque: float


@dataclass(frozen=True)
class AgentConfig:
    name: str
    roomba: Roomba
    policy: FeedbackPoseTracking
    body_color: str
    heading_color: str
    trail_color: str


@dataclass
class AgentState:
    config: AgentConfig
    initial_position: np.ndarray
    initial_direction: np.ndarray
    trail: deque[tuple[float, float]]
    position_error_history: deque[float]
    heading_error_history: deque[float]
    speed_history: deque[float]
    torque_history: deque[float]
    last_diag: Diagnostics = field(
        default_factory=lambda: Diagnostics(
            position_error=0.0,
            heading_error=0.0,
            speed=0.0,
            ref_speed=0.0,
            force_norm=0.0,
            torque=0.0,
        )
    )


class InfinityPath:
    def __init__(self, params: InfinityPathParams) -> None:
        self.params = params
        self._shape_factor = 0.5

    def evaluate(self, t: float) -> ReferenceState:
        a = self.params.size
        w = self.params.angular_speed
        wt = w * t

        x = self.params.center_x + a * math.sin(wt)
        y = self.params.center_y + self._shape_factor * a * math.sin(2.0 * wt)

        vx = a * w * math.cos(wt)
        vy = 2.0 * self._shape_factor * a * w * math.cos(2.0 * wt)

        ax = -a * w * w * math.sin(wt)
        ay = -4.0 * self._shape_factor * a * w * w * math.sin(2.0 * wt)

        theta = math.atan2(vy, vx)
        omega = (vx * ay - vy * ax) / (vx * vx + vy * vy + 1e-9)

        return ReferenceState(
            position=np.array([x, y], dtype=float),
            velocity=np.array([vx, vy], dtype=float),
            theta=theta,
            omega=omega,
        )

    def build_preview(self, num_points: int = 280) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for i in range(num_points):
            phase = (2.0 * math.pi * i) / max(num_points - 1, 1)
            x = self.params.center_x + self.params.size * math.sin(phase)
            y = self.params.center_y + self._shape_factor * self.params.size * math.sin(
                2.0 * phase
            )
            points.append((x, y))
        return points


class RoombaPathFollowingModel:
    def __init__(
        self,
        agents: Sequence[AgentConfig],
        stepper: SymplecticEulerForward,
        path: InfinityPath,
        params: SimulationParams,
    ) -> None:
        self.stepper = stepper
        self.path = path
        self.params = params
        self.time = 0.0
        self.current_ref = self.path.evaluate(self.time)
        self.path_preview = self.path.build_preview()

        self.agents: list[AgentState] = []
        for agent_cfg in agents:
            roomba = agent_cfg.roomba
            trail = deque(maxlen=self.params.trail_max_len)
            trail.append((float(roomba.position[0]), float(roomba.position[1])))
            self.agents.append(
                AgentState(
                    config=agent_cfg,
                    initial_position=roomba.position.copy(),
                    initial_direction=roomba.direction.copy(),
                    trail=trail,
                    position_error_history=deque(maxlen=self.params.diag_window),
                    heading_error_history=deque(maxlen=self.params.diag_window),
                    speed_history=deque(maxlen=self.params.diag_window),
                    torque_history=deque(maxlen=self.params.diag_window),
                )
            )

        self.ref_speed_history: deque[float] = deque(maxlen=self.params.diag_window)

    def reset(self) -> None:
        self.time = 0.0
        self.current_ref = self.path.evaluate(self.time)
        self.ref_speed_history.clear()

        for agent in self.agents:
            roomba = agent.config.roomba
            roomba.position[:] = agent.initial_position
            roomba.direction[:] = agent.initial_direction
            roomba.velocity[:] = 0.0
            roomba.acceleration[:] = 0.0
            roomba.omega[:] = 0.0
            roomba.alpha[:] = 0.0
            roomba.external_forces[:] = 0.0
            roomba.external_torques[:] = 0.0
            agent.last_diag = Diagnostics(
                position_error=0.0,
                heading_error=0.0,
                speed=0.0,
                ref_speed=0.0,
                force_norm=0.0,
                torque=0.0,
            )
            agent.trail.clear()
            agent.trail.append((float(roomba.position[0]), float(roomba.position[1])))
            agent.position_error_history.clear()
            agent.heading_error_history.clear()
            agent.speed_history.clear()
            agent.torque_history.clear()

    def step_frame(self) -> None:
        for _ in range(self.params.substeps_per_frame):
            self.current_ref = self.path.evaluate(self.time)
            ref_speed = float(np.linalg.norm(self.current_ref.velocity))

            for agent in self.agents:
                roomba = agent.config.roomba
                theta = math.atan2(float(roomba.direction[1]), float(roomba.direction[0]))
                omega = float(roomba.omega[0])
                position = roomba.position.copy()
                velocity = roomba.velocity.copy()

                control = agent.config.policy.compute_control(
                    position=position,
                    velocity=velocity,
                    theta=theta,
                    omega=omega,
                    reference=self.current_ref,
                    mass=float(roomba.mass),
                    inertia=float(roomba.inertia),
                )

                roomba.external_forces[:] = control.force
                roomba.external_torques[:] = control.torque
                self.stepper.step_single_instance(
                    roomba, np.float64(self.time), np.float64(self.params.dt)
                )

                e_pos = self.current_ref.position - roomba.position
                e_pos_norm = float(np.linalg.norm(e_pos))
                new_theta = math.atan2(float(roomba.direction[1]), float(roomba.direction[0]))
                e_heading = wrap_angle(self.current_ref.theta - new_theta)
                speed = float(np.linalg.norm(roomba.velocity))
                force_norm = float(np.linalg.norm(control.force))

                agent.last_diag = Diagnostics(
                    position_error=e_pos_norm,
                    heading_error=e_heading,
                    speed=speed,
                    ref_speed=ref_speed,
                    force_norm=force_norm,
                    torque=control.torque,
                )

                agent.position_error_history.append(e_pos_norm)
                agent.heading_error_history.append(e_heading)
                agent.speed_history.append(speed)
                agent.torque_history.append(control.torque)
                agent.trail.append((float(roomba.position[0]), float(roomba.position[1])))

            self.ref_speed_history.append(ref_speed)
            self.time += self.params.dt

    def get_object_poses(self) -> Sequence[ObjectPose2D]:
        poses: list[ObjectPose2D] = []
        for agent in self.agents:
            roomba = agent.config.roomba
            heading = roomba.direction / (np.linalg.norm(roomba.direction) + 1e-12)
            poses.append(
                ObjectPose2D(
                    x=float(roomba.position[0]),
                    y=float(roomba.position[1]),
                    dir_x=float(heading[0]),
                    dir_y=float(heading[1]),
                    radius=self.params.robot_draw_radius,
                    heading_length=self.params.heading_length,
                    body_color=agent.config.body_color,
                    heading_color=agent.config.heading_color,
                    tag=agent.config.name,
                )
            )
        return poses

    def get_target_pose(self) -> TargetPose2D:
        return TargetPose2D(
            x=float(self.current_ref.position[0]),
            y=float(self.current_ref.position[1]),
            theta=float(self.current_ref.theta),
            marker_radius=7.0,
            heading_length=28.0,
        )

    def get_trails(self) -> Sequence[Trail2D]:
        trails: list[Trail2D] = [Trail2D(points=self.path_preview, color="#495057", width=2)]
        for agent in self.agents:
            trails.append(Trail2D(points=list(agent.trail), color=agent.config.trail_color, width=3))
        return trails

    def _series_for_agents(self, key: str, label_suffix: str) -> list[PlotSeries]:
        series: list[PlotSeries] = []
        for agent in self.agents:
            if key == "position":
                values = list(agent.position_error_history)
            elif key == "heading":
                values = list(agent.heading_error_history)
            elif key == "speed":
                values = list(agent.speed_history)
            elif key == "torque":
                values = list(agent.torque_history)
            else:
                values = []
            series.append(
                PlotSeries(
                    values=values,
                    color=agent.config.trail_color,
                    label=f"{agent.config.name} {label_suffix}".strip(),
                )
            )
        return series

    def get_plotting_data(self) -> Sequence[PlotPanel]:
        speed_series = self._series_for_agents("speed", "|v|")
        speed_series.append(
            PlotSeries(values=list(self.ref_speed_history), color="#94d82d", label="reference |v|")
        )

        return [
            PlotPanel(
                title="Position Error",
                series=self._series_for_agents("position", "||p_ref - p||"),
                fixed_range=(0.0, self.path.params.size * 0.8),
            ),
            PlotPanel(
                title="Speed Tracking",
                series=speed_series,
                fixed_range=(0.0, self.path.params.size * self.path.params.angular_speed * 1.4),
            ),
            PlotPanel(
                title="Heading Error",
                series=self._series_for_agents("heading", "theta err"),
                fixed_range=(-math.pi, math.pi),
            ),
            PlotPanel(
                title="Control Torque",
                series=self._series_for_agents("torque", "tau"),
            ),
        ]

    def get_hud_text(self) -> str:
        lines = [
            "controls: r reset | t pause/resume",
            f"t = {self.time:7.2f} s",
            f"ref = ({self.current_ref.position[0]:7.1f}, {self.current_ref.position[1]:7.1f})",
        ]
        for agent in self.agents:
            roomba = agent.config.roomba
            lines.append(
                (
                    f"{agent.config.name}: pos=({roomba.position[0]:6.1f},{roomba.position[1]:6.1f}) "
                    f"err={agent.last_diag.position_error:6.2f} "
                    f"v={agent.last_diag.speed:6.2f} "
                    f"tau={agent.last_diag.torque:7.2f}"
                )
            )
        return "\n".join(lines)
