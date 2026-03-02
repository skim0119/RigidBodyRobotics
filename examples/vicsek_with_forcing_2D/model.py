from typing import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import elastica as ea
import elastica_rigid as er
import elastica_plugins as ep

from elastica_rigid.visualize.tk_app import (
    PlotPanel,
    PlotSeries,
    Trail2D,
    TrianglePose2D,
)

from config import FlockingConfig
from telemetry import SwarmTelemetry

from elastica_plugins.neighbor_search import CellListNeighborSearch2D


class Simulator(
    ea.BaseSystemCollection,
    ea.Forcing,
    ea.CallBacks,
    ep.Flocking,
):
    pass


@dataclass
class SwarmState:
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    direction: NDArray[np.float64]
    omega: NDArray[np.float64]
    time: float = 0.0
    step: int = 0


class TelemetryCallback(ea.CallBackBaseClass):
    def __init__(
        self,
        step_skip: int,
        state: SwarmState,
        telemetry: SwarmTelemetry,
    ) -> None:
        super().__init__()
        self.every = step_skip
        self.state = state
        self.telemetry = telemetry

    def make_callback(self, system, time, current_step: int) -> None:
        if current_step % self.every != 0:
            return

        for i, robot in enumerate(system):
            self.state.position[:, i] = robot.position[:, 0]
            self.state.velocity[:, i] = robot.velocity[:, 0]
            self.state.direction[:, i] = robot.direction[:, 0]
            self.state.omega[i] = float(robot.omega[0])

        self.state.time = float(time)
        self.state.step = int(current_step)

        order = float(np.linalg.norm(np.mean(self.state.direction, axis=1)))
        mean_speed = float(np.mean(np.linalg.norm(self.state.velocity, axis=0)))
        self.telemetry.record(
            t=self.state.time,
            order=order,
            speed=mean_speed,
            avg_neighbors=None,
        )


def create_robot(config: FlockingConfig, rng: np.random.Generator) -> er.SE2RigidBody:
    """
    Return random robot according to the config.
    """
    lx, ly = config.box_size
    position = rng.uniform(0.0, 1.0, size=(2,))
    position[0] *= lx
    position[1] *= ly

    theta = rng.uniform(-np.pi, np.pi)
    direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)

    return er.SE2RigidBody.create_body(
        initial_position=position,
        initial_direction=direction,
        mass=float(config.mass),
        inertia=float(config.inertia),
    )


def _build_simulator(
    config: FlockingConfig,
    rng: np.random.Generator,
    telemetry: SwarmTelemetry,
) -> tuple[Simulator, SwarmState]:
    simulator = Simulator()
    simulator.enable_block_supports(er.SE2RigidBody, er.MemoryBlockSE2Body)

    robots = []
    for _ in range(config.n_bodies):
        robot = create_robot(config, rng)
        simulator.append(robot)
        robots.append(robot)

    simulator.configure_flocking(
        flocking_block=er.MemoryBlockSE2Body,
        box_size=config.box_size,
    ).using(
        ep.VicsekModel,
        box_size=config.box_size,
        config=config,
        rng=rng,
    )

    state = SwarmState(
        position=np.zeros((2, config.n_bodies), dtype=np.float64),
        velocity=np.zeros((2, config.n_bodies), dtype=np.float64),
        direction=np.zeros((2, config.n_bodies), dtype=np.float64),
        omega=np.zeros((config.n_bodies,), dtype=np.float64),
    )
    for i, robot in enumerate(robots):
        state.position[:, i] = robot.position[:, 0]
        state.velocity[:, i] = robot.velocity[:, 0]
        state.direction[:, i] = robot.direction[:, 0]
        state.omega[i] = float(robot.omega[0])

    simulator.collect_diagnostics(robots).using(
        TelemetryCallback,
        step_skip=1,
        state=state,
        telemetry=telemetry,
    )
    simulator.finalize()

    return simulator, state


def run_demo() -> None:
    from tqdm import tqdm

    config = FlockingConfig()
    rng = np.random.default_rng(config.seed)
    simulator, _ = _build_simulator(config, rng, SwarmTelemetry(maxlen=32))
    stepper = ea.PositionVerlet()

    time = 0.0
    for _ in tqdm(range(10), desc="simulation"):
        time = stepper.step(simulator, time, config.dt)


class SimulationModel:
    def __init__(self, config: FlockingConfig) -> None:
        self.config = config
        self.params = self.config  # Mutable UI controls feed VicsekModel via config
        self.running = True
        self._rng = np.random.default_rng(self.config.seed)

        self.telemetry = SwarmTelemetry(maxlen=self.config.telemetry_max_points)
        self.stepper = ea.PositionVerlet()
        self.search = CellListNeighborSearch2D(
            box_size=np.asarray(self.config.box_size, dtype=np.float64),
            radius=self.params.align_radius,
        )

        self.time = 0.0
        self.step_count = 0
        self.viewport_width = 760
        self.viewport_height = 600
        self.playback_rate = 1.0
        self.is_lagging = False

        self.simulator: Simulator
        self.state: SwarmState
        self._rebuild_simulation()

    def _rebuild_simulation(self) -> None:
        self.simulator, self.state = _build_simulator(self.config, self._rng, self.telemetry)
        self.time = 0.0
        self.step_count = 0

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.config.seed)
        self.telemetry.reset()
        self._rebuild_simulation()

    def step(self) -> None:
        self.search.box_size = np.asarray(self.params.box_size, dtype=np.float64)
        self.search.set_radius(self.params.align_radius)

        self.time = self.stepper.step(self.simulator, self.time, self.config.dt)
        self.step_count += 1

        neighbors = self.search.query_all(self.state.position.T)
        avg_neighbors = float(np.mean([max(0, len(nb) - 1) for nb in neighbors]))
        if len(self.telemetry.avg_neighbors) > 0:
            self.telemetry.avg_neighbors[-1] = avg_neighbors

    def set_viewport(self, width: int, height: int) -> None:
        self.viewport_width = int(width)
        self.viewport_height = int(height)

    def set_runtime_status(self, playback_rate: float, is_lagging: bool) -> None:
        self.playback_rate = max(0.0, float(playback_rate))
        self.is_lagging = bool(is_lagging)

    def world_to_screen(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        width: int,
        height: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        lx, ly = self.params.box_size
        scale = min(width / lx, height / ly)
        offset_x = 0.5 * (width - scale * lx)
        offset_y = 0.5 * (height - scale * ly)

        sx = offset_x + scale * x
        sy = height - (offset_y + scale * y)
        return sx, sy, scale

    def get_object_poses(self) -> Sequence[TrianglePose2D]:
        n = self.config.n_bodies
        stride = max(1, n // self.config.max_render_agents)
        idx = np.arange(0, n, stride, dtype=np.int64)

        pos = self.state.position[:, idx]
        vel = self.state.velocity[:, idx]
        d1 = self.state.direction[:, idx]

        sx, sy, _ = self.world_to_screen(
            pos[0],
            pos[1],
            self.viewport_width,
            self.viewport_height,
        )

        poses: list[TrianglePose2D] = []
        for k in range(idx.size):
            poses.append(
                TrianglePose2D(
                    x=float(sx[k]),
                    y=float(sy[k]),
                    dir_x=float(d1[0, k]),
                    dir_y=float(-d1[1, k]),
                    radius=self.config.body_radius_px,
                    heading_length=self.config.heading_length_px,
                    body_color="#57cc99",
                    heading_color="#ffd166",
                    vel_x=float(vel[0, k]),
                    vel_y=float(-vel[1, k]),
                    velocity_scale=self.config.velocity_scale_px,
                    velocity_color="#ef476f",
                    draw_velocity=True,
                )
            )

        return poses

    def get_target_pose(self) -> None:
        return None

    def get_trails(self) -> Sequence[Trail2D]:
        return [Trail2D(points=[])]

    def get_plotting_data(self) -> Sequence[PlotPanel]:
        return [
            PlotPanel(
                title="Order Parameter",
                series=[
                    PlotSeries(
                        values=list(self.telemetry.order_parameter),
                        color="#4cc9f0",
                        label="phi",
                    )
                ],
                fixed_range=(0.0, 1.0),
            ),
            PlotPanel(
                title="Mean Speed",
                series=[
                    PlotSeries(
                        values=list(self.telemetry.mean_speed),
                        color="#f72585",
                        label="|v|",
                    )
                ],
            ),
            PlotPanel(
                title="Avg Neighbors",
                series=[
                    PlotSeries(
                        values=list(self.telemetry.avg_neighbors),
                        color="#ffd166",
                        label="k_avg",
                    )
                ],
            ),
        ]

    def get_hud_text(self) -> str:
        if not self.running:
            rt_text = "Realtime paused"
        elif self.is_lagging:
            rt_text = f"Realtime ... {self.playback_rate:.2f}x"
        else:
            rt_text = f"Realtime {self.playback_rate:.2f}x"

        return (
            f"N={self.config.n_bodies}  t={self.time:6.2f}s  dt={self.config.dt:.3f}\n"
            f"{rt_text}\n"
            f"r_align={self.params.align_radius:.2f}  noise={self.params.noise:.2f}  v0={self.params.target_speed:.2f}\n"
            f"k_theta={self.params.k_theta:.2f}  k_v={self.params.k_v:.2f}\n"
            "Controls: wheel radius | q/a noise | w/s speed | e/d k_theta | t/g k_v | space pause | r reset"
        )


if __name__ == "__main__":
    run_demo()
