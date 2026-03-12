"""Common utilities for Project 2 estimation/control tasks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict

import numpy as np
from numpy.typing import NDArray

import elastica as ea
import elastica_rigid as er
from elastica.external_forces import NoForces
from elastica_rigid.external_forces import (
    OpenLoopForce,
    compute_wheel_forces_to_external,
)


def project2_results_dir() -> Path:
    """Get and create the project2 results directory."""
    results = Path("results_project2")
    results.mkdir(parents=True, exist_ok=True)
    return results


@dataclass(frozen=True)
class RobotParams:
    """Physical parameters for the planar differential-drive robot."""

    mass: float = 2.0  # kg
    inertia: float = 0.05  # kg m^2
    radius: float = 0.2  # m
    width: float = 0.15  # m


DEFAULT_ROBOT_PARAMS = RobotParams()
DEFAULT_DT = 1.0e-3
SIM_TIME = 10.0

# Open-loop schedule arrays for Task 1/2:
# 2-3s forward, 3-6s coast, 6-7s in-place rotation
OPEN_LOOP_TIME_INTERVALS = np.array(
    [
        [2.0, 3.0],
        [6.0, 7.0],
    ],
    dtype=np.float64,
)
OPEN_LOOP_LEFT_WHEEL_FORCES = np.array(
    [
        [1.0, 0.0],
        [1.0, 0.0],
    ],
    dtype=np.float64,
)
OPEN_LOOP_RIGHT_WHEEL_FORCES = np.array(
    [
        [1.0, 0.0],
        [-1.0, 0.0],
    ],
    dtype=np.float64,
)


class Simulator(
    ea.BaseSystemCollection,
    ea.Forcing,
    ea.CallBacks,
):
    """Project-local simulator wrapper using the Project 1 mixin pattern."""


class CallBack(ea.CallBackBaseClass):
    """
    Tracks linear speed w_t = ||η_t||/m = ||v_t|| and angular velocity ω_t = l_t/I.
    """

    def __init__(self, step_skip: int, callback_params: dict):
        super().__init__()
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["position"].append(system.position[:, 0].copy())
            self.callback_params["direction"].append(system.direction[:, 0].copy())
            # Linear speed w_t = ||v_t|| (same as ||η_t||/m since η = m*v)
            self.callback_params["linear_speed"].append(
                np.linalg.norm(system.velocity[:, 0].copy())
            )
            # Angular velocity ω_t (rad/s)
            self.callback_params["angular_velocity"].append(float(system.omega[0]))
            return


def make_simulator(
    *,
    initial_position: NDArray[np.float64] | None = None,
    initial_direction: NDArray[np.float64] | None = None,
    params: RobotParams = DEFAULT_ROBOT_PARAMS,
) -> tuple[Simulator, er.Roomba, DefaultDict[str, list]]:
    """Create a simulator configured for Roomba systems."""
    sim = Simulator()
    sim.append_allowed_types(er.Roomba)

    if initial_position is None:
        initial_position = np.array([0.0, 0.0], dtype=np.float64)
    if initial_direction is None:
        initial_direction = np.array([1.0, 0.0], dtype=np.float64)

    robot = er.Roomba.create_robot(
        initial_position=initial_position,
        initial_direction=initial_direction,
        mass=params.mass,
        inertia=params.inertia,
        radius=params.radius,
        width=params.width,
    )

    sim.append(robot)
    sim.add_forcing_to(robot).using(
        OpenLoopForce,
        OPEN_LOOP_TIME_INTERVALS,
        OPEN_LOOP_LEFT_WHEEL_FORCES,
        OPEN_LOOP_RIGHT_WHEEL_FORCES,
    )

    logs = defaultdict(list)
    sim.collect_diagnostics(robot).using(CallBack, step_skip=1, callback_params=logs)

    return sim, robot, logs


def wrap_angle_array(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wrap angle array to [-pi, pi)."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi

