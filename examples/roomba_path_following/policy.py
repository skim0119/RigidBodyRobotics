from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ReferenceState:
    position: np.ndarray
    velocity: np.ndarray
    theta: float
    omega: float


@dataclass(frozen=True)
class ControlOutput:
    force: np.ndarray
    torque: float


@dataclass(frozen=True)
class FeedbackControlParams:
    kp_position: float = 0.25
    kd_position: float = 1.1
    kp_heading: float = 10.0
    kd_heading: float = 2.2
    max_force: float = 220.0
    max_torque: float = 180.0


def wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


class FeedbackPoseTracking:
    def __init__(self, params: FeedbackControlParams) -> None:
        self.params = params

    def compute_control(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        theta: float,
        omega: float,
        reference: ReferenceState,
        mass: float,
        inertia: float,
    ) -> ControlOutput:
        e_pos = reference.position - position
        e_vel = reference.velocity - velocity

        acc_cmd = self.params.kp_position * e_pos + self.params.kd_position * e_vel
        force = mass * acc_cmd
        force_norm = float(np.linalg.norm(force))
        if force_norm > self.params.max_force:
            force *= self.params.max_force / max(force_norm, 1e-12)

        e_theta = wrap_angle(reference.theta - theta)
        alpha_cmd = self.params.kp_heading * e_theta + self.params.kd_heading * (
            reference.omega - omega
        )
        torque = float(inertia * alpha_cmd)
        torque = float(np.clip(torque, -self.params.max_torque, self.params.max_torque))

        return ControlOutput(force=force, torque=torque)
