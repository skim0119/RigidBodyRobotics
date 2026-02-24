"""Pose-tracking policy and control tuning for the SE(2) Roomba example."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(frozen=True)
class ControlConfig:
    # Per-wheel thrust saturation [N]
    max_wheel_thrust: float = 180.0

    # Pose-tracking behavior knobs
    rho_stop: float = 16.0
    k_rho: float = 1.4
    k_alpha: float = 4.8
    k_theta: float = 2.2

    # Velocity/turn-rate tracking gains (dynamic inversion)
    k_v: float = 5.0
    k_omega: float = 3.5

    # Command envelope / smoothness limits
    v_max: float = 520.0
    omega_max: float = 6.0
    max_du_per_step: float = 24.0

    # Optional behavior toggle
    allow_backward_motion: bool = False


DEFAULT_CONTROL_CONFIG = ControlConfig()


class SimplePoseTracking:
    def __init__(self, config: ControlConfig = DEFAULT_CONTROL_CONFIG) -> None:
        self.config = config

    def compute_action(
        self,
        *,
        roomba,
        target_x: float,
        target_y: float,
        target_theta: float,
        prev_u_left: float,
        prev_u_right: float,
    ) -> tuple[float, float]:
        x = roomba.position.copy()
        v = roomba.velocity.copy()
        theta = math.atan2(float(roomba.direction[1]), float(roomba.direction[0]))
        omega = float(roomba.omega[0])
        d1 = roomba.direction.copy()

        dx = target_x - float(x[0])
        dy = target_y - float(x[1])
        rho = math.hypot(dx, dy)

        theta_to_target = math.atan2(dy, dx) if rho > 1e-9 else theta
        alpha = _wrap_angle(theta_to_target - theta)
        theta_err = _wrap_angle(target_theta - theta)
        drive_sign = 1.0

        if self.config.allow_backward_motion and abs(alpha) > (0.5 * math.pi):
            drive_sign = -1.0
            alpha = _wrap_angle(alpha - math.copysign(math.pi, alpha))

        if rho < self.config.rho_stop:
            v_des = 0.0
            omega_des = self.config.k_theta * theta_err
        else:
            v_des = drive_sign * self.config.k_rho * rho
            if abs(alpha) > 1.2:
                v_des = 0.0
            omega_des = (
                self.config.k_alpha * alpha
                + self.config.k_theta
                * theta_err
                * (self.config.rho_stop / (rho + self.config.rho_stop))
            )

        v_des = float(np.clip(v_des, -self.config.v_max, self.config.v_max))
        omega_des = float(
            np.clip(omega_des, -self.config.omega_max, self.config.omega_max)
        )

        v_forward = float(np.dot(v, d1))
        a_forward_des = self.config.k_v * (v_des - v_forward)
        alpha_des = self.config.k_omega * (omega_des - omega)

        m = float(roomba.mass)
        inertia = float(roomba.inertia)
        width = float(roomba.width)
        u_sum_des = m * a_forward_des
        u_diff_des = (2.0 * inertia / width) * alpha_des

        u_left_nom = 0.5 * (u_sum_des - u_diff_des)
        u_right_nom = 0.5 * (u_sum_des + u_diff_des)

        u_left = float(
            np.clip(
                u_left_nom, -self.config.max_wheel_thrust, self.config.max_wheel_thrust
            )
        )
        u_right = float(
            np.clip(
                u_right_nom,
                -self.config.max_wheel_thrust,
                self.config.max_wheel_thrust,
            )
        )

        u_left = float(
            np.clip(
                u_left,
                prev_u_left - self.config.max_du_per_step,
                prev_u_left + self.config.max_du_per_step,
            )
        )
        u_right = float(
            np.clip(
                u_right,
                prev_u_right - self.config.max_du_per_step,
                prev_u_right + self.config.max_du_per_step,
            )
        )

        return u_left, u_right


# --- NUMBA WRAPPED FUNCTION ---


@njit(cache=True)
def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
