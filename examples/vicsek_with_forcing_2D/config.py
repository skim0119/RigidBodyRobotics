from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FlockingConfig:
    n_bodies: int = 500
    box_size: tuple[float, float] = (20.0, 20.0)
    dt: float = 0.01
    substeps_per_frame: int = 2
    seed: int = 42

    mass: float = 1.0
    inertia: float = 0.2
    initial_speed: float = 1.0

    align_radius: float = 0.9
    noise: float = 0.25
    target_speed: float = 1.0
    k_theta: float = 9.0
    c_omega: float = 1.4
    k_v: float = 3.0
    c_v: float = 0.6
    c_perp: float = 1.5
    max_force: float = 25.0
    max_torque: float = 25.0

    max_render_agents: int = 1200
    body_radius_px: float = 3.5
    heading_length_px: float = 8.0
    velocity_scale_px: float = 4.0

    telemetry_max_points: int = 50


@dataclass
class ControlConfig:
    radius_step: float = 0.05
    noise_step: float = 0.02
    speed_step: float = 0.05
    gain_step: float = 0.2

    min_align_radius: float = 0.05
    max_align_radius: float = 4.0
    min_noise: float = 0.0
    max_noise: float = 1.5
    min_speed: float = 0.0
    max_speed: float = 4.0

    # Realtime playback controls
    ui_tick_ms: int = 1
    max_steps_per_tick: int = 12
    max_accumulator_seconds: float = 0.5
    playback_ema_alpha: float = 0.2
