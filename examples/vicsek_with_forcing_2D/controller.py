from __future__ import annotations

import time
import tkinter as tk

from config import ControlConfig
from model import SimulationModel


class Controller:
    def __init__(
        self,
        model: SimulationModel,
        view,
        control_config: ControlConfig | None = None,
    ) -> None:
        self.model = model
        self.view = view
        self.control_config = ControlConfig() if control_config is None else control_config
        self._last_wall_time = time.perf_counter()
        self._sim_accumulator = 0.0
        self._playback_rate = 1.0
        self._rate_window_wall = 0.0
        self._rate_window_sim = 0.0
        self._rate_window_seconds = 0.5

        self._bind_inputs()

    def _bind_inputs(self) -> None:
        root = self.view.winfo_toplevel()
        root.bind("<space>", self._on_toggle_pause)
        root.bind("<r>", self._on_reset)
        root.bind("<R>", self._on_reset)

        root.bind("<q>", lambda _: self._adjust_noise(+self.control_config.noise_step))
        root.bind("<a>", lambda _: self._adjust_noise(-self.control_config.noise_step))
        root.bind("<w>", lambda _: self._adjust_speed(+self.control_config.speed_step))
        root.bind("<s>", lambda _: self._adjust_speed(-self.control_config.speed_step))
        root.bind("<e>", lambda _: self._adjust_k_theta(+self.control_config.gain_step))
        root.bind("<d>", lambda _: self._adjust_k_theta(-self.control_config.gain_step))
        root.bind("<t>", lambda _: self._adjust_k_v(+self.control_config.gain_step))
        root.bind("<g>", lambda _: self._adjust_k_v(-self.control_config.gain_step))

        root.bind("<MouseWheel>", self._on_mouse_wheel)
        root.bind(
            "<Button-4>", lambda _: self._adjust_radius(+self.control_config.radius_step)
        )
        root.bind(
            "<Button-5>", lambda _: self._adjust_radius(-self.control_config.radius_step)
        )

    def run(self) -> None:
        self._last_wall_time = time.perf_counter()
        self._tick()

    def _tick(self) -> None:
        now = time.perf_counter()
        wall_dt = max(1e-9, now - self._last_wall_time)
        self._last_wall_time = now

        is_lagging = False
        sim_advanced = 0.0
        if self.model.running:
            self._sim_accumulator += wall_dt
            self._sim_accumulator = min(
                self._sim_accumulator,
                self.control_config.max_accumulator_seconds,
            )
            desired_steps = int(self._sim_accumulator / self.model.config.dt)
            steps = min(desired_steps, self.control_config.max_steps_per_tick)
            is_lagging = desired_steps > self.control_config.max_steps_per_tick

            for _ in range(steps):
                self.model.step()
            sim_advanced = steps * self.model.config.dt
            self._sim_accumulator -= sim_advanced

            # Use a wall-time window to avoid per-tick quantization bias
            # (many ticks can have zero sim steps).
            self._rate_window_wall += wall_dt
            self._rate_window_sim += sim_advanced
            if self._rate_window_wall >= self._rate_window_seconds:
                self._playback_rate = self._rate_window_sim / max(
                    1e-9, self._rate_window_wall
                )
                self._rate_window_wall = 0.0
                self._rate_window_sim = 0.0

        self.model.set_runtime_status(self._playback_rate, is_lagging)
        self.model.set_viewport(
            self.view.get_left_panel_width(),
            self.view.get_left_panel_height(),
        )
        self.view.render(self.model)
        self.view.after(self.control_config.ui_tick_ms, self._tick)

    def _on_toggle_pause(self, _: tk.Event) -> None:
        self.model.running = not self.model.running

    def _on_reset(self, _: tk.Event) -> None:
        self.model.reset()
        self._sim_accumulator = 0.0
        self._playback_rate = 1.0
        self._rate_window_wall = 0.0
        self._rate_window_sim = 0.0
        self._last_wall_time = time.perf_counter()

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        delta = getattr(event, "delta", 0)
        if delta > 0:
            self._adjust_radius(+self.control_config.radius_step)
        elif delta < 0:
            self._adjust_radius(-self.control_config.radius_step)

    def _adjust_radius(self, delta: float) -> None:
        self.model.params.align_radius = min(
            self.control_config.max_align_radius,
            max(
                self.control_config.min_align_radius,
                self.model.params.align_radius + delta,
            ),
        )

    def _adjust_noise(self, delta: float) -> None:
        self.model.params.noise = min(
            self.control_config.max_noise,
            max(self.control_config.min_noise, self.model.params.noise + delta),
        )

    def _adjust_speed(self, delta: float) -> None:
        self.model.params.target_speed = min(
            self.control_config.max_speed,
            max(self.control_config.min_speed, self.model.params.target_speed + delta),
        )

    def _adjust_k_theta(self, delta: float) -> None:
        self.model.params.k_theta = max(0.0, self.model.params.k_theta + delta)

    def _adjust_k_v(self, delta: float) -> None:
        self.model.params.k_v = max(0.0, self.model.params.k_v + delta)
