from __future__ import annotations
from typing import Callable

import math
import tkinter as tk

from model import SimulationModel
from elastica_rigid.visualize.tk_app import TkView2D


class TkRunner:
    def __init__(
        self,
        root: tk.Tk,
        *,
        step_fn: Callable[[], None],
        render_fn: Callable[[], None],
        frame_ms: int = 16,
    ) -> None:
        self.root = root
        self.step_fn = step_fn
        self.render_fn = render_fn
        self.frame_ms = frame_ms
        self.running = False

    def run(self) -> None:
        if self.running:
            return
        self.running = True
        self._tick()

    def pause(self) -> None:
        self.running = False

    def step(self) -> None:
        self.step_fn()
        self.render_fn()

    def _tick(self) -> None:
        if not self.running:
            return
        self.step_fn()
        self.render_fn()
        self.root.after(self.frame_ms, self._tick)


class Controller:
    def __init__(self, model: SimulationModel, view: TkView2D) -> None:
        root: tk.TK = view.root
        self.model = model
        self.view = view

        self.runner = TkRunner(
            root,
            step_fn=self.model.step,
            render_fn=lambda: self.view.render(self.model),
            frame_ms=16,
        )

        # Bind keys
        root.bind("<Motion>", self._on_motion)
        root.bind("<MouseWheel>", self._on_mousewheel)  # macOS / Windows
        root.bind("<Button-4>", self._on_scroll_up)  # Linux scroll up
        root.bind("<Button-5>", self._on_scroll_down)  # Linux scroll down
        root.bind("<r>", self._on_reset)
        root.bind("<R>", self._on_reset)

    def run(self) -> None:
        self.runner.run()

    def pause(self) -> None:
        self.runner.pause()

    def step(self) -> None:
        self.runner.step()

    def reset(self) -> None:
        self.model.reset()
        self.view.render(self.model)

    def _on_motion(self, event: tk.Event) -> None:
        left_origin_x = self.view.left_canvas.winfo_rootx()
        left_origin_y = self.view.left_canvas.winfo_rooty()
        x_local = float(event.x_root - left_origin_x)
        y_local = float(event.y_root - left_origin_y)
        left_w = self.view.get_left_panel_width()
        left_h = self.view.get_left_panel_height()
        x = float(max(0.0, min(x_local, left_w - 1)))
        y = float(max(0.0, min(y_local, left_h - 1)))
        self.model.set_target_position(x, y)

    def _on_mousewheel(self, event: tk.Event) -> None:
        delta = float(getattr(event, "delta", 0.0))
        self.model.adjust_target_theta(0.004 * delta)

    def _on_scroll_up(self, _: tk.Event) -> None:
        self.model.adjust_target_theta(math.radians(6.0))

    def _on_scroll_down(self, _: tk.Event) -> None:
        self.model.adjust_target_theta(-math.radians(6.0))

    def _on_reset(self, _: tk.Event) -> None:
        self.reset()
