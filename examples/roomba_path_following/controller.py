from __future__ import annotations

import tkinter as tk

from elastica_rigid.visualize.tk_app import TkView2D

from model import RoombaPathFollowingModel


class Controller:
    def __init__(
        self,
        root: tk.Tk,
        model: RoombaPathFollowingModel,
        view: TkView2D,
        fps: int,
    ) -> None:
        self.root = root
        self.model = model
        self.view = view
        self.frame_delay_ms = max(1, int(1000 / fps))
        self.running = True
        self.paused = False

        self.root.bind("<r>", self._on_reset)
        self.root.bind("<R>", self._on_reset)
        self.root.bind("<t>", self._on_toggle_pause)
        self.root.bind("<T>", self._on_toggle_pause)

    def run(self) -> None:
        self._tick()

    def _tick(self) -> None:
        if not self.running or not self.root.winfo_exists():
            return
        if not self.paused:
            self.model.step_frame()
        self.view.render(self.model)
        self.root.after(self.frame_delay_ms, self._tick)

    def _on_reset(self, _: tk.Event) -> None:
        self.model.reset()

    def _on_toggle_pause(self, _: tk.Event) -> None:
        self.paused = not self.paused
