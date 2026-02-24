"""MVC (Model-View-Controller) architecture

Controls:
- Move cursor: desired position target.
- Two-finger scroll: desired orientation target.
- Press R: reset robot state and target.
- Press Esc: quit.

Run:
    python examples/trackpad_control/how-to-use-TkView2D.py
"""

from __future__ import annotations

import tkinter as tk

import numpy as np

import elastica_rigid as er

from elastica_rigid.visualize.tk_app import TkView2D
from elastica_rigid.visualize.tk_app import (
    ObjectPose2D,
    PlotPanel,
    TargetPose2D,
    Trail2D,
)


class DummyModel:
    def get_object_poses(self):
        return [
            ObjectPose2D(
                x=120, y=120, dir_x=1.0, dir_y=0.0, radius=20, heading_length=30
            ),
            ObjectPose2D(
                x=300, y=320, dir_x=0.0, dir_y=-1.0, radius=10, heading_length=25
            ),
        ]

    def get_target_pose(self):
        return TargetPose2D(x=200, y=150, theta=0.3, marker_radius=6, heading_length=30)

    def get_trails(self):
        return [Trail2D(points=[])]

    def get_plotting_data(self):
        return [PlotPanel(title="Empty", series=[])]

    def get_hud_text(self):
        return "Hellow World"


def main() -> None:
    root = tk.Tk()
    root.title("Some title for the window.")
    view = TkView2D(root)
    view.render(DummyModel())
    root.mainloop()


if __name__ == "__main__":
    main()
