"""MVC (Model-View-Controller) architecture

Shows how to:
- implement the model query interface,
- render multiple objects,
- attach text labels with ObjectPose2D(name="...").

Run:
    python examples/se3_control_sandbox/how-to-use-TkView2D.py
"""

from __future__ import annotations

import tkinter as tk

from elastica_rigid.visualize.tk_app import TkView2D
from elastica_rigid.visualize.tk_app import (
    ObjectPose2D,
    PlotPanel,
    TargetPose2D,
    Trail2D,
)


class DummyModel:
    """Small model that satisfies TkView2D's query protocol."""

    def get_object_poses(self):
        return [
            ObjectPose2D(
                x=140,
                y=140,
                dir_x=1.0,
                dir_y=0.2,
                radius=20,
                heading_length=34,
                name="Robot A",  # name is drawn next to the object
            ),
            ObjectPose2D(
                x=280,
                y=320,
                dir_x=-0.3,
                dir_y=-1.0,
                radius=14,
                heading_length=26,
                name="Target Follower",
            ),
            ObjectPose2D(
                x=420,
                y=210,
                dir_x=0.0,
                dir_y=1.0,
                radius=12,
                heading_length=24,
                # name defaults to "" => no text drawn
            ),
        ]

    def get_target_pose(self):
        return TargetPose2D(x=220, y=180, theta=0.6, marker_radius=7, heading_length=34)

    def get_trails(self):
        return [
            Trail2D(points=[]),
        ]

    def get_plotting_data(self):
        return [PlotPanel(title="Empty", series=[])]

    def get_hud_text(self):
        return "Hellow World"


def main() -> None:
    root = tk.Tk()
    root.title("TkView2D Tutorial")
    view = TkView2D(root)
    view.render(DummyModel())
    root.mainloop()


if __name__ == "__main__":
    main()
