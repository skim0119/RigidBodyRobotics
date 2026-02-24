"""Trackpad-driven optimal control playground for a Roomba model.

MVC (Model-View-Controller) architecture.

Controls:
- Move cursor: desired position target.
- Two-finger scroll: desired orientation target.
- Press R: reset robot state and target.
- Press Esc: quit.

Run:
    python examples/se3_control_sandbox/run.py
"""

from __future__ import annotations

import tkinter as tk

import numpy as np

import elastica_rigid as er
from elastica_rigid import DEFAULT_UI_CONFIG

from controller import Controller
from model import SimulationModel
from policy import SimplePoseTracking


def main() -> None:
    # Tk Init
    root = tk.Tk()
    root.title("Cool stuffs. I'm bored.")

    initial_x = DEFAULT_UI_CONFIG.left_panel_width * 0.5
    initial_y = DEFAULT_UI_CONFIG.window_height * 0.5

    # Define Policy
    policy = SimplePoseTracking()

    # Define simulation model
    roomba = er.Roomba.create_robot(
        initial_position=np.array([initial_x, initial_y]),
        initial_direction=np.array([1.0, 0.0]),
        mass=1.0,
        inertia=1.0,
        radius=0.06,
        width=0.25,
    )
    stepper = er.SymplecticEulerForward()
    model = SimulationModel(
        roomba=roomba,
        stepper=stepper,
        policy=policy,
        initial_x=initial_x,
        initial_y=initial_y,
    )

    # Viewer
    view = er.TkView2D(root)

    # Controller
    controller = Controller(model, view)
    controller.run()

    # Run
    root.mainloop()


if __name__ == "__main__":
    main()
