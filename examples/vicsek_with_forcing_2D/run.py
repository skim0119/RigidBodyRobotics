import tkinter as tk

import elastica_rigid as er

from config import ControlConfig, FlockingConfig
from controller import Controller
from model import SimulationModel


def main() -> None:
    root = tk.Tk()
    root.title("Vicsek Force MVC")

    model_config = FlockingConfig()
    control_config = ControlConfig()
    model = SimulationModel(model_config)
    view = er.TkView2D(root)
    controller = Controller(model, view, control_config)
    controller.run()

    root.mainloop()


if __name__ == "__main__":
    main()
