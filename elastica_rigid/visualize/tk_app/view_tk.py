from __future__ import annotations

from dataclasses import dataclass
import math
import tkinter as tk

from .characters import BaseCharacter2D
from .config import UiConfig, DEFAULT_UI_CONFIG
from .protocol import (
    ModelProtocol,
    PlotPanel,
    TargetPose2D,
    Trail2D,
)

# TODO: Maybe move to common style file?
COLOR_GRAPH_BG = "#1b2229"
COLOR_GRAPH_AXIS = "#8d99ae"
COLOR_GRAPH_TEXT = "#ced4da"
COLOR_BACKGROUND = "#101419"
COLOR_TEXT = "#f8f9fa"


@dataclass
class _ColumnSpec:
    width: int
    expand: bool
    canvas: tk.Canvas | None = None


class TkView2D(tk.Frame):
    """Tk-based 2D visualization with two columns layout.

    Column 0 is the main simulation view and expands with the window.
    Column 1 is for diagnostic plottings.

    Example:
        ```python
        import tkinter as tk
        from elastica_rigid.visualize.tk_app import TkView2D
        from elastica_rigid.visualize.tk_app import CirclePose2D

        class DummyModel:
            def get_object_poses(self):
                return [CirclePose2D(x=120, y=120, dir_x=1.0, dir_y=0.0, radius=20, heading_length=30)]

            def get_target_pose(self):
                return TargetPose2D(x=200, y=150, theta=0.3, marker_radius=6, heading_length=30)

            def get_trails(self):
                return [Trail2D(points=[])]

            def get_plotting_data(self):
                return [PlotPanel(title="Empty", series=[])]

            def get_hud_text(self):
                return "Hellow World"

        root = tk.Tk()
        view = TkView2D(root)
        view.render(DummyModel())
        root.mainloop()
        ```
    """

    def __init__(
        self,
        parent: tk.Tk,
        ui_config: UiConfig = DEFAULT_UI_CONFIG,
    ) -> None:
        """Initialize the two-panel Tk view and key bindings."""
        super().__init__(parent, bg=COLOR_BACKGROUND)
        self.root = parent
        self.ui = ui_config

        self._columns: list[_ColumnSpec] = [
            _ColumnSpec(width=self.ui.left_panel_width, expand=True),
            _ColumnSpec(
                width=self.ui.window_width - self.ui.left_panel_width, expand=False
            ),
        ]

        # Toggle handles and flags
        self._show_hud = True
        self._show_plots = True
        self.grid_rowconfigure(0, weight=1)
        self.pack(fill="both", expand=True)
        self.bind_all("<h>", self._on_toggle_hud)
        self.bind_all("<H>", self._on_toggle_hud)
        self.bind_all("<p>", self._on_toggle_plots)
        self.bind_all("<P>", self._on_toggle_plots)
        self.bind_all("<Escape>", self._on_escape)

        # Retained-mode caches for left canvas.
        self._object_items: dict[int, BaseCharacter2D] = {}
        self._trail_items: dict[int, int] = {}
        self._target_marker_id: int | None = None
        self._target_heading_id: int | None = None
        self._hud_id: int | None = None

        # Initialize
        self._build_columns()

    def get_hud_text(self) -> str:
        return "Keys: h toggle HUD | p toggle plots | Esc quit"

    def get_left_panel_width(self) -> int:
        """Return current drawable width of the left panel in pixels."""
        width = int(self.left_canvas.winfo_width())
        if width <= 1:
            width = int(self._columns[0].width)
        return max(width, 1)

    def get_left_panel_height(self) -> int:
        """Return current drawable height of the left panel in pixels."""
        height = int(self.left_canvas.winfo_height())
        if height <= 1:
            height = int(self.ui.window_height)
        return max(height, 1)

    def render(self, model: ModelProtocol) -> None:
        """Draw one frame from model query data."""
        # Left column uses retained-mode updates to reduce item churn.
        trails = list(model.get_trails() or [])
        self._draw_trails(trails)
        self._draw_target(model.get_target_pose())
        objects = list(model.get_object_poses() or [])
        self._draw_objects(objects)
        if self._show_hud:
            self_hud = self.get_hud_text()
            model_hud = model.get_hud_text()
            self._draw_hud(self_hud, model_hud)
        else:
            self._draw_hud()

        # Right column remains immediate-mode redraw.
        self.right_canvas.delete("all")

        # Right Column
        if self._show_plots and (_plot_data := model.get_plotting_data()):
            self._draw_graphs(_plot_data)

    # -- Toggles --

    def _on_toggle_hud(self, _: tk.Event) -> None:
        self._show_hud = not self._show_hud

    def _on_toggle_plots(self, _: tk.Event) -> None:
        self._show_plots = not self._show_plots
        self._apply_plot_visibility_layout()

    def _on_escape(self, _: tk.Event) -> None:
        self.winfo_toplevel().destroy()

    # -- Visibility Handles --

    def _apply_plot_visibility_layout(self) -> None:
        if self._show_plots:
            self.right_canvas.grid()
            self.grid_columnconfigure(1, minsize=self._columns[1].width)
            self.right_canvas.configure(width=self._columns[1].width)
        else:
            self.right_canvas.grid_remove()
            self.grid_columnconfigure(1, minsize=0)
            self.right_canvas.configure(width=0)

    def _build_columns(self) -> None:
        for idx, spec in enumerate(self._columns):
            weight = 1 if spec.expand else 0
            self.grid_columnconfigure(idx, weight=weight, minsize=spec.width)
            canvas = tk.Canvas(
                self,
                width=spec.width,
                height=self.ui.window_height,
                bg=COLOR_BACKGROUND,
                highlightthickness=0,
            )
            canvas.grid(row=0, column=idx, sticky="nsew")
            spec.canvas = canvas

        self.left_canvas = self._columns[0].canvas
        self.right_canvas = self._columns[1].canvas
        self._apply_plot_visibility_layout()

    def resize_column(self, column_index: int, size: int) -> TkView2D:
        """Resize a column width in pixels."""
        if column_index < 0 or column_index >= len(self._columns):
            raise IndexError(f"Invalid column index: {column_index}")
        if size <= 0:
            raise ValueError("Column size must be positive")

        spec = self._columns[column_index]
        spec.width = int(size)
        self.grid_columnconfigure(column_index, minsize=spec.width)
        if spec.canvas is not None:
            spec.canvas.configure(width=spec.width)
        return self

    # -- Drawing --

    def _draw_trails(self, trails: list[Trail2D]) -> None:
        for idx, trail in enumerate(trails):
            points_flat: list[float] = []
            for x, y in trail.points:
                points_flat.extend([x, y])

            item_id = self._trail_items.get(idx)
            if len(points_flat) < 4:
                if item_id is not None:
                    self.left_canvas.coords(item_id, 0, 0, 0, 0)
                continue

            if item_id is None:
                item_id = self.left_canvas.create_line(
                    *points_flat,
                    fill=trail.color,
                    width=trail.width,
                )
                self._trail_items[idx] = item_id
            else:
                self.left_canvas.coords(item_id, *points_flat)
                self.left_canvas.itemconfigure(
                    item_id, fill=trail.color, width=trail.width
                )

        stale = [k for k in self._trail_items if k >= len(trails)]
        for k in stale:
            self.left_canvas.delete(self._trail_items[k])
            del self._trail_items[k]

    def _draw_target(self, target: TargetPose2D | None) -> None:
        if target is None:
            if self._target_marker_id is not None:
                self.left_canvas.delete(self._target_marker_id)
                self._target_marker_id = None
            if self._target_heading_id is not None:
                self.left_canvas.delete(self._target_heading_id)
                self._target_heading_id = None
            return

        d = target.marker_radius
        marker_coords = (target.x - d, target.y - d, target.x + d, target.y + d)
        hx = target.x + target.heading_length * math.cos(target.theta)
        hy = target.y + target.heading_length * math.sin(target.theta)

        if self._target_marker_id is None:
            self._target_marker_id = self.left_canvas.create_oval(
                *marker_coords,
                outline=target.marker_color,
                width=2,
            )
        else:
            self.left_canvas.coords(self._target_marker_id, *marker_coords)
            self.left_canvas.itemconfigure(
                self._target_marker_id,
                outline=target.marker_color,
                width=2,
            )

        if self._target_heading_id is None:
            self._target_heading_id = self.left_canvas.create_line(
                target.x,
                target.y,
                hx,
                hy,
                fill=target.heading_color,
                width=3,
                dash=(8, 6),
                arrow=tk.LAST,
                arrowshape=(10, 12, 4),
            )
        else:
            self.left_canvas.coords(self._target_heading_id, target.x, target.y, hx, hy)
            self.left_canvas.itemconfigure(
                self._target_heading_id,
                fill=target.heading_color,
                width=3,
                dash=(8, 6),
            )

    def _draw_objects(self, objects: list[BaseCharacter2D]) -> None:
        for idx, pose in enumerate(objects):
            prev = self._object_items.get(idx)
            if prev is None:  # Does not exist
                pose.draw(self.left_canvas, idx)
                self._object_items[idx] = pose
                continue

            if type(prev) is not type(pose):
                prev.delete(self.left_canvas, idx)
                pose.draw(self.left_canvas, idx)
                self._object_items[idx] = pose
            else:
                pose.move(self.left_canvas, idx)
                self._object_items[idx] = pose

        stale = [k for k in self._object_items if k >= len(objects)]
        for k in stale:
            self._object_items[k].delete(self.left_canvas, k)
            del self._object_items[k]

    def _draw_line_plot(
        self,
        canvas: tk.Canvas,
        x0: float,
        y0: float,
        width: float,
        height: float,
        panel: PlotPanel,
    ) -> None:
        canvas.create_rectangle(
            x0,
            y0,
            x0 + width,
            y0 + height,
            fill=COLOR_GRAPH_BG,
            outline=COLOR_GRAPH_AXIS,
            width=1,
        )
        canvas.create_text(
            x0 + 8,
            y0 + 8,
            text=panel.title,
            fill=COLOR_GRAPH_TEXT,
            font=("Menlo", 11),
            anchor="nw",
        )

        # Draw legend horizontally below title, wrapping to next row if needed.
        legend_x = x0 + 8
        legend_y = y0 + 24
        legend_cursor_x = legend_x
        legend_cursor_y = legend_y
        legend_max_x = x0 + width - 12
        legend_row_h = 14
        legend_rows = 1
        for s in panel.series:
            entry_w = max(70, 30 + 7 * len(s.label))
            if legend_cursor_x + entry_w > legend_max_x:
                legend_cursor_x = legend_x
                legend_cursor_y += legend_row_h
                legend_rows += 1
            y = legend_cursor_y
            canvas.create_line(
                legend_cursor_x, y, legend_cursor_x + 16, y, fill=s.color, width=3
            )
            canvas.create_text(
                legend_cursor_x + 22,
                y,
                text=s.label,
                fill=COLOR_GRAPH_TEXT,
                font=("Menlo", 10),
                anchor="w",
            )
            legend_cursor_x += entry_w

        left_pad = 12.0
        right_pad = 8.0
        top_pad = 28.0 + legend_row_h * legend_rows + 4.0
        bottom_pad = 12.0
        px0 = x0 + left_pad
        px1 = x0 + width - right_pad
        py0 = y0 + top_pad
        py1 = y0 + height - bottom_pad
        canvas.create_rectangle(px0, py0, px1, py1, outline=COLOR_GRAPH_AXIS, width=1)

        data = [v for s in panel.series for v in s.values]
        if not data:
            return
        if panel.fixed_range is None:
            ymin = min(data)
            ymax = max(data)
            if math.isclose(ymin, ymax):
                ymin -= 1.0
                ymax += 1.0
            pad = 0.1 * (ymax - ymin)
            ymin -= pad
            ymax += pad
        else:
            ymin, ymax = panel.fixed_range

        span = ymax - ymin
        if span <= 1e-12:
            span = 1.0
            ymin -= 0.5

        max_len = max((len(s.values) for s in panel.series), default=1)
        if max_len <= 1:
            max_len = 2

        y_zero = py1 - ((0.0 - ymin) / span) * (py1 - py0)
        canvas.create_line(px0, y_zero, px1, y_zero, fill="#495057", dash=(3, 3))

        for s in panel.series:
            if len(s.values) < 2:
                continue
            points: list[float] = []
            for i, val in enumerate(s.values):
                x = px0 + (i / (max_len - 1)) * (px1 - px0)
                y = py1 - ((val - ymin) / span) * (py1 - py0)
                points.extend([x, y])
            canvas.create_line(*points, fill=s.color, width=2)

    def _get_right_panel_width(self) -> int:
        width = int(self.right_canvas.winfo_width())
        return width if width > 1 else (self.ui.window_width - self.ui.left_panel_width)

    def _draw_graphs(self, panels: list[PlotPanel]) -> None:
        panel_x = 14
        panel_w = self._get_right_panel_width() - 28
        gap = 10
        available_h = self.ui.window_height - 14 - 14 - gap * (len(panels) - 1)
        graph_h = max(120, int(available_h / len(panels)))

        for i, panel in enumerate(panels):
            y = 14 + i * (graph_h + gap)
            self._draw_line_plot(self.right_canvas, panel_x, y, panel_w, graph_h, panel)

    def _draw_hud(self, *huds: str) -> None:
        hud_text = "\n".join(huds)

        if self._hud_id is None:
            # Create new hud
            self._hud_id = self.left_canvas.create_text(
                14,
                14,
                text=hud_text,
                fill=COLOR_TEXT,
                font=("Menlo", 14),
                anchor="nw",
            )
        else:
            # Change existing hud
            self.left_canvas.coords(self._hud_id, 14, 14)
            self.left_canvas.itemconfigure(self._hud_id, text=hud_text)
