from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import tkinter as tk


COLOR_OBJECT_BODY = "#4cc9f0"
COLOR_OBJECT_HEADING = "#f9844a"
COLOR_OBJECT_VELOCITY = "#ef476f"
COLOR_TEXT = "#f8f9fa"


def _normalized_direction(dx: float, dy: float) -> tuple[float, float]:
    norm = math.hypot(dx, dy)
    if norm < 1e-12:
        return 1.0, 0.0
    return dx / norm, dy / norm


def _first_item(canvas: tk.Canvas, tag: str) -> int | None:
    # tk canvas does not prevent stacking tags
    ids = canvas.find_withtag(tag)
    if not ids:
        return None
    return int(ids[0])


@dataclass(frozen=True, eq=True)
class BaseCharacter2D(ABC):
    """Pose + drawable behavior for retained Tk canvas rendering."""

    name: str = ""

    def _group_tag(self, char_idx: int) -> str:
        return f"char:{char_idx}"

    def _subtag(self, char_idx: int, name: str) -> str:
        return f"char:{char_idx}:{name}"

    @abstractmethod
    def draw(self, canvas: tk.Canvas, char_idx: int) -> None:
        """Create canvas items for this character."""

    @abstractmethod
    def move(self, canvas: tk.Canvas, char_idx: int) -> None:
        """Update geometry and style for this character."""

    @abstractmethod
    def delete(self, canvas: tk.Canvas, char_idx: int) -> None:
        """Delete all canvas items for this character."""


@dataclass(frozen=True)
class CirclePose2D(BaseCharacter2D):
    """Circle body with heading orientation."""

    x: float = 0.0
    y: float = 0.0
    dir_x: float = 1.0
    dir_y: float = 0.0
    radius: float = 10.0
    heading_length: float = 15.0
    body_color: str = COLOR_OBJECT_BODY
    heading_color: str = COLOR_OBJECT_HEADING

    def draw(self, canvas: tk.Canvas, char_idx: int) -> None:
        g = self._group_tag(char_idx)
        body = self._subtag(char_idx, "body")
        heading = self._subtag(char_idx, "heading")
        label = self._subtag(char_idx, "label")
        dx, dy = _normalized_direction(self.dir_x, self.dir_y)

        canvas.create_oval(
            self.x - self.radius,
            self.y - self.radius,
            self.x + self.radius,
            self.y + self.radius,
            fill=self.body_color,
            outline="",
            tags=(g, body),
        )
        canvas.create_line(
            self.x,
            self.y,
            self.x + self.heading_length * dx,
            self.y + self.heading_length * dy,
            fill=self.heading_color,
            width=4,
            arrow=tk.LAST,
            arrowshape=(11, 14, 5),
            tags=(g, heading),
        )
        if self.name:
            canvas.create_text(
                self.x + self.radius + 8,
                self.y - self.radius - 2,
                text=self.name,
                fill=COLOR_TEXT,
                font=("Menlo", 11),
                anchor="sw",
                tags=(g, label),
            )

    def move(self, canvas: tk.Canvas, char_idx: int) -> None:
        g = self._group_tag(char_idx)
        if not canvas.find_withtag(g):
            self.draw(canvas, char_idx)
            return

        body = _first_item(canvas, self._subtag(char_idx, "body"))
        heading = _first_item(canvas, self._subtag(char_idx, "heading"))
        label = _first_item(canvas, self._subtag(char_idx, "label"))
        dx, dy = _normalized_direction(self.dir_x, self.dir_y)

        if body is not None:
            canvas.coords(
                body,
                self.x - self.radius,
                self.y - self.radius,
                self.x + self.radius,
                self.y + self.radius,
            )
            canvas.itemconfigure(body, fill=self.body_color, outline="")

        if heading is not None:
            canvas.coords(
                heading,
                self.x,
                self.y,
                self.x + self.heading_length * dx,
                self.y + self.heading_length * dy,
            )
            canvas.itemconfigure(heading, fill=self.heading_color, width=4)

        label_tag = self._subtag(char_idx, "label")
        if self.name:
            tx = self.x + self.radius + 8
            ty = self.y - self.radius - 2
            if label is None:
                canvas.create_text(
                    tx,
                    ty,
                    text=self.name,
                    fill=COLOR_TEXT,
                    font=("Menlo", 11),
                    anchor="sw",
                    tags=(g, label_tag),
                )
            else:
                canvas.coords(label, tx, ty)
                canvas.itemconfigure(label, text=self.name)
        elif label is not None:
            canvas.delete(label)

    def delete(self, canvas: tk.Canvas, char_idx: int) -> None:
        canvas.delete(self._group_tag(char_idx))


@dataclass(frozen=True)
class TrianglePose2D(BaseCharacter2D):
    """Triangle body with orientation and velocity direction."""

    x: float = 0.0
    y: float = 0.0
    dir_x: float = 1.0
    dir_y: float = 0.0
    radius: float = 6.0
    heading_length: float = 10.0
    vel_x: float = 0.0
    vel_y: float = 0.0
    velocity_scale: float = 1.0
    body_color: str = COLOR_OBJECT_BODY
    heading_color: str = COLOR_OBJECT_HEADING
    velocity_color: str = COLOR_OBJECT_VELOCITY
    draw_velocity: bool = True

    def _triangle_coords(self) -> tuple[float, float, float, float, float, float]:
        dx, dy = _normalized_direction(self.dir_x, self.dir_y)
        tip_x = self.x + self.heading_length * dx
        tip_y = self.y + self.heading_length * dy
        base_cx = self.x - 0.55 * self.heading_length * dx
        base_cy = self.y - 0.55 * self.heading_length * dy
        px, py = -dy, dx
        half_w = self.radius
        left_x = base_cx + half_w * px
        left_y = base_cy + half_w * py
        right_x = base_cx - half_w * px
        right_y = base_cy - half_w * py
        return tip_x, tip_y, left_x, left_y, right_x, right_y

    def draw(self, canvas: tk.Canvas, char_idx: int) -> None:
        g = self._group_tag(char_idx)
        body = self._subtag(char_idx, "body")
        vel = self._subtag(char_idx, "velocity")
        label = self._subtag(char_idx, "label")

        canvas.create_polygon(
            *self._triangle_coords(),
            fill=self.body_color,
            outline=self.heading_color,
            width=1,
            tags=(g, body),
        )
        if self.draw_velocity:
            canvas.create_line(
                self.x,
                self.y,
                self.x + self.velocity_scale * self.vel_x,
                self.y + self.velocity_scale * self.vel_y,
                fill=self.velocity_color,
                width=2,
                arrow=tk.LAST,
                arrowshape=(8, 10, 4),
                tags=(g, vel),
            )
        if self.name:
            canvas.create_text(
                self.x + self.radius + 8,
                self.y - self.radius - 2,
                text=self.name,
                fill=COLOR_TEXT,
                font=("Menlo", 11),
                anchor="sw",
                tags=(g, label),
            )

    def move(self, canvas: tk.Canvas, char_idx: int) -> None:
        g = self._group_tag(char_idx)
        if not canvas.find_withtag(g):
            self.draw(canvas, char_idx)
            return

        body = _first_item(canvas, self._subtag(char_idx, "body"))
        vel = _first_item(canvas, self._subtag(char_idx, "velocity"))
        label = _first_item(canvas, self._subtag(char_idx, "label"))

        if body is not None:
            canvas.coords(body, *self._triangle_coords())
            canvas.itemconfigure(
                body,
                fill=self.body_color,
                outline=self.heading_color,
                width=1,
            )

        vel_tag = self._subtag(char_idx, "velocity")
        if self.draw_velocity:
            x1 = self.x + self.velocity_scale * self.vel_x
            y1 = self.y + self.velocity_scale * self.vel_y
            if vel is None:
                canvas.create_line(
                    self.x,
                    self.y,
                    x1,
                    y1,
                    fill=self.velocity_color,
                    width=2,
                    arrow=tk.LAST,
                    arrowshape=(8, 10, 4),
                    tags=(g, vel_tag),
                )
            else:
                canvas.coords(vel, self.x, self.y, x1, y1)
                canvas.itemconfigure(vel, fill=self.velocity_color, width=2)
        elif vel is not None:
            canvas.delete(vel)

        label_tag = self._subtag(char_idx, "label")
        if self.name:
            tx = self.x + self.radius + 8
            ty = self.y - self.radius - 2
            if label is None:
                canvas.create_text(
                    tx,
                    ty,
                    text=self.name,
                    fill=COLOR_TEXT,
                    font=("Menlo", 11),
                    anchor="sw",
                    tags=(g, label_tag),
                )
            else:
                canvas.coords(label, tx, ty)
                canvas.itemconfigure(label, text=self.name)
        elif label is not None:
            canvas.delete(label)

    def delete(self, canvas: tk.Canvas, char_idx: int) -> None:
        canvas.delete(self._group_tag(char_idx))


# Backward-compat alias: historical ObjectPose2D behaved like a circle payload.
ObjectPose2D = CirclePose2D
