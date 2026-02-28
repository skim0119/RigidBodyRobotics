from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .characters import BaseCharacter2D


COLOR_TARGET_MARKER = "#90be6d"
COLOR_TARGET_HEADING = "#90be6d"
COLOR_TRAIL = "#6c757d"


@dataclass(frozen=True)
class TargetPose2D:
    """Renderable target marker and heading."""

    x: float
    y: float
    theta: float
    marker_radius: float
    heading_length: float

    marker_color: str = COLOR_TARGET_MARKER
    heading_color: str = COLOR_TARGET_HEADING


@dataclass(frozen=True)
class Trail2D:
    """Line strip data for trajectory rendering."""

    points: Sequence[tuple[float, float]]
    color: str = COLOR_TRAIL
    width: int = 2


@dataclass(frozen=True)
class PlotSeries:
    """Single line series for a plot panel."""

    values: Sequence[float]
    color: str = "#4dabf7"
    label: str = "series"


@dataclass(frozen=True)
class PlotPanel:
    """Plot panel metadata and series payload."""

    title: str
    series: Sequence[PlotSeries]
    fixed_range: tuple[float, float] | None = None


class ModelProtocol(Protocol):
    """Minimal query contract required by TkView2D."""

    def get_object_poses(self) -> Sequence[BaseCharacter2D]: ...

    def get_target_pose(self) -> TargetPose2D | None: ...

    def get_trails(self) -> Sequence[Trail2D]: ...

    def get_plotting_data(self) -> Sequence[PlotPanel]: ...

    def get_hud_text(self) -> str: ...
