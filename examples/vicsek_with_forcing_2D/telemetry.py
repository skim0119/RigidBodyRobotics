from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class SwarmTelemetry:
    """Ring-buffer telemetry for Vicsek example diagnostics."""

    maxlen: int = 400
    time: deque[float] = field(default_factory=lambda: deque(maxlen=400))
    order_parameter: deque[float] = field(default_factory=lambda: deque(maxlen=400))
    mean_speed: deque[float] = field(default_factory=lambda: deque(maxlen=400))
    avg_neighbors: deque[float] = field(default_factory=lambda: deque(maxlen=400))
    trail_x: deque[float] = field(default_factory=lambda: deque(maxlen=400))
    trail_y: deque[float] = field(default_factory=lambda: deque(maxlen=400))

    def __post_init__(self) -> None:
        self.time = deque(maxlen=self.maxlen)
        self.order_parameter = deque(maxlen=self.maxlen)
        self.mean_speed = deque(maxlen=self.maxlen)
        self.avg_neighbors = deque(maxlen=self.maxlen)
        self.trail_x = deque(maxlen=self.maxlen)
        self.trail_y = deque(maxlen=self.maxlen)

    def record(
        self,
        *,
        t: float,
        order: float,
        speed: float,
        avg_neighbors: float | None = None,
        focal_position: NDArray[np.float64] | None = None,
    ) -> None:
        self.time.append(float(t))
        self.order_parameter.append(float(order))
        self.mean_speed.append(float(speed))
        if avg_neighbors is None:
            self.avg_neighbors.append(np.nan)
        else:
            self.avg_neighbors.append(float(avg_neighbors))
        if focal_position is not None and focal_position.shape[0] >= 2:
            self.trail_x.append(float(focal_position[0]))
            self.trail_y.append(float(focal_position[1]))

    def reset(self) -> None:
        self.time.clear()
        self.order_parameter.clear()
        self.mean_speed.clear()
        self.avg_neighbors.clear()
        self.trail_x.clear()
        self.trail_y.clear()

    @property
    def trail_points_world(self) -> list[tuple[float, float]]:
        return list(zip(self.trail_x, self.trail_y))
