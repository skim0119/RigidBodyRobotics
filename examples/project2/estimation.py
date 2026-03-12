"""State estimators for Project 2 sensing and control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from common import wrap_angle


@dataclass
class GyroIntegrator:
    """Heading obtained by direct gyro integration (for diagnostics/baselines)."""

    dt: float
    theta: float

    def update(self, y_gyro: float, *, bias_compensation: float = 0.0) -> float:
        self.theta = wrap_angle(self.theta + (y_gyro - bias_compensation) * self.dt)
        return self.theta
