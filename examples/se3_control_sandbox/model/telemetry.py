from __future__ import annotations

from collections import deque
import math
from typing import Deque


class Telemetry:
    def __init__(self, history_size: int, trail_size: int) -> None:
        self.trail_size = trail_size
        self.trail: list[tuple[float, float]] = []

        self.left_power_hist: Deque[float] = deque(maxlen=history_size)
        self.right_power_hist: Deque[float] = deque(maxlen=history_size)
        self.robot_theta_hist: Deque[float] = deque(maxlen=history_size)
        self.target_theta_hist: Deque[float] = deque(maxlen=history_size)
        self.robot_x_hist: Deque[float] = deque(maxlen=history_size)
        self.target_x_hist: Deque[float] = deque(maxlen=history_size)
        self.robot_y_hist: Deque[float] = deque(maxlen=history_size)
        self.target_y_hist: Deque[float] = deque(maxlen=history_size)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def clear(self) -> None:
        self.trail.clear()
        self.left_power_hist.clear()
        self.right_power_hist.clear()
        self.robot_theta_hist.clear()
        self.target_theta_hist.clear()
        self.robot_x_hist.clear()
        self.target_x_hist.clear()
        self.robot_y_hist.clear()
        self.target_y_hist.clear()

    def push_pose(self, x: float, y: float) -> None:
        self.trail.append((x, y))
        if len(self.trail) > self.trail_size:
            self.trail.pop(0)

    def record(
        self,
        *,
        left_power: float,
        right_power: float,
        robot_theta: float,
        target_theta: float,
        robot_x: float,
        target_x: float,
        robot_y: float,
        target_y: float,
    ) -> None:
        self.left_power_hist.append(left_power)
        self.right_power_hist.append(right_power)
        self.robot_theta_hist.append(math.degrees(robot_theta))
        self.target_theta_hist.append(math.degrees(self._wrap_angle(target_theta)))
        self.robot_x_hist.append(robot_x)
        self.target_x_hist.append(target_x)
        self.robot_y_hist.append(robot_y)
        self.target_y_hist.append(target_y)
