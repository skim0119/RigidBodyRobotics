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


@dataclass
class LowPassPositionFilter:
    """First-order low-pass position estimator for GPS measurements."""

    alpha_x: float
    xhat: NDArray[np.float64]

    def __post_init__(self) -> None:
        self.alpha_x = float(self.alpha_x)
        self.xhat = np.asarray(self.xhat, dtype=np.float64).copy()
        if self.xhat.shape != (2,):
            raise ValueError(f"xhat must have shape (2,), got {self.xhat.shape}")

    def update(self, y_gps: NDArray[np.float64]) -> NDArray[np.float64]:
        """xhat_{k+1} = (1-alpha)*xhat_k + alpha*y_gps_k."""
        y = np.asarray(y_gps, dtype=np.float64)
        self.xhat = (1.0 - self.alpha_x) * self.xhat + self.alpha_x * y
        return self.xhat.copy()

    @staticmethod
    def steady_state_variance(alpha_x: float, sigma_gps2: float) -> float:
        """Steady-state scalar variance: alpha/(2-alpha) * sigma_gps^2."""
        alpha = float(alpha_x)
        return float((alpha / (2.0 - alpha)) * sigma_gps2)


@dataclass
class ComplementaryHeadingFilter:
    """Complementary heading filter with optional online gyro-bias estimation."""

    alpha_theta: float
    dt: float
    theta_hat: float
    estimate_bias: bool = False
    gamma_beta: float = 0.0
    beta_hat: float = 0.0

    def __post_init__(self) -> None:
        self.alpha_theta = float(self.alpha_theta)
        self.dt = float(self.dt)
        self.theta_hat = float(self.theta_hat)
        self.gamma_beta = float(self.gamma_beta)
        self.beta_hat = float(self.beta_hat)

    def update(self, y_compass: float, y_gyro: float) -> float:
        """Update heading estimate according to the selected complementary form."""
        if self.estimate_bias:
            theta_gyro_pred = wrap_angle(
                self.theta_hat + (float(y_gyro) - self.beta_hat) * self.dt
            )
            theta_err = wrap_angle(float(y_compass) - theta_gyro_pred)
            self.theta_hat = wrap_angle(
                theta_gyro_pred + (1.0 - self.alpha_theta) * theta_err
            )
            self.beta_hat += self.gamma_beta * theta_err * self.dt
            return self.theta_hat

        theta_gyro_pred = wrap_angle(self.theta_hat + float(y_gyro) * self.dt)
        theta_err = wrap_angle(float(y_compass) - theta_gyro_pred)
        # Equivalent complementary correction on S^1; avoids direct averaging artifacts.
        self.theta_hat = wrap_angle(
            theta_gyro_pred + (1.0 - self.alpha_theta) * theta_err
        )
        return self.theta_hat
