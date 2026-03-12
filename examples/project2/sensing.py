"""Sensor models for GPS, compass, and gyro measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import elastica as ea
from common import wrap_angle


@dataclass(frozen=True)
class GPSCompassNoiseConfig:
    """Noise config for GPS + compass measurements."""

    sigma_gps2: float = 0.04
    sigma_compass2: float = 0.25


@dataclass(frozen=True)
class GyroscopeNoiseConfig:
    """Noise and bias-prior config for gyroscope measurements."""

    sigma_gyro2: float = 0.04
    mu_beta: float = 0.1
    sigma_beta2: float = 0.01


DEFAULT_GPS_COMPASS_NOISE = GPSCompassNoiseConfig()
DEFAULT_GYROSCOPE_NOISE = GyroscopeNoiseConfig()


class GPSCompassSensor(ea.CallBackBaseClass):
    """GPS + compass sensor model with callback logging support."""

    def __init__(
        self,
        step_skip: int,
        callback_params: dict,
        seed: int,
        *,
        config: GPSCompassNoiseConfig = DEFAULT_GPS_COMPASS_NOISE,
    ) -> None:
        super().__init__()
        self.every = step_skip
        self.callback_params = callback_params

        self._rng = np.random.default_rng(seed)
        self._sigma_gps = np.sqrt(config.sigma_gps2)
        self._sigma_compass = np.sqrt(config.sigma_compass2)
        self.config = config

    def sample(
        self,
        position_true: NDArray[np.float64],
        theta_true: float,
    ) -> tuple[NDArray[np.float64], float]:
        """Sample (y_gps, y_compass) from true position and heading."""
        n_gps = self._rng.normal(0.0, self._sigma_gps, size=2)
        y_gps = position_true + n_gps

        n_compass = self._rng.normal(0.0, self._sigma_compass)
        y_compass = wrap_angle(theta_true + n_compass)
        return y_gps, y_compass

    def make_callback(self, system, time, current_step: int) -> None:
        if current_step % self.every != 0:
            return

        x_true = system.position[:, 0].copy()
        d1 = system.direction[:, 0].copy()
        theta_true = np.arctan2(d1[1], d1[0])

        y_gps, y_compass = self.sample(x_true, theta_true)

        self.callback_params["time"].append(float(time))
        self.callback_params["y_gps"].append(y_gps)
        self.callback_params["y_compass"].append(y_compass)


class GyroscopeSensor(ea.CallBackBaseClass):
    """Gyro sensor model with persistent bias and callback logging support."""

    def __init__(
        self,
        step_skip: int,
        callback_params: dict,
        seed: int,
        *,
        config: GyroscopeNoiseConfig = DEFAULT_GYROSCOPE_NOISE,
    ) -> None:
        super().__init__()
        self.every = step_skip
        self.callback_params = callback_params

        self._rng = np.random.default_rng(seed)
        self._rng_beta = np.random.default_rng(seed * 2)
        self._sigma_gyro = np.sqrt(config.sigma_gyro2)
        self._sigma_beta = np.sqrt(config.sigma_beta2)
        self.config = config

    def sample(self, omega_true: float) -> float:
        """Sample y_gyro = omega + beta + n_gyro."""
        n_gyro = self._rng.normal(0.0, self._sigma_gyro)
        beta = self._rng_beta.normal(self.config.mu_beta, self._sigma_beta)
        return float(omega_true + beta + n_gyro)

    def make_callback(self, system, time, current_step: int) -> None:
        if current_step % self.every != 0:
            return

        omega_true = float(system.omega[0])
        y_gyro = self.sample(omega_true)

        self.callback_params["time"].append(float(time))
        self.callback_params["y_gyro"].append(y_gyro)
