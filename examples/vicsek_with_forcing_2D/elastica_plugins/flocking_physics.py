from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numba import njit

from elastica_rigid import MemoryBlockSE2Body
from .neighbor_search import CellListNeighborSearch2D


class FlockingPhysics(ABC):
    """
    Abstract base class for block-wise collision physics.
    """

    @abstractmethod
    def compute_attraction(self, system: MemoryBlockSE2Body, time: np.float64) -> None:
        """
        TODO docstring
        """


class VicsekModel(FlockingPhysics):
    """
    Linear spring-dashpot collision physics.
    """

    def __init__(
        self,
        box_size: tuple[float, float] | np.ndarray,
        config: Any | None = None,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> None:
        self._config = config
        self._sync_from_config()

        if rng is None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = rng

        self._neighbor_search = CellListNeighborSearch2D(
            box_size=np.asarray(box_size, dtype=np.float64),
            radius=self.align_radius,
        )

    def compute_attraction(self, system: MemoryBlockSE2Body, time: np.float64) -> None:
        self._sync_from_config()
        self._neighbor_search.set_radius(self.align_radius)

        positions = system.position[:2, :].T
        neighbors = self._neighbor_search.query_all(positions)

        n = positions.shape[0]
        neighbors_offsets = np.empty(n + 1, dtype=np.int64)
        neighbors_offsets[0] = 0
        for i in range(n):
            neighbors_offsets[i + 1] = neighbors_offsets[i] + len(neighbors[i])

        neighbors_flat = np.empty(neighbors_offsets[-1], dtype=np.int64)
        for i in range(n):
            start = neighbors_offsets[i]
            end = neighbors_offsets[i + 1]
            neighbors_flat[start:end] = neighbors[i]

        d1 = np.asarray(system.direction[:2, :], dtype=np.float64)
        v = np.asarray(system.velocity[:2, :], dtype=np.float64)
        omega = system.omega
        if self.noise > 0.0:
            noise_terms = self._rng.uniform(-self.noise, self.noise, size=n)
        else:
            noise_terms = np.zeros((n,), dtype=np.float64)

        forces, torques = _compute_vicsek_force_torque(
            d1,
            v,
            omega,
            neighbors_flat,
            neighbors_offsets,
            noise_terms,
            self.target_speed,
            self.k_theta,
            self.c_omega,
            self.k_v,
            self.c_v,
            self.c_perp,
            self.max_force,
            self.max_torque,
        )

        system.external_forces[:, :] = 0.0
        system.external_forces[:2, :] = forces
        system.external_torques[:] = 0.0
        system.external_torques[:] = torques

    def _sync_from_config(self) -> None:
        cfg = self._config
        self.align_radius = float(getattr(cfg, "align_radius"))
        self.noise = float(getattr(cfg, "noise"))
        self.target_speed = float(getattr(cfg, "target_speed"))
        self.k_theta = float(getattr(cfg, "k_theta"))
        self.c_omega = float(getattr(cfg, "c_omega"))
        self.k_v = float(getattr(cfg, "k_v"))
        self.c_v = float(getattr(cfg, "c_v"))
        self.c_perp = float(getattr(cfg, "c_perp"))
        self.max_force = float(getattr(cfg, "max_force"))
        self.max_torque = float(getattr(cfg, "max_torque"))


@njit(cache=True)  # type: ignore
def _compute_vicsek_force_torque(
    d1: np.ndarray,
    v: np.ndarray,
    omega: np.ndarray,
    neighbors_flat: np.ndarray,
    neighbors_offsets: np.ndarray,
    noise_terms: np.ndarray,
    target_speed: float,
    k_theta: float,
    c_omega: float,
    k_v: float,
    c_v: float,
    c_perp: float,
    max_force: float,
    max_torque: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = d1.shape[1]
    theta = np.empty((n,), dtype=np.float64)
    psi = np.empty((n,), dtype=np.float64)

    for i in range(n):
        theta[i] = np.arctan2(d1[1, i], d1[0, i])

    for i in range(n):
        sx = 0.0
        sy = 0.0
        start = neighbors_offsets[i]
        end = neighbors_offsets[i + 1]
        for p in range(start, end):
            j = neighbors_flat[p]
            sx += d1[0, j]
            sy += d1[1, j]
        if abs(sx) + abs(sy) < 1e-12:
            psi[i] = theta[i]
        else:
            psi[i] = np.arctan2(sy, sx)
        psi[i] += noise_terms[i]

    forces = np.empty((2, n), dtype=np.float64)
    torques = np.empty((n,), dtype=np.float64)
    for i in range(n):
        heading_error = (psi[i] - theta[i] + np.pi) % (2.0 * np.pi) - np.pi

        v_parallel = v[0, i] * d1[0, i] + v[1, i] * d1[1, i]
        v_perp_x = v[0, i] - d1[0, i] * v_parallel
        v_perp_y = v[1, i] - d1[1, i] * v_parallel

        forward = k_v * (target_speed - v_parallel) - c_v * v_parallel
        fx = d1[0, i] * forward - c_perp * v_perp_x
        fy = d1[1, i] * forward - c_perp * v_perp_y

        fnorm = np.sqrt(fx * fx + fy * fy)
        if fnorm > max_force:
            scale = max_force / fnorm
            fx *= scale
            fy *= scale

        tau = k_theta * heading_error - c_omega * omega[i]
        if tau > max_torque:
            tau = max_torque
        elif tau < -max_torque:
            tau = -max_torque

        forces[0, i] = fx
        forces[1, i] = fy
        torques[i] = tau

    return forces, torques
