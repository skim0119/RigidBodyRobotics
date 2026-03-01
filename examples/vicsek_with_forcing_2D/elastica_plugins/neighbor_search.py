from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):  # type: ignore
        def _decorator(func):
            return func

        return _decorator


@njit(cache=True)  # type: ignore
def _build_cell_linked_list(
    positions: NDArray[np.float64],
    box_size: NDArray[np.float64],
    radius: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64], int, int, float, float]:
    cell_size = radius
    nx = max(1, int(np.floor(box_size[0] / cell_size)))
    ny = max(1, int(np.floor(box_size[1] / cell_size)))
    inv_dx = nx / box_size[0]
    inv_dy = ny / box_size[1]

    n = positions.shape[0]
    n_cells = nx * ny
    cell_counts = np.zeros(n_cells, dtype=np.int64)
    cell_ids = np.empty(n, dtype=np.int64)

    for i in range(n):
        ix = int(np.floor(positions[i, 0] * inv_dx)) % nx
        iy = int(np.floor(positions[i, 1] * inv_dy)) % ny
        cell = ix + nx * iy
        cell_ids[i] = cell
        cell_counts[cell] += 1

    cell_offsets = np.empty(n_cells + 1, dtype=np.int64)
    cell_offsets[0] = 0
    for c in range(n_cells):
        cell_offsets[c + 1] = cell_offsets[c] + cell_counts[c]

    cell_members = np.empty(n, dtype=np.int64)
    cursor = cell_offsets[:-1].copy()
    for i in range(n):
        cell = cell_ids[i]
        cell_members[cursor[cell]] = i
        cursor[cell] += 1

    return cell_offsets, cell_members, nx, ny, inv_dx, inv_dy


@njit(cache=True)  # type: ignore
def _query_all_flat(
    positions: NDArray[np.float64],
    box_size: NDArray[np.float64],
    radius: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    n = positions.shape[0]
    r2 = radius * radius
    lx = box_size[0]
    ly = box_size[1]

    cell_offsets, cell_members, nx, ny, inv_dx, inv_dy = _build_cell_linked_list(
        positions, box_size, radius
    )
    counts = np.zeros(n, dtype=np.int64)

    for i in range(n):
        px = positions[i, 0]
        py = positions[i, 1]
        ix = int(np.floor(px * inv_dx)) % nx
        iy = int(np.floor(py * inv_dy)) % ny

        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                cell = ((ix + ddx) % nx) + nx * ((iy + ddy) % ny)
                start = cell_offsets[cell]
                end = cell_offsets[cell + 1]
                for p in range(start, end):
                    j = cell_members[p]
                    dx = positions[j, 0] - px
                    dx -= np.round(dx / lx) * lx
                    dy = positions[j, 1] - py
                    dy -= np.round(dy / ly) * ly
                    if dx * dx + dy * dy <= r2:
                        counts[i] += 1

    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    for i in range(n):
        offsets[i + 1] = offsets[i] + counts[i]

    flat = np.empty(offsets[n], dtype=np.int64)
    cursor = offsets[:-1].copy()

    for i in range(n):
        px = positions[i, 0]
        py = positions[i, 1]
        ix = int(np.floor(px * inv_dx)) % nx
        iy = int(np.floor(py * inv_dy)) % ny

        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                cell = ((ix + ddx) % nx) + nx * ((iy + ddy) % ny)
                start = cell_offsets[cell]
                end = cell_offsets[cell + 1]
                for p in range(start, end):
                    j = cell_members[p]
                    dx = positions[j, 0] - px
                    dx -= np.round(dx / lx) * lx
                    dy = positions[j, 1] - py
                    dy -= np.round(dy / ly) * ly
                    if dx * dx + dy * dy <= r2:
                        flat[cursor[i]] = j
                        cursor[i] += 1

    return flat, offsets


@njit(cache=True)  # type: ignore
def _brute_force_query_all_flat(
    positions: NDArray[np.float64],
    box_size: NDArray[np.float64],
    radius: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    n = positions.shape[0]
    r2 = radius * radius
    lx = box_size[0]
    ly = box_size[1]

    counts = np.zeros(n, dtype=np.int64)
    for i in range(n):
        px = positions[i, 0]
        py = positions[i, 1]
        for j in range(n):
            dx = positions[j, 0] - px
            dx -= np.round(dx / lx) * lx
            dy = positions[j, 1] - py
            dy -= np.round(dy / ly) * ly
            if dx * dx + dy * dy <= r2:
                counts[i] += 1

    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    for i in range(n):
        offsets[i + 1] = offsets[i] + counts[i]

    flat = np.empty(offsets[n], dtype=np.int64)
    cursor = offsets[:-1].copy()
    for i in range(n):
        px = positions[i, 0]
        py = positions[i, 1]
        for j in range(n):
            dx = positions[j, 0] - px
            dx -= np.round(dx / lx) * lx
            dy = positions[j, 1] - py
            dy -= np.round(dy / ly) * ly
            if dx * dx + dy * dy <= r2:
                flat[cursor[i]] = j
                cursor[i] += 1

    return flat, offsets


@dataclass
class CellListNeighborSearch2D:
    """Periodic 2D cell-list neighbor search for radius queries."""

    box_size: NDArray[np.float64]
    radius: float

    def __post_init__(self) -> None:
        self.box_size = np.asarray(self.box_size, dtype=np.float64)
        self.radius = float(self.radius)
        if self.box_size.shape != (2,):
            raise ValueError("box_size must have shape (2,)")
        if self.radius <= 0.0:
            raise ValueError("radius must be > 0")

    def set_radius(self, radius: float) -> None:
        self.radius = max(float(radius), 1e-6)

    def periodic_displacement(
        self,
        origin: NDArray[np.float64],
        other: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        disp = other - origin
        for k in range(2):
            length = self.box_size[k]
            disp[k] -= np.round(disp[k] / length) * length
        return disp

    def query_all(self, positions: NDArray[np.float64]) -> list[list[int]]:
        """Return inclusive neighbor lists (each body includes itself)."""
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape (N, 2)")
        pos = np.asarray(positions, dtype=np.float64)
        flat, offsets = _query_all_flat(pos, self.box_size, self.radius)
        n = pos.shape[0]
        return [flat[offsets[i] : offsets[i + 1]].tolist() for i in range(n)]

    def brute_force_query_all(self, positions: NDArray[np.float64]) -> list[list[int]]:
        """Reference implementation used for correctness tests."""
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape (N, 2)")
        pos = np.asarray(positions, dtype=np.float64)
        flat, offsets = _brute_force_query_all_flat(pos, self.box_size, self.radius)
        n = pos.shape[0]
        return [flat[offsets[i] : offsets[i + 1]].tolist() for i in range(n)]
