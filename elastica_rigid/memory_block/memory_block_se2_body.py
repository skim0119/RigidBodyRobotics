"""SE(2) memory block for Roomba-like planar rigid bodies."""

from __future__ import annotations

from math import ceil
from typing import Literal

import numpy as np
from ..body.roomba import SE2RigidBody

from elastica.typing import SystemIdxType

_VECTOR_FIELDS: tuple[str, ...] = (
    "position",
    "direction",
    "velocity",
    "acceleration",
    "external_forces",
)

_SCALAR_FIELDS: tuple[str, ...] = (
    "mass",
    "inertia",
    "omega",
    "alpha",
    "external_torques",
)


class MemoryBlockSE2Body(SE2RigidBody):
    """
    Memory block for SE(2) rigid bodies.

    Public collections use:
    - vectors: `(2, N)`
    - scalars: `(N,)`
    """

    STORAGE_LAYOUT: Literal["soa", "aos"] = "soa"
    BLOCKING_POLICY: Literal["single", "segmented"] = "single"
    SEGMENT_SIZE: int = 1024
    _LOCKED_CONFIG: bool = False

    @classmethod
    def configure(
        cls,
        *,
        storage_layout: Literal["soa", "aos"] | None = None,
        blocking_policy: Literal["single", "segmented"] | None = None,
        segment_size: int | None = None,
    ) -> type["MemoryBlockSE2Body"]:
        """
        Return a configured MemoryBlockSE2Body class.

        Example
        -------
        MemoryBlockAOS = MemoryBlockSE2Body.configure(storage_layout="aos")
        block = MemoryBlockAOS(systems, system_idx_list)
        """
        layout = cls.STORAGE_LAYOUT if storage_layout is None else storage_layout
        policy = cls.BLOCKING_POLICY if blocking_policy is None else blocking_policy
        seg = cls.SEGMENT_SIZE if segment_size is None else int(segment_size)

        if layout not in ("soa", "aos"):
            raise ValueError("storage_layout must be one of {'soa', 'aos'}")
        if policy not in ("single", "segmented"):
            raise ValueError("blocking_policy must be one of {'single', 'segmented'}")
        if policy == "segmented" and seg <= 0:
            raise ValueError("segment_size must be > 0 for segmented blocking")

        configured_name = f"{cls.__name__}_Configured_{layout}_{policy}_{seg}"
        return type(
            configured_name,
            (cls,),
            {
                "STORAGE_LAYOUT": layout,
                "BLOCKING_POLICY": policy,
                "SEGMENT_SIZE": seg,
                "_LOCKED_CONFIG": True,
            },
        )

    def __init__(
        self,
        systems: list[SE2RigidBody],
        system_idx_list: list[SystemIdxType],
        *,
        storage_layout: Literal["soa", "aos"] | None = None,
        blocking_policy: Literal["single", "segmented"] | None = None,
        segment_size: int | None = None,
    ) -> None:
        if self.__class__._LOCKED_CONFIG and any(
            v is not None for v in (storage_layout, blocking_policy, segment_size)
        ):
            raise ValueError(
                "This configured class has fixed memory-block parameters. "
                "Instantiate without per-instance overrides."
            )

        self.n_systems = len(systems)
        self.n_elems = self.n_systems
        self.n_nodes = self.n_elems
        self.system_idx_list = np.asarray(system_idx_list, dtype=np.int32)
        self.storage_layout = (
            self.__class__.STORAGE_LAYOUT if storage_layout is None else storage_layout
        )
        self.blocking_policy = (
            self.__class__.BLOCKING_POLICY
            if blocking_policy is None
            else blocking_policy
        )
        self.segment_size = (
            int(self.__class__.SEGMENT_SIZE)
            if segment_size is None
            else int(segment_size)
        )

        if self.storage_layout not in ("soa", "aos"):
            raise ValueError("storage_layout must be one of {'soa', 'aos'}")
        if self.blocking_policy not in ("single", "segmented"):
            raise ValueError("blocking_policy must be one of {'single', 'segmented'}")
        if self.n_elems < 0:
            raise ValueError("number of systems cannot be negative")
        if self.blocking_policy == "segmented" and self.segment_size <= 0:
            raise ValueError("segment_size must be > 0 for segmented blocking")

        self._allocate_vector_storage()
        self._allocate_scalar_storage()
        self._create_collection_views()
        self._initialize_from_systems_and_swap_views(systems)
        self._setup_segment_slices()
        self._setup_stepper_views()

    def _allocate_vector_storage(self) -> None:
        n_vec = len(_VECTOR_FIELDS)
        n = self.n_elems
        if self.blocking_policy == "single":
            if self.storage_layout == "soa":
                # [field, dim, elem]
                self._vector_block = np.zeros((n_vec, 2, n), dtype=np.float64)
            else:
                # [elem, field, dim]
                self._vector_block = np.zeros((n, n_vec, 2), dtype=np.float64)
            return

        n_seg = ceil(n / self.segment_size) if n > 0 else 0
        block = self.segment_size
        if self.storage_layout == "soa":
            # [seg, field, dim, local_elem]
            self._vector_block = np.zeros((n_seg, n_vec, 2, block), dtype=np.float64)
        else:
            # [seg, local_elem, field, dim]
            self._vector_block = np.zeros((n_seg, block, n_vec, 2), dtype=np.float64)

    def _allocate_scalar_storage(self) -> None:
        n_sca = len(_SCALAR_FIELDS)
        n = self.n_elems
        if self.blocking_policy == "single":
            if self.storage_layout == "soa":
                # [field, elem]
                self._scalar_block = np.zeros((n_sca, n), dtype=np.float64)
            else:
                # [elem, field]
                self._scalar_block = np.zeros((n, n_sca), dtype=np.float64)
            return

        n_seg = ceil(n / self.segment_size) if n > 0 else 0
        block = self.segment_size
        if self.storage_layout == "soa":
            # [seg, field, local_elem]
            self._scalar_block = np.zeros((n_seg, n_sca, block), dtype=np.float64)
        else:
            # [seg, local_elem, field]
            self._scalar_block = np.zeros((n_seg, block, n_sca), dtype=np.float64)

    def _vector_field_view(self, field_idx: int) -> np.ndarray:
        n = self.n_elems
        if self.blocking_policy == "single":
            if self.storage_layout == "soa":
                return self._vector_block[field_idx]
            return self._vector_block[:, field_idx, :].T

        if self.storage_layout == "soa":
            packed = self._vector_block[:, field_idx, :, :]  # (S,2,B)
            view = packed.transpose(1, 0, 2).reshape(2, -1)
        else:
            packed = self._vector_block[:, :, field_idx, :]  # (S,B,2)
            view = packed.transpose(2, 0, 1).reshape(2, -1)
        return view[:, :n]

    def _scalar_field_view(self, field_idx: int) -> np.ndarray:
        n = self.n_elems
        if self.blocking_policy == "single":
            if self.storage_layout == "soa":
                return self._scalar_block[field_idx]
            return self._scalar_block[:, field_idx]

        if self.storage_layout == "soa":
            packed = self._scalar_block[:, field_idx, :]  # (S,B)
        else:
            packed = self._scalar_block[:, :, field_idx]  # (S,B)
        return packed.reshape(-1)[:n]

    def _create_collection_views(self) -> None:
        for idx, name in enumerate(_VECTOR_FIELDS):
            setattr(self, name, self._vector_field_view(idx))
        for idx, name in enumerate(_SCALAR_FIELDS):
            setattr(self, name, self._scalar_field_view(idx))

    def _setup_segment_slices(self) -> None:
        if self.n_elems == 0:
            self.segment_slices: tuple[slice, ...] = ()
            return
        if self.blocking_policy == "single":
            self.segment_slices = (slice(0, self.n_elems),)
            return
        starts = range(0, self.n_elems, self.segment_size)
        self.segment_slices = tuple(
            slice(s, min(s + self.segment_size, self.n_elems)) for s in starts
        )

    def _setup_stepper_views(self) -> None:
        self.v_w_collection = np.vstack((self.velocity, self.omega[None, :]))
        self.dvdt_dwdt_collection = np.vstack((self.acceleration, self.alpha[None, :]))

    @staticmethod
    def _read_vector2(system: SE2RigidBody, field: str) -> np.ndarray:
        raw = np.asarray(getattr(system, field), dtype=np.float64)
        if raw.shape == (2, 1):
            return raw
        if raw.shape == (2,):
            return raw.reshape(2, 1)
        if raw.shape == (1, 2):
            return raw.T
        if raw.ndim == 2 and raw.shape[0] == 2 and raw.shape[1] >= 1:
            return raw[:, :1]
        raise ValueError(
            f"Expected {field} to have shape (2,1) or (2,), got {raw.shape}"
        )

    @staticmethod
    def _read_scalar(system: SE2RigidBody, field: str) -> float:
        raw = np.asarray(getattr(system, field), dtype=np.float64)
        if raw.ndim == 0:
            return float(raw)
        if raw.ndim == 1 and raw.shape[0] >= 1:
            return float(raw[0])
        raise ValueError(
            f"Expected {field} to be scalar or shape (1,), got {raw.shape}"
        )

    @staticmethod
    def _swap_vector_view(
        system: SE2RigidBody,
        field: str,
        source: np.ndarray,
        system_idx: int,
    ) -> None:
        setattr(system, field, source[:, system_idx : system_idx + 1])

    @staticmethod
    def _swap_scalar_view(
        system: SE2RigidBody,
        field: str,
        source: np.ndarray,
        system_idx: int,
    ) -> None:
        setattr(system, field, source[system_idx : system_idx + 1])

    def _initialize_from_systems_and_swap_views(
        self, systems: list[SE2RigidBody]
    ) -> None:
        for i, system in enumerate(systems):
            for field in _VECTOR_FIELDS:
                collection = getattr(self, field)
                collection[:, i : i + 1] = self._read_vector2(system, field)
                self._swap_vector_view(system, field, collection, i)

            for field in _SCALAR_FIELDS:
                collection = getattr(self, field)
                collection[i] = self._read_scalar(system, field)
                self._swap_scalar_view(system, field, collection, i)

    def iter_vector_segments(self, name: str):
        """Yield `(2, segment_len)` views over one vector collection."""
        if name not in _VECTOR_FIELDS:
            raise KeyError(f"Unknown vector collection '{name}'")
        arr = getattr(self, name)
        for slc in self.segment_slices:
            yield arr[:, slc]

    def iter_scalar_segments(self, name: str):
        """Yield `(segment_len,)` views over one scalar collection."""
        if name not in _SCALAR_FIELDS:
            raise KeyError(f"Unknown scalar collection '{name}'")
        arr = getattr(self, name)
        for slc in self.segment_slices:
            yield arr[slc]
