from __future__ import annotations

__all__ = ["Flocking"]
from typing import Any, Type
import functools

import numpy as np

from elastica.typing import SystemIdxType
from elastica.modules.protocol import SystemCollectionProtocol, ModuleProtocol

from .flocking_physics import FlockingPhysics
from .equations import nb_mod_position_inplace


def _periodic_position(system, time, boundary):
    """Apply periodic wrapping to a block system position array."""
    nb_mod_position_inplace(system.position, boundary)


class Flocking(SystemCollectionProtocol):
    """
    Flocking orchestration mixin for block-based SE(2) simulations.

    This module wires a flocking-physics object into PyElastica's feature groups:
    - during `finalize`, locate the requested memory block system
    - during each synchronize stage, call `physics.compute_attraction(system=block)`
    - optionally, apply periodic wrapping in constrain-values stage

    Expected physics interface is `FlockingPhysics`, which must implement
    `compute_attraction(system)`.

    Usage
    -----
    1. Add this mixin to your simulator class.
    2. Enable block support for your SE(2) body type.
    3. Register flocking model with `configure_flocking(...).using(...)`.

    >>> import elastica as ea
    >>> import elastica_rigid as er
    >>> import elastica_plugins as ep
    ...
    >>> class Simulator(ea.BaseSystemCollection, ep.Flocking, ...):
    ...     pass
    ...
    >>> simulator = Simulator()
    >>> simulator.enable_block_supports(er.SE2RigidBody, er.MemoryBlockSE2Body)
    >>> simulator.configure_flocking(
    ...     flocking_block=er.MemoryBlockSE2Body,
    ...     box_size=(20.0, 20.0),
    ... ).using(MyFlockingPhysics, ...)
    """

    def __init__(self) -> None:
        super().__init__()
        self._feature_group_finalize.append(self._finalize_flocking)

        # Set during configure_flocking(...)
        self._flocking_controller: "_FlockingController"
        self._flocking_block_type = None
        self._boundary: tuple[float, float] | None = None

    def configure_flocking(
        self, *, flocking_block, box_size: tuple[float, float]
    ) -> ModuleProtocol:
        """
        Register a flocking block type and return a builder for flocking physics.

        Parameters
        ----------
        flocking_block
            Memory block class to target during finalize.
        box_size
            Periodic domain size `(Lx, Ly)` used for position wrapping.
        """
        self._flocking_block_type = flocking_block
        self._boundary = box_size

        self._flocking_controller = _FlockingController()
        self._feature_group_synchronize.append_id(self._flocking_controller)
        self._feature_group_constrain_values.append_id(self._flocking_controller)

        return self._flocking_controller

    def _finalize_flocking(self) -> None:
        """Resolve controller and attach flocking operators to target block systems."""
        if self._flocking_block_type is None:
            raise RuntimeError(
                "Flocking block type is not configured. "
                "Call `configure_flocking(...)` before finalize."
            )

        controller = self._flocking_controller.instantiate()

        for sys in self.final_systems():
            if not isinstance(sys, self._flocking_block_type):
                continue
            block = sys

            _compute = functools.partial(controller.compute_attraction, system=block)
            self._feature_group_synchronize.add_operators(
                self._flocking_controller, [_compute]
            )

            _periodic = functools.partial(
                _periodic_position, system=block, boundary=self._boundary
            )
            self._feature_group_constrain_values.add_operators(
                self._flocking_controller, [_periodic]
            )

        del self._flocking_controller


class _FlockingController:
    """Builder object used by `.configure_flocking(...).using(...)`."""

    _args: Any
    _kwargs: Any

    def using(self, cls: Type[FlockingPhysics], *args: Any, **kwargs: Any) -> None:
        """
        Select and parameterize the flocking-physics implementation.

        Parameters
        ----------
        cls: Type[FlockingPhysics]
            Flocking physics class.
        *args: Any
            Positional arguments forwarded to `cls`.
        **kwargs: Any
            Keyword arguments forwarded to `cls`.
        """
        assert issubclass(cls, FlockingPhysics), (
            f"{cls} is not a valid flocking physics. "
            "Did you forget to derive from FlockingPhysics?"
        )
        self._cls = cls
        self._args = args
        self._kwargs = kwargs

    def id(self) -> SystemIdxType:
        return None  # type: ignore

    def instantiate(self) -> "FlockingPhysics":
        """Construct the configured flocking-physics instance."""
        if not hasattr(self, "_cls"):
            raise RuntimeError(
                "No flocking physics was provided. "
                "Did you forget to call `.using(...)`?"
            )

        try:
            return self._cls(*self._args, **self._kwargs)
        except (TypeError, IndexError):
            raise TypeError(
                "Unable to construct flocking physics class.\n"
                "Did you provide all required parameters?"
            )
