import numpy as np
import pytest

from elastica_rigid.memory_block.memory_block_se2_body import MemoryBlockSE2Body


class _DummySE2System:
    def __init__(self, i: int):
        # Vector state aliases used by Roomba-like systems.
        self.position = np.array([[i + 0.1], [i + 0.2]], dtype=np.float64)
        self.direction = np.array([[1.0], [0.0]], dtype=np.float64)
        self.velocity = np.array([[0.5], [0.25]], dtype=np.float64)
        self.acceleration = np.array([[0.0], [0.0]], dtype=np.float64)
        self.external_forces = np.array([[0.0], [0.0]], dtype=np.float64)

        # Scalar state aliases used by Roomba-like systems.
        self.mass = np.array([1.0 + i], dtype=np.float64)
        self.inertia = np.array([0.2 + i], dtype=np.float64)
        self.radius = np.array([0.3], dtype=np.float64)
        self.width = np.array([0.4], dtype=np.float64)
        self.omega = np.array([0.0], dtype=np.float64)
        self.alpha = np.array([0.0], dtype=np.float64)
        self.external_torques = np.array([0.0], dtype=np.float64)


def _build_dummy_systems(n: int) -> list[_DummySE2System]:
    return [_DummySE2System(i) for i in range(n)]


def test_reference_swap_default_memory_block() -> None:
    systems = _build_dummy_systems(4)
    block = MemoryBlockSE2Body(systems, list(range(4)))

    # System references should be swapped to block-backed views.
    assert np.shares_memory(systems[0].position, block.position)
    assert np.shares_memory(systems[1].velocity, block.velocity)
    assert np.shares_memory(systems[2].omega, block.omega)
    assert np.shares_memory(systems[3].mass, block.mass)

    # Mutation through block must be visible through per-system references.
    block.position[0, 0] = 42.0
    block.omega[1] = 3.5
    assert systems[0].position[0, 0] == pytest.approx(42.0)
    assert systems[1].omega[0] == pytest.approx(3.5)

    # Mutation through per-system view must be visible in block collections.
    systems[2].velocity[1, 0] = -7.0
    systems[3].mass[0] = 99.0
    assert block.velocity[1, 2] == pytest.approx(-7.0)
    assert block.mass[3] == pytest.approx(99.0)


def test_reference_swap_with_configured_subclass_segmented() -> None:
    systems = _build_dummy_systems(5)
    configured_cls = MemoryBlockSE2Body.configure(
        storage_layout="aos",
        blocking_policy="segmented",
        segment_size=2,
    )
    block = configured_cls(systems, list(range(5)))

    assert configured_cls.STORAGE_LAYOUT == "aos"
    assert configured_cls.BLOCKING_POLICY == "segmented"
    assert configured_cls.SEGMENT_SIZE == 2
    assert block.storage_layout == "aos"
    assert block.blocking_policy == "segmented"
    assert block.segment_size == 2

    # Reference swap still holds under segmented AoS layout.
    assert np.shares_memory(systems[4].position, block.position)
    assert np.shares_memory(systems[4].external_torques, block.external_torques)

    block.position[1, 4] = 11.0
    systems[0].external_torques[0] = -2.0
    assert systems[4].position[1, 0] == pytest.approx(11.0)
    assert block.external_torques[0] == pytest.approx(-2.0)

    # Configured classes lock instance-level overrides.
    with pytest.raises(ValueError):
        configured_cls(systems, list(range(5)), storage_layout="soa")

