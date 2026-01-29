"""Unit tests for elastica_rigid.external_forces."""

import numpy as np

from elastica_rigid.external_forces import compute_wheel_forces_to_external


class TestComputeWheelForcesToExternal:
    """Numerical checks for compute_wheel_forces_to_external."""

    def test_forward_direction_symmetric_wheels(self) -> None:
        """d1 = [1,0]: net force along d1, zero torque when left == right."""
        direction = np.array([1.0, 0.0])
        left = np.array([1.0, 0.0])
        right = np.array([1.0, 0.0])
        track_width = np.float64(0.2)
        external_forces = np.array([0.0, 0.0])
        external_torques = np.array([0.0, 0.0])

        compute_wheel_forces_to_external(
            direction, left, right, track_width, external_forces, external_torques
        )

        # director @ (left+right) = I @ [2,0] = [2,0] in world frame
        np.testing.assert_allclose(external_forces, [2.0, 0.0])
        # (track_width/2) * (-left + right) = 0.1 * [0,0] = [0,0]
        np.testing.assert_allclose(external_torques, [0.0, 0.0])

    def test_rotated_direction(self) -> None:
        """d1 = [0,1] (90° from x): net force along d2, torque from wheel diff."""
        direction = np.array([0.0, 1.0])
        left = np.array([1.0, 0.0])
        right = np.array([0.0, 0.0])
        track_width = np.float64(0.2)
        external_forces = np.array([0.0, 0.0])
        external_torques = np.array([0.0, 0.0])

        compute_wheel_forces_to_external(
            direction, left, right, track_width, external_forces, external_torques
        )

        # d2 = R @ d1 = [-1, 0]. director = [d1 | d2] = [[0,-1],[1,0]].
        # (left+right) = [1,0]. director @ [1,0] = first column = d1 = [0,1].
        np.testing.assert_allclose(external_forces, [0.0, 1.0])
        # (track_width/2)*(-left+right) = 0.1 * [-1, 0] = [-0.1, 0]
        np.testing.assert_allclose(external_torques, [-0.1, 0.0])

    def test_adds_to_existing(self) -> None:
        """Function adds to external_forces and external_torques (does not replace)."""
        direction = np.array([1.0, 0.0])
        left = np.array([1.0, 0.0])
        right = np.array([0.0, 0.0])
        track_width = np.float64(0.2)
        external_forces = np.array([1.0, 2.0])
        external_torques = np.array([0.5, 0.0])

        compute_wheel_forces_to_external(
            direction, left, right, track_width, external_forces, external_torques
        )

        np.testing.assert_allclose(external_forces, [2.0, 2.0])  # [1,2] + [1,0]
        np.testing.assert_allclose(external_torques, [0.4, 0.0])  # [0.5,0] + [-0.1,0]
