"""Unit tests for elastica_rigid.body.equations."""

import numpy as np

from elastica_rigid.body.equations import (
    _update_accelerations,
    _update_rigid_SO2_dynamic_state,
    _update_rigid_SO2_kinematic_state,
    _zeroed_out_external_forces_and_torques,
)


class TestUpdateRigidSO2KinematicState:
    """Numerical checks for _update_rigid_SO2_kinematic_state."""

    def test_position_update(self) -> None:
        """Position advances by prefac * velocity."""
        prefac = 1.0
        position = np.array([0.0, 0.0])
        velocity = np.array([2.0, 1.0])
        direction = np.array([1.0, 0.0])
        omega = np.array([0.0])

        _update_rigid_SO2_kinematic_state(prefac, position, velocity, direction, omega)

        np.testing.assert_allclose(position, [2.0, 1.0])
        np.testing.assert_allclose(direction, [1.0, 0.0])  # no rotation

    def test_direction_rotation_90_deg(self) -> None:
        """Direction rotates by theta = prefac * omega (90 deg)."""
        prefac = 1.0
        position = np.array([0.0, 0.0])
        velocity = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        omega = np.array([np.pi / 2])

        _update_rigid_SO2_kinematic_state(prefac, position, velocity, direction, omega)

        np.testing.assert_allclose(direction, [0.0, 1.0], atol=1e-14)


class TestUpdateRigidSO2DynamicState:
    """Numerical checks for _update_rigid_SO2_dynamic_state."""

    def test_velocity_and_omega_update(self) -> None:
        """Velocity and omega advance by prefac * acceleration and prefac * alpha."""
        prefac = 0.5
        velocity = np.array([1.0, 0.0])
        acceleration = np.array([2.0, 0.0])
        omega = np.array([0.5])
        alpha = np.array([1.0])

        _update_rigid_SO2_dynamic_state(prefac, velocity, acceleration, omega, alpha)

        np.testing.assert_allclose(velocity, [2.0, 0.0])  # 1 + 0.5*2
        np.testing.assert_allclose(omega, [1.0])  # 0.5 + 0.5*1


class TestUpdateAccelerations:
    """Numerical checks for _update_accelerations."""

    def test_acceleration_equals_force_over_mass(self) -> None:
        """Acceleration = external_forces / mass, alpha = external_torques / inertia."""
        mass = 2.0
        inertia = 0.5
        acceleration = np.array([0.0, 0.0])
        alpha = np.array([0.0])
        external_forces = np.array([4.0, 2.0])
        external_torques = np.array([1.0])

        _update_accelerations(
            acceleration,
            alpha,
            np.array(mass),
            np.array(inertia),
            external_forces,
            external_torques,
        )

        np.testing.assert_allclose(acceleration, [2.0, 1.0])
        np.testing.assert_allclose(alpha, [2.0])


class TestZeroedOutExternalForcesAndTorques:
    """Numerical checks for _zeroed_out_external_forces_and_torques."""

    def test_arrays_zeroed(self) -> None:
        """External forces and torques arrays are set to zero."""
        external_forces = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        external_torques = np.array([[1.0], [2.0], [3.0]])

        _zeroed_out_external_forces_and_torques(external_forces, external_torques)

        np.testing.assert_array_equal(external_forces, 0.0)
        np.testing.assert_array_equal(external_torques, 0.0)
