"""Unit tests for elastica_rigid.external_forces."""

import numpy as np

from elastica_rigid.external_forces import (
    boundary_penetration_forces,
    closest_point_on_aabb,
    compute_friction_force_mag_dir,
    compute_potential_field_wheel_forces,
    compute_single_wheel_friction,
    compute_wheel_forces_to_external,
    contact_force_circle_vs_aabb,
    interp_piecewise_linear,
    point_in_friction_region,
    torque_z_from_force_2d,
    wheel_velocity_2d,
)


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


class TestComputePotentialFieldWheelForces:
    """Shape and run tests for compute_potential_field_wheel_forces."""

    def test_output_shapes(self) -> None:
        x = np.array([1.0, 2.0], dtype=np.float64)
        d1 = np.array([1.0, 0.0], dtype=np.float64)
        K = np.float64(0.5)
        left, right = compute_potential_field_wheel_forces(x, d1, K)
        assert left.shape == (2,)
        assert right.shape == (2,)

    def test_runs_for_valid_input(self) -> None:
        x = np.array([0.0, 0.0], dtype=np.float64)
        d1 = np.array([0.0, 1.0], dtype=np.float64)
        left, right = compute_potential_field_wheel_forces(x, d1, np.float64(1.0))
        np.testing.assert_allclose(left, [0.0, 0.0])
        np.testing.assert_allclose(right, [0.0, 0.0])


class TestInterpPiecewiseLinear:
    """Shape and run tests for interp_piecewise_linear."""

    def test_returns_scalar(self) -> None:
        t = np.float64(1.0)
        times = np.array([0.0, 2.0], dtype=np.float64)
        values = np.array([0.0, 1.0], dtype=np.float64)
        out = interp_piecewise_linear(t, times, values)
        assert np.isscalar(out) or getattr(out, "shape", None) == ()

    def test_runs_extrapolate_left(self) -> None:
        out = interp_piecewise_linear(
            np.float64(-1.0),
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([10.0, 20.0], dtype=np.float64),
        )
        assert out == 10.0

    def test_runs_interior(self) -> None:
        out = interp_piecewise_linear(
            np.float64(0.5),
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([0.0, 2.0], dtype=np.float64),
        )
        assert out == 1.0


class TestClosestPointOnAabb:
    """Shape and run tests for closest_point_on_aabb."""

    def test_output_shape(self) -> None:
        p = np.array([0.5, 0.5], dtype=np.float64)
        mins = np.array([0.0, 0.0], dtype=np.float64)
        maxs = np.array([1.0, 1.0], dtype=np.float64)
        out = closest_point_on_aabb(p, mins, maxs)
        assert out.shape == (2,)

    def test_runs_inside(self) -> None:
        p = np.array([0.5, 0.5], dtype=np.float64)
        mins = np.array([0.0, 0.0], dtype=np.float64)
        maxs = np.array([1.0, 1.0], dtype=np.float64)
        out = closest_point_on_aabb(p, mins, maxs)
        np.testing.assert_allclose(out, [0.5, 0.5])


class TestContactForceCircleVsAabb:
    """Shape and run tests for contact_force_circle_vs_aabb."""

    def test_output_shape(self) -> None:
        center = np.array([0.5, 0.5], dtype=np.float64)
        radius = np.float64(0.1)
        mins = np.array([0.0, 0.0], dtype=np.float64)
        maxs = np.array([1.0, 1.0], dtype=np.float64)
        out = contact_force_circle_vs_aabb(
            center, radius, mins, maxs, np.float64(1000.0), np.float64(1e-12)
        )
        assert out.shape == (2,)

    def test_runs_no_overlap(self) -> None:
        center = np.array([2.0, 2.0], dtype=np.float64)
        out = contact_force_circle_vs_aabb(
            center,
            np.float64(0.1),
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
            np.float64(1000.0),
            np.float64(1e-12),
        )
        np.testing.assert_allclose(out, [0.0, 0.0])


class TestPointInFrictionRegion:
    """Run tests for point_in_friction_region."""

    def test_returns_bool(self) -> None:
        x = np.array([0.0, 5.0], dtype=np.float64)
        out = point_in_friction_region(x, np.float64(-4.0 / 3.0), np.float64(4.0))
        assert out is True

    def test_below_line(self) -> None:
        x = np.array([0.0, 0.0], dtype=np.float64)
        out = point_in_friction_region(x, np.float64(-4.0 / 3.0), np.float64(4.0))
        assert out is False


class TestComputeFrictionForceMagDir:
    """Shape and run tests for compute_friction_force_mag_dir."""

    def test_output_shapes(self) -> None:
        v = np.array([1.0, 0.0], dtype=np.float64)
        f_mag, f_dir = compute_friction_force_mag_dir(
            v, np.float64(1.0), np.float64(0.5), np.float64(1.0), np.float64(1e-12)
        )
        assert np.isscalar(f_mag) or getattr(f_mag, "shape", None) == ()
        assert f_dir.shape == (2,)

    def test_zero_speed_returns_zero(self) -> None:
        v = np.array([0.0, 0.0], dtype=np.float64)
        f_mag, f_dir = compute_friction_force_mag_dir(
            v, np.float64(0.0), np.float64(0.5), np.float64(1.0), np.float64(1e-12)
        )
        assert f_mag == 0.0
        np.testing.assert_allclose(f_dir, [0.0, 0.0])


class TestComputeSingleWheelFriction:
    """Shape and run tests for compute_single_wheel_friction."""

    def test_output_shapes(self) -> None:
        x_wheel = np.array([0.0, 5.0], dtype=np.float64)
        v = np.array([1.0, 0.0], dtype=np.float64)
        omega = np.float64(0.1)
        d1 = np.array([1.0, 0.0], dtype=np.float64)
        half_width = np.float64(0.1)
        f_mag, f_dir = compute_single_wheel_friction(
            x_wheel,
            v,
            omega,
            d1,
            half_width,
            np.float64(1.0),
            np.float64(-4.0 / 3.0),
            np.float64(4.0),
            np.float64(0.5),
            np.float64(1.0),
            np.float64(1e-12),
        )
        assert np.isscalar(f_mag) or getattr(f_mag, "shape", None) == ()
        assert f_dir.shape == (2,)

    def test_runs_outside_region_returns_zero(self) -> None:
        x_wheel = np.array([0.0, 0.0], dtype=np.float64)
        f_mag, f_dir = compute_single_wheel_friction(
            x_wheel,
            np.array([1.0, 0.0], dtype=np.float64),
            np.float64(0.1),
            np.array([1.0, 0.0], dtype=np.float64),
            np.float64(0.1),
            np.float64(1.0),
            np.float64(-4.0 / 3.0),
            np.float64(4.0),
            np.float64(0.5),
            np.float64(1.0),
            np.float64(1e-12),
        )
        assert f_mag == 0.0
        np.testing.assert_allclose(f_dir, [0.0, 0.0])


class TestWheelVelocity2d:
    """Shape and run tests for wheel_velocity_2d."""

    def test_output_shape(self) -> None:
        v = np.array([1.0, 0.0], dtype=np.float64)
        out = wheel_velocity_2d(
            v,
            np.float64(0.1),
            np.array([1.0, 0.0], dtype=np.float64),
            np.float64(0.1),
            np.float64(1.0),
        )
        assert out.shape == (2,)

    def test_runs_zero_omega(self) -> None:
        v = np.array([1.0, 0.0], dtype=np.float64)
        out = wheel_velocity_2d(
            v,
            np.float64(0.0),
            np.array([1.0, 0.0], dtype=np.float64),
            np.float64(0.1),
            np.float64(1.0),
        )
        np.testing.assert_allclose(out, [1.0, 0.0])


class TestTorqueZFromForce2d:
    """Shape and run tests for torque_z_from_force_2d."""

    def test_returns_scalar(self) -> None:
        lever = np.array([0.1, 0.0], dtype=np.float64)
        f_dir = np.array([-1.0, 0.0], dtype=np.float64)
        out = torque_z_from_force_2d(lever, f_dir, np.float64(1.0))
        assert np.isscalar(out) or getattr(out, "shape", None) == ()

    def test_runs(self) -> None:
        lever = np.array([1.0, 0.0], dtype=np.float64)
        f_dir = np.array([0.0, 1.0], dtype=np.float64)
        out = torque_z_from_force_2d(lever, f_dir, np.float64(1.0))
        assert out == 1.0  # cross([1,0], [0,1]) = 1


class TestBoundaryPenetrationForces:
    """Shape and run tests for boundary_penetration_forces."""

    def test_output_shape(self) -> None:
        center = np.array([0.5, 0.5], dtype=np.float64)
        out = boundary_penetration_forces(
            center,
            np.float64(0.1),
            np.float64(0.0),
            np.float64(1.0),
            np.float64(0.0),
            np.float64(1.0),
            np.float64(1000.0),
        )
        assert out.shape == (2,)

    def test_runs_inside_bounds(self) -> None:
        center = np.array([0.5, 0.5], dtype=np.float64)
        out = boundary_penetration_forces(
            center,
            np.float64(0.1),
            np.float64(0.0),
            np.float64(1.0),
            np.float64(0.0),
            np.float64(1.0),
            np.float64(1000.0),
        )
        np.testing.assert_allclose(out, [0.0, 0.0])
