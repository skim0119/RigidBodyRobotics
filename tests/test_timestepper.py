"""Unit tests for elastica_rigid.timestepper."""

import numpy as np

from elastica_rigid import ExplicitEulerForward, Roomba, SymplecticEulerForward


def _make_roomba() -> Roomba:
    """Roomba at origin, facing x, with known mass/inertia for predictable steps."""
    return Roomba(
        position=np.array([0.0, 0.0]),
        direction=np.array([1.0, 0.0]),
        mass=1.0,
        inertia=1.0,
        radius=0.1,
        width=0.2,
        initial_velocity=np.array([0.0, 0.0]),
        initial_omega=np.array([0.0]),
    )


class TestExplicitEulerForward:
    """Numerical checks for ExplicitEulerForward.step_single_instance."""

    def test_returns_time_plus_dt(self) -> None:
        """Step returns time + dt."""
        stepper = ExplicitEulerForward()
        system = _make_roomba()
        time = np.float64(0.0)
        dt = np.float64(0.01)

        t_next = stepper.step_single_instance(system, time, dt)

        assert t_next == time + dt

    def test_one_step_updates_state(self) -> None:
        """One step with constant force updates position and velocity."""
        stepper = ExplicitEulerForward()
        system = _make_roomba()
        system.external_forces[:] = [1.0, 0.0]
        system.external_torques[:] = [0.0]
        time = np.float64(0.0)
        dt = np.float64(0.1)

        stepper.step_single_instance(system, time, dt)

        # a = F/m = 1, v_new = v + dt*a = 0.1, x_new = x + dt*v = 0.01 (explicit Euler)
        np.testing.assert_allclose(system.velocity, [0.1, 0.0])
        np.testing.assert_allclose(system.position, [0.01, 0.0])


class TestSymplecticEulerForward:
    """Numerical checks for SymplecticEulerForward.step_single_instance."""

    def test_returns_time_plus_dt(self) -> None:
        """Step returns time + dt."""
        stepper = SymplecticEulerForward()
        system = _make_roomba()
        time = np.float64(0.0)
        dt = np.float64(0.01)

        t_next = stepper.step_single_instance(system, time, dt)

        assert t_next == time + dt

    def test_one_step_updates_state(self) -> None:
        """One step (dynamics then kinematics) updates state."""
        stepper = SymplecticEulerForward()
        system = _make_roomba()
        system.external_forces[:] = [1.0, 0.0]
        system.external_torques[:] = [0.0]
        time = np.float64(0.0)
        dt = np.float64(0.1)

        stepper.step_single_instance(system, time, dt)

        # Symplectic: v_new = v + dt*a = 0.1, then x_new = x + dt*v_new = 0.01
        np.testing.assert_allclose(system.velocity, [0.1, 0.0])
        np.testing.assert_allclose(system.position, [0.01, 0.0])
