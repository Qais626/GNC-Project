"""
===============================================================================
GNC PROJECT - Attitude Control Test Suite
===============================================================================
Tests for attitude controllers: PID zero-error and step response, LQR gain
properties, sliding-mode surface convergence, detumble mode, pointing accuracy,
reaction wheel saturation, and thruster minimum impulse bit.

Uses project actuator models from control.actuators and attitude dynamics
from dynamics.attitude_dynamics.
===============================================================================
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from numpy.testing import assert_allclose

from control.actuators import ReactionWheel, ReactionWheelArray, Thruster
from core.quaternion import Quaternion


# =============================================================================
# Inline PID Controller
# =============================================================================

class PIDController:
    """Simple 3-axis PID attitude controller for testing."""

    def __init__(self, Kp, Ki, Kd, dt=0.01):
        self.Kp = np.asarray(Kp, dtype=float)
        self.Ki = np.asarray(Ki, dtype=float)
        self.Kd = np.asarray(Kd, dtype=float)
        self.dt = dt
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)

    def compute(self, error, error_rate=None):
        """
        Compute PID control torque.

        Parameters
        ----------
        error : (3,) attitude error vector (rad)
        error_rate : (3,) angular velocity error (rad/s), optional

        Returns
        -------
        torque : (3,) control torque (Nm)
        """
        error = np.asarray(error, dtype=float)
        self.integral += error * self.dt

        if error_rate is not None:
            derivative = np.asarray(error_rate, dtype=float)
        else:
            derivative = (error - self.prev_error) / self.dt

        self.prev_error = error.copy()

        torque = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return torque


# =============================================================================
# Inline LQR gain computation
# =============================================================================

def compute_lqr_gain(A, B, Q_lqr, R_lqr, n_iter=200):
    """
    Compute LQR gain via iterative discrete-time algebraic Riccati equation.

    Uses value iteration: P_{k+1} = Q + A^T P_k A - A^T P_k B (R + B^T P_k B)^{-1} B^T P_k A

    Returns K such that u = -K x.
    """
    P = Q_lqr.copy()
    for _ in range(n_iter):
        S = R_lqr + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        P_new = Q_lqr + A.T @ P @ A - A.T @ P @ B @ K
        P_new = 0.5 * (P_new + P_new.T)
        if np.linalg.norm(P_new - P) < 1e-12:
            break
        P = P_new
    K = np.linalg.solve(R_lqr + B.T @ P @ B, B.T @ P @ A)
    return K, P


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def pid():
    """Return a PID controller with typical spacecraft gains."""
    return PIDController(
        Kp=[10.0, 10.0, 10.0],
        Ki=[0.1, 0.1, 0.1],
        Kd=[5.0, 5.0, 5.0],
        dt=0.01,
    )


@pytest.fixture
def reaction_wheel():
    """Return a single reaction wheel with default parameters."""
    return ReactionWheel(
        max_torque=0.2,
        max_momentum=50.0,
        wheel_inertia=0.05,
    )


@pytest.fixture
def rw_array():
    """Return a 4-wheel pyramid array."""
    return ReactionWheelArray(max_torque=0.2, max_momentum=50.0)


@pytest.fixture
def thruster():
    """Return a single thruster with known parameters."""
    return Thruster(
        nominal_thrust=22.0,
        isp=230.0,
        min_impulse_bit=0.020,
    )


# =============================================================================
# Test: PID zero error
# =============================================================================

class TestPIDZeroError:
    """Tests for PID controller with zero error input."""

    def test_pid_zero_error(self, pid):
        """Zero error input should produce zero output torque."""
        error = np.zeros(3)
        error_rate = np.zeros(3)
        torque = pid.compute(error, error_rate)
        assert_allclose(torque, np.zeros(3), atol=1e-15)

    def test_pid_zero_error_multiple_steps(self, pid):
        """Zero error over multiple steps should always produce zero output."""
        for _ in range(10):
            torque = pid.compute(np.zeros(3), np.zeros(3))
            assert_allclose(torque, np.zeros(3), atol=1e-15)


# =============================================================================
# Test: PID step response
# =============================================================================

class TestPIDStepResponse:
    """Tests for PID step response behavior."""

    def test_pid_step_response(self, pid):
        """Step error should produce proportional + derivative response."""
        error = np.array([0.1, 0.0, 0.0])
        error_rate = np.array([0.0, 0.0, 0.0])

        torque = pid.compute(error, error_rate)

        # First step: torque should be dominated by proportional term
        # T = Kp * error + Ki * integral * dt + Kd * error_rate
        # Integral on first step is error * dt = 0.1 * 0.01 = 0.001
        expected_p = 10.0 * 0.1  # 1.0
        expected_i = 0.1 * 0.001  # 0.0001
        expected_d = 5.0 * 0.0  # 0.0
        expected = expected_p + expected_i + expected_d

        assert_allclose(torque[0], expected, rtol=1e-10)
        assert torque[0] > 0, "Torque should be positive for positive error"

    def test_pid_derivative_response(self, pid):
        """Non-zero error rate should contribute to the output."""
        error = np.array([0.0, 0.0, 0.0])
        error_rate = np.array([0.5, 0.0, 0.0])

        torque = pid.compute(error, error_rate)

        # Should have derivative contribution only
        assert torque[0] == pytest.approx(5.0 * 0.5, rel=1e-10)


# =============================================================================
# Test: LQR gain properties
# =============================================================================

class TestLQRGain:
    """Tests for LQR gain matrix properties."""

    def test_lqr_gain_symmetric(self):
        """The Riccati solution P should be symmetric positive definite."""
        # Simple double-integrator model (attitude: angle + rate)
        dt = 0.01
        A = np.array([[1, dt], [0, 1]])
        B = np.array([[0], [dt]])
        Q_lqr = np.diag([100.0, 1.0])
        R_lqr = np.array([[1.0]])

        K, P = compute_lqr_gain(A, B, Q_lqr, R_lqr)

        # P should be symmetric
        assert_allclose(P, P.T, atol=1e-10)

        # P should be positive definite
        eigvals = np.linalg.eigvalsh(P)
        assert np.all(eigvals > 0), f"P has non-positive eigenvalues: {eigvals}"

        # K should have appropriate dimensions
        assert K.shape == (1, 2)

    def test_lqr_stabilizes_system(self):
        """Closed-loop system A - B*K should be stable (eigenvalues inside unit circle)."""
        dt = 0.01
        A = np.array([[1, dt], [0, 1]])
        B = np.array([[0], [dt]])
        Q_lqr = np.diag([100.0, 1.0])
        R_lqr = np.array([[1.0]])

        K, _ = compute_lqr_gain(A, B, Q_lqr, R_lqr)
        A_cl = A - B @ K

        eigvals = np.linalg.eigvals(A_cl)
        assert np.all(np.abs(eigvals) < 1.0), (
            f"Closed-loop eigenvalues not stable: {eigvals}"
        )


# =============================================================================
# Test: Sliding mode convergence
# =============================================================================

class TestSlidingMode:
    """Tests for sliding mode controller reaching the sliding surface."""

    def test_sliding_mode_reaches_surface(self):
        """The sliding variable should converge to zero."""
        # Simplified 1-DOF attitude dynamics: theta_ddot = u/I
        I_zz = 4800.0
        dt = 0.01
        theta = np.radians(5.0)  # initial pointing error
        theta_dot = np.radians(0.5)  # initial angular rate

        # Sliding surface: s = theta_dot + lambda * theta
        lam = 2.0  # sliding surface slope
        eta = 5.0  # reaching gain

        s_history = []

        for _ in range(2000):
            s = theta_dot + lam * theta
            s_history.append(abs(s))

            # Sliding mode control law: u = -I * (lambda * theta_dot + eta * sign(s))
            u = -I_zz * (lam * theta_dot + eta * np.sign(s))

            # Saturation
            u = np.clip(u, -100.0, 100.0)

            # Integrate dynamics
            theta_ddot = u / I_zz
            theta_dot += theta_ddot * dt
            theta += theta_dot * dt

        # Sliding variable should be near zero at the end
        assert abs(s_history[-1]) < 0.01, (
            f"|s| = {s_history[-1]:.4f} did not converge to zero"
        )
        # It should have decreased from the initial value
        assert s_history[-1] < s_history[0]


# =============================================================================
# Test: Detumble mode
# =============================================================================

class TestDetumble:
    """Tests for detumble mode reducing high angular rates."""

    def test_attitude_system_detumble(self):
        """High angular rate should be reduced by a simple detumble (B-dot) law."""
        I = np.diag([3500.0, 4200.0, 4800.0])
        I_inv = np.linalg.inv(I)
        omega = np.array([0.1, -0.15, 0.08])  # High rate for a spacecraft (rad/s)
        dt = 0.1
        K_detumble = 5000.0  # Detumble gain

        initial_rate = np.linalg.norm(omega)

        for _ in range(5000):
            # Simple proportional detumble: torque = -K * omega
            torque = -K_detumble * omega
            # Euler's equation (simplified, no gyroscopic for small omega)
            omega_dot = I_inv @ (torque - np.cross(omega, I @ omega))
            omega += omega_dot * dt

        final_rate = np.linalg.norm(omega)
        assert final_rate < initial_rate * 0.01, (
            f"Rate not reduced enough: {final_rate:.6f} vs initial {initial_rate:.6f}"
        )


# =============================================================================
# Test: Pointing accuracy
# =============================================================================

class TestPointingAccuracy:
    """Tests for steady-state pointing accuracy."""

    def test_pointing_accuracy(self):
        """Controller should achieve < 0.1 deg pointing accuracy in steady state."""
        I = np.diag([3500.0, 4200.0, 4800.0])
        I_inv = np.linalg.inv(I)
        dt = 0.01

        q_current = Quaternion.from_euler(np.radians(1.0), np.radians(0.5),
                                           np.radians(-0.3))
        q_desired = Quaternion.identity()
        omega = np.zeros(3)

        Kp = np.array([200.0, 250.0, 300.0])
        Kd = np.array([400.0, 500.0, 600.0])

        for _ in range(10000):
            # Error quaternion
            q_err = q_current.error_quaternion(q_desired)
            err_vec = q_err.vector * 2.0  # small-angle approximation

            # PD control
            torque = -Kp * err_vec - Kd * omega

            # Dynamics
            omega_dot = I_inv @ (torque - np.cross(omega, I @ omega))
            omega += omega_dot * dt
            q_current = q_current.propagate(omega, dt)

        # Check pointing error
        error_angle_rad = q_current.angle_to(q_desired)
        error_angle_deg = np.degrees(error_angle_rad)
        assert error_angle_deg < 0.1, (
            f"Pointing error {error_angle_deg:.4f} deg exceeds 0.1 deg"
        )


# =============================================================================
# Test: Reaction wheel saturation
# =============================================================================

class TestReactionWheelSaturation:
    """Tests for reaction wheel momentum saturation."""

    def test_reaction_wheel_saturation(self, reaction_wheel):
        """Wheel should stop producing useful torque at max momentum."""
        rw = reaction_wheel
        # Spin the wheel to near saturation
        rw.current_speed = rw.max_momentum / rw.wheel_inertia * 0.99
        rw.accumulated_momentum = rw.wheel_inertia * rw.current_speed

        # Try to command more torque in the same direction
        actual = rw.command_torque(0.2)
        rw.update_state(actual, 1.0)

        # The wheel should be at or very near saturation
        assert rw.is_saturated() or abs(rw.get_momentum()) >= rw.max_momentum * 0.99

    def test_saturated_wheel_limits_torque(self, reaction_wheel):
        """Once saturated, commanded torque should be clamped."""
        rw = reaction_wheel
        rw.current_speed = rw.max_momentum / rw.wheel_inertia
        rw.accumulated_momentum = rw.max_momentum

        # Command a large positive torque (same direction as spin)
        actual = rw.command_torque(0.2)
        # The actual torque should be zero or very small in the same direction
        # because the wheel cannot accelerate further
        assert actual <= 0.01


# =============================================================================
# Test: Thruster minimum impulse bit
# =============================================================================

class TestThrusterMinimumImpulse:
    """Tests for thruster minimum impulse bit enforcement."""

    def test_thruster_minimum_impulse(self, thruster):
        """Thruster should enforce minimum impulse bit for short firings."""
        # Command a very short firing (1 ms, below minimum impulse bit of 20 ms)
        np.random.seed(42)
        impulse = thruster.fire(0.001)

        # The actual firing should be at least min_impulse_bit
        # Propellant consumed should correspond to min_impulse_bit, not 0.001 s
        prop = thruster.get_propellant_consumed(0.001)
        prop_min = thruster.mass_flow_rate * thruster.min_impulse_bit
        assert_allclose(prop, prop_min, rtol=1e-10)

    def test_thruster_long_firing_not_clamped(self, thruster):
        """A firing longer than min_impulse_bit should not be clamped."""
        np.random.seed(42)
        duration = 0.1  # 100 ms, well above min_impulse_bit
        prop = thruster.get_propellant_consumed(duration)
        expected = thruster.mass_flow_rate * duration
        assert_allclose(prop, expected, rtol=1e-10)

    @pytest.mark.parametrize("duration", [0.001, 0.005, 0.010, 0.019])
    def test_short_firings_all_clamp_to_min(self, thruster, duration):
        """Any duration shorter than min_impulse_bit clamps to min_impulse_bit."""
        prop = thruster.get_propellant_consumed(duration)
        prop_min = thruster.mass_flow_rate * thruster.min_impulse_bit
        assert_allclose(prop, prop_min, rtol=1e-10)
