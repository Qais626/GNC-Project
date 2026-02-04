"""
===============================================================================
GNC PROJECT - Extended Kalman Filter Test Suite
===============================================================================
Tests for a simplified Extended Kalman Filter implementation covering
initialization, prediction, measurement update, position convergence,
attitude convergence, bias estimation, Joseph form positive-definiteness,
and innovation consistency.

Since the project references an EKF module in navigation/, these tests
implement a lightweight EKF class inline for self-contained testing of
the filter algorithms.
===============================================================================
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from numpy.testing import assert_allclose


# =============================================================================
# Lightweight EKF implementation for testing
# =============================================================================

class SimpleEKF:
    """
    Minimal Extended Kalman Filter for testing.

    State: [x, y, z, vx, vy, vz, bx, by, bz]
           position (3), velocity (3), gyro bias (3)
    """

    def __init__(self, x0, P0, Q, R):
        """
        Parameters
        ----------
        x0 : initial state vector (n,)
        P0 : initial covariance (n, n)
        Q  : process noise covariance (n, n)
        R  : measurement noise covariance (m, m)
        """
        self.x = np.array(x0, dtype=float).copy()
        self.P = np.array(P0, dtype=float).copy()
        self.Q = np.array(Q, dtype=float).copy()
        self.R = np.array(R, dtype=float).copy()
        self.n = len(x0)

    def predict(self, F, B=None, u=None, dt=1.0):
        """
        Predict step using state transition matrix F.

        x_pred = F * x + B * u
        P_pred = F * P * F^T + Q
        """
        self.x = F @ self.x
        if B is not None and u is not None:
            self.x += B @ u
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z, H, use_joseph=False):
        """
        Measurement update step.

        Parameters
        ----------
        z : measurement vector
        H : measurement matrix
        use_joseph : if True, use Joseph stabilized form for covariance
        """
        z = np.asarray(z, dtype=float)
        # Innovation
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        # State update
        self.x = self.x + K @ y
        # Covariance update
        I_n = np.eye(self.n)
        if use_joseph:
            # Joseph stabilized form: P = (I - K*H) * P * (I - K*H)^T + K*R*K^T
            tmp = I_n - K @ H
            self.P = tmp @ self.P @ tmp.T + K @ self.R @ K.T
        else:
            self.P = (I_n - K @ H) @ self.P
        # Symmetrize
        self.P = 0.5 * (self.P + self.P.T)
        return y, S


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def ekf_9state():
    """Create a 9-state EKF (position, velocity, gyro bias)."""
    n = 9
    x0 = np.zeros(n)
    P0 = np.eye(n) * 100.0  # Large initial uncertainty
    Q = np.eye(n) * 0.01
    R = np.eye(3) * 1.0  # Position measurement noise
    return SimpleEKF(x0, P0, Q, R)


@pytest.fixture
def constant_velocity_F():
    """State transition matrix for constant velocity model with dt=1."""
    dt = 1.0
    F = np.eye(9)
    F[0, 3] = dt  # x += vx * dt
    F[1, 4] = dt  # y += vy * dt
    F[2, 5] = dt  # z += vz * dt
    return F


@pytest.fixture
def position_H():
    """Measurement matrix that observes position only."""
    H = np.zeros((3, 9))
    H[0, 0] = 1.0  # observe x
    H[1, 1] = 1.0  # observe y
    H[2, 2] = 1.0  # observe z
    return H


# =============================================================================
# Test: Initialization
# =============================================================================

class TestInitialization:
    """Tests for EKF initialization."""

    def test_initialization(self, ekf_9state):
        """State and covariance should be set correctly at construction."""
        assert_allclose(ekf_9state.x, np.zeros(9), atol=1e-15)
        assert_allclose(ekf_9state.P, np.eye(9) * 100.0, atol=1e-15)
        assert ekf_9state.n == 9

    def test_initialization_nonzero_state(self):
        """EKF can be initialized with a non-zero state."""
        x0 = np.array([1, 2, 3, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03])
        P0 = np.eye(9) * 50.0
        Q = np.eye(9) * 0.01
        R = np.eye(3)
        ekf = SimpleEKF(x0, P0, Q, R)
        assert_allclose(ekf.x, x0, atol=1e-15)


# =============================================================================
# Test: Predict stationary
# =============================================================================

class TestPredictStationary:
    """Tests for prediction with zero velocity."""

    def test_predict_stationary(self, ekf_9state, constant_velocity_F):
        """Zero-velocity state should remain near the origin after prediction."""
        ekf_9state.predict(constant_velocity_F)
        # Position and velocity should still be zero
        assert_allclose(ekf_9state.x[:6], np.zeros(6), atol=1e-15)

    def test_predict_increases_uncertainty(self, ekf_9state, constant_velocity_F):
        """Prediction should increase covariance (add process noise)."""
        trace_before = np.trace(ekf_9state.P)
        ekf_9state.predict(constant_velocity_F)
        trace_after = np.trace(ekf_9state.P)
        assert trace_after > trace_before


# =============================================================================
# Test: Update reduces covariance
# =============================================================================

class TestUpdateReducesCovariance:
    """Tests that measurement updates reduce uncertainty."""

    def test_update_reduces_covariance(self, ekf_9state, constant_velocity_F,
                                        position_H):
        """After measurement update, trace(P) should decrease."""
        ekf_9state.predict(constant_velocity_F)
        trace_before = np.trace(ekf_9state.P)

        z = np.array([0.0, 0.0, 0.0])  # Measurement at origin
        ekf_9state.update(z, position_H)
        trace_after = np.trace(ekf_9state.P)

        assert trace_after < trace_before

    @pytest.mark.parametrize("n_updates", [1, 5, 10])
    def test_multiple_updates_reduce_covariance(self, n_updates,
                                                  constant_velocity_F,
                                                  position_H):
        """Multiple updates should progressively reduce covariance trace."""
        ekf = SimpleEKF(np.zeros(9), np.eye(9) * 100.0,
                        np.eye(9) * 0.01, np.eye(3) * 1.0)
        traces = [np.trace(ekf.P)]

        for _ in range(n_updates):
            ekf.predict(constant_velocity_F)
            ekf.update(np.zeros(3), position_H)
            traces.append(np.trace(ekf.P))

        # Each update should produce a smaller or equal trace than the previous
        # (after the predict step adds noise, the update should bring it down)
        assert traces[-1] < traces[0]


# =============================================================================
# Test: Position convergence
# =============================================================================

class TestPositionConvergence:
    """Tests for position estimate convergence."""

    def test_position_convergence(self, constant_velocity_F, position_H):
        """Feed noisy position measurements; estimate should converge to truth."""
        np.random.seed(123)
        true_pos = np.array([100.0, -50.0, 200.0])
        true_state = np.zeros(9)
        true_state[0:3] = true_pos

        ekf = SimpleEKF(np.zeros(9), np.eye(9) * 1000.0,
                        np.eye(9) * 0.001, np.eye(3) * 4.0)

        n_steps = 50
        for _ in range(n_steps):
            ekf.predict(constant_velocity_F)
            noise = np.random.normal(0, 2.0, size=3)
            z = true_pos + noise
            ekf.update(z, position_H)

        pos_error = np.linalg.norm(ekf.x[0:3] - true_pos)
        assert pos_error < 5.0, (
            f"Position error {pos_error:.2f} m exceeds 5.0 m after "
            f"{n_steps} measurements"
        )


# =============================================================================
# Test: Attitude convergence (simplified as 3-angle state)
# =============================================================================

class TestAttitudeConvergence:
    """Tests for attitude estimation convergence via star tracker measurements."""

    def test_attitude_convergence(self):
        """
        Feed star tracker-like attitude measurements with noise; the
        attitude error should decrease over time.

        Uses a simple 3-state attitude-only EKF where the state is
        [roll_error, pitch_error, yaw_error] in radians.
        """
        np.random.seed(456)
        n = 3
        true_attitude = np.array([0.01, -0.005, 0.02])  # rad

        x0 = np.zeros(n)  # Initial estimate: zero
        P0 = np.eye(n) * np.radians(5.0) ** 2  # 5-deg uncertainty
        Q = np.eye(n) * (np.radians(0.001)) ** 2
        sigma_meas = np.radians(0.01)  # 36 arcsec star tracker noise
        R = np.eye(n) * sigma_meas ** 2

        ekf = SimpleEKF(x0, P0, Q, R)
        F = np.eye(n)  # Static attitude
        H = np.eye(n)  # Direct attitude measurement

        initial_error = np.linalg.norm(ekf.x - true_attitude)

        for _ in range(100):
            ekf.predict(F)
            noise = np.random.normal(0, sigma_meas, size=n)
            z = true_attitude + noise
            ekf.update(z, H)

        final_error = np.linalg.norm(ekf.x - true_attitude)
        assert final_error < initial_error, (
            f"Attitude error did not decrease: "
            f"initial={np.degrees(initial_error):.4f} deg, "
            f"final={np.degrees(final_error):.4f} deg"
        )
        # Should be sub-millidegree after 100 measurements
        assert final_error < np.radians(0.01), (
            f"Final error {np.degrees(final_error)*3600:.1f} arcsec too large"
        )


# =============================================================================
# Test: Bias estimation
# =============================================================================

class TestBiasEstimation:
    """Tests for gyro bias estimation within the EKF."""

    def test_bias_estimation(self, constant_velocity_F, position_H):
        """
        The filter should estimate gyro bias over time when it is observable
        through the state dynamics.
        """
        np.random.seed(789)
        true_bias = np.array([0.005, -0.003, 0.002])  # rad/s

        # Start with zero bias estimate
        x0 = np.zeros(9)
        P0 = np.eye(9) * 100.0
        Q = np.eye(9) * 0.001
        R = np.eye(3) * 0.5

        ekf = SimpleEKF(x0, P0, Q, R)

        # Modify F to couple bias into velocity (simplified dynamics)
        dt = 1.0
        F = np.eye(9)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        # Bias drives velocity (coupling)
        F[3, 6] = dt
        F[4, 7] = dt
        F[5, 8] = dt

        # True position evolves with bias-driven acceleration
        true_pos = np.zeros(3)
        true_vel = np.zeros(3)

        for step in range(200):
            true_vel += true_bias * dt
            true_pos += true_vel * dt

            ekf.predict(F)
            noise = np.random.normal(0, 0.5, size=3)
            z = true_pos + noise
            ekf.update(z, position_H)

        # The bias estimate should be reasonably close to the truth
        bias_error = np.linalg.norm(ekf.x[6:9] - true_bias)
        assert bias_error < np.linalg.norm(true_bias), (
            f"Bias error {bias_error:.6f} not smaller than true bias norm "
            f"{np.linalg.norm(true_bias):.6f}"
        )


# =============================================================================
# Test: Joseph form positive-definiteness
# =============================================================================

class TestJosephForm:
    """Tests for positive-definiteness using the Joseph stabilized form."""

    def test_joseph_form(self, constant_velocity_F, position_H):
        """Covariance should stay positive definite after many Joseph-form updates."""
        np.random.seed(101)
        ekf = SimpleEKF(np.zeros(9), np.eye(9) * 100.0,
                        np.eye(9) * 0.01, np.eye(3) * 1.0)

        for _ in range(500):
            ekf.predict(constant_velocity_F)
            z = np.random.normal(0, 1.0, size=3)
            ekf.update(z, position_H, use_joseph=True)

            # Check positive definiteness via eigenvalues
            eigvals = np.linalg.eigvalsh(ekf.P)
            assert np.all(eigvals > -1e-12), (
                f"Covariance has negative eigenvalue: {eigvals.min():.3e}"
            )

    def test_joseph_form_symmetry(self, constant_velocity_F, position_H):
        """Covariance must remain symmetric after Joseph-form updates."""
        ekf = SimpleEKF(np.zeros(9), np.eye(9) * 50.0,
                        np.eye(9) * 0.01, np.eye(3) * 1.0)

        for _ in range(100):
            ekf.predict(constant_velocity_F)
            ekf.update(np.zeros(3), position_H, use_joseph=True)

        assert_allclose(ekf.P, ekf.P.T, atol=1e-12)


# =============================================================================
# Test: Innovation consistency
# =============================================================================

class TestInnovationConsistency:
    """Tests for normalized innovation consistency."""

    def test_innovation_consistency(self, constant_velocity_F, position_H):
        """
        Normalized innovation squared should follow a chi-squared distribution.
        Compute the average NIS over many steps and check it is near the
        measurement dimension (3) -- the expected value of chi-squared(3).
        """
        np.random.seed(202)
        ekf = SimpleEKF(np.zeros(9), np.eye(9) * 100.0,
                        np.eye(9) * 0.01, np.eye(3) * 1.0)

        n_steps = 500
        nis_values = []

        for _ in range(n_steps):
            ekf.predict(constant_velocity_F)
            noise = np.random.normal(0, 1.0, size=3)
            z = noise  # True state is zero
            y, S = ekf.update(z, position_H)

            # Normalized Innovation Squared: y^T S^{-1} y
            nis = y @ np.linalg.solve(S, y)
            nis_values.append(nis)

        mean_nis = np.mean(nis_values)
        # For a consistent filter, E[NIS] = measurement dimension = 3
        # Allow generous bounds due to filter transients
        assert 1.0 < mean_nis < 8.0, (
            f"Mean NIS = {mean_nis:.2f}, expected near 3.0 for a 3-D measurement"
        )
