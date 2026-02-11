"""
===============================================================================
Unscented Kalman Filter (UKF) for Spacecraft Navigation
===============================================================================
Same 15-state formulation as EKF but uses sigma point propagation instead
of Jacobian linearization. Better for highly nonlinear dynamics (e.g., near
Jupiter with strong J2, or during powered maneuvers).

State vector (15 states):
    x = [position(3), velocity(3), attitude_error(3), gyro_bias(3), accel_bias(3)]

Key advantages over EKF:
    - Captures mean and covariance to 2nd order (vs 1st order for EKF)
    - No Jacobian computation needed (avoids analytical derivatives)
    - Better for highly nonlinear systems
    - More computationally expensive (2n+1 propagations per step)

Sigma Point Strategy:
    Generate 2n+1 = 31 sigma points from state mean and covariance.
    Propagate each through the nonlinear dynamics.
    Reconstruct mean and covariance from propagated sigma points.

References:
    - Julier & Uhlmann, "A New Extension of the Kalman Filter to
      Nonlinear Systems," 1997
    - Van der Merwe, "Sigma-Point Kalman Filters for Probabilistic
      Inference in Dynamic State-Space Models," 2004
===============================================================================
"""

import numpy as np
from typing import Callable, Optional, Tuple


class UKF:
    """
    Unscented Kalman Filter for spacecraft state estimation.

    Uses the scaled unscented transform with tuning parameters
    (alpha, beta, kappa) to select sigma points that capture the
    mean and covariance of the state distribution.

    Args:
        x0: Initial state estimate (15-vector)
        P0: Initial covariance matrix (15x15)
        Q: Process noise covariance (15x15)
        R: Default measurement noise covariance
        alpha: Spread of sigma points (typically 1e-4 to 1)
        beta: Prior knowledge parameter (2 is optimal for Gaussian)
        kappa: Secondary scaling parameter (typically 0 or 3-n)
    """

    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray,
                 R: np.ndarray, alpha: float = 1e-3, beta: float = 2.0,
                 kappa: float = 0.0):
        self.n = len(x0)  # State dimension (15)
        self.x = x0.copy().astype(float)
        self.P = P0.copy().astype(float)
        self.Q = Q.copy().astype(float)
        self.R_default = R.copy().astype(float)

        # --- Tuning parameters ---
        # alpha: determines spread of sigma points around mean
        #   Small alpha (1e-3): sigma points close to mean, good for
        #   mildly nonlinear systems
        #   Large alpha (1.0): sigma points spread wider
        self.alpha = alpha

        # beta: incorporates prior knowledge of distribution
        #   beta = 2 is optimal for Gaussian distributions
        self.beta = beta

        # kappa: secondary scaling parameter
        #   kappa = 3 - n is a common choice
        #   kappa = 0 is another common choice
        self.kappa = kappa

        # --- Derived parameters ---
        # lambda: composite scaling parameter
        self.lambda_ = alpha ** 2 * (self.n + kappa) - self.n

        # Total number of sigma points: 2n + 1
        self.n_sigma = 2 * self.n + 1

        # --- Compute weights ---
        # Weight for mean (Wm) and covariance (Wc)
        self.Wm = np.zeros(self.n_sigma)
        self.Wc = np.zeros(self.n_sigma)

        # Central sigma point weights
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + \
                      (1.0 - alpha ** 2 + beta)

        # Remaining sigma point weights (all equal)
        w = 1.0 / (2.0 * (self.n + self.lambda_))
        self.Wm[1:] = w
        self.Wc[1:] = w

        # Reference quaternion for attitude tracking
        self.q_ref = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]

        # Step counter
        self.step_count = 0

    def compute_sigma_points(self, x: np.ndarray,
                              P: np.ndarray) -> np.ndarray:
        """
        Generate 2n+1 sigma points from state mean and covariance.

        The sigma points are selected deterministically to capture
        the mean and covariance of the state distribution:
            X_0 = x                         (central point)
            X_i = x + sqrt((n+lambda)*P)_i  (i = 1..n)
            X_{n+i} = x - sqrt((n+lambda)*P)_i  (i = 1..n)

        where sqrt() denotes the matrix square root (Cholesky decomposition).

        Args:
            x: State mean (n-vector)
            P: State covariance (nxn matrix)

        Returns:
            Sigma points array of shape (2n+1, n)
        """
        n = len(x)
        sigma_points = np.zeros((2 * n + 1, n))

        # Central sigma point = mean
        sigma_points[0] = x

        # Matrix square root using Cholesky decomposition
        # S = cholesky((n + lambda) * P), so S*S^T = (n + lambda) * P
        try:
            # Ensure P is positive definite by adding small diagonal
            P_scaled = (n + self.lambda_) * P
            # Add small regularization for numerical stability
            P_scaled += np.eye(n) * 1e-10
            S = np.linalg.cholesky(P_scaled)
        except np.linalg.LinAlgError:
            # Fallback: use eigendecomposition if Cholesky fails
            eigvals, eigvecs = np.linalg.eigh(P_scaled)
            eigvals = np.maximum(eigvals, 1e-10)
            S = eigvecs @ np.diag(np.sqrt(eigvals))

        # Generate sigma points symmetrically around mean
        for i in range(n):
            sigma_points[i + 1] = x + S[:, i]
            sigma_points[n + i + 1] = x - S[:, i]

        return sigma_points

    def predict(self, process_model: Callable, dt: float):
        """
        UKF prediction step (time update).

        1. Generate sigma points from current state and covariance
        2. Propagate each sigma point through nonlinear dynamics
        3. Compute predicted mean and covariance from propagated points

        The key advantage: no Jacobian needed! The nonlinear function
        is evaluated directly at the sigma points.

        Args:
            process_model: Function f(state, dt) -> propagated_state
                          Must accept a 15-vector and time step,
                          return a 15-vector
            dt: Time step in seconds
        """
        # Step 1: Generate sigma points
        sigma_pts = self.compute_sigma_points(self.x, self.P)

        # Step 2: Propagate sigma points through nonlinear dynamics
        # Each sigma point is transformed: X_pred[i] = f(X[i], dt)
        sigma_pred = np.zeros_like(sigma_pts)
        for i in range(self.n_sigma):
            sigma_pred[i] = process_model(sigma_pts[i], dt)

        # Step 3: Compute predicted mean
        # x_pred = sum_i Wm[i] * X_pred[i]
        self.x = np.zeros(self.n)
        for i in range(self.n_sigma):
            self.x += self.Wm[i] * sigma_pred[i]

        # Step 4: Compute predicted covariance
        # P_pred = sum_i Wc[i] * (X_pred[i] - x_pred)(X_pred[i] - x_pred)^T + Q
        self.P = np.zeros((self.n, self.n))
        for i in range(self.n_sigma):
            diff = sigma_pred[i] - self.x
            self.P += self.Wc[i] * np.outer(diff, diff)
        self.P += self.Q

        # Ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)

        self.step_count += 1

    def update(self, measurement: np.ndarray,
               measurement_model: Callable,
               R: Optional[np.ndarray] = None):
        """
        UKF measurement update step.

        1. Generate sigma points from predicted state
        2. Transform sigma points through measurement model
        3. Compute innovation, cross-covariance, and Kalman gain
        4. Update state and covariance

        Args:
            measurement: Actual measurement vector (m-vector)
            measurement_model: Function h(state) -> predicted_measurement
                              Maps state vector to measurement space
            R: Measurement noise covariance (mxm). Uses default if None.
        """
        if R is None:
            R = self.R_default

        m = len(measurement)  # Measurement dimension

        # Step 1: Generate sigma points from current (predicted) state
        sigma_pts = self.compute_sigma_points(self.x, self.P)

        # Step 2: Transform sigma points through measurement model
        # Z[i] = h(X[i])
        Z = np.zeros((self.n_sigma, m))
        for i in range(self.n_sigma):
            Z[i] = measurement_model(sigma_pts[i])

        # Step 3a: Predicted measurement mean
        # z_pred = sum_i Wm[i] * Z[i]
        z_pred = np.zeros(m)
        for i in range(self.n_sigma):
            z_pred += self.Wm[i] * Z[i]

        # Step 3b: Innovation covariance
        # Pzz = sum_i Wc[i] * (Z[i] - z_pred)(Z[i] - z_pred)^T + R
        Pzz = np.zeros((m, m))
        for i in range(self.n_sigma):
            dz = Z[i] - z_pred
            Pzz += self.Wc[i] * np.outer(dz, dz)
        Pzz += R

        # Step 3c: Cross-covariance between state and measurement
        # Pxz = sum_i Wc[i] * (X[i] - x_pred)(Z[i] - z_pred)^T
        Pxz = np.zeros((self.n, m))
        for i in range(self.n_sigma):
            dx = sigma_pts[i] - self.x
            dz = Z[i] - z_pred
            Pxz += self.Wc[i] * np.outer(dx, dz)

        # Step 4a: Kalman gain
        # K = Pxz * Pzz^{-1}
        K = Pxz @ np.linalg.inv(Pzz)

        # Step 4b: State update
        # x = x + K * (z - z_pred)
        innovation = measurement - z_pred
        self.x = self.x + K @ innovation

        # Step 4c: Covariance update
        # P = P - K * Pzz * K^T
        self.P = self.P - K @ Pzz @ K.T

        # Ensure symmetry and positive-definiteness
        self.P = 0.5 * (self.P + self.P.T)
        # Small regularization
        min_eig = np.min(np.linalg.eigvalsh(self.P))
        if min_eig < 1e-10:
            self.P += np.eye(self.n) * (1e-10 - min_eig)

    def update_position(self, z_pos: np.ndarray,
                        R_pos: np.ndarray):
        """
        Convenience: measurement update using position observation.

        Measurement model: h(x) = x[0:3] (position components)

        Args:
            z_pos: Measured position (3-vector, meters)
            R_pos: Position measurement noise covariance (3x3)
        """
        def h_pos(state):
            return state[0:3]
        self.update(z_pos, h_pos, R_pos)

    def update_velocity(self, z_vel: np.ndarray,
                        R_vel: np.ndarray):
        """
        Convenience: measurement update using velocity observation.

        Measurement model: h(x) = x[3:6] (velocity components)

        Args:
            z_vel: Measured velocity (3-vector, m/s)
            R_vel: Velocity measurement noise covariance (3x3)
        """
        def h_vel(state):
            return state[3:6]
        self.update(z_vel, h_vel, R_vel)

    def update_attitude(self, z_att_error: np.ndarray,
                        R_att: np.ndarray):
        """
        Convenience: measurement update using attitude observation.

        The attitude error is expressed as a 3-component rotation
        vector (small angle approximation).

        Args:
            z_att_error: Measured attitude error (3-vector, radians)
            R_att: Attitude measurement noise covariance (3x3)
        """
        def h_att(state):
            return state[6:9]
        self.update(z_att_error, h_att, R_att)

    def get_state(self) -> np.ndarray:
        """Return current state estimate (15-vector copy)."""
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        """Return current covariance matrix (15x15 copy)."""
        return self.P.copy()

    def get_position(self) -> np.ndarray:
        """Return estimated position (3-vector, meters)."""
        return self.x[0:3].copy()

    def get_velocity(self) -> np.ndarray:
        """Return estimated velocity (3-vector, m/s)."""
        return self.x[3:6].copy()

    def get_attitude_error(self) -> np.ndarray:
        """Return estimated attitude error (3-vector, radians)."""
        return self.x[6:9].copy()

    def get_gyro_bias(self) -> np.ndarray:
        """Return estimated gyroscope bias (3-vector, rad/s)."""
        return self.x[9:12].copy()

    def get_accel_bias(self) -> np.ndarray:
        """Return estimated accelerometer bias (3-vector, m/s^2)."""
        return self.x[12:15].copy()

    def get_attitude_quaternion(self) -> np.ndarray:
        """
        Return reference attitude quaternion [w, x, y, z].

        The small attitude error from the filter state is folded
        into the reference quaternion.
        """
        # Fold attitude error into reference quaternion
        err = self.x[6:9]
        err_mag = np.linalg.norm(err)

        if err_mag > 1e-10:
            # Convert error vector to quaternion: dq = [cos(|e|/2), sin(|e|/2)*e/|e|]
            half_angle = err_mag / 2.0
            dq = np.array([
                np.cos(half_angle),
                np.sin(half_angle) * err[0] / err_mag,
                np.sin(half_angle) * err[1] / err_mag,
                np.sin(half_angle) * err[2] / err_mag
            ])
        else:
            # Small angle: dq ~ [1, e/2]
            dq = np.array([1.0, err[0] / 2.0, err[1] / 2.0, err[2] / 2.0])
            dq /= np.linalg.norm(dq)

        # Quaternion multiply: q_new = dq * q_ref
        q = self._quat_multiply(dq, self.q_ref)
        return q / np.linalg.norm(q)

    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Hamilton quaternion product q1*q2 with convention [w,x,y,z]."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])

    def get_sigma_point_spread(self) -> float:
        """
        Diagnostic: compute average distance of sigma points from mean.
        Useful for checking if sigma points are well-distributed.
        """
        sigma_pts = self.compute_sigma_points(self.x, self.P)
        distances = [np.linalg.norm(sigma_pts[i] - self.x)
                     for i in range(1, self.n_sigma)]
        return np.mean(distances)
