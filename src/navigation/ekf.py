"""
===============================================================================
GNC PROJECT - Extended Kalman Filter (EKF) for Spacecraft Navigation
===============================================================================

Implements a 15-state Extended Kalman Filter with a multiplicative quaternion
error formulation for spacecraft attitude estimation. This is the standard
approach used by JPL, NASA GSFC, and ESA for spacecraft navigation.

State Vector (15 elements)
--------------------------
    x[0:3]   = position [x, y, z]        (meters, inertial frame)
    x[3:6]   = velocity [vx, vy, vz]     (m/s, inertial frame)
    x[6:9]   = attitude error [ex, ey, ez] (radians, small-angle approximation)
    x[9:12]  = gyroscope bias [bx, by, bz] (rad/s)
    x[12:15] = accelerometer bias [ax, ay, az] (m/s^2)

Multiplicative Quaternion Error Formulation
-------------------------------------------
The attitude is NOT stored as a 4-element quaternion in the state vector.
Instead, we maintain a separate reference quaternion q_ref and estimate
only a 3-element small-angle error vector delta_theta in the state vector.
This has two critical advantages:

1. **Avoids the quaternion normalization constraint**: A 4-element quaternion
   in a Kalman filter would violate the unit-norm constraint because the
   Gaussian distribution assumed by the KF does not respect this nonlinear
   constraint. The 3-element error parameterization has no such constraint.

2. **Minimal representation**: 3 parameters for 3 DOF of rotation, avoiding
   the overparameterization of the 4-element quaternion.

The relationship between the true quaternion q_true, the reference quaternion
q_ref, and the error quaternion delta_q is:

    q_true = delta_q (*) q_ref

where delta_q is constructed from the small-angle error:

    delta_q ~= [1, 0.5 * delta_theta_x, 0.5 * delta_theta_y, 0.5 * delta_theta_z]

This approximation is valid when delta_theta is small (< ~15 degrees), which
is maintained by the "reset" step: after every update, the error is folded
back into q_ref and reset to zero.

Joseph Form Covariance Update
-----------------------------
The standard covariance update P = (I - K*H) * P is numerically unstable
because floating-point errors can make P non-positive-definite. The Joseph
(stabilized) form guarantees symmetry and positive-definiteness:

    P = (I - K*H) * P * (I - K*H)^T + K * R * K^T

This is more expensive (two matrix multiplies instead of one) but essential
for long-duration missions where the filter runs for months.

References
----------
    [1] Markley & Crassidis, "Fundamentals of Spacecraft Attitude
        Determination and Control", Springer, 2014, Ch. 6.
    [2] Lefferts, Markley, & Shuster, "Kalman Filtering for Spacecraft
        Attitude Estimation", JGCD, 1982.
    [3] Crassidis, Markley, & Cheng, "Survey of Nonlinear Attitude
        Estimation Methods", JGCD, 2007.
    [4] Brown & Hwang, "Introduction to Random Signals and Applied
        Kalman Filtering", 4th ed., Wiley, 2012.

===============================================================================
"""

import numpy as np
from core.quaternion import Quaternion


def _skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Construct the 3x3 skew-symmetric (cross-product) matrix from a 3-vector.

    For a vector v = [vx, vy, vz], the skew-symmetric matrix [v x] is:

        [v x] = |  0   -vz   vy |
                |  vz   0   -vx |
                | -vy   vx   0  |

    such that [v x] * u = v x u (cross product) for any vector u.

    This matrix appears throughout the EKF dynamics:
    - In the attitude error propagation: d(delta_theta)/dt = -[omega x] * delta_theta
    - In the gravity gradient terms
    - In converting between angular velocity and attitude rates

    Parameters
    ----------
    v : np.ndarray
        3-element vector.

    Returns
    -------
    np.ndarray
        3x3 skew-symmetric matrix.
    """
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0]
    ], dtype=np.float64)


class EKF:
    """
    Extended Kalman Filter with multiplicative quaternion error formulation.

    This filter estimates spacecraft position, velocity, attitude, and sensor
    biases from noisy measurements. It is the workhorse of spacecraft
    navigation, used on missions from Apollo to Mars 2020 Perseverance.

    The EKF linearizes the nonlinear dynamics and measurement models about
    the current state estimate, computing Jacobians (F and H matrices) at
    each time step. This first-order approximation is accurate when:
    - The state uncertainty is small relative to the nonlinearity
    - The time step is small enough to capture the dynamics

    For highly nonlinear scenarios (e.g., close planetary encounters), the
    UKF may provide better performance without requiring Jacobian derivation.

    Attributes
    ----------
    x : np.ndarray
        15-element state vector.
    P : np.ndarray
        15x15 state error covariance matrix.
    Q : np.ndarray
        15x15 process noise covariance matrix.
    q_ref : Quaternion
        Reference quaternion representing the current best estimate of
        the spacecraft attitude. Updated during attitude measurement
        updates and during the predict step.
    dt : float
        Default time step for predict (can be overridden per call).

    Examples
    --------
    >>> # Initialize EKF at origin with identity attitude
    >>> x0 = np.zeros(15)
    >>> P0 = np.diag([100.0]*3 + [1.0]*3 + [0.01]*3 + [1e-6]*3 + [1e-4]*3)
    >>> Q = np.diag([0.1]*3 + [0.01]*3 + [1e-6]*3 + [1e-8]*3 + [1e-6]*3)
    >>> ekf = EKF(x0, P0, Q, dt=1.0)
    >>> ekf.predict(omega_measured=np.array([0.01, 0.0, 0.0]),
    ...             accel_measured=np.array([0.0, 0.0, -9.81]))
    """

    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray,
                 dt: float = 1.0) -> None:
        """
        Initialize the Extended Kalman Filter.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state vector (15 elements).
            [position(3), velocity(3), attitude_error(3),
             gyro_bias(3), accel_bias(3)]
        P0 : np.ndarray
            Initial state error covariance matrix (15x15).
            Diagonal elements represent the squared uncertainty in each state.
            Off-diagonal elements represent correlations between states.
            Must be symmetric positive-definite.
        Q : np.ndarray
            Process noise covariance matrix (15x15).
            Models the uncertainty in the dynamics model:
            - Position process noise: from unmodeled forces (solar pressure, etc.)
            - Velocity process noise: from unmodeled accelerations
            - Attitude process noise: from gyro noise
            - Gyro bias process noise: from bias instability (random walk)
            - Accel bias process noise: from accelerometer bias instability
        dt : float, optional
            Default propagation time step in seconds. Default 1.0 s.

        Raises
        ------
        ValueError
            If dimensions are inconsistent.
        """
        # --- Validate inputs ---
        self.x = np.array(x0, dtype=np.float64).flatten()
        if self.x.shape[0] != 15:
            raise ValueError(
                f"State vector must have 15 elements, got {self.x.shape[0]}. "
                "Expected: [pos(3), vel(3), att_err(3), gyro_bias(3), accel_bias(3)]"
            )

        self.P = np.array(P0, dtype=np.float64)
        if self.P.shape != (15, 15):
            raise ValueError(
                f"Covariance matrix must be 15x15, got {self.P.shape}"
            )

        self.Q = np.array(Q, dtype=np.float64)
        if self.Q.shape != (15, 15):
            raise ValueError(
                f"Process noise matrix must be 15x15, got {self.Q.shape}"
            )

        self.dt = dt

        # --- Initialize reference quaternion to identity ---
        # The attitude error in the state vector is relative to this quaternion.
        # At initialization, we assume the initial attitude is known (identity),
        # and any initial attitude error is captured in x[6:9].
        self.q_ref = Quaternion.identity()

        # --- Ensure initial attitude error is zero ---
        # If the user provided a nonzero initial attitude error, fold it into
        # the reference quaternion immediately for consistency.
        if np.linalg.norm(self.x[6:9]) > 1e-10:
            delta_theta = self.x[6:9]
            # Construct error quaternion from small angle approximation:
            # delta_q ~= [1, 0.5*delta_theta]  (first-order approximation)
            delta_q = Quaternion(
                1.0,
                0.5 * delta_theta[0],
                0.5 * delta_theta[1],
                0.5 * delta_theta[2]
            )
            # Fold the error into the reference: q_ref_new = delta_q * q_ref
            self.q_ref = delta_q * self.q_ref
            # Reset the error to zero
            self.x[6:9] = 0.0

    # =========================================================================
    # PREDICT STEP
    # =========================================================================

    def predict(self, omega_measured: np.ndarray, accel_measured: np.ndarray,
                dt: float = None) -> None:
        """
        Propagate the state and covariance forward by one time step.

        This is the "time update" or "predict" step of the Kalman filter.
        It uses the IMU measurements (gyroscope and accelerometer) to
        propagate the state estimate forward in time, while accounting
        for the estimated sensor biases.

        The predict step has 6 sub-steps:
        1. Remove estimated biases from IMU measurements
        2. Propagate position using velocity (kinematics)
        3. Propagate velocity using corrected acceleration (dynamics)
        4. Propagate the reference quaternion using corrected angular velocity
        5. Compute the state transition matrix F (linearized dynamics)
        6. Propagate the error covariance P = F*P*F^T + Q

        Parameters
        ----------
        omega_measured : np.ndarray
            Measured angular velocity from the gyroscope [wx, wy, wz] in rad/s.
            This includes the true angular velocity PLUS gyro bias PLUS noise:
                omega_measured = omega_true + bias_gyro + noise_gyro
        accel_measured : np.ndarray
            Measured specific force from the accelerometer [ax, ay, az] in m/s^2.
            This includes the true specific force PLUS accel bias PLUS noise:
                accel_measured = accel_true + bias_accel + noise_accel
            Note: In free-fall, the accelerometer reads zero (not -g).
        dt : float, optional
            Time step in seconds. If None, uses the default dt.

        Notes
        -----
        The gyroscope measures angular velocity in the BODY frame. The
        accelerometer measures specific force (non-gravitational acceleration)
        in the body frame. Gravity must be modeled separately in the dynamics.

        For the position and velocity propagation, we use a simple Euler
        step. For higher accuracy, this should be replaced with RK4
        integration, especially for long time steps or highly dynamic
        trajectories (e.g., during powered flight).
        """
        if dt is None:
            dt = self.dt

        omega_measured = np.asarray(omega_measured, dtype=np.float64)
        accel_measured = np.asarray(accel_measured, dtype=np.float64)

        # =====================================================================
        # Step 1: Remove estimated biases from IMU measurements
        # =====================================================================
        # The filter maintains estimates of the gyro and accelerometer biases
        # (states 9-11 and 12-14). By subtracting these estimates, we get
        # a better approximation of the true angular velocity and acceleration.
        #
        # As the filter converges, the bias estimates improve, and the
        # corrected measurements become more accurate.
        # =====================================================================
        gyro_bias = self.x[9:12]      # Current gyro bias estimate (rad/s)
        accel_bias = self.x[12:15]    # Current accel bias estimate (m/s^2)

        omega_corrected = omega_measured - gyro_bias
        accel_corrected = accel_measured - accel_bias

        # =====================================================================
        # Step 2: Propagate position using velocity (kinematic equation)
        # =====================================================================
        # x_new = x_old + v * dt  (simple Euler integration)
        #
        # This is the fundamental kinematic relation: position changes at
        # the rate of velocity. For higher accuracy, use:
        #   x_new = x_old + v * dt + 0.5 * a * dt^2  (includes acceleration)
        # but the EKF's covariance propagation accounts for this truncation
        # error through the process noise Q.
        # =====================================================================
        self.x[0:3] += self.x[3:6] * dt

        # =====================================================================
        # Step 3: Propagate velocity using corrected acceleration
        # =====================================================================
        # v_new = v_old + a_corrected * dt
        #
        # The corrected acceleration is the specific force measured by the
        # accelerometer (with bias removed). In orbit, the accelerometer
        # measures only non-gravitational forces (thrust, drag, solar pressure).
        # Gravity is a "free fall" acceleration that the accelerometer does not
        # sense. Gravity effects on the orbit must be modeled separately in
        # the dynamics (handled by the state transition matrix F).
        #
        # Note: The accelerometer measurement is in the body frame, so it
        # should be rotated to the inertial frame using q_ref before adding
        # to velocity. For simplicity, we assume the measurement has already
        # been transformed, or the rotation is handled externally.
        # =====================================================================
        self.x[3:6] += accel_corrected * dt

        # =====================================================================
        # Step 4: Propagate reference quaternion using corrected angular velocity
        # =====================================================================
        # The reference quaternion is updated using the corrected gyroscope
        # measurement. We construct a small rotation quaternion from the
        # angular velocity and multiply it with the current reference.
        #
        # For small angles (omega * dt << 1 radian), the rotation quaternion is:
        #   delta_q = [cos(|omega|*dt/2), sin(|omega|*dt/2) * omega_hat]
        #
        # Using Quaternion.from_rotation_vector handles this exactly, including
        # the case of zero angular velocity.
        # =====================================================================
        rotation_vector = omega_corrected * dt
        delta_q_propagate = Quaternion.from_rotation_vector(rotation_vector)

        # Apply the rotation: q_ref_new = q_ref_old * delta_q
        # (body-frame angular velocity -> right-multiply)
        self.q_ref = self.q_ref * delta_q_propagate

        # =====================================================================
        # Step 5: Reset attitude error states to zero
        # =====================================================================
        # After propagating the reference quaternion, the attitude error should
        # be reset to zero. The error states represent the deviation of the
        # true attitude from the reference. Since we just updated the reference
        # to match our best estimate, the error is zero by definition.
        # =====================================================================
        self.x[6:9] = 0.0

        # =====================================================================
        # Step 6: Compute the state transition matrix F (15x15 Jacobian)
        # =====================================================================
        # F = partial(f) / partial(x), where f is the nonlinear state
        # propagation function. This linearization is the core of the EKF.
        #
        # The F matrix has a specific block structure reflecting the physics:
        #
        #   F = | I   I*dt   0       0       0    |  position rows
        #       | 0   I      0       0      -I*dt |  velocity rows
        #       | 0   0      Phi_att 0       0    |  attitude error rows
        #       | 0   0      0       I       0    |  gyro bias rows
        #       | 0   0      0       0       I    |  accel bias rows
        #
        # where:
        # - Phi_att = I - [omega_corrected x] * dt
        #   (attitude error transition due to rotation)
        # - The -I*dt in velocity/accel_bias coupling means that accel bias
        #   directly affects velocity propagation
        # - Gyro and accel biases are modeled as random walks (F = I)
        # =====================================================================
        F = np.eye(15, dtype=np.float64)

        # Position is driven by velocity: dx/dt = v => F[pos, vel] = I * dt
        F[0:3, 3:6] = np.eye(3) * dt

        # Velocity is affected by accelerometer bias:
        # dv/dt = a_measured - bias_accel
        # => partial(v_new)/partial(bias_accel) = -I * dt
        F[3:6, 12:15] = -np.eye(3) * dt

        # Attitude error propagation:
        # d(delta_theta)/dt = -[omega_corrected x] * delta_theta - delta_bias_gyro
        # => F[att, att] = I - [omega_corrected x] * dt
        # The skew-symmetric matrix captures how rotation couples the error axes
        omega_skew = _skew_symmetric(omega_corrected)
        F[6:9, 6:9] = np.eye(3) - omega_skew * dt

        # Attitude error is driven by gyro bias:
        # => F[att, gyro_bias] = -I * dt
        F[6:9, 9:12] = -np.eye(3) * dt

        # Gyro and accel biases: random walk model => F = I (already set)
        # d(bias)/dt = noise => F[bias, bias] = I
        # This means biases persist but slowly drift over time.

        # =====================================================================
        # Step 7: Propagate the error covariance
        # =====================================================================
        # P_new = F * P_old * F^T + Q
        #
        # This is the fundamental covariance propagation equation. It has two
        # terms:
        # 1. F * P * F^T: How the existing uncertainty transforms through
        #    the dynamics. If the dynamics amplify errors (e.g., gravity
        #    divergence), this term grows.
        # 2. Q: Additional uncertainty injected by process noise (unmodeled
        #    forces, sensor noise that drives the state). This prevents the
        #    filter from becoming overconfident.
        #
        # The Q matrix should be tuned based on:
        # - Position: unmodeled accelerations * dt^2 / 2 (e.g., solar pressure)
        # - Velocity: unmodeled accelerations * dt
        # - Attitude: gyro angle random walk * sqrt(dt)
        # - Gyro bias: gyro bias instability * dt
        # - Accel bias: accelerometer bias instability * dt
        # =====================================================================
        self.P = F @ self.P @ F.T + self.Q

        # Enforce symmetry (floating-point errors can break this over time)
        # P should always be symmetric; forcing it prevents slow drift
        self.P = 0.5 * (self.P + self.P.T)

    # =========================================================================
    # MEASUREMENT UPDATE: POSITION
    # =========================================================================

    def update_position(self, z_position: np.ndarray,
                        R_pos: np.ndarray) -> None:
        """
        Update the state with a position measurement (e.g., from GPS, DSN
        ranging, star tracker + ephemeris, or optical navigation).

        The measurement model is:
            z = H * x + noise
            z_position = x[0:3] + noise

        so the measurement matrix H is simply:
            H = [I(3), 0(3x3), 0(3x3), 0(3x3), 0(3x3)]  (3x15)

        This is a linear measurement model, so the EKF update is exact
        (no linearization error).

        Parameters
        ----------
        z_position : np.ndarray
            Measured position [x, y, z] in meters (inertial frame).
        R_pos : np.ndarray
            Measurement noise covariance matrix (3x3).
            Diagonal elements are the squared position uncertainty in each axis.
            For DSN ranging at Jupiter: ~10-100 km uncertainty per axis.
            For GPS in LEO: ~10-100 m uncertainty per axis.

        Notes
        -----
        The Joseph form is used for the covariance update to maintain
        numerical stability. See class docstring for explanation.
        """
        z_position = np.asarray(z_position, dtype=np.float64).flatten()
        R_pos = np.asarray(R_pos, dtype=np.float64)

        # --- Measurement matrix H (3x15) ---
        # z = [I(3) | 0 | 0 | 0 | 0] * x = x[0:3]
        # Position measurement directly observes the position states.
        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)

        # --- Innovation (measurement residual) ---
        # innovation = z - H*x = z_position - x[0:3]
        # This is the difference between what we measured and what we predicted.
        # A large innovation means the measurement disagrees with our estimate.
        innovation = z_position - self.x[0:3]

        # --- Innovation covariance ---
        # S = H * P * H^T + R
        # This combines the state uncertainty projected into measurement space
        # (H*P*H^T) with the measurement noise (R). The Kalman gain trades off
        # between trusting the model (small P) and trusting the measurement (small R).
        S = H @ self.P @ H.T + R_pos

        # --- Kalman gain ---
        # K = P * H^T * S^{-1}
        # The gain determines how much of the innovation to apply to the state.
        # K ~ I means "trust the measurement" (R << H*P*H^T)
        # K ~ 0 means "trust the model" (R >> H*P*H^T)
        K = self.P @ H.T @ np.linalg.inv(S)

        # --- State update ---
        # x_new = x_old + K * innovation
        # This shifts our state estimate toward the measurement, weighted by
        # the Kalman gain. The innovation is projected into all 15 states
        # through K, allowing a position measurement to improve velocity
        # estimates (through the position-velocity cross-correlation in P).
        self.x += K @ innovation

        # --- Covariance update (Joseph form) ---
        # P_new = (I - K*H) * P_old * (I - K*H)^T + K * R * K^T
        #
        # This is algebraically equivalent to the standard form:
        #   P_new = (I - K*H) * P_old
        # but numerically superior because:
        # 1. The result is guaranteed symmetric (A * A^T is always symmetric)
        # 2. The result is guaranteed positive semi-definite
        # 3. The K*R*K^T term ensures we never underestimate the uncertainty
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_pos @ K.T

        # Enforce symmetry as a final safeguard
        self.P = 0.5 * (self.P + self.P.T)

    # =========================================================================
    # MEASUREMENT UPDATE: ATTITUDE (QUATERNION)
    # =========================================================================

    def update_attitude(self, z_quaternion: 'Quaternion',
                        R_att: np.ndarray) -> None:
        """
        Update the state with an attitude measurement (e.g., from star
        tracker or Sun sensor).

        This is the core of the multiplicative quaternion EKF. The measured
        quaternion is compared to the reference quaternion to compute a
        small-angle error, which is then processed through the standard
        Kalman update equations.

        The measurement model:
            delta_q = z_quaternion * q_ref^{-1}
            innovation = 2 * delta_q.vector_part  (small-angle approximation)

        The factor of 2 comes from the half-angle encoding in quaternions:
            delta_q ~= [1, 0.5 * delta_theta]
            => delta_theta = 2 * delta_q.vector

        The measurement matrix:
            H = [0(3x3), 0(3x3), I(3), 0(3x3), 0(3x3)]  (3x15)

        Parameters
        ----------
        z_quaternion : Quaternion
            Measured attitude quaternion from the star tracker or other
            attitude sensor.
        R_att : np.ndarray
            Attitude measurement noise covariance (3x3) in radians^2.
            For a typical star tracker: diag([1e-6, 1e-6, 1e-4]) rad^2
            (arcsecond-level cross-boresight, arcminute-level roll).

        Notes
        -----
        After the update, the attitude error is "reset" by folding it back
        into the reference quaternion. This is essential for the multiplicative
        formulation to work correctly:
        1. Compute delta_q from the updated error states x[6:9]
        2. Update: q_ref_new = delta_q * q_ref_old
        3. Reset: x[6:9] = 0

        This reset step keeps the attitude error small, ensuring the
        small-angle approximation remains valid.
        """
        R_att = np.asarray(R_att, dtype=np.float64)

        # --- Compute error quaternion ---
        # delta_q = z_measured * q_ref^{-1}
        # This gives the rotation from the reference attitude to the measured
        # attitude. If they are identical, delta_q = identity = [1, 0, 0, 0].
        delta_q = z_quaternion * self.q_ref.inverse()

        # --- Extract small-angle innovation ---
        # For a small rotation, the error quaternion is approximately:
        #   delta_q ~= [1, 0.5*delta_theta_x, 0.5*delta_theta_y, 0.5*delta_theta_z]
        # Therefore:
        #   delta_theta = 2 * [delta_q.x, delta_q.y, delta_q.z]
        #
        # This is the key step that converts the 4-element quaternion
        # measurement into a 3-element innovation suitable for the Kalman
        # update. The factor of 2 converts from "half-angle" quaternion space
        # to "full-angle" physical space.
        innovation = 2.0 * delta_q.vector

        # --- Measurement matrix H (3x15) ---
        # The attitude error states are x[6:9], so H selects those columns.
        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 6:9] = np.eye(3)

        # --- Innovation covariance ---
        # S = H * P * H^T + R_att
        S = H @ self.P @ H.T + R_att

        # --- Kalman gain ---
        # K = P * H^T * S^{-1}
        K = self.P @ H.T @ np.linalg.inv(S)

        # --- State update ---
        # x_new = x_old + K * innovation
        # This updates ALL 15 states, not just the attitude error.
        # The cross-correlations in P allow the attitude measurement to
        # improve gyro bias estimates (critical for long-term accuracy).
        self.x += K @ innovation

        # --- Covariance update (Joseph form) ---
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_att @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # =====================================================================
        # ATTITUDE RESET STEP (essential for multiplicative formulation)
        # =====================================================================
        # After the Kalman update, the attitude error states x[6:9] contain
        # the updated error. We must fold this error back into the reference
        # quaternion to keep the error small for the next cycle.
        #
        # 1. Construct the error quaternion from x[6:9]:
        #    delta_q = [1, 0.5*ex, 0.5*ey, 0.5*ez].normalize()
        # 2. Update the reference: q_ref = delta_q * q_ref
        # 3. Reset: x[6:9] = 0
        #
        # This is why it is called "multiplicative": the error is applied
        # multiplicatively (quaternion product) rather than additively.
        # =====================================================================
        delta_theta = self.x[6:9]
        delta_q_reset = Quaternion(
            1.0,
            0.5 * delta_theta[0],
            0.5 * delta_theta[1],
            0.5 * delta_theta[2]
        )
        # Normalize to maintain unit quaternion constraint
        delta_q_reset = delta_q_reset.normalize()

        # Fold the error into the reference quaternion
        self.q_ref = delta_q_reset * self.q_ref

        # Reset the attitude error to zero
        self.x[6:9] = 0.0

    # =========================================================================
    # MEASUREMENT UPDATE: VELOCITY
    # =========================================================================

    def update_velocity(self, z_vel: np.ndarray, R_vel: np.ndarray) -> None:
        """
        Update the state with a velocity measurement (e.g., from Doppler
        ranging, GPS velocity, or radar).

        The measurement model:
            z_velocity = x[3:6] + noise

        so the measurement matrix is:
            H = [0(3x3), I(3), 0(3x3), 0(3x3), 0(3x3)]  (3x15)

        Parameters
        ----------
        z_vel : np.ndarray
            Measured velocity [vx, vy, vz] in m/s (inertial frame).
        R_vel : np.ndarray
            Velocity measurement noise covariance (3x3) in (m/s)^2.
            For DSN Doppler at Jupiter: ~0.1-1.0 mm/s accuracy per axis.
            For GPS in LEO: ~0.01-0.1 m/s accuracy per axis.

        Notes
        -----
        DSN Doppler measurements are one of the most precise navigation
        observables available for deep-space missions, achieving sub-mm/s
        accuracy. This is because Doppler measures the line-of-sight
        velocity component directly from the carrier frequency shift,
        and the DSN transmitters/receivers have extremely stable oscillators
        (hydrogen maser frequency standards with stability ~10^-15).
        """
        z_vel = np.asarray(z_vel, dtype=np.float64).flatten()
        R_vel = np.asarray(R_vel, dtype=np.float64)

        # --- Measurement matrix H (3x15) ---
        # Velocity states are x[3:6]
        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 3:6] = np.eye(3)

        # --- Innovation ---
        # innovation = z_vel - H * x = z_vel - x[3:6]
        innovation = z_vel - self.x[3:6]

        # --- Innovation covariance ---
        S = H @ self.P @ H.T + R_vel

        # --- Kalman gain ---
        K = self.P @ H.T @ np.linalg.inv(S)

        # --- State update ---
        self.x += K @ innovation

        # --- Joseph form covariance update ---
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_vel @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # =========================================================================
    # STATE ACCESS METHODS
    # =========================================================================

    def get_state(self) -> np.ndarray:
        """
        Return a copy of the full 15-element state vector.

        Returns
        -------
        np.ndarray
            Copy of [position(3), velocity(3), attitude_error(3),
                     gyro_bias(3), accel_bias(3)].
            Note: attitude_error should be near zero after each update
            cycle (due to the reset step).
        """
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        """
        Return a copy of the 15x15 state error covariance matrix.

        The diagonal elements represent the variance (squared uncertainty)
        of each state. The square roots of the diagonals give the 1-sigma
        uncertainties:
        - P[0,0]^0.5 = position X uncertainty (meters)
        - P[3,3]^0.5 = velocity X uncertainty (m/s)
        - P[6,6]^0.5 = attitude error X uncertainty (radians)
        - P[9,9]^0.5 = gyro bias X uncertainty (rad/s)
        - P[12,12]^0.5 = accel bias X uncertainty (m/s^2)

        Returns
        -------
        np.ndarray
            Copy of the 15x15 covariance matrix.
        """
        return self.P.copy()

    def get_position(self) -> np.ndarray:
        """
        Return the estimated position [x, y, z] in meters.

        Returns
        -------
        np.ndarray
            3-element position vector in the inertial frame.
        """
        return self.x[0:3].copy()

    def get_velocity(self) -> np.ndarray:
        """
        Return the estimated velocity [vx, vy, vz] in m/s.

        Returns
        -------
        np.ndarray
            3-element velocity vector in the inertial frame.
        """
        return self.x[3:6].copy()

    def get_attitude_quaternion(self) -> 'Quaternion':
        """
        Return the estimated attitude as a quaternion.

        This returns the reference quaternion q_ref, which is the best
        estimate of the spacecraft attitude. The attitude error states
        x[6:9] should be near zero (they are reset after each update),
        so q_ref alone represents the full attitude estimate.

        Returns
        -------
        Quaternion
            Estimated attitude quaternion (scalar-first convention).
        """
        return self.q_ref

    def get_gyro_bias(self) -> np.ndarray:
        """
        Return the estimated gyroscope bias [bx, by, bz] in rad/s.

        The gyro bias drifts slowly over time (hours to days) due to
        temperature changes, aging, and other effects. Estimating and
        removing this bias is one of the most important functions of the
        navigation filter, as even a small bias (1e-4 rad/s ~ 0.006 deg/s)
        accumulates to significant attitude error over time:
        - After 1 hour: 0.006 * 3600 = 21.6 degrees of drift!

        Returns
        -------
        np.ndarray
            3-element gyro bias vector in rad/s (body frame).
        """
        return self.x[9:12].copy()

    def get_accel_bias(self) -> np.ndarray:
        """
        Return the estimated accelerometer bias [bx, by, bz] in m/s^2.

        Returns
        -------
        np.ndarray
            3-element accelerometer bias vector in m/s^2 (body frame).
        """
        return self.x[12:15].copy()

    # =========================================================================
    # FILTER HEALTH DIAGNOSTICS
    # =========================================================================

    def get_innovation_consistency(self, innovation: np.ndarray,
                                   S: np.ndarray) -> float:
        """
        Compute the normalized innovation squared (NIS) for filter consistency
        checking.

        The NIS is defined as:
            NIS = innovation^T * S^{-1} * innovation

        For a properly tuned filter, the NIS should follow a chi-squared
        distribution with n degrees of freedom (where n is the measurement
        dimension). This provides a statistical test for filter health:

        - If NIS is consistently too high: the filter is overconfident
          (Q or R too small). Measurements are being rejected too aggressively.
        - If NIS is consistently too low: the filter is underconfident
          (Q or R too large). The filter is not extracting enough information
          from measurements.
        - If NIS is erratic (sometimes very high): the dynamics model may be
          wrong, or there are measurement outliers that need rejection.

        The 95% confidence interval for chi-squared(n) is approximately:
        - n=3: [0.35, 7.81]
        - n=6: [1.64, 12.59]

        Parameters
        ----------
        innovation : np.ndarray
            Measurement innovation vector (z - H*x).
        S : np.ndarray
            Innovation covariance matrix (H*P*H^T + R).

        Returns
        -------
        float
            Normalized innovation squared. Should be approximately n
            (the measurement dimension) on average for a consistent filter.

        Notes
        -----
        This metric is used in real missions for:
        - Real-time filter health monitoring
        - Measurement rejection (if NIS > threshold, reject the measurement)
        - Post-flight filter tuning (adjust Q and R to achieve consistency)
        - Fault detection (sudden NIS spike indicates sensor failure)
        """
        innovation = np.asarray(innovation, dtype=np.float64).flatten()
        S = np.asarray(S, dtype=np.float64)

        # NIS = y^T * S^{-1} * y
        # Using solve instead of explicit inverse for numerical stability:
        # S^{-1} * y = solve(S, y)
        S_inv_innovation = np.linalg.solve(S, innovation)
        nis = float(innovation @ S_inv_innovation)

        return nis
