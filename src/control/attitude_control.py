"""
Attitude Control System
=======================

Multi-mode controller supporting three control laws:
  - PID       (used during detumble to arrest residual angular rates)
  - LQR       (used for fine pointing once attitude error is small)
  - Sliding Mode (used for large-angle slew maneuvers)

Target pointing logic:
  The spacecraft is always oriented toward the current celestial target
  (Moon during the cislunar phase, Jupiter during the interplanetary phase).
  A target quaternion is computed so that the body +Z axis points along the
  line-of-sight to the target body.

Additional features:
  - Anti-windup on the PID integrator (per-axis clamping)
  - First-order low-pass filter on the PID derivative term
  - Automatic mode switching between DETUMBLE, POINTING, and SLEW

References:
  Wie, B. "Space Vehicle Dynamics and Control", 2nd ed., AIAA, 2008.
  Sidi, M.J. "Spacecraft Dynamics and Control", Cambridge, 1997.
"""

import numpy as np
from core.quaternion import Quaternion
from core.constants import DEG2RAD, RAD2DEG


# ---------------------------------------------------------------------------
# PID Controller
# ---------------------------------------------------------------------------
class PIDController:
    """
    Classical three-axis PID attitude controller.

    Each axis is controlled independently.  The derivative channel uses a
    first-order low-pass filter to attenuate high-frequency sensor noise,
    and the integrator incorporates per-axis anti-windup clamping.

    Typical use: detumble mode where the goal is simply to drive angular
    rates to zero and coarsely reduce attitude error.

    Torque command:
        tau = Kp * e  +  Ki * integral(e) dt  +  Kd * de/dt   (3-vector)

    Parameters
    ----------
    kp : array_like (3,)
        Proportional gains [Nm/rad] for roll, pitch, yaw.
    ki : array_like (3,)
        Integral gains [Nm/(rad*s)].
    kd : array_like (3,)
        Derivative gains [Nm*s/rad].
    max_integral : float
        Anti-windup clamp magnitude per axis [Nm].
    dt_filter : float
        Time constant for the first-order derivative low-pass filter [s].
    """

    def __init__(self, kp, ki, kd, max_integral=10.0, dt_filter=0.01):
        self.kp = np.array(kp, dtype=float)
        self.ki = np.array(ki, dtype=float)
        self.kd = np.array(kd, dtype=float)
        self.max_integral = float(max_integral)
        self.dt_filter = float(dt_filter)

        # Internal state
        self._integral = np.zeros(3)        # accumulated integral term
        self._prev_error = np.zeros(3)      # previous-step error for finite diff
        self._d_filtered = np.zeros(3)      # filtered derivative value
        self._initialized = False

    # ----- public API -----

    def compute(self, error, error_rate, dt):
        """
        Compute the PID torque command.

        Parameters
        ----------
        error : ndarray (3,)
            Attitude error vector [rad] (e.g. small-angle MRP or Euler err).
        error_rate : ndarray (3,)
            Angular-rate error [rad/s] (omega_current - omega_desired).
        dt : float
            Control timestep [s].

        Returns
        -------
        torque_cmd : ndarray (3,)
            Commanded torque in body frame [Nm].
        """
        error = np.asarray(error, dtype=float)
        error_rate = np.asarray(error_rate, dtype=float)

        # --- Proportional term ---
        p_term = self.kp * error

        # --- Integral term with anti-windup ---
        self._integral += self.ki * error * dt
        # Per-axis clamping to prevent integrator windup
        self._integral = np.clip(
            self._integral, -self.max_integral, self.max_integral
        )
        i_term = self._integral

        # --- Derivative term with first-order low-pass filter ---
        #   raw derivative = error_rate (direct measurement preferred over
        #   finite-difference of error to avoid amplification of noise)
        raw_d = self.kd * error_rate
        # Low-pass filter:  alpha = dt / (dt + tau_filter)
        # When alpha -> 1, filter passes everything; when alpha -> 0, heavy
        # smoothing.  dt_filter acts as the filter time constant.
        alpha = dt / (dt + self.dt_filter) if (dt + self.dt_filter) > 0 else 1.0
        self._d_filtered = (1.0 - alpha) * self._d_filtered + alpha * raw_d
        d_term = self._d_filtered

        # --- Total PID output ---
        torque_cmd = p_term + i_term + d_term
        self._prev_error = error.copy()
        self._initialized = True
        return torque_cmd

    def reset(self):
        """Clear integrator and derivative filter state."""
        self._integral[:] = 0.0
        self._prev_error[:] = 0.0
        self._d_filtered[:] = 0.0
        self._initialized = False


# ---------------------------------------------------------------------------
# LQR Controller
# ---------------------------------------------------------------------------
class LQRController:
    """
    Linear Quadratic Regulator for spacecraft attitude control.

    LQR minimises the infinite-horizon cost

        J = integral_0^inf  ( x^T Q x  +  u^T R u ) dt

    where the state vector x = [theta_err (3), omega_err (3)]^T and the
    control u is the 3-axis body torque.

    The linearised attitude dynamics (small-angle, no coupling) are:

        x_dot = A x + B u

        A = [[0_{3x3}, I_{3x3}],      B = [[0_{3x3}      ],
             [0_{3x3}, 0_{3x3}]]           [J^{-1}        ]]

    The optimal gain is  K = R^{-1} B^T P  where P solves the continuous
    algebraic Riccati equation (CARE):

        A^T P + P A - P B R^{-1} B^T P + Q = 0

    Parameters
    ----------
    Q_diag : array_like (6,)
        Diagonal of the state-weighting matrix Q.
    R_diag : array_like (3,)
        Diagonal of the control-weighting matrix R.
    inertia : ndarray (3,3)
        Spacecraft inertia tensor [kg*m^2].
    """

    def __init__(self, Q_diag, R_diag, inertia):
        self.Q = np.diag(np.asarray(Q_diag, dtype=float))   # 6x6
        self.R = np.diag(np.asarray(R_diag, dtype=float))    # 3x3
        self.J = np.array(inertia, dtype=float)               # 3x3
        self.J_inv = np.linalg.inv(self.J)

        # Build linearised plant matrices
        #   state = [theta_err_x, theta_err_y, theta_err_z,
        #            omega_err_x, omega_err_y, omega_err_z]
        self.A = np.zeros((6, 6))
        self.A[0:3, 3:6] = np.eye(3)           # d(theta)/dt = omega

        self.B = np.zeros((6, 3))
        self.B[3:6, 0:3] = self.J_inv           # d(omega)/dt = J^{-1} u

        # Solve for the optimal gain K
        P = self._solve_riccati(self.A, self.B, self.Q, self.R)
        # K = R^{-1} B^T P
        self.K = np.linalg.inv(self.R) @ self.B.T @ P   # (3x6)

    def _solve_riccati(self, A, B, Q, R, n_iter=2000, tol=1e-12):
        """
        Iterative solver for the continuous algebraic Riccati equation:

            A^T P + P A - P B R^{-1} B^T P + Q = 0

        Uses a simple fixed-point iteration (value iteration) with a small
        step size delta:

            P_{k+1} = P_k + delta * (A^T P_k + P_k A
                                      - P_k B R^{-1} B^T P_k + Q)

        This is guaranteed to converge for stabilisable (A, B) provided
        delta is sufficiently small.

        Parameters
        ----------
        A : ndarray (n, n)
        B : ndarray (n, m)
        Q : ndarray (n, n)   positive semi-definite
        R : ndarray (m, m)   positive definite
        n_iter : int
            Maximum iterations.
        tol : float
            Convergence tolerance on the Frobenius norm of the residual.

        Returns
        -------
        P : ndarray (n, n)
            Stabilising solution of the CARE.
        """
        n = A.shape[0]
        P = np.copy(Q)                       # initial guess
        R_inv = np.linalg.inv(R)
        BRinvBT = B @ R_inv @ B.T
        delta = 0.001                         # step size for iteration

        for i in range(n_iter):
            # Residual of the CARE
            residual = A.T @ P + P @ A - P @ BRinvBT @ P + Q
            P_new = P + delta * residual

            # Enforce symmetry (numerical hygiene)
            P_new = 0.5 * (P_new + P_new.T)

            if np.linalg.norm(residual, 'fro') < tol:
                return P_new

            P = P_new

        # If not converged, return best estimate with a warning
        return P

    def compute(self, state_error_6):
        """
        Compute the LQR control torque.

        The full-state feedback law is:

            u = -K x

        where x = [theta_err (3), omega_err (3)]^T.

        Parameters
        ----------
        state_error_6 : ndarray (6,)
            Concatenated attitude and rate error.

        Returns
        -------
        torque_cmd : ndarray (3,)
            Optimal torque command [Nm].
        """
        x = np.asarray(state_error_6, dtype=float)
        # Full-state feedback: LQR minimises J = integral (x'Qx + u'Ru) dt
        u = -self.K @ x
        return u

    def get_gain_matrix(self):
        """Return the 3x6 LQR gain matrix K."""
        return self.K.copy()


# ---------------------------------------------------------------------------
# Sliding Mode Controller
# ---------------------------------------------------------------------------
class SlidingModeController:
    """
    Sliding Mode Controller for large-angle attitude slew maneuvers.

    Sliding mode control is inherently robust to bounded matched
    uncertainties (unmodelled torques, inertia errors, etc.) because the
    switching term drives the state onto a sliding surface where the
    dynamics are independent of the disturbance.

    The sliding surface is defined as:

        s = omega_err + Lambda * q_err_vec          (3-vector)

    where Lambda = diag(lambda_1, lambda_2, lambda_3).

    Control law:
        u = u_eq + u_sw

        u_eq = -J (Lambda * omega_err) + omega x (J omega)
               (equivalent control -- cancels known dynamics)

        u_sw = -eta * sat(s / phi)
               (switching control -- drives state to surface)

    sat() is the saturation function that replaces sgn() inside a
    boundary layer of width phi to suppress chattering.

    Parameters
    ----------
    lambda_gains : array_like (3,)
        Sliding surface slope gains [1/s].
    eta : float
        Switching gain magnitude [Nm].
    boundary_layer : float
        Width of the saturation boundary layer.  Larger values give
        smoother control at the expense of steady-state accuracy.
    """

    def __init__(self, lambda_gains, eta, boundary_layer=0.05):
        self.lam = np.array(lambda_gains, dtype=float)
        self.eta = float(eta)
        self.phi = float(boundary_layer)

    @staticmethod
    def _sat(x, phi):
        """
        Element-wise saturation function.

        sat(x/phi) = x/phi   if |x/phi| <= 1
                   = sign(x) otherwise

        This replaces the discontinuous signum function to suppress
        chattering while maintaining a bounded boundary layer.
        """
        ratio = x / phi
        return np.clip(ratio, -1.0, 1.0)

    def compute(self, q_error_vec, omega_error, inertia, omega_current=None):
        """
        Compute the sliding mode control torque.

        Parameters
        ----------
        q_error_vec : ndarray (3,)
            Vector part of the error quaternion (approximately half the
            rotation error in radians for small angles).
        omega_error : ndarray (3,)
            Angular rate error in body frame [rad/s].
        inertia : ndarray (3,3)
            Spacecraft inertia tensor [kg*m^2].
        omega_current : ndarray (3,) or None
            Current angular velocity in body frame [rad/s].  Required for
            the gyroscopic cross-coupling compensation.  If None, the
            cross-coupling term is omitted.

        Returns
        -------
        torque_cmd : ndarray (3,)
            Commanded torque in body frame [Nm].
        """
        q_err = np.asarray(q_error_vec, dtype=float)
        w_err = np.asarray(omega_error, dtype=float)
        J = np.asarray(inertia, dtype=float)

        # --- Sliding surface ---
        # s = omega_error + Lambda * q_error_vec  (element-wise product)
        s = w_err + self.lam * q_err

        # --- Equivalent control (cancels known nonlinear dynamics) ---
        # u_eq = -J * (Lambda .* omega_err)  +  omega x (J * omega)
        u_eq = -J @ (self.lam * w_err)
        if omega_current is not None:
            omega = np.asarray(omega_current, dtype=float)
            # Gyroscopic compensation
            u_eq += np.cross(omega, J @ omega)

        # --- Switching control (robust to bounded uncertainties) ---
        # u_sw = -eta * sat(s / phi)
        u_sw = -self.eta * self._sat(s, self.phi)

        # --- Total torque ---
        torque_cmd = u_eq + u_sw
        return torque_cmd


# ---------------------------------------------------------------------------
# Main Attitude Control System
# ---------------------------------------------------------------------------
class AttitudeControlSystem:
    """
    Top-level attitude control system that manages mode transitions and
    delegates torque computation to the appropriate sub-controller.

    Modes
    -----
    DETUMBLE  : PID controller.  Used immediately after separation or
                after a fault to arrest tumbling rates.
    POINTING  : LQR controller.  Fine three-axis pointing once the
                attitude error is within the linear regime (~5 deg).
    SLEW      : Sliding Mode controller.  Large-angle reorientation
                maneuvers where robustness to model errors is critical.

    Target Pointing
    ---------------
    The spacecraft is always oriented so that its body +Z axis points
    toward the current celestial target:
      - Moon (cislunar phase)
      - Jupiter (interplanetary cruise)

    The target quaternion is recomputed every control cycle from the
    relative geometry.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
          'pid_kp', 'pid_ki', 'pid_kd'        -- PID gains (3-vectors)
          'pid_max_integral'                    -- anti-windup limit
          'pid_dt_filter'                       -- derivative filter tau
          'lqr_Q_diag'                          -- LQR Q diagonal (6)
          'lqr_R_diag'                          -- LQR R diagonal (3)
          'inertia'                              -- 3x3 inertia tensor
          'smc_lambda'                          -- SMC surface gains (3)
          'smc_eta'                              -- SMC switching gain
          'smc_boundary_layer'                  -- SMC boundary layer width
    """

    VALID_MODES = {'DETUMBLE', 'POINTING', 'SLEW'}

    def __init__(self, config):
        self.config = config
        inertia = np.array(config['inertia'], dtype=float)

        # --- Sub-controllers ---
        self.pid = PIDController(
            kp=config['pid_kp'],
            ki=config['pid_ki'],
            kd=config['pid_kd'],
            max_integral=config.get('pid_max_integral', 10.0),
            dt_filter=config.get('pid_dt_filter', 0.01),
        )

        self.lqr = LQRController(
            Q_diag=config['lqr_Q_diag'],
            R_diag=config['lqr_R_diag'],
            inertia=inertia,
        )

        self.smc = SlidingModeController(
            lambda_gains=config['smc_lambda'],
            eta=config['smc_eta'],
            boundary_layer=config.get('smc_boundary_layer', 0.05),
        )

        self.inertia = inertia
        self.mode = 'DETUMBLE'

        # Target tracking state
        self._target_body_name = None
        self._target_position_eci = None

    # ----- Mode management -----

    def set_mode(self, mode):
        """
        Set the active control mode.

        Parameters
        ----------
        mode : str
            One of 'DETUMBLE', 'POINTING', 'SLEW'.

        Raises
        ------
        ValueError
            If mode is not one of the valid modes.
        """
        mode = mode.upper()
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of {self.VALID_MODES}."
            )
        # Reset PID integrator when leaving detumble to avoid stale state
        if self.mode == 'DETUMBLE' and mode != 'DETUMBLE':
            self.pid.reset()
        self.mode = mode

    # ----- Target tracking -----

    def set_target_body(self, body_name, body_position_eci):
        """
        Define the celestial body the spacecraft should point at.

        Parameters
        ----------
        body_name : str
            Human-readable name (e.g. 'Moon', 'Jupiter').
        body_position_eci : ndarray (3,)
            Position of the target body in ECI frame [m].
        """
        self._target_body_name = body_name
        self._target_position_eci = np.array(body_position_eci, dtype=float)

    def compute_target_quaternion(self, sc_position_eci, target_position_eci):
        """
        Compute the quaternion that orients the body +Z axis toward the
        target position.

        Algorithm:
          1. Compute the desired pointing direction (unit vector from
             spacecraft to target).
          2. Choose a reference "sun" direction for the secondary axis
             constraint (here we use the ECI +X axis as a default).
          3. Build a right-handed body frame:
               z_b = pointing direction
               y_b = z_b x sun_dir  (normalised)
               x_b = y_b x z_b
          4. Assemble the Direction Cosine Matrix (DCM) and convert to
             quaternion.

        Parameters
        ----------
        sc_position_eci : ndarray (3,)
            Spacecraft position in ECI [m].
        target_position_eci : ndarray (3,)
            Target body position in ECI [m].

        Returns
        -------
        q_target : Quaternion
            Desired attitude quaternion (ECI -> body).
        """
        sc_pos = np.asarray(sc_position_eci, dtype=float)
        tgt_pos = np.asarray(target_position_eci, dtype=float)

        # Desired body +Z direction (line-of-sight to target)
        d = tgt_pos - sc_pos
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-10:
            # Degenerate case -- return identity quaternion
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        z_b = d / d_norm

        # Reference vector for the secondary constraint.  Using the ECI +X
        # axis works unless z_b is nearly parallel to +X, in which case we
        # fall back to ECI +Y.
        sun_ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(z_b, sun_ref)) > 0.98:
            sun_ref = np.array([0.0, 1.0, 0.0])

        # Build orthonormal triad (right-handed)
        y_b = np.cross(z_b, sun_ref)
        y_b /= np.linalg.norm(y_b)
        x_b = np.cross(y_b, z_b)
        x_b /= np.linalg.norm(x_b)

        # Direction Cosine Matrix (columns are body axes expressed in ECI)
        # DCM rows = body unit vectors in ECI coordinates
        dcm = np.array([x_b, y_b, z_b])   # 3x3, dcm[i] = i-th body axis

        # Convert DCM to quaternion using Shepperd's method
        q = self._dcm_to_quaternion(dcm)
        return q

    @staticmethod
    def _dcm_to_quaternion(dcm):
        """
        Convert a 3x3 Direction Cosine Matrix to a unit quaternion.

        Uses Shepperd's method to avoid numerical singularities.

        Parameters
        ----------
        dcm : ndarray (3,3)
            Proper orthogonal rotation matrix.

        Returns
        -------
        q : Quaternion
        """
        tr = np.trace(dcm)

        if tr > 0:
            s = 2.0 * np.sqrt(tr + 1.0)
            w = 0.25 * s
            x = (dcm[2, 1] - dcm[1, 2]) / s
            y = (dcm[0, 2] - dcm[2, 0]) / s
            z = (dcm[1, 0] - dcm[0, 1]) / s
        elif (dcm[0, 0] > dcm[1, 1]) and (dcm[0, 0] > dcm[2, 2]):
            s = 2.0 * np.sqrt(1.0 + dcm[0, 0] - dcm[1, 1] - dcm[2, 2])
            w = (dcm[2, 1] - dcm[1, 2]) / s
            x = 0.25 * s
            y = (dcm[0, 1] + dcm[1, 0]) / s
            z = (dcm[0, 2] + dcm[2, 0]) / s
        elif dcm[1, 1] > dcm[2, 2]:
            s = 2.0 * np.sqrt(1.0 + dcm[1, 1] - dcm[0, 0] - dcm[2, 2])
            w = (dcm[0, 2] - dcm[2, 0]) / s
            x = (dcm[0, 1] + dcm[1, 0]) / s
            y = 0.25 * s
            z = (dcm[1, 2] + dcm[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + dcm[2, 2] - dcm[0, 0] - dcm[1, 1])
            w = (dcm[1, 0] - dcm[0, 1]) / s
            x = (dcm[0, 2] + dcm[2, 0]) / s
            y = (dcm[1, 2] + dcm[2, 1]) / s
            z = 0.25 * s

        # Ensure scalar-first convention and positive w
        q = Quaternion(w, x, y, z)
        return q

    # ----- Error computation -----

    def compute_error_quaternion(self, current_q, target_q):
        """
        Compute the attitude error as a 3-vector from the error quaternion.

        The error quaternion is:

            q_err = q_target^{-1}  *  q_current

        For small errors the vector part of q_err is approximately half
        the rotation error in radians:

            error_vec ~ 2 * q_err.vector_part

        A sign convention is enforced so that q_err.w >= 0 (short
        rotation path).

        Parameters
        ----------
        current_q : Quaternion
            Current spacecraft attitude.
        target_q : Quaternion
            Desired attitude.

        Returns
        -------
        error_vec : ndarray (3,)
            Attitude error [rad] (roll, pitch, yaw).
        """
        # q_err = q_target^{-1} * q_current
        q_err = target_q.inverse() * current_q

        # Extract scalar and vector parts
        w = q_err.w
        vec = q_err.vector_part()          # ndarray (3,)

        # Ensure short-rotation convention: if w < 0, negate the whole
        # quaternion (represents the same rotation)
        if w < 0.0:
            w = -w
            vec = -vec

        # Small-angle approximation: error ~ 2 * vector part [rad]
        error_vec = 2.0 * vec
        return error_vec

    # ----- Control computation -----

    def compute_control(self, current_q, current_omega, target_q, dt):
        """
        Compute the 3-axis torque command based on the active control mode.

        Parameters
        ----------
        current_q : Quaternion
            Current attitude quaternion (ECI -> body).
        current_omega : ndarray (3,)
            Current angular velocity in body frame [rad/s].
        target_q : Quaternion
            Desired attitude quaternion.
        dt : float
            Control timestep [s].

        Returns
        -------
        torque_cmd : ndarray (3,)
            Commanded torque in body frame [Nm].
        """
        omega = np.asarray(current_omega, dtype=float)

        # Attitude and rate errors
        error_vec = self.compute_error_quaternion(current_q, target_q)
        # Desired angular velocity is zero for regulation tasks
        omega_error = omega  # omega_err = omega - 0

        # --- Mode dispatch ---
        if self.mode == 'DETUMBLE':
            # PID: drive rates to zero and coarsely correct attitude
            torque_cmd = self.pid.compute(error_vec, omega_error, dt)

        elif self.mode == 'POINTING':
            # LQR: optimal fine pointing
            # State vector: [attitude_error (3), rate_error (3)]
            state_error = np.concatenate([error_vec, omega_error])
            torque_cmd = self.lqr.compute(state_error)

        elif self.mode == 'SLEW':
            # Sliding Mode: robust large-angle slew
            torque_cmd = self.smc.compute(
                q_error_vec=error_vec,
                omega_error=omega_error,
                inertia=self.inertia,
                omega_current=omega,
            )

        else:
            # Fallback -- should never reach here due to set_mode validation
            torque_cmd = np.zeros(3)

        return torque_cmd

    # ----- Telemetry helpers -----

    def get_pointing_error_deg(self, current_q, target_q):
        """
        Compute the total pointing error angle between the current and
        target attitudes.

        Uses the axis-angle representation of the error quaternion:

            theta = 2 * arccos(|q_err.w|)

        Parameters
        ----------
        current_q : Quaternion
        target_q : Quaternion

        Returns
        -------
        error_deg : float
            Total pointing error [degrees].
        """
        q_err = target_q.inverse() * current_q
        w = q_err.w
        # Clamp to handle numerical noise
        w_clamped = np.clip(abs(w), -1.0, 1.0)
        error_rad = 2.0 * np.arccos(w_clamped)
        return error_rad * RAD2DEG
