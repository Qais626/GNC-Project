"""
Optimal Control Methods
=======================

Advanced, robust, and optimal control algorithms for spacecraft attitude:

  - **LQG** (Linear Quadratic Gaussian)
        Combines an LQR optimal state-feedback law with a Kalman filter
        (optimal linear estimator).  By the *separation principle*, the
        resulting output-feedback controller is optimal for the LQG cost.

  - **H-infinity** (simplified)
        Minimises the worst-case (L2) gain from external disturbances to
        the regulated output.  More robust than LQR/LQG when the plant
        model is uncertain, at the cost of performance.

  - **MPC** (Model Predictive Control)
        Solves a finite-horizon optimal control problem at every timestep
        and applies only the first control action (receding-horizon
        strategy).  Naturally handles state and input constraints.

All three controllers operate on the linearised 6-state attitude model:

    x = [theta_err (3), omega_err (3)]^T
    u = [tau_x, tau_y, tau_z]^T              (body torques)

    x_dot = A x + B u + w        (process noise / disturbance)
    y     = C x + v              (measurement noise)

References
----------
  Stengel, R.F. "Optimal Control and Estimation", Dover, 1994.
  Zhou, K., Doyle, J.C. "Essentials of Robust Control", Prentice Hall, 1998.
  Rawlings, Mayne, Diehl, "Model Predictive Control", 2nd ed., 2017.
"""

import numpy as np
from core.constants import DEG2RAD


# ===================================================================
# LQG Controller
# ===================================================================

class LQGController:
    """
    Linear Quadratic Gaussian (LQG) controller.

    Structure:
        1. **Kalman Filter** (predictor-corrector) estimates the full state
           from noisy measurements y = C x + v.
        2. **LQR** gain is applied to the state estimate to produce the
           optimal control u = -K x_hat.

    Separation Principle
    --------------------
    The optimal estimator (Kalman filter) and the optimal regulator (LQR)
    can be designed independently; their cascade is the optimal
    output-feedback controller for the LQG cost:

        J = E{ integral_0^inf (x^T Q x + u^T R u) dt }

    subject to process noise w ~ N(0, Q_kalman) and measurement noise
    v ~ N(0, R_kalman).

    Parameters
    ----------
    A : ndarray (n, n)
        State transition matrix (continuous-time).
    B : ndarray (n, m)
        Input matrix.
    C : ndarray (p, n)
        Observation matrix.
    Q_lqr : ndarray (n, n)
        State-weighting matrix for LQR.
    R_lqr : ndarray (m, m)
        Control-weighting matrix for LQR.
    Q_kalman : ndarray (n, n)
        Process noise covariance (drives Kalman prediction uncertainty).
    R_kalman : ndarray (p, p)
        Measurement noise covariance.
    """

    def __init__(self, A, B, C, Q_lqr, R_lqr, Q_kalman, R_kalman):
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.C = np.array(C, dtype=float)

        self.Q_lqr = np.array(Q_lqr, dtype=float)
        self.R_lqr = np.array(R_lqr, dtype=float)
        self.Q_kalman = np.array(Q_kalman, dtype=float)
        self.R_kalman = np.array(R_kalman, dtype=float)

        n = self.A.shape[0]
        m = self.B.shape[1]
        p = self.C.shape[0]

        # ----- LQR gain -----
        # Solve the CARE:  A^T P + P A - P B R^{-1} B^T P + Q = 0
        P_lqr = self._solve_riccati_iterative(
            self.A, self.B, self.Q_lqr, self.R_lqr
        )
        # K = R^{-1} B^T P
        self.K = np.linalg.inv(self.R_lqr) @ self.B.T @ P_lqr   # (m x n)

        # ----- Kalman filter gain -----
        # The Kalman filter CARE is the dual of the LQR CARE:
        #   A P + P A^T - P C^T R_k^{-1} C P + Q_k = 0
        # We solve it by noting it has the same form with (A, B, Q, R)
        # replaced by (A^T, C^T, Q_k, R_k).
        P_kalman = self._solve_riccati_iterative(
            self.A.T, self.C.T, self.Q_kalman, self.R_kalman
        )
        # Kalman gain:  L = P C^T R_k^{-1}
        self.L = P_kalman @ self.C.T @ np.linalg.inv(self.R_kalman)  # (n x p)

        # ----- Estimator state -----
        self.x_hat = np.zeros(n)                # state estimate
        self.P_est = np.copy(self.Q_kalman)     # estimation covariance

    # ----- Riccati solver (shared with LQR file but kept self-contained) -----

    @staticmethod
    def _solve_riccati_iterative(A, B, Q, R, n_iter=1000, tol=1e-12):
        """
        Iterative solver for the continuous algebraic Riccati equation:

            A^T P + P A - P B R^{-1} B^T P + Q = 0

        Fixed-point iteration with step size delta:

            P_{k+1} = P_k + delta * (A^T P_k + P_k A
                                      - P_k B R^{-1} B^T P_k + Q)

        Convergence is guaranteed for stabilisable (A, B) with a
        sufficiently small delta.

        Parameters
        ----------
        A : ndarray (n, n)
        B : ndarray (n, m)
        Q : ndarray (n, n)   positive semi-definite
        R : ndarray (m, m)   positive definite
        n_iter : int
        tol : float

        Returns
        -------
        P : ndarray (n, n)
        """
        n = A.shape[0]
        P = np.copy(Q)
        R_inv = np.linalg.inv(R)
        BRinvBT = B @ R_inv @ B.T
        delta = 0.001

        for _ in range(n_iter):
            residual = A.T @ P + P @ A - P @ BRinvBT @ P + Q
            P_new = P + delta * residual
            P_new = 0.5 * (P_new + P_new.T)      # enforce symmetry

            if np.linalg.norm(residual, 'fro') < tol:
                return P_new
            P = P_new

        return P

    # ----- Public API -----

    def compute(self, y_measurement, dt):
        """
        Run one cycle of the LQG controller (Kalman update + LQR).

        Steps:
          1. **Predict** the state one step forward using the model.
          2. **Update** the estimate with the new measurement.
          3. **Compute** the control law u = -K x_hat.

        Parameters
        ----------
        y_measurement : ndarray (p,)
            Current sensor measurement vector.
        dt : float
            Timestep [s].

        Returns
        -------
        u : ndarray (m,)
            Control command (body torque).
        """
        y = np.asarray(y_measurement, dtype=float)

        # ---- Kalman Predict (propagate state and covariance) ----
        # Euler integration of the continuous model:
        #   x_hat_minus = x_hat + (A x_hat + B u_prev) * dt
        # (u_prev is the last applied control, stored implicitly in x_hat)
        x_hat_minus = self.x_hat + (self.A @ self.x_hat) * dt
        P_minus = (self.P_est
                   + (self.A @ self.P_est + self.P_est @ self.A.T
                      + self.Q_kalman) * dt)

        # ---- Kalman Update (correct with measurement) ----
        # Innovation
        y_pred = self.C @ x_hat_minus
        innovation = y - y_pred

        # Innovation covariance
        S = self.C @ P_minus @ self.C.T + self.R_kalman

        # Kalman gain (recomputed each step for the time-varying covariance)
        K_kalman = P_minus @ self.C.T @ np.linalg.inv(S)

        # State update
        self.x_hat = x_hat_minus + K_kalman @ innovation

        # Covariance update (Joseph form for numerical stability)
        I_n = np.eye(self.A.shape[0])
        temp = I_n - K_kalman @ self.C
        self.P_est = (temp @ P_minus @ temp.T
                      + K_kalman @ self.R_kalman @ K_kalman.T)

        # ---- LQR control from estimated state ----
        # u = -K x_hat   (separation principle: optimal estimator +
        #                  optimal regulator = optimal output feedback)
        u = -self.K @ self.x_hat
        return u

    def get_state_estimate(self):
        """Return a copy of the current state estimate."""
        return self.x_hat.copy()

    def get_estimation_covariance(self):
        """Return a copy of the estimation error covariance."""
        return self.P_est.copy()


# ===================================================================
# H-infinity Controller
# ===================================================================

class HInfinityController:
    """
    Simplified H-infinity (H-inf) state-feedback controller.

    H-infinity control minimises the worst-case (supremum over all
    frequencies) gain from the exogenous disturbance w to the regulated
    output z:

        ||T_{zw}||_inf < gamma

    where gamma is a user-specified upper bound on the closed-loop
    H-infinity norm.

    A *smaller* gamma demands more robustness but yields a more
    conservative (less performant) controller.  If gamma is chosen too
    small, no stabilising solution to the Riccati equation exists and the
    problem is infeasible.

    The state-feedback gain is obtained from a modified Riccati equation:

        A^T X + X A + Q + X (1/gamma^2 * I - B R^{-1} B^T) X = 0

    The control law is:

        u = -K_hinf x,    K_hinf = R^{-1} B^T X

    Parameters
    ----------
    A : ndarray (n, n)
        System state matrix.
    B : ndarray (n, m)
        Control input matrix.
    C : ndarray (p, n)
        Output matrix (defines the regulated output z = C x).
    gamma : float
        H-infinity norm bound.  Must be large enough for a solution to
        exist.  Typical starting value: 5.0 -- 10.0.
    """

    def __init__(self, A, B, C, gamma=5.0):
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.C = np.array(C, dtype=float)
        self.gamma = float(gamma)

        n = self.A.shape[0]
        m = self.B.shape[1]

        # Default weighting matrices
        # Q penalises regulated output energy:  Q = C^T C
        self.Q = self.C.T @ self.C
        # R penalises control effort (identity by default)
        self.R = np.eye(m)

        # Solve the modified Riccati equation
        X = self._solve_h_inf_riccati(
            self.A, self.B, self.C, self.Q, self.gamma
        )

        # H-inf state-feedback gain:  K = R^{-1} B^T X
        self.K_hinf = np.linalg.inv(self.R) @ self.B.T @ X   # (m x n)

    def _solve_h_inf_riccati(self, A, B, C, Q, gamma,
                              n_iter=2000, tol=1e-12):
        """
        Solve the H-infinity Riccati equation:

            A^T X + X A + Q + X (1/gamma^2 * I_n - B R^{-1} B^T) X = 0

        The term (1/gamma^2 * I - B R^{-1} B^T) competes: the I/gamma^2
        part destabilises (models worst-case disturbance), while
        B R^{-1} B^T stabilises (control authority).  A solution exists
        only if gamma is large enough.

        Uses the same fixed-point iteration as the standard CARE solver.

        Parameters
        ----------
        A, B, C : system matrices
        Q : ndarray (n, n)
        gamma : float

        Returns
        -------
        X : ndarray (n, n)
            Stabilising solution.

        Raises
        ------
        RuntimeError
            If the iteration diverges (gamma too small).
        """
        n = A.shape[0]
        m = B.shape[1]

        R_inv = np.linalg.inv(self.R)
        BRinvBT = B @ R_inv @ B.T

        # The combined matrix that replaces B R^{-1} B^T in the standard
        # CARE.  Negative eigenvalues of M indicate the disturbance
        # channel dominates -- which is the essence of H-inf design.
        #
        # M = B R^{-1} B^T  -  (1/gamma^2) I_n
        M = BRinvBT - (1.0 / gamma**2) * np.eye(n)

        X = np.copy(Q)
        delta = 0.0005       # smaller step for the harder H-inf Riccati

        for iteration in range(n_iter):
            # Residual of the H-inf CARE
            residual = A.T @ X + X @ A + Q - X @ M @ X
            X_new = X + delta * residual
            X_new = 0.5 * (X_new + X_new.T)

            res_norm = np.linalg.norm(residual, 'fro')

            # Divergence check: if X grows without bound, gamma is too small
            if np.any(np.isnan(X_new)) or np.linalg.norm(X_new, 'fro') > 1e15:
                raise RuntimeError(
                    f"H-inf Riccati diverged at iteration {iteration}.  "
                    f"gamma = {gamma:.3f} is likely too small for this plant.  "
                    f"Increase gamma and retry."
                )

            if res_norm < tol:
                return X_new

            X = X_new

        # Return best available solution even if not fully converged
        return X

    # ----- Public API -----

    def compute(self, state):
        """
        Compute the H-infinity control torque.

            u = -K_hinf x

        Parameters
        ----------
        state : ndarray (n,)
            Full state vector [theta_err (3), omega_err (3)].

        Returns
        -------
        u : ndarray (m,)
            Control torque [Nm].
        """
        x = np.asarray(state, dtype=float)
        u = -self.K_hinf @ x
        return u

    def set_gamma(self, gamma):
        """
        Update the robustness parameter gamma and recompute the gain.

        A smaller gamma increases robustness to disturbances but degrades
        nominal performance.  If gamma is too small, the Riccati solver
        will raise a RuntimeError.

        Parameters
        ----------
        gamma : float
            New H-infinity norm bound (must be positive).
        """
        self.gamma = float(gamma)
        X = self._solve_h_inf_riccati(
            self.A, self.B, self.C, self.Q, self.gamma
        )
        self.K_hinf = np.linalg.inv(self.R) @ self.B.T @ X

    def get_gain_matrix(self):
        """Return the H-inf gain matrix K_hinf."""
        return self.K_hinf.copy()


# ===================================================================
# MPC Controller
# ===================================================================

class MPCController:
    """
    Model Predictive Controller (MPC) for spacecraft attitude.

    At each timestep, MPC solves a finite-horizon quadratic programme:

        min   sum_{k=0}^{N-1} [ (x_k - x_ref)^T Q (x_k - x_ref)
                                + u_k^T R u_k ]
              + (x_N - x_ref)^T Q_f (x_N - x_ref)

        s.t.  x_{k+1} = A_d x_k + B_d u_k
              u_min <= u_k <= u_max        (torque limits)

    Only the *first* control action u_0 is applied; the problem is
    re-solved at the next timestep with updated state measurements.  This
    **receding-horizon** strategy provides implicit feedback and adapts to
    disturbances.

    In this simplified implementation, the QP is converted to an
    unconstrained least-squares problem (no inequality constraints on u).
    This makes the solution a direct matrix computation rather than
    requiring a QP solver.  Constraint handling can be added by
    substituting a proper QP solver (e.g. OSQP).

    Parameters
    ----------
    A : ndarray (n, n)
        Continuous-time state matrix.
    B : ndarray (n, m)
        Continuous-time input matrix.
    Q : ndarray (n, n)
        Stage state-weighting matrix.
    R : ndarray (m, m)
        Control-weighting matrix.
    N_horizon : int
        Prediction horizon length (number of steps).
    dt : float
        Discretisation timestep [s].
    u_max : float or None
        Symmetric torque saturation limit [Nm].  If None, unconstrained.
    """

    def __init__(self, A, B, Q, R, N_horizon=20, dt=1.0, u_max=None):
        self.A_c = np.array(A, dtype=float)
        self.B_c = np.array(B, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)
        self.N = int(N_horizon)
        self.dt = float(dt)
        self.u_max = u_max

        n = self.A_c.shape[0]
        m = self.B_c.shape[1]
        self.n = n
        self.m = m

        # Discretise via first-order Euler hold:
        #   A_d = I + A * dt
        #   B_d = B * dt
        # (For higher fidelity, use matrix exponential:
        #   A_d = expm(A*dt), B_d = integral_0^dt expm(A*s) ds * B )
        self.A_d = np.eye(n) + self.A_c * self.dt
        self.B_d = self.B_c * self.dt

        # Terminal cost:  Q_f = Q  (simple choice; a Lyapunov-based
        # terminal cost would give stronger stability guarantees)
        self.Q_f = np.copy(self.Q)

        # Pre-build the condensed QP matrices for the prediction horizon.
        # This is done once and reused at each control step.
        self._build_prediction_matrices()

    def _build_prediction_matrices(self):
        """
        Build the condensed (stacked) prediction matrices so that the
        full state trajectory X = [x_1, ..., x_N]^T can be written as:

            X = Phi * x_0  +  Gamma * U

        where U = [u_0, ..., u_{N-1}]^T is the stacked control sequence.

        Phi : (N*n, n)
        Gamma : (N*n, N*m)

        The QP cost can then be expressed as:

            J = X^T Q_bar X + U^T R_bar U
              = (Phi x0 + Gamma U - X_ref)^T Q_bar (...)  + U^T R_bar U

        which is a standard unconstrained least-squares in U.
        """
        n, m, N = self.n, self.m, self.N
        A_d, B_d = self.A_d, self.B_d

        # Phi: propagation of free (unforced) dynamics
        Phi = np.zeros((N * n, n))
        A_pow = np.eye(n)
        for k in range(N):
            A_pow = A_pow @ A_d     # A_d^{k+1}
            Phi[k * n:(k + 1) * n, :] = A_pow

        # Gamma: effect of controls on future states
        Gamma = np.zeros((N * n, N * m))
        for k in range(N):
            for j in range(k + 1):
                # x_{k+1} depends on u_j via A_d^{k-j} B_d
                row_start = k * n
                row_end = (k + 1) * n
                col_start = j * m
                col_end = (j + 1) * m

                power = k - j
                A_pow_kj = np.linalg.matrix_power(A_d, power)
                Gamma[row_start:row_end, col_start:col_end] = A_pow_kj @ B_d

        # Stacked cost matrices
        Q_bar = np.zeros((N * n, N * n))
        for k in range(N - 1):
            Q_bar[k * n:(k + 1) * n, k * n:(k + 1) * n] = self.Q
        # Terminal stage uses Q_f
        Q_bar[(N - 1) * n:N * n, (N - 1) * n:N * n] = self.Q_f

        R_bar = np.zeros((N * m, N * m))
        for k in range(N):
            R_bar[k * m:(k + 1) * m, k * m:(k + 1) * m] = self.R

        # Store for use in compute()
        self._Phi = Phi
        self._Gamma = Gamma
        self._Q_bar = Q_bar
        self._R_bar = R_bar

    def compute(self, x_current, x_reference):
        """
        Solve the MPC problem and return the first control action.

        The unconstrained solution is:

            U* = (Gamma^T Q_bar Gamma + R_bar)^{-1}
                 * Gamma^T Q_bar (X_ref - Phi x_0)

        Only u_0 (the first m entries of U*) is applied.

        Parameters
        ----------
        x_current : ndarray (n,)
            Current state measurement.
        x_reference : ndarray (n,)
            Desired reference state (constant over the horizon).

        Returns
        -------
        u : ndarray (m,)
            First control action to apply.
        """
        x0 = np.asarray(x_current, dtype=float)
        x_ref = np.asarray(x_reference, dtype=float)

        N, n, m = self.N, self.n, self.m
        Phi = self._Phi
        Gamma = self._Gamma
        Q_bar = self._Q_bar
        R_bar = self._R_bar

        # Build stacked reference: X_ref = [x_ref, x_ref, ..., x_ref]^T
        X_ref = np.tile(x_ref, N)

        # Free-response trajectory (no control)
        X_free = Phi @ x0

        # Error from reference under free response
        E = X_ref - X_free

        # ----- Unconstrained QP solution -----
        # Hessian:  H = Gamma^T Q_bar Gamma + R_bar
        H = Gamma.T @ Q_bar @ Gamma + R_bar

        # Gradient (negated):  f = Gamma^T Q_bar E
        f = Gamma.T @ Q_bar @ E

        # Optimal control sequence:  U* = H^{-1} f
        # Using least-squares for numerical robustness
        U_star, _, _, _ = np.linalg.lstsq(H, f, rcond=None)

        # Extract first control action (receding horizon)
        u = U_star[0:m]

        # ----- Optional: apply torque saturation -----
        # MPC can handle constraints explicitly.  Here we apply a simple
        # post-hoc clamp.  A proper implementation would add inequality
        # constraints to the QP and solve with an active-set or
        # interior-point method.
        if self.u_max is not None:
            u = np.clip(u, -self.u_max, self.u_max)

        return u

    def compute_trajectory(self, x_current, x_reference):
        """
        Solve the MPC problem and return the full predicted state and
        control trajectories (useful for visualisation / analysis).

        Parameters
        ----------
        x_current : ndarray (n,)
        x_reference : ndarray (n,)

        Returns
        -------
        X_traj : ndarray (N, n)
            Predicted state trajectory.
        U_traj : ndarray (N, m)
            Optimal control sequence.
        """
        x0 = np.asarray(x_current, dtype=float)
        x_ref = np.asarray(x_reference, dtype=float)

        N, n, m = self.N, self.n, self.m
        Phi = self._Phi
        Gamma = self._Gamma
        Q_bar = self._Q_bar
        R_bar = self._R_bar

        X_ref = np.tile(x_ref, N)
        X_free = Phi @ x0
        E = X_ref - X_free

        H = Gamma.T @ Q_bar @ Gamma + R_bar
        f = Gamma.T @ Q_bar @ E
        U_star, _, _, _ = np.linalg.lstsq(H, f, rcond=None)

        if self.u_max is not None:
            U_star = np.clip(U_star, -self.u_max, self.u_max)

        # Reconstruct state trajectory
        X_pred = Phi @ x0 + Gamma @ U_star

        X_traj = X_pred.reshape(N, n)
        U_traj = U_star.reshape(N, m)
        return X_traj, U_traj

    def set_horizon(self, N_horizon):
        """
        Change the prediction horizon and rebuild internal matrices.

        Parameters
        ----------
        N_horizon : int
            New prediction horizon length.
        """
        self.N = int(N_horizon)
        self._build_prediction_matrices()

    def set_weights(self, Q=None, R=None):
        """
        Update the cost matrices and rebuild internal matrices.

        Parameters
        ----------
        Q : ndarray (n, n) or None
        R : ndarray (m, m) or None
        """
        if Q is not None:
            self.Q = np.array(Q, dtype=float)
            self.Q_f = np.copy(self.Q)
        if R is not None:
            self.R = np.array(R, dtype=float)
        self._build_prediction_matrices()
