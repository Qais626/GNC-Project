"""
attitude_predictor.py - Neural Network Attitude State Prediction
================================================================

This module provides a feedforward neural network that predicts a spacecraft's
attitude state *N* time steps into the future given the current state (quaternion,
angular velocity) and known external torques.

Motivation
----------
Traditional attitude propagation uses numerical integration of Euler's
rotational equations of motion (typically with a 4th-order Runge-Kutta
integrator).  While highly accurate, RK4 propagation can be computationally
expensive when:

* The prediction horizon is long and the time step must be small for
  numerical stability (e.g. 0.001 s steps over 10 s = 10 000 evaluations).
* Many Monte-Carlo trajectories must be propagated for covariance analysis
  or fault detection.
* The flight computer is radiation-hardened and slow.

A trained neural network evaluates in *O(1)* -- a fixed number of matrix
multiplications independent of the horizon length -- making it orders of
magnitude faster than numerical integration for the same prediction interval.

The trade-off is accuracy: the NN is an *approximation*.  In practice it
would be used for rapid screening (e.g. "will we violate pointing
constraints in the next 10 s?") with full RK4 propagation reserved for
high-fidelity maneuver planning.

Architecture
------------
::

    Input(10)  -->  Dense(64, ReLU)  -->  Dense(32, ReLU)  -->  Output(7, linear)

Input vector (10 elements):
    q0, q1, q2, q3          -- attitude quaternion (scalar-first convention)
    wx, wy, wz              -- angular velocity in body frame (rad/s)
    Tx, Ty, Tz              -- known external torque in body frame (N-m)

Output vector (7 elements):
    q0', q1', q2', q3'      -- predicted future quaternion
    wx', wy', wz'           -- predicted future angular velocity

The network is implemented using ``SimpleNeuralNet`` from the sibling
``anomaly_detection`` module (pure NumPy, no deep-learning frameworks).

Training Data Generation
------------------------
We generate training pairs by propagating Euler's equations with RK4
from randomised initial conditions.  Each sample consists of:

    X = [q, omega, torque]       (input at t=0)
    y = [q_future, omega_future] (state at t=N*dt)

This is a supervised regression task.

References
----------
* Sidi, *Spacecraft Dynamics and Control*, Ch. 4 (Euler's equations).
* Crassidis & Junkins, *Optimal Estimation of Dynamic Systems*, App. F
  (quaternion kinematics).
* Izzo & Oezdemir, "Machine Learning and Evolutionary Techniques in
  Interplanetary Trajectory Design", Ch. 8.

Author : GNC Autonomy Team
Date   : 2025
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (safe for headless servers)
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
#  Import or inline the SimpleNeuralNet
# ---------------------------------------------------------------------------
# We attempt to import from the sibling module.  If that fails (e.g. the
# module has not been installed or the import path is not configured), we
# fall back to a self-contained copy.

try:
    from autonomy.anomaly_detection import SimpleNeuralNet
except ImportError:
    try:
        from anomaly_detection import SimpleNeuralNet
    except ImportError:
        # ---- Self-contained fallback (identical to anomaly_detection.py) ----
        class SimpleNeuralNet:  # type: ignore[no-redef]
            """Minimal feedforward neural network in pure NumPy.

            See ``anomaly_detection.SimpleNeuralNet`` for full documentation.
            This is a self-contained copy so that ``attitude_predictor.py``
            can be used standalone.
            """

            def __init__(
                self,
                layer_sizes: List[int],
                learning_rate: float = 0.001,
                seed: Optional[int] = None,
            ) -> None:
                self.layer_sizes = list(layer_sizes)
                self.learning_rate = learning_rate
                self.n_layers = len(layer_sizes) - 1

                rng = np.random.default_rng(seed)
                self.weights: List[np.ndarray] = []
                self.biases: List[np.ndarray] = []
                for i in range(self.n_layers):
                    n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
                    std = np.sqrt(2.0 / (n_in + n_out))
                    self.weights.append(rng.normal(0, std, (n_in, n_out)))
                    self.biases.append(np.zeros((1, n_out)))
                self._z_cache: List[np.ndarray] = []
                self._a_cache: List[np.ndarray] = []

            @staticmethod
            def _relu(z):
                return np.maximum(0.0, z)

            @staticmethod
            def _relu_derivative(z):
                return (z > 0).astype(z.dtype)

            def forward(self, X):
                self._z_cache, self._a_cache = [], [X]
                a = X
                for i in range(self.n_layers):
                    z = a @ self.weights[i] + self.biases[i]
                    self._z_cache.append(z)
                    a = self._relu(z) if i < self.n_layers - 1 else z
                    self._a_cache.append(a)
                return a

            def backward(self, y_true):
                bs = y_true.shape[0]
                y_pred = self._a_cache[-1]
                loss = 0.5 * np.mean((y_pred - y_true) ** 2)
                delta = (y_pred - y_true) / bs
                for i in reversed(range(self.n_layers)):
                    a_prev = self._a_cache[i]
                    dW = a_prev.T @ delta
                    db = np.sum(delta, axis=0, keepdims=True)
                    self.weights[i] -= self.learning_rate * dW
                    self.biases[i] -= self.learning_rate * db
                    if i > 0:
                        delta = (delta @ self.weights[i].T) * self._relu_derivative(
                            self._z_cache[i - 1]
                        )
                return float(loss)

            def predict(self, X):
                return self.forward(X)

            def get_weights(self):
                return [{"W": W.copy(), "b": b.copy()} for W, b in zip(self.weights, self.biases)]

            def set_weights(self, params):
                for i, p in enumerate(params):
                    self.weights[i] = p["W"].copy()
                    self.biases[i] = p["b"].copy()


# ---------------------------------------------------------------------------
#  Attitude Dynamics Utilities (for training data generation)
# ---------------------------------------------------------------------------

def _quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalise a quaternion to unit length.

    Quaternion convention: [q0, q1, q2, q3] where q0 is the scalar part.
    """
    norm = np.linalg.norm(q)
    if norm < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def _quat_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Compute the time derivative of the attitude quaternion.

    The quaternion kinematic equation is:

        dq/dt = 0.5 * Omega(omega) * q

    where Omega is the 4x4 skew-symmetric matrix built from the angular
    velocity vector omega = [wx, wy, wz]:

        Omega = [[ 0,   -wx, -wy, -wz],
                 [ wx,   0,   wz, -wy],
                 [ wy,  -wz,  0,   wx],
                 [ wz,   wy, -wx,  0 ]]

    Parameters
    ----------
    q : ndarray (4,) -- current quaternion [q0, q1, q2, q3]
    omega : ndarray (3,) -- angular velocity [wx, wy, wz] (rad/s)

    Returns
    -------
    ndarray (4,) -- dq/dt
    """
    wx, wy, wz = omega
    Omega = np.array([
        [0,   -wx, -wy, -wz],
        [wx,   0,   wz, -wy],
        [wy,  -wz,  0,   wx],
        [wz,   wy, -wx,  0],
    ])
    return 0.5 * Omega @ q


def _euler_equations(omega: np.ndarray, torque: np.ndarray, inertia: np.ndarray) -> np.ndarray:
    """Euler's rotational equations of motion.

    I * domega/dt = torque - omega x (I * omega)

    Parameters
    ----------
    omega : ndarray (3,) -- angular velocity in body frame
    torque : ndarray (3,) -- total external torque in body frame (N-m)
    inertia : ndarray (3,) -- principal moments of inertia [Ixx, Iyy, Izz]

    Returns
    -------
    ndarray (3,) -- domega/dt (angular acceleration)
    """
    I = inertia
    # omega x (I * omega) -- gyroscopic coupling term
    h = I * omega  # angular momentum (diagonal inertia)
    cross = np.array([
        omega[1] * h[2] - omega[2] * h[1],
        omega[2] * h[0] - omega[0] * h[2],
        omega[0] * h[1] - omega[1] * h[0],
    ])
    return (torque - cross) / I


def _rk4_step(
    q: np.ndarray,
    omega: np.ndarray,
    torque: np.ndarray,
    inertia: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate the attitude state by one RK4 step.

    This is the classical 4th-order Runge-Kutta integrator applied to the
    coupled quaternion kinematics + Euler equations system.

    Parameters
    ----------
    q : ndarray (4,) -- current quaternion
    omega : ndarray (3,) -- current angular velocity
    torque : ndarray (3,) -- external torque (assumed constant over dt)
    inertia : ndarray (3,) -- principal moments of inertia
    dt : float -- time step (s)

    Returns
    -------
    q_new : ndarray (4,) -- propagated quaternion (normalised)
    omega_new : ndarray (3,) -- propagated angular velocity
    """
    # k1
    dq1 = _quat_derivative(q, omega)
    dw1 = _euler_equations(omega, torque, inertia)

    # k2
    q2 = q + 0.5 * dt * dq1
    w2 = omega + 0.5 * dt * dw1
    dq2 = _quat_derivative(q2, w2)
    dw2 = _euler_equations(w2, torque, inertia)

    # k3
    q3 = q + 0.5 * dt * dq2
    w3 = omega + 0.5 * dt * dw2
    dq3 = _quat_derivative(q3, w3)
    dw3 = _euler_equations(w3, torque, inertia)

    # k4
    q4 = q + dt * dq3
    w4 = omega + dt * dw3
    dq4 = _quat_derivative(q4, w4)
    dw4 = _euler_equations(w4, torque, inertia)

    # Combine
    q_new = q + (dt / 6.0) * (dq1 + 2 * dq2 + 2 * dq3 + dq4)
    omega_new = omega + (dt / 6.0) * (dw1 + 2 * dw2 + 2 * dw3 + dw4)

    # Re-normalise quaternion to prevent drift
    q_new = _quat_normalize(q_new)

    return q_new, omega_new


def propagate_attitude(
    q0: np.ndarray,
    omega0: np.ndarray,
    torque: np.ndarray,
    inertia: np.ndarray,
    dt: float,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate attitude state N steps forward using RK4.

    Parameters
    ----------
    q0 : ndarray (4,) -- initial quaternion
    omega0 : ndarray (3,) -- initial angular velocity
    torque : ndarray (3,) -- constant external torque
    inertia : ndarray (3,) -- principal moments of inertia
    dt : float -- integration time step (s)
    n_steps : int -- number of steps to propagate

    Returns
    -------
    q_final : ndarray (4,)
    omega_final : ndarray (3,)
    """
    q = q0.copy()
    omega = omega0.copy()
    for _ in range(n_steps):
        q, omega = _rk4_step(q, omega, torque, inertia, dt)
    return q, omega


# ---------------------------------------------------------------------------
#  Attitude Predictor
# ---------------------------------------------------------------------------

class AttitudePredictor:
    """Neural network for rapid spacecraft attitude state prediction.

    Why Use a Neural Network?
    -------------------------
    Given input state x_0 = [q, omega, torque] at time t, we want to predict
    the output state y = [q_N, omega_N] at time t + N*dt.

    An RK4 propagator must evaluate the dynamics equations 4*N times.  The
    neural network evaluates a fixed set of matrix multiplications regardless
    of N -- this can be 100-1000x faster for large N.

    The accuracy is of course lower (typically ~0.1-1% relative error after
    training), but this is sufficient for many operational decisions:
    * Quick constraint-violation screening
    * Initial guess for optimisation-based maneuver planning
    * Real-time attitude estimation when the propagation budget is tight

    Architecture
    ------------
    Input(10) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Output(7, linear)

    The input is the 10-element state/torque vector; the output is the 7-
    element predicted future state.

    Data Normalisation
    ------------------
    Inputs and outputs are z-score normalised using training-set statistics
    to improve training stability.

    Parameters
    ----------
    input_dim : int
        Input dimensionality (default 10: 4 quat + 3 omega + 3 torque).
    output_dim : int
        Output dimensionality (default 7: 4 quat + 3 omega).
    hidden_dims : tuple of int
        Hidden layer sizes (default (64, 32)).
    prediction_horizon : int
        Number of RK4 steps into the future that the network predicts
        (used for training data generation).
    dt : float
        Integration time step for the RK4 propagator (seconds).
    learning_rate : float
        Gradient descent step size.
    inertia : ndarray (3,) or None
        Principal moments of inertia for training data generation.
        If None, defaults to a typical small satellite.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 7,
        hidden_dims: Tuple[int, ...] = (64, 32),
        prediction_horizon: int = 10,
        dt: float = 0.1,
        learning_rate: float = 0.001,
        inertia: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.prediction_horizon = prediction_horizon
        self.dt = dt
        self._seed = seed

        # Principal moments of inertia (default: ~10 kg small satellite)
        self.inertia = (
            np.asarray(inertia, dtype=np.float64)
            if inertia is not None
            else np.array([0.04, 0.06, 0.08])  # kg*m^2
        )

        # Build network
        layer_sizes = [input_dim] + list(hidden_dims) + [output_dim]
        self.network = SimpleNeuralNet(
            layer_sizes=layer_sizes,
            learning_rate=learning_rate,
            seed=seed,
        )

        # Normalisation statistics (fitted during training)
        self._X_mean: Optional[np.ndarray] = None
        self._X_std: Optional[np.ndarray] = None
        self._y_mean: Optional[np.ndarray] = None
        self._y_std: Optional[np.ndarray] = None

        # Training history
        self.loss_history: List[float] = []

    # ---- Normalisation helpers ------------------------------------------------

    def _fit_normaliser(self, X: np.ndarray, y: np.ndarray) -> None:
        """Compute z-score normalisation parameters from training data."""
        self._X_mean = np.mean(X, axis=0)
        self._X_std = np.std(X, axis=0)
        self._X_std[self._X_std < 1e-12] = 1.0

        self._y_mean = np.mean(y, axis=0)
        self._y_std = np.std(y, axis=0)
        self._y_std[self._y_std < 1e-12] = 1.0

    def _normalise_X(self, X: np.ndarray) -> np.ndarray:
        return (X - self._X_mean) / self._X_std

    def _normalise_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self._y_mean) / self._y_std

    def _denormalise_y(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * self._y_std + self._y_mean

    # ---- Training data generation --------------------------------------------

    def generate_training_data(
        self,
        num_samples: int = 10000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training pairs from attitude dynamics.

        For each sample we:
        1. Draw a random unit quaternion (uniformly distributed on SO(3)).
        2. Draw a random angular velocity from a reasonable range.
        3. Draw a random external torque from a reasonable range.
        4. Propagate the attitude state ``prediction_horizon`` steps
           forward using the RK4 integrator.
        5. Record (input, output) = ([q, omega, torque], [q_future, omega_future]).

        Parameters
        ----------
        num_samples : int
            Number of training pairs to generate.

        Returns
        -------
        X : ndarray (num_samples, 10)
        y : ndarray (num_samples, 7)
        """
        rng = np.random.default_rng(self._seed)

        X = np.zeros((num_samples, self.input_dim))
        y = np.zeros((num_samples, self.output_dim))

        for i in range(num_samples):
            # Random unit quaternion (uniformly distributed on the 3-sphere).
            # Method: draw 4 Gaussian samples and normalise.
            q0 = rng.standard_normal(4)
            q0 = _quat_normalize(q0)

            # Random angular velocity: typical small-sat rates up to ~5 deg/s
            omega0 = rng.uniform(-0.1, 0.1, size=3)  # rad/s

            # Random external torque: small disturbance torques
            torque = rng.uniform(-0.001, 0.001, size=3)  # N-m

            # Propagate forward
            q_future, omega_future = propagate_attitude(
                q0, omega0, torque, self.inertia, self.dt, self.prediction_horizon
            )

            # Assemble input and output
            X[i] = np.concatenate([q0, omega0, torque])
            y[i] = np.concatenate([q_future, omega_future])

        return X, y

    # ---- Training ------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 200,
        batch_size: int = 32,
        verbose: bool = True,
        print_every: int = 20,
    ) -> List[float]:
        """Train the attitude predictor network.

        Parameters
        ----------
        X_train : ndarray (n_samples, input_dim)
        y_train : ndarray (n_samples, output_dim)
        epochs : int
        batch_size : int
        verbose : bool
        print_every : int

        Returns
        -------
        list of float
            Per-epoch average loss.
        """
        assert X_train.shape[1] == self.input_dim
        assert y_train.shape[1] == self.output_dim

        n_samples = X_train.shape[0]
        rng = np.random.default_rng(self._seed)

        # Fit and apply normalisation
        self._fit_normaliser(X_train, y_train)
        X_norm = self._normalise_X(X_train)
        y_norm = self._normalise_y(y_train)

        self.loss_history = []
        t_start = time.time()

        for epoch in range(epochs):
            indices = rng.permutation(n_samples)
            X_shuf = X_norm[indices]
            y_shuf = y_norm[indices]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                X_batch = X_shuf[start : start + batch_size]
                y_batch = y_shuf[start : start + batch_size]

                self.network.forward(X_batch)
                batch_loss = self.network.backward(y_batch)

                epoch_loss += batch_loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.loss_history.append(avg_loss)

            if verbose and (epoch % print_every == 0 or epoch == epochs - 1):
                print(
                    f"  [AttPredictor] Epoch {epoch:4d}/{epochs}  "
                    f"loss = {avg_loss:.6f}"
                )

        elapsed = time.time() - t_start
        if verbose:
            print(f"  [AttPredictor] Training complete in {elapsed:.1f}s")

        return self.loss_history

    # ---- Prediction ----------------------------------------------------------

    def predict(self, current_state: np.ndarray) -> np.ndarray:
        """Predict the future attitude state.

        Parameters
        ----------
        current_state : ndarray of shape (10,) or (n, 10)
            [q0, q1, q2, q3, wx, wy, wz, Tx, Ty, Tz]

        Returns
        -------
        ndarray of shape (7,) or (n, 7)
            [q0', q1', q2', q3', wx', wy', wz'] at t + N*dt
        """
        single = current_state.ndim == 1
        X = np.atleast_2d(current_state)

        if self._X_mean is None:
            raise RuntimeError("Model has not been trained.  Call train() first.")

        X_norm = self._normalise_X(X)
        y_norm = self.network.predict(X_norm)
        y = self._denormalise_y(y_norm)

        # Re-normalise the quaternion part of the output to ensure it is a
        # valid unit quaternion.
        for i in range(y.shape[0]):
            y[i, :4] = _quat_normalize(y[i, :4])

        return y[0] if single else y

    # ---- Evaluation ----------------------------------------------------------

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate prediction accuracy on a test set.

        Returns
        -------
        dict with keys:
            mse          -- Mean Squared Error over all outputs
            mae          -- Mean Absolute Error over all outputs
            max_error    -- Maximum absolute error across all samples/outputs
            quat_mse     -- MSE on quaternion components only
            omega_mse    -- MSE on angular velocity components only
        """
        y_pred = self.predict(X_test)
        residuals = y_test - y_pred

        mse = float(np.mean(residuals ** 2))
        mae = float(np.mean(np.abs(residuals)))
        max_err = float(np.max(np.abs(residuals)))

        quat_mse = float(np.mean(residuals[:, :4] ** 2))
        omega_mse = float(np.mean(residuals[:, 4:] ** 2))

        return {
            "mse": mse,
            "mae": mae,
            "max_error": max_err,
            "quat_mse": quat_mse,
            "omega_mse": omega_mse,
        }

    # ---- Comparison with RK4 propagator --------------------------------------

    def compare_with_propagator(
        self,
        test_cases: Optional[np.ndarray] = None,
        num_cases: int = 100,
    ) -> "pd.DataFrame":
        """Compare ML prediction speed and accuracy against RK4 propagation.

        For each test case we run both the neural network and the RK4
        propagator and record:
        * Prediction error (MSE between NN output and RK4 ground truth)
        * Wall-clock time for each method

        Parameters
        ----------
        test_cases : ndarray (n, 10) or None
            If None, generate random test cases.
        num_cases : int
            Number of test cases to generate (if test_cases is None).

        Returns
        -------
        pd.DataFrame
            Columns: nn_mse, rk4_time_us, nn_time_us, speedup
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas is required for compare_with_propagator()")

        rng = np.random.default_rng(self._seed)

        if test_cases is None:
            # Generate random test inputs
            test_cases = np.zeros((num_cases, self.input_dim))
            for i in range(num_cases):
                q = _quat_normalize(rng.standard_normal(4))
                omega = rng.uniform(-0.1, 0.1, size=3)
                torque = rng.uniform(-0.001, 0.001, size=3)
                test_cases[i] = np.concatenate([q, omega, torque])

        results = []
        for i in range(test_cases.shape[0]):
            x = test_cases[i]
            q0, omega0, torque = x[:4], x[4:7], x[7:10]

            # --- RK4 propagation ---
            t0 = time.perf_counter()
            q_rk4, omega_rk4 = propagate_attitude(
                q0, omega0, torque, self.inertia, self.dt, self.prediction_horizon
            )
            rk4_time = (time.perf_counter() - t0) * 1e6  # microseconds

            y_rk4 = np.concatenate([q_rk4, omega_rk4])

            # --- Neural network prediction ---
            t0 = time.perf_counter()
            y_nn = self.predict(x)
            nn_time = (time.perf_counter() - t0) * 1e6

            # Error
            nn_mse = float(np.mean((y_nn - y_rk4) ** 2))
            speedup = rk4_time / max(nn_time, 0.01)

            results.append({
                "nn_mse": nn_mse,
                "rk4_time_us": rk4_time,
                "nn_time_us": nn_time,
                "speedup": speedup,
            })

        return pd.DataFrame(results)

    # ---- Plotting utilities --------------------------------------------------

    def plot_training_curve(
        self,
        filepath: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """Plot loss vs epoch (training curve).

        Parameters
        ----------
        filepath : str or None
            If provided, save the figure to this path (PNG, PDF, etc.).
        show : bool
            If True, call plt.show() (only works in interactive environments).
        """
        if not _HAS_MATPLOTLIB:
            print("  [AttPredictor] matplotlib not available; skipping plot.")
            return
        if not self.loss_history:
            print("  [AttPredictor] No training history to plot.")
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.loss_history, linewidth=1.5, color="steelblue")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss (normalised)")
        ax.set_title("Attitude Predictor - Training Curve")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        if filepath:
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            print(f"  [AttPredictor] Training curve saved to {filepath}")
        if show:
            plt.show()
        plt.close(fig)

    def plot_prediction_comparison(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        num_samples: int = 5,
        filepath: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """Plot predicted vs true output for a few test samples.

        Creates a figure with 7 subplots (one per output dimension) showing
        the true and predicted values as scatter points.

        Parameters
        ----------
        X_test, y_test : test data arrays
        num_samples : int -- number of samples to highlight
        filepath : str or None
        show : bool
        """
        if not _HAS_MATPLOTLIB:
            print("  [AttPredictor] matplotlib not available; skipping plot.")
            return

        y_pred = self.predict(X_test)
        labels = ["q0", "q1", "q2", "q3", "wx", "wy", "wz"]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for j in range(self.output_dim):
            ax = axes[j]
            ax.scatter(
                y_test[:, j], y_pred[:, j],
                s=4, alpha=0.3, color="steelblue", label="all"
            )
            # Perfect prediction line
            lims = [
                min(y_test[:, j].min(), y_pred[:, j].min()),
                max(y_test[:, j].max(), y_pred[:, j].max()),
            ]
            ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
            ax.set_xlabel(f"True {labels[j]}")
            ax.set_ylabel(f"Pred {labels[j]}")
            ax.set_title(labels[j])
            ax.grid(True, alpha=0.2)

        # Hide the unused 8th subplot
        axes[7].set_visible(False)

        fig.suptitle("Attitude Predictor - Predicted vs True", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if filepath:
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            print(f"  [AttPredictor] Comparison plot saved to {filepath}")
        if show:
            plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
#  Demonstration / self-test
# ---------------------------------------------------------------------------

def _demo() -> None:
    """Full demonstration: generate data, train, evaluate, and compare."""
    print("=" * 70)
    print("  Attitude Predictor - Neural Network Demo")
    print("=" * 70)

    predictor = AttitudePredictor(
        input_dim=10,
        output_dim=7,
        hidden_dims=(64, 32),
        prediction_horizon=10,
        dt=0.1,
        learning_rate=0.001,
        seed=42,
    )

    # --- Generate training and test data ---
    print("\n--- Generating training data (RK4 propagation) ---")
    t0 = time.time()
    X_train, y_train = predictor.generate_training_data(num_samples=5000)
    print(f"  Generated {X_train.shape[0]} training samples in {time.time()-t0:.1f}s")
    print(f"  X shape: {X_train.shape},  y shape: {y_train.shape}")

    t0 = time.time()
    X_test, y_test = predictor.generate_training_data(num_samples=1000)
    print(f"  Generated {X_test.shape[0]} test samples in {time.time()-t0:.1f}s")

    # --- Train ---
    print("\n--- Training ---")
    predictor.train(X_train, y_train, epochs=100, batch_size=64, verbose=True, print_every=20)

    # --- Evaluate ---
    print("\n--- Evaluation on test set ---")
    metrics = predictor.evaluate(X_test, y_test)
    for key, val in metrics.items():
        print(f"  {key:15s}: {val:.8f}")

    # --- Compare with RK4 ---
    if _HAS_PANDAS:
        print("\n--- Speed comparison: NN vs RK4 (50 test cases) ---")
        df = predictor.compare_with_propagator(num_cases=50)
        print(f"  Average NN MSE:     {df['nn_mse'].mean():.8f}")
        print(f"  Median RK4 time:    {df['rk4_time_us'].median():.1f} us")
        print(f"  Median NN time:     {df['nn_time_us'].median():.1f} us")
        print(f"  Median speedup:     {df['speedup'].median():.1f}x")
        print(f"\n  Full comparison DataFrame:\n{df.describe().to_string()}")
    else:
        print("\n  [pandas not available; skipping speed comparison]")

    # --- Plot training curve ---
    predictor.plot_training_curve(filepath=None, show=False)
    print("\n  (Training curve plot generated; pass filepath to save.)")

    # --- Single prediction example ---
    print("\n--- Single prediction example ---")
    # Random initial state
    q0 = _quat_normalize(np.array([1.0, 0.0, 0.0, 0.0]))
    omega0 = np.array([0.01, -0.02, 0.005])
    torque = np.array([0.0001, -0.0002, 0.00005])
    x = np.concatenate([q0, omega0, torque])

    y_nn = predictor.predict(x)
    q_rk4, omega_rk4 = propagate_attitude(
        q0, omega0, torque, predictor.inertia, predictor.dt, predictor.prediction_horizon
    )
    y_rk4 = np.concatenate([q_rk4, omega_rk4])

    print(f"  Input state:      {x}")
    print(f"  NN prediction:    {y_nn}")
    print(f"  RK4 ground truth: {y_rk4}")
    print(f"  Absolute error:   {np.abs(y_nn - y_rk4)}")
    print(f"  MSE:              {np.mean((y_nn - y_rk4)**2):.10f}")

    print("\n" + "=" * 70)
    print("  Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
