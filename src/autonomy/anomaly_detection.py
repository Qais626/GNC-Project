"""
anomaly_detection.py - ML-Based Anomaly Detection for Spacecraft Sensor Health
===============================================================================

This module provides two complementary anomaly-detection strategies for
monitoring spacecraft sensor telemetry in real time:

1. **SensorAnomalyDetector** (autoencoder approach)
   - Trains a small autoencoder on *nominal* (healthy) sensor data.
   - At inference time the reconstruction error is computed; if it exceeds
     a learned threshold the reading is flagged as anomalous.
   - The autoencoder is implemented entirely in NumPy (no PyTorch /
     TensorFlow) so the code is self-contained and transparent.

2. **StatisticalAnomalyDetector** (Mahalanobis distance approach)
   - Maintains a running mean and covariance of the sensor vector.
   - Uses the Mahalanobis distance and a chi-squared threshold to decide
     whether a new reading is anomalous.
   - Lighter-weight than the autoencoder and requires no training phase;
     useful as a first-pass filter or as a fallback.

Both detectors return a tuple ``(is_anomaly: bool, score: float)`` so they
share a common interface and can be swapped transparently.

References
----------
* Goodfellow, Bengio, Courville - *Deep Learning*, Ch. 14 (Autoencoders)
* De Maesschalck et al., "The Mahalanobis distance", *Chemometrics and
  Intelligent Laboratory Systems*, 2000.
* Chandola, Banerjee, Kumar - "Anomaly Detection: A Survey", *ACM Computing
  Surveys*, 2009.

Author : GNC Autonomy Team
Date   : 2025
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
#  Helper: Pure-NumPy Neural Network
# ---------------------------------------------------------------------------

class SimpleNeuralNet:
    """A minimal fully-connected feedforward neural network in pure NumPy.

    This class exists so that the autoencoder can be built without any deep-
    learning framework.  It supports:

    * Arbitrary layer sizes (constructor takes a list of ints).
    * Xavier / Glorot weight initialisation.
    * Forward pass with ReLU activations on hidden layers and a *linear*
      output layer (appropriate for regression / reconstruction tasks).
    * Back-propagation via the chain rule with vanilla gradient descent.

    Why pure NumPy?
    ---------------
    On a spacecraft flight computer we cannot assume that PyTorch or
    TensorFlow are available.  A small NumPy network can be compiled with
    Cython or Numba and run on radiation-hardened processors.  It also makes
    every matrix operation auditable, which is important for mission-critical
    software.

    Parameters
    ----------
    layer_sizes : list of int
        Number of neurons in each layer, e.g. ``[n_inputs, 64, 16, 64, n_outputs]``.
    learning_rate : float, optional
        Step size for gradient descent (default 0.001).
    seed : int or None, optional
        Random seed for reproducibility of weight initialisation.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        learning_rate: float = 0.001,
        seed: Optional[int] = None,
    ) -> None:
        self.layer_sizes = list(layer_sizes)
        self.learning_rate = learning_rate
        self.n_layers = len(layer_sizes) - 1  # number of weight matrices

        rng = np.random.default_rng(seed)

        # ----- Xavier / Glorot initialisation -----
        # For a layer connecting n_in -> n_out the weights are drawn from
        # N(0, sqrt(2 / (n_in + n_out))).  This keeps the variance of
        # activations roughly constant across layers, preventing vanishing
        # or exploding gradients in moderately deep networks.
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(self.n_layers):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            std = np.sqrt(2.0 / (n_in + n_out))
            W = rng.normal(0.0, std, size=(n_in, n_out))
            b = np.zeros((1, n_out))
            self.weights.append(W)
            self.biases.append(b)

        # Caches filled during forward pass for use in back-propagation.
        self._z_cache: List[np.ndarray] = []   # pre-activation values
        self._a_cache: List[np.ndarray] = []   # post-activation values

    # ---- Activation functions ------------------------------------------------

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit: max(0, z).

        ReLU is the most common hidden-layer activation because it is cheap
        to compute and does not saturate for positive inputs, which helps
        gradient flow during back-propagation.
        """
        return np.maximum(0.0, z)

    @staticmethod
    def _relu_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU.  d/dz max(0, z) = 1 if z > 0 else 0.

        Technically the derivative is undefined at z = 0, but by convention
        we set it to 0 there (sub-gradient).
        """
        return (z > 0).astype(z.dtype)

    # ---- Forward pass --------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the network output for a batch of inputs.

        Parameters
        ----------
        X : ndarray of shape (batch_size, n_inputs)

        Returns
        -------
        ndarray of shape (batch_size, n_outputs)

        Notes
        -----
        Hidden layers use ReLU; the output layer is linear (no activation).
        This is standard for regression / reconstruction targets that can
        take arbitrary real values.
        """
        self._z_cache = []
        self._a_cache = [X]  # a[0] = input

        a = X
        for i in range(self.n_layers):
            # z = a @ W + b   (affine transformation)
            z = a @ self.weights[i] + self.biases[i]
            self._z_cache.append(z)

            if i < self.n_layers - 1:
                # Hidden layer -> ReLU activation
                a = self._relu(z)
            else:
                # Output layer -> linear (identity) activation
                a = z

            self._a_cache.append(a)

        return a

    # ---- Back-propagation ----------------------------------------------------

    def backward(self, y_true: np.ndarray) -> float:
        """Run back-propagation and update weights via gradient descent.

        Uses Mean Squared Error (MSE) as the loss function:

            L = (1 / 2N) * sum((y_pred - y_true)^2)

        The factor of 1/2 simplifies the gradient to (y_pred - y_true) / N.

        Parameters
        ----------
        y_true : ndarray of shape (batch_size, n_outputs)
            Target / ground-truth values.

        Returns
        -------
        float
            The scalar MSE loss for this batch (before the weight update).

        Algorithm (chain rule)
        ----------------------
        Starting from the output layer and moving backward:

        1.  delta_L = dL/dz_L = (a_L - y_true) / N        (linear output)
        2.  For hidden layer l (going backward):
                delta_l = (delta_{l+1} @ W_{l+1}^T) * relu'(z_l)
        3.  Gradients for each layer's weights and biases:
                dW_l = a_{l-1}^T @ delta_l
                db_l = sum(delta_l, axis=0)
        4.  Update:  W_l -= lr * dW_l,  b_l -= lr * db_l
        """
        batch_size = y_true.shape[0]
        y_pred = self._a_cache[-1]

        # MSE loss (for reporting)
        loss = 0.5 * np.mean((y_pred - y_true) ** 2)

        # ---- Output layer delta ----
        # dL/dz_L = (y_pred - y_true) / N   (linear output, MSE loss)
        delta = (y_pred - y_true) / batch_size

        # ---- Propagate backward through layers ----
        for i in reversed(range(self.n_layers)):
            a_prev = self._a_cache[i]  # activation of layer before this one

            # Gradients for this layer's parameters
            dW = a_prev.T @ delta
            db = np.sum(delta, axis=0, keepdims=True)

            # Update parameters (vanilla SGD)
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

            # Propagate delta to the previous layer (if not at input layer)
            if i > 0:
                delta = (delta @ self.weights[i].T) * self._relu_derivative(
                    self._z_cache[i - 1]
                )

        return float(loss)

    # ---- Convenience ---------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Alias for ``forward`` that makes intent clearer at call sites."""
        return self.forward(X)

    def get_weights(self) -> List[Dict[str, np.ndarray]]:
        """Return a copy of all weights and biases (for serialisation)."""
        return [
            {"W": W.copy(), "b": b.copy()}
            for W, b in zip(self.weights, self.biases)
        ]

    def set_weights(self, params: List[Dict[str, np.ndarray]]) -> None:
        """Restore weights and biases from a previously saved snapshot."""
        for i, p in enumerate(params):
            self.weights[i] = p["W"].copy()
            self.biases[i] = p["b"].copy()


# ---------------------------------------------------------------------------
#  Autoencoder-Based Anomaly Detector
# ---------------------------------------------------------------------------

@dataclass
class AnomalyResult:
    """Container for a single anomaly-detection result."""
    is_anomaly: bool
    score: float                # reconstruction error or Mahalanobis distance
    threshold: float            # threshold used for this decision
    details: Dict = field(default_factory=dict)


class SensorAnomalyDetector:
    """Autoencoder-based anomaly detector for spacecraft sensor telemetry.

    How It Works
    ------------
    An *autoencoder* is a neural network trained to reconstruct its own input.
    The network has a "bottleneck" hidden layer that is much smaller than the
    input, forcing it to learn a compressed representation of the data.

    When trained on *nominal* (healthy) sensor readings the autoencoder
    learns the manifold of normal behaviour.  If a new reading comes from a
    different distribution (e.g. a failing gyroscope, a stuck thruster valve)
    the reconstruction will be poor and the *reconstruction error* will be
    large.  We compare this error to a threshold derived from the training
    data to make an anomaly decision.

    Architecture
    ------------
    ::

        Input(n_sensors)  -->  Dense(64, ReLU)
                          -->  Dense(hidden_dim, ReLU)   [bottleneck]
                          -->  Dense(64, ReLU)
                          -->  Output(n_sensors, linear)

    The bottleneck dimension (default 16) controls the compression ratio.
    A smaller bottleneck forces the network to keep only the most important
    features, improving anomaly separation at the cost of reconstruction
    fidelity on nominal data.

    Threshold Selection
    -------------------
    After training we compute the reconstruction error on the training set
    and set the threshold to ``mean + k * std`` of these errors, where *k*
    is typically 3 (corresponding to ~99.7% of a Gaussian distribution).
    The threshold can later be adapted with ``update_threshold``.

    Parameters
    ----------
    n_sensors : int
        Dimensionality of the sensor vector.
    hidden_dim : int
        Size of the bottleneck layer (default 16).
    learning_rate : float
        Gradient descent step size (default 0.001).
    threshold_sigma : float
        Number of standard deviations above mean training error for the
        anomaly threshold (default 3.0).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_sensors: int,
        hidden_dim: int = 16,
        learning_rate: float = 0.001,
        threshold_sigma: float = 3.0,
        seed: Optional[int] = None,
    ) -> None:
        self.n_sensors = n_sensors
        self.hidden_dim = hidden_dim
        self.threshold_sigma = threshold_sigma
        self._seed = seed

        # Build the autoencoder network:
        #   Input -> 64 -> hidden_dim -> 64 -> Output
        self.network = SimpleNeuralNet(
            layer_sizes=[n_sensors, 64, hidden_dim, 64, n_sensors],
            learning_rate=learning_rate,
            seed=seed,
        )

        # Anomaly threshold (set during training)
        self._threshold: float = float("inf")
        self._train_error_mean: float = 0.0
        self._train_error_std: float = 1.0

        # Data normalisation parameters (z-score)
        self._data_mean: Optional[np.ndarray] = None
        self._data_std: Optional[np.ndarray] = None

        # Training history
        self.loss_history: List[float] = []

    # ---- Data normalisation helpers ----------------------------------------

    def _fit_normaliser(self, data: np.ndarray) -> None:
        """Compute per-feature mean and std for z-score normalisation.

        Normalising the input is important because sensor channels may have
        wildly different scales (e.g. temperature in Kelvin vs angular rate
        in rad/s).  Without normalisation the autoencoder would be dominated
        by the largest-magnitude features.
        """
        self._data_mean = np.mean(data, axis=0)
        self._data_std = np.std(data, axis=0)
        # Avoid division by zero for constant features.
        self._data_std[self._data_std < 1e-12] = 1.0

    def _normalise(self, data: np.ndarray) -> np.ndarray:
        """Apply z-score normalisation using stored statistics."""
        if self._data_mean is None or self._data_std is None:
            raise RuntimeError("Normaliser has not been fitted yet.  Call train() first.")
        return (data - self._data_mean) / self._data_std

    # ---- Training -----------------------------------------------------------

    def train(
        self,
        nominal_data: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> List[float]:
        """Train the autoencoder on nominal (healthy) sensor data.

        Parameters
        ----------
        nominal_data : ndarray of shape (n_samples, n_sensors)
            Matrix of sensor readings that are known to be nominal.
        epochs : int
            Number of full passes over the training set.
        batch_size : int
            Mini-batch size for stochastic gradient descent.
        verbose : bool
            If True, print loss every 10 epochs.

        Returns
        -------
        list of float
            Per-epoch average loss values (training curve).

        Training Procedure
        ------------------
        1. Normalise the data (z-score).
        2. For each epoch, shuffle the data and iterate over mini-batches.
        3. For each batch run forward -> backward -> update weights.
        4. After training, compute reconstruction errors on the full training
           set and derive the anomaly threshold.
        """
        if nominal_data.ndim == 1:
            nominal_data = nominal_data.reshape(1, -1)
        assert nominal_data.shape[1] == self.n_sensors, (
            f"Expected {self.n_sensors} sensor channels, got {nominal_data.shape[1]}"
        )

        n_samples = nominal_data.shape[0]
        rng = np.random.default_rng(self._seed)

        # Step 1 - fit and apply normalisation
        self._fit_normaliser(nominal_data)
        X = self._normalise(nominal_data)

        self.loss_history = []

        # Step 2 - training loop
        for epoch in range(epochs):
            # Shuffle training data each epoch to reduce variance of SGD
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                X_batch = X_shuffled[start : start + batch_size]

                # Forward pass: reconstruct input from itself
                _ = self.network.forward(X_batch)

                # Backward pass: minimise MSE between input and reconstruction
                batch_loss = self.network.backward(X_batch)

                epoch_loss += batch_loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.loss_history.append(avg_loss)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"  [AnomalyDetector] Epoch {epoch:4d}/{epochs}  loss = {avg_loss:.6f}")

        # Step 3 - set anomaly threshold from training reconstruction errors
        self._compute_threshold(X)

        if verbose:
            print(f"  [AnomalyDetector] Training complete.  Threshold = {self._threshold:.6f}")

        return self.loss_history

    def _compute_threshold(self, X_normalised: np.ndarray) -> None:
        """Derive the anomaly threshold from training-set reconstruction errors.

        We compute the per-sample MSE, then set:

            threshold = mean(errors) + threshold_sigma * std(errors)

        This is a simple but effective heuristic: under the assumption that
        reconstruction errors on nominal data are approximately Gaussian,
        ``threshold_sigma = 3`` corresponds to a false-positive rate of
        roughly 0.3%.
        """
        reconstructed = self.network.predict(X_normalised)
        errors = np.mean((X_normalised - reconstructed) ** 2, axis=1)

        self._train_error_mean = float(np.mean(errors))
        self._train_error_std = float(np.std(errors))

        self._threshold = (
            self._train_error_mean + self.threshold_sigma * self._train_error_std
        )

    # ---- Inference -----------------------------------------------------------

    def detect(self, sensor_reading: np.ndarray) -> Tuple[bool, float]:
        """Determine whether a sensor reading is anomalous.

        Parameters
        ----------
        sensor_reading : ndarray of shape (n_sensors,) or (n_samples, n_sensors)
            One or more sensor readings.

        Returns
        -------
        is_anomaly : bool
            True if the reconstruction error exceeds the threshold.
        anomaly_score : float
            The reconstruction MSE (higher = more anomalous).
            For batches this is the *maximum* score in the batch.
        """
        x = np.atleast_2d(sensor_reading)
        x_norm = self._normalise(x)
        x_recon = self.network.predict(x_norm)

        # Per-sample reconstruction error
        errors = np.mean((x_norm - x_recon) ** 2, axis=1)

        # Take the worst-case error in the batch
        score = float(np.max(errors))
        is_anomaly = score > self._threshold

        return is_anomaly, score

    def detect_detailed(self, sensor_reading: np.ndarray) -> AnomalyResult:
        """Like ``detect`` but returns an ``AnomalyResult`` with extra info."""
        x = np.atleast_2d(sensor_reading)
        x_norm = self._normalise(x)
        x_recon = self.network.predict(x_norm)

        per_sample_error = np.mean((x_norm - x_recon) ** 2, axis=1)
        per_feature_error = np.mean((x_norm - x_recon) ** 2, axis=0)

        score = float(np.max(per_sample_error))
        return AnomalyResult(
            is_anomaly=(score > self._threshold),
            score=score,
            threshold=self._threshold,
            details={
                "per_sample_errors": per_sample_error.tolist(),
                "per_feature_errors": per_feature_error.tolist(),
                "reconstruction": x_recon.tolist(),
            },
        )

    # ---- Threshold management -----------------------------------------------

    def get_threshold(self) -> float:
        """Return the current anomaly decision threshold."""
        return self._threshold

    def update_threshold(self, new_nominal_data: np.ndarray) -> float:
        """Adaptively adjust the threshold using new nominal data.

        This is useful during long missions where the "normal" sensor profile
        may drift slowly (e.g. thermal environment changes as the spacecraft
        moves along its orbit).

        The new threshold is a weighted combination of the old and new
        statistics (exponential moving average with alpha = 0.3).

        Parameters
        ----------
        new_nominal_data : ndarray of shape (n_samples, n_sensors)

        Returns
        -------
        float
            The updated threshold.
        """
        alpha = 0.3  # blending factor (higher = more weight on new data)

        x = np.atleast_2d(new_nominal_data)
        x_norm = self._normalise(x)
        x_recon = self.network.predict(x_norm)
        errors = np.mean((x_norm - x_recon) ** 2, axis=1)

        new_mean = float(np.mean(errors))
        new_std = float(np.std(errors))

        # Exponential moving average update
        self._train_error_mean = (
            (1 - alpha) * self._train_error_mean + alpha * new_mean
        )
        self._train_error_std = (
            (1 - alpha) * self._train_error_std + alpha * new_std
        )

        self._threshold = (
            self._train_error_mean + self.threshold_sigma * self._train_error_std
        )
        return self._threshold

    # ---- Serialisation -------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Save the detector state (weights, normalisation, threshold) to disk."""
        path = Path(filepath)
        state = {
            "n_sensors": self.n_sensors,
            "hidden_dim": self.hidden_dim,
            "threshold": self._threshold,
            "train_error_mean": self._train_error_mean,
            "train_error_std": self._train_error_std,
            "threshold_sigma": self.threshold_sigma,
            "data_mean": self._data_mean.tolist() if self._data_mean is not None else None,
            "data_std": self._data_std.tolist() if self._data_std is not None else None,
            "weights": [
                {"W": p["W"].tolist(), "b": p["b"].tolist()}
                for p in self.network.get_weights()
            ],
        }
        path.write_text(json.dumps(state, indent=2))

    def load(self, filepath: str) -> None:
        """Restore detector state from a previously saved file."""
        path = Path(filepath)
        state = json.loads(path.read_text())

        self._threshold = state["threshold"]
        self._train_error_mean = state["train_error_mean"]
        self._train_error_std = state["train_error_std"]
        self.threshold_sigma = state["threshold_sigma"]

        if state["data_mean"] is not None:
            self._data_mean = np.array(state["data_mean"])
            self._data_std = np.array(state["data_std"])

        params = [
            {"W": np.array(p["W"]), "b": np.array(p["b"])}
            for p in state["weights"]
        ]
        self.network.set_weights(params)


# ---------------------------------------------------------------------------
#  Statistical (Mahalanobis) Anomaly Detector
# ---------------------------------------------------------------------------

class StatisticalAnomalyDetector:
    """Multivariate anomaly detector using the Mahalanobis distance.

    How It Works
    ------------
    The **Mahalanobis distance** generalises the concept of "how many standard
    deviations away" to multiple dimensions, accounting for correlations
    between features.  For a measurement vector *x*, a mean vector *mu*, and
    a covariance matrix *Sigma*:

        D_M(x) = sqrt( (x - mu)^T  Sigma^{-1}  (x - mu) )

    Under the assumption that the sensor vector is multivariate Gaussian, the
    *squared* Mahalanobis distance follows a chi-squared distribution with
    ``n_sensors`` degrees of freedom.  We use the chi-squared CDF to set the
    anomaly threshold at a specified significance level (default alpha = 0.001,
    i.e. 99.9% of nominal data should be below the threshold).

    Running Statistics
    ------------------
    Instead of storing all historical data, we use Welford's online algorithm
    to maintain a running mean and covariance.  This makes the detector
    suitable for real-time telemetry streams with bounded memory.

    Parameters
    ----------
    n_sensors : int
        Dimensionality of the sensor vector.
    significance : float
        Significance level for the chi-squared test (default 0.001).
        Lower values -> fewer false positives but may miss subtle anomalies.
    min_samples : int
        Minimum number of update() calls before detection is active.
        Until this count is reached, detect() always returns (False, 0.0).
    """

    def __init__(
        self,
        n_sensors: int,
        significance: float = 0.001,
        min_samples: int = 30,
    ) -> None:
        self.n_sensors = n_sensors
        self.significance = significance
        self.min_samples = min_samples

        # Running statistics (Welford's algorithm)
        self._count: int = 0
        self._mean = np.zeros(n_sensors)
        self._M2 = np.zeros((n_sensors, n_sensors))  # sum of outer products of deviations

        # Chi-squared threshold: P(chi2 > threshold) = significance
        # For n_sensors degrees of freedom.
        self._threshold = float(
            stats.chi2.ppf(1.0 - significance, df=n_sensors)
        )

        # Cache for the inverse covariance (recomputed when needed)
        self._cov_inv: Optional[np.ndarray] = None
        self._cov_inv_dirty: bool = True

    # ---- Running statistics (Welford's online algorithm) --------------------

    def update(self, reading: np.ndarray) -> None:
        """Incorporate a new sensor reading into the running statistics.

        Uses Welford's online algorithm, which is numerically stable for
        computing variance in a single pass:

            count += 1
            delta  = x - old_mean
            mean  += delta / count
            delta2 = x - new_mean
            M2    += outer(delta, delta2)

        The covariance matrix is then  Sigma = M2 / (count - 1).

        Parameters
        ----------
        reading : ndarray of shape (n_sensors,)
        """
        x = np.asarray(reading, dtype=np.float64).ravel()
        assert x.shape[0] == self.n_sensors

        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._M2 += np.outer(delta, delta2)

        # Invalidate cached inverse covariance
        self._cov_inv_dirty = True

    def batch_update(self, data: np.ndarray) -> None:
        """Update running statistics with a batch of readings."""
        for row in np.atleast_2d(data):
            self.update(row)

    def _get_covariance(self) -> np.ndarray:
        """Return the current sample covariance matrix."""
        if self._count < 2:
            return np.eye(self.n_sensors)
        return self._M2 / (self._count - 1)

    def _get_cov_inv(self) -> np.ndarray:
        """Return the inverse of the covariance matrix (cached)."""
        if self._cov_inv_dirty or self._cov_inv is None:
            cov = self._get_covariance()
            # Regularise to avoid singular matrix issues (add small diagonal)
            cov_reg = cov + 1e-6 * np.eye(self.n_sensors)
            try:
                self._cov_inv = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                warnings.warn(
                    "Covariance matrix is singular even after regularisation; "
                    "falling back to pseudo-inverse.",
                    RuntimeWarning,
                )
                self._cov_inv = np.linalg.pinv(cov_reg)
            self._cov_inv_dirty = False
        return self._cov_inv

    # ---- Detection -----------------------------------------------------------

    def _mahalanobis_sq(self, x: np.ndarray) -> float:
        """Compute the squared Mahalanobis distance of x from the mean.

        D^2 = (x - mu)^T  Sigma^{-1}  (x - mu)

        This quantity follows a chi-squared distribution with n_sensors
        degrees of freedom when x is drawn from N(mu, Sigma).
        """
        diff = x - self._mean
        cov_inv = self._get_cov_inv()
        return float(diff @ cov_inv @ diff)

    def detect(self, reading: np.ndarray) -> Tuple[bool, float]:
        """Determine whether a sensor reading is anomalous.

        Parameters
        ----------
        reading : ndarray of shape (n_sensors,)

        Returns
        -------
        is_anomaly : bool
            True if squared Mahalanobis distance exceeds chi-squared threshold.
        score : float
            The squared Mahalanobis distance.
        """
        if self._count < self.min_samples:
            # Not enough data to make a reliable decision
            return False, 0.0

        x = np.asarray(reading, dtype=np.float64).ravel()
        d2 = self._mahalanobis_sq(x)
        is_anomaly = d2 > self._threshold
        return is_anomaly, d2

    def detect_batch(self, data: np.ndarray) -> List[Tuple[bool, float]]:
        """Run detection on multiple readings at once."""
        return [self.detect(row) for row in np.atleast_2d(data)]

    # ---- Accessors -----------------------------------------------------------

    @property
    def threshold(self) -> float:
        """Current chi-squared anomaly threshold."""
        return self._threshold

    @property
    def mean(self) -> np.ndarray:
        """Running mean of the sensor vector."""
        return self._mean.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Current estimate of the sensor covariance matrix."""
        return self._get_covariance()

    @property
    def sample_count(self) -> int:
        """Number of samples incorporated so far."""
        return self._count


# ---------------------------------------------------------------------------
#  Demonstration / self-test
# ---------------------------------------------------------------------------

def _demo() -> None:
    """Quick demonstration of both detectors on synthetic data."""
    print("=" * 70)
    print("  Anomaly Detection Demo - Spacecraft Sensor Health Monitoring")
    print("=" * 70)

    rng = np.random.default_rng(42)
    n_sensors = 6
    n_train = 500
    n_test_nominal = 50
    n_test_anomaly = 10

    # --- Generate synthetic nominal data ---
    # Simulate 6 sensor channels with some correlation structure.
    mean = np.array([300.0, 0.01, -0.02, 1.5, 0.0, 100.0])
    cov = np.diag([5.0, 0.001, 0.001, 0.1, 0.001, 10.0])
    nominal_train = rng.multivariate_normal(mean, cov, size=n_train)
    nominal_test = rng.multivariate_normal(mean, cov, size=n_test_nominal)

    # Anomalous data: one channel has a large offset (simulates a stuck sensor)
    anomaly_test = rng.multivariate_normal(mean, cov, size=n_test_anomaly)
    anomaly_test[:, 0] += 50.0   # temperature channel jumps by 50 K
    anomaly_test[:, 3] *= 10.0   # angular rate spikes

    # --- Autoencoder detector ---
    print("\n--- SensorAnomalyDetector (Autoencoder) ---")
    ae_det = SensorAnomalyDetector(n_sensors=n_sensors, hidden_dim=8, seed=42)
    ae_det.train(nominal_train, epochs=60, batch_size=32, verbose=True)

    print(f"\n  Threshold: {ae_det.get_threshold():.6f}")
    print("\n  Nominal test samples:")
    for i in range(min(5, n_test_nominal)):
        flag, score = ae_det.detect(nominal_test[i])
        print(f"    sample {i}: anomaly={flag}, score={score:.6f}")

    print("\n  Anomalous test samples:")
    for i in range(n_test_anomaly):
        flag, score = ae_det.detect(anomaly_test[i])
        print(f"    sample {i}: anomaly={flag}, score={score:.6f}")

    # --- Statistical detector ---
    print("\n--- StatisticalAnomalyDetector (Mahalanobis) ---")
    stat_det = StatisticalAnomalyDetector(n_sensors=n_sensors, significance=0.001)
    stat_det.batch_update(nominal_train)

    print(f"  Threshold (chi2): {stat_det.threshold:.4f}")
    print(f"  Samples seen:     {stat_det.sample_count}")

    print("\n  Nominal test samples:")
    for i in range(min(5, n_test_nominal)):
        flag, score = stat_det.detect(nominal_test[i])
        print(f"    sample {i}: anomaly={flag}, score={score:.4f}")

    print("\n  Anomalous test samples:")
    for i in range(n_test_anomaly):
        flag, score = stat_det.detect(anomaly_test[i])
        print(f"    sample {i}: anomaly={flag}, score={score:.4f}")

    print("\n" + "=" * 70)
    print("  Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
