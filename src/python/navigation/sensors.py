"""
===============================================================================
GNC PROJECT - Sensor Models
===============================================================================
Realistic sensor models with noise injection, bias drift, and failure modes
for spacecraft navigation. Each sensor class encapsulates:
  - Truth-to-measurement transformation with configurable noise
  - Operational constraints (FOV, altitude, angular rate limits)
  - Health monitoring

Sensors implemented:
  IMU              -- Inertial Measurement Unit (gyro + accelerometer)
  StarTracker      -- Stellar attitude reference sensor
  SunSensor        -- Coarse Sun direction sensor
  GPSReceiver      -- GNSS position/velocity (LEO only)
  DeepSpaceNetwork -- Range, range-rate, angles from ground stations

All noise is injected via numpy.random. Constructors accept configuration
dictionaries so parameters can be loaded from mission config files.
===============================================================================
"""

import numpy as np
from numpy.linalg import norm

from core.constants import SPEED_OF_LIGHT, ARCSEC2RAD, DEG2RAD, AU


# =============================================================================
# IMU (Inertial Measurement Unit)
# =============================================================================

class IMU:
    """
    Six-degree-of-freedom inertial measurement unit model combining a
    three-axis gyroscope and a three-axis accelerometer.

    Gyroscope error model:
        - Bias instability (deg/hr) -- slowly-varying residual bias
        - Angle Random Walk, ARW (deg/sqrt(hr)) -- white noise on angular rate
        - Scale factor error (ppm) -- multiplicative gain error per axis
        - Axis misalignment (arcsec) -- small rotation between true and
          sensor frames, modelled as a skew-symmetric matrix

    Accelerometer error model:
        - Bias (m/s^2) -- constant residual bias per axis
        - Velocity Random Walk, VRW (m/s/sqrt(hr)) -- white noise on accel
        - Scale factor error (ppm) -- multiplicative gain error per axis

    Biases are propagated between calls with a first-order Gauss-Markov
    random walk so that they drift realistically over time.

    Parameters (passed as config dict):
        gyro_bias_instability   : float  -- deg/hr  (1-sigma)
        gyro_arw                : float  -- deg/sqrt(hr) (1-sigma)
        gyro_scale_factor_ppm   : float  -- parts per million (1-sigma)
        gyro_misalignment_arcsec: float  -- arcsec (1-sigma per off-diagonal)
        accel_bias              : float  -- m/s^2 (1-sigma)
        accel_vrw               : float  -- m/s/sqrt(hr) (1-sigma)
        accel_scale_factor_ppm  : float  -- ppm (1-sigma)
        dt                      : float  -- nominal sample period (s)
        seed                    : int    -- (optional) RNG seed
    """

    def __init__(self, config: dict):
        # ----- Random number generator -----
        seed = config.get("seed", None)
        self.rng = np.random.default_rng(seed)

        # ----- Gyroscope parameters (convert to SI / rad) -----
        # Bias instability: deg/hr -> rad/s
        self.gyro_bias_sigma = (
            config.get("gyro_bias_instability", 0.01) * DEG2RAD / 3600.0
        )
        # ARW: deg/sqrt(hr) -> rad/sqrt(s)
        self.gyro_arw = (
            config.get("gyro_arw", 0.002) * DEG2RAD / np.sqrt(3600.0)
        )
        # Scale factor (dimensionless fraction from ppm)
        self.gyro_sf_sigma = config.get("gyro_scale_factor_ppm", 5.0) * 1.0e-6
        # Misalignment: arcsec -> rad
        self.gyro_misalign_sigma = (
            config.get("gyro_misalignment_arcsec", 1.0) * ARCSEC2RAD
        )

        # ----- Accelerometer parameters -----
        self.accel_bias_sigma = config.get("accel_bias", 1.0e-4)  # m/s^2
        # VRW: m/s/sqrt(hr) -> m/s/sqrt(s)
        self.accel_vrw = (
            config.get("accel_vrw", 0.003) / np.sqrt(3600.0)
        )
        self.accel_sf_sigma = config.get("accel_scale_factor_ppm", 5.0) * 1.0e-6

        # ----- Timing -----
        self.dt = config.get("dt", 0.01)  # seconds

        # ----- Initialize internal bias states -----
        self._init_biases()

        # ----- Build one-time misalignment matrix -----
        # Skew-symmetric part representing small-angle axis cross-coupling
        mis = self.rng.normal(0.0, self.gyro_misalign_sigma, size=3)
        self.misalignment_matrix = np.array([
            [1.0,     -mis[2],  mis[1]],
            [mis[2],   1.0,    -mis[0]],
            [-mis[1],  mis[0],  1.0   ],
        ])

        # Scale factor errors (drawn once, fixed for sensor lifetime)
        self.gyro_scale_factors = 1.0 + self.rng.normal(
            0.0, self.gyro_sf_sigma, size=3
        )
        self.accel_scale_factors = 1.0 + self.rng.normal(
            0.0, self.accel_sf_sigma, size=3
        )

        # Health flag
        self._healthy = True

    # --------------------------------------------------------------------- #
    def _init_biases(self):
        """Draw initial bias values and set bias random-walk state."""
        self.gyro_bias = self.rng.normal(0.0, self.gyro_bias_sigma, size=3)
        self.accel_bias = self.rng.normal(0.0, self.accel_bias_sigma, size=3)

    # --------------------------------------------------------------------- #
    def measure(self, true_omega: np.ndarray, true_accel: np.ndarray):
        """
        Produce a noisy IMU measurement from the true angular rate and
        specific-force vectors (both in body frame).

        The gyro measurement model is:
            omega_meas = M * diag(sf) * true_omega + bias + noise
        where M is the misalignment matrix, sf are scale factors, and noise
        is white with PSD derived from the angle random walk.

        The accel measurement model is:
            accel_meas = diag(sf) * true_accel + bias + noise

        Between calls the biases undergo a random walk:
            bias(k+1) = bias(k) + w,   w ~ N(0, sigma * sqrt(dt))

        Parameters
        ----------
        true_omega : (3,) ndarray -- true angular velocity in body frame (rad/s)
        true_accel : (3,) ndarray -- true specific force in body frame (m/s^2)

        Returns
        -------
        omega_meas : (3,) ndarray -- measured angular velocity (rad/s)
        accel_meas : (3,) ndarray -- measured specific force (m/s^2)
        """
        if not self._healthy:
            return None, None

        dt = self.dt

        # ---- Propagate gyro bias with random walk ----
        self.gyro_bias += self.rng.normal(
            0.0, self.gyro_bias_sigma * np.sqrt(dt), size=3
        )

        # ---- Gyro measurement ----
        omega_scaled = self.gyro_scale_factors * true_omega
        omega_misaligned = self.misalignment_matrix @ omega_scaled
        gyro_noise = self.rng.normal(0.0, self.gyro_arw / np.sqrt(dt), size=3)
        omega_meas = omega_misaligned + self.gyro_bias + gyro_noise

        # ---- Propagate accel bias with random walk ----
        self.accel_bias += self.rng.normal(
            0.0, self.accel_bias_sigma * np.sqrt(dt), size=3
        )

        # ---- Accelerometer measurement ----
        accel_scaled = self.accel_scale_factors * true_accel
        accel_noise = self.rng.normal(0.0, self.accel_vrw / np.sqrt(dt), size=3)
        accel_meas = accel_scaled + self.accel_bias + accel_noise

        return omega_meas, accel_meas

    # --------------------------------------------------------------------- #
    def reset(self):
        """Re-initialize biases (simulates a sensor power-cycle)."""
        self._init_biases()
        self._healthy = True

    # --------------------------------------------------------------------- #
    def get_health_status(self) -> dict:
        """
        Return a summary of the IMU internal state for telemetry.

        Returns
        -------
        dict with keys:
            healthy        : bool
            gyro_bias_norm : float -- norm of current gyro bias (rad/s)
            accel_bias_norm: float -- norm of current accel bias (m/s^2)
        """
        return {
            "healthy": self._healthy,
            "gyro_bias_norm": float(norm(self.gyro_bias)),
            "accel_bias_norm": float(norm(self.accel_bias)),
        }


# =============================================================================
# Star Tracker
# =============================================================================

class StarTracker:
    """
    Star tracker sensor model. Returns an attitude quaternion measurement
    corrupted by small-angle Gaussian noise.

    Failure modes:
        - Sun exclusion: measurement unavailable if the Sun direction in
          body frame falls within the sensor boresight exclusion cone.
        - Moon exclusion: same for the Moon.
        - Angular rate limit: measurement unavailable when the spacecraft
          angular rate magnitude exceeds a threshold (star images streak).

    Parameters (config dict):
        noise_sigma_arcsec : float -- 1-sigma attitude noise per axis (arcsec)
        sun_exclusion_deg  : float -- half-angle exclusion cone for Sun (deg)
        moon_exclusion_deg : float -- half-angle exclusion cone for Moon (deg)
        max_angular_rate   : float -- max body rate for valid image (deg/s)
        boresight          : (3,) array-like -- sensor boresight in body frame
        fov_half_angle_deg : float -- sensor field of view half-angle (deg)
        seed               : int   -- (optional) RNG seed
    """

    def __init__(self, config: dict):
        seed = config.get("seed", None)
        self.rng = np.random.default_rng(seed)

        # Noise: arcsec -> rad (per axis, small-angle approximation)
        self.noise_sigma = (
            config.get("noise_sigma_arcsec", 5.0) * ARCSEC2RAD
        )

        # Exclusion half-angles (deg -> rad)
        self.sun_exclusion = config.get("sun_exclusion_deg", 25.0) * DEG2RAD
        self.moon_exclusion = config.get("moon_exclusion_deg", 15.0) * DEG2RAD

        # Maximum angular rate for a valid exposure (deg/s -> rad/s)
        self.max_omega = config.get("max_angular_rate", 1.0) * DEG2RAD

        # Boresight direction in body frame (default +Z)
        boresight = np.asarray(config.get("boresight", [0.0, 0.0, 1.0]),
                               dtype=float)
        self.boresight = boresight / norm(boresight)

        self.fov_half_angle = config.get("fov_half_angle_deg", 10.0) * DEG2RAD

    # --------------------------------------------------------------------- #
    @staticmethod
    def _quat_mult(q, r):
        """
        Hamilton quaternion product  q (*) r.
        Convention: q = [qx, qy, qz, qw]  (scalar-last).
        """
        q0, q1, q2, q3 = q
        r0, r1, r2, r3 = r
        return np.array([
            q3 * r0 + q0 * r3 + q1 * r2 - q2 * r1,
            q3 * r1 - q0 * r2 + q1 * r3 + q2 * r0,
            q3 * r2 + q0 * r1 - q1 * r0 + q2 * r3,
            q3 * r3 - q0 * r0 - q1 * r1 - q2 * r2,
        ])

    # --------------------------------------------------------------------- #
    def _in_exclusion_zone(self, direction_body: np.ndarray,
                           half_angle: float) -> bool:
        """
        Check if a direction vector (in body frame) falls within a
        circular exclusion cone centred on the sensor boresight.
        """
        if direction_body is None:
            return False
        d = np.asarray(direction_body, dtype=float)
        d_norm = norm(d)
        if d_norm < 1.0e-12:
            return False
        cos_angle = np.dot(self.boresight, d / d_norm)
        # Clamp for numerical safety
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return angle < half_angle

    # --------------------------------------------------------------------- #
    def measure(self, true_quaternion: np.ndarray,
                sun_direction_body: np.ndarray,
                moon_direction_body: np.ndarray,
                omega_mag: float):
        """
        Produce a noisy attitude quaternion or None if the measurement is
        unavailable.

        Parameters
        ----------
        true_quaternion     : (4,) ndarray  -- true attitude [qx, qy, qz, qw]
        sun_direction_body  : (3,) ndarray  -- Sun unit vector in body frame
        moon_direction_body : (3,) ndarray  -- Moon unit vector in body frame
        omega_mag           : float         -- body angular-rate magnitude (rad/s)

        Returns
        -------
        (4,) ndarray or None
            Noisy quaternion [qx, qy, qz, qw] if valid, else None.
        """
        # -- Check Sun exclusion --
        if self._in_exclusion_zone(sun_direction_body, self.sun_exclusion):
            return None

        # -- Check Moon exclusion --
        if self._in_exclusion_zone(moon_direction_body, self.moon_exclusion):
            return None

        # -- Check angular rate limit --
        if omega_mag > self.max_omega:
            return None

        # -- Generate small-angle noise quaternion --
        # delta_theta is a 3-element rotation error vector (rad)
        delta_theta = self.rng.normal(0.0, self.noise_sigma, size=3)
        half = 0.5 * delta_theta
        dq_vec = half                        # sin(|half|) ~ |half| for small angles
        dq_scalar = 1.0 - 0.5 * np.dot(half, half)  # cos(|half|) ~ 1 - |half|^2/2
        dq = np.array([dq_vec[0], dq_vec[1], dq_vec[2], dq_scalar])
        dq /= norm(dq)

        # -- Compose: measured = true (*) dq --
        q_meas = self._quat_mult(true_quaternion, dq)
        # Ensure scalar part positive (canonical form)
        if q_meas[3] < 0.0:
            q_meas = -q_meas

        return q_meas


# =============================================================================
# Sun Sensor
# =============================================================================

class SunSensor:
    """
    Coarse Sun sensor that returns the Sun direction unit vector in the body
    frame when the Sun is within the sensor field of view.

    Parameters (config dict):
        noise_sigma_deg : float -- 1-sigma noise on each axis (deg)
        fov_half_angle_deg : float -- half-angle of the conical FOV (deg)
        boresight       : (3,) array-like -- sensor boresight in body frame
        seed            : int   -- (optional) RNG seed
    """

    def __init__(self, config: dict):
        seed = config.get("seed", None)
        self.rng = np.random.default_rng(seed)

        self.noise_sigma = config.get("noise_sigma_deg", 0.5) * DEG2RAD
        self.fov_half_angle = config.get("fov_half_angle_deg", 60.0) * DEG2RAD

        boresight = np.asarray(config.get("boresight", [1.0, 0.0, 0.0]),
                               dtype=float)
        self.boresight = boresight / norm(boresight)

    # --------------------------------------------------------------------- #
    def measure(self, sun_direction_true_body: np.ndarray):
        """
        Return a noisy Sun direction unit vector or None if the Sun is
        outside the FOV.

        Parameters
        ----------
        sun_direction_true_body : (3,) ndarray -- true Sun unit vector in body

        Returns
        -------
        (3,) ndarray or None
            Noisy unit vector if Sun is visible, else None.
        """
        d = np.asarray(sun_direction_true_body, dtype=float)
        d_norm = norm(d)
        if d_norm < 1.0e-12:
            return None

        d_unit = d / d_norm

        # Check FOV
        cos_angle = np.clip(np.dot(self.boresight, d_unit), -1.0, 1.0)
        if np.arccos(cos_angle) > self.fov_half_angle:
            return None

        # Add noise: perturb each component then re-normalise
        noise = self.rng.normal(0.0, self.noise_sigma, size=3)
        d_noisy = d_unit + noise
        d_noisy_norm = norm(d_noisy)
        if d_noisy_norm < 1.0e-12:
            return None

        return d_noisy / d_noisy_norm


# =============================================================================
# GPS Receiver
# =============================================================================

class GPSReceiver:
    """
    GPS / GNSS receiver model. Provides position and velocity measurements
    in ECEF frame. Only operates below a maximum altitude (GPS constellation
    is designed for surface and LEO users; above ~3000 km the geometry
    degrades rapidly and signals are typically unavailable).

    Parameters (config dict):
        pos_noise_sigma : float -- 1-sigma position noise per axis (m)
        vel_noise_sigma : float -- 1-sigma velocity noise per axis (m/s)
        max_altitude    : float -- maximum operating altitude (m)
        seed            : int   -- (optional) RNG seed
    """

    def __init__(self, config: dict):
        seed = config.get("seed", None)
        self.rng = np.random.default_rng(seed)

        self.pos_noise_sigma = config.get("pos_noise_sigma", 5.0)      # m
        self.vel_noise_sigma = config.get("vel_noise_sigma", 0.05)     # m/s
        self.max_altitude = config.get("max_altitude", 3000.0e3)       # 3 000 km

    # --------------------------------------------------------------------- #
    def measure(self, true_pos_ecef: np.ndarray,
                true_vel_ecef: np.ndarray,
                altitude: float):
        """
        Return noisy ECEF position and velocity, or (None, None) if above
        the maximum operating altitude.

        Parameters
        ----------
        true_pos_ecef : (3,) ndarray -- true ECEF position (m)
        true_vel_ecef : (3,) ndarray -- true ECEF velocity (m/s)
        altitude      : float        -- geodetic altitude (m)

        Returns
        -------
        pos_meas : (3,) ndarray or None -- measured position (m)
        vel_meas : (3,) ndarray or None -- measured velocity (m/s)
        """
        if altitude > self.max_altitude:
            return None, None

        pos_noise = self.rng.normal(0.0, self.pos_noise_sigma, size=3)
        vel_noise = self.rng.normal(0.0, self.vel_noise_sigma, size=3)

        pos_meas = np.asarray(true_pos_ecef, dtype=float) + pos_noise
        vel_meas = np.asarray(true_vel_ecef, dtype=float) + vel_noise

        return pos_meas, vel_meas


# =============================================================================
# Deep Space Network
# =============================================================================

class DeepSpaceNetwork:
    """
    Model of ground-based Deep Space Network (DSN) tracking measurements.

    Provides:
        - Range (two-way light-time ranging) -- metres
        - Range-rate (Doppler) -- m/s
        - Angles (azimuth, elevation) -- rad

    Signal degradation: measurement noise increases proportionally to the
    square of the distance from Earth (inverse-square law on received power).
    A reference distance (1 AU) is used to normalise noise levels.

    Communication delay is modelled as the one-way light-time.

    Parameters (config dict):
        range_noise_1au       : float -- 1-sigma range noise at 1 AU (m)
        range_rate_noise_1au  : float -- 1-sigma range-rate noise at 1 AU (m/s)
        angle_noise_1au_arcsec: float -- 1-sigma angle noise at 1 AU (arcsec)
        seed                  : int   -- (optional) RNG seed
    """

    def __init__(self, config: dict):
        seed = config.get("seed", None)
        self.rng = np.random.default_rng(seed)

        # Reference noise levels at 1 AU
        self.range_noise_1au = config.get("range_noise_1au", 1.0)           # m
        self.range_rate_noise_1au = config.get("range_rate_noise_1au", 0.1) # mm/s -> 0.1 m/s default
        self.angle_noise_1au = (
            config.get("angle_noise_1au_arcsec", 0.5) * ARCSEC2RAD
        )

    # --------------------------------------------------------------------- #
    def _distance_scale(self, distance_from_earth: float) -> float:
        """
        Compute noise scale factor.  Noise ~ distance^2 / (1 AU)^2
        because received power drops with the inverse square of distance,
        so the standard deviation of the measurement noise rises linearly
        with distance (sigma ~ 1/sqrt(SNR) ~ distance).

        However DSN measurement accuracy scales approximately as distance
        for range and range-rate (not distance^2) because we are measuring
        the standard deviation, not variance.  We use:
            scale = distance / 1 AU
        """
        return max(distance_from_earth / AU, 1.0)

    # --------------------------------------------------------------------- #
    def measure(self, true_range: float, true_range_rate: float,
                true_angles: np.ndarray, distance_from_earth: float) -> dict:
        """
        Produce noisy DSN measurements.

        Parameters
        ----------
        true_range          : float        -- true one-way range (m)
        true_range_rate     : float        -- true range-rate / Doppler (m/s)
        true_angles         : (2,) ndarray -- [azimuth, elevation] (rad)
        distance_from_earth : float        -- distance from Earth (m)

        Returns
        -------
        dict with keys:
            range       : float  -- measured range (m)
            range_rate  : float  -- measured range-rate (m/s)
            angles      : (2,) ndarray -- [az, el] (rad)
            delay       : float  -- one-way light-time delay (s)
            noise_scale : float  -- scale factor applied to noise
        """
        scale = self._distance_scale(distance_from_earth)

        # Range measurement
        range_noise = self.rng.normal(0.0, self.range_noise_1au * scale)
        range_meas = true_range + range_noise

        # Range-rate (Doppler) measurement
        rr_noise = self.rng.normal(0.0, self.range_rate_noise_1au * scale)
        range_rate_meas = true_range_rate + rr_noise

        # Angles measurement
        angle_noise = self.rng.normal(0.0, self.angle_noise_1au * scale, size=2)
        angles_meas = np.asarray(true_angles, dtype=float) + angle_noise

        # One-way light-time delay
        delay = distance_from_earth / SPEED_OF_LIGHT

        return {
            "range": float(range_meas),
            "range_rate": float(range_rate_meas),
            "angles": angles_meas,
            "delay": float(delay),
            "noise_scale": float(scale),
        }
