"""
===============================================================================
GNC PROJECT - Orbital Mechanics Engine
===============================================================================
N-body propagation with J2/J3 perturbations, Lambert solver,
Hohmann/bi-elliptic transfers, Keplerian <-> Cartesian conversions.

This module is the computational heart of the trajectory subsystem.  It
provides:

    1. **State representation** -- The OrbitalState dataclass bundles
       position, velocity, time, and reference frame into a single
       immutable snapshot that can be passed between modules.

    2. **Numerical propagation** -- Fixed-step RK4 and adaptive
       Dormand-Prince (RKDP / RK45) integrators that advance the
       equations of motion through time.

    3. **Force models** -- Two-body gravity, J2/J3 zonal harmonics,
       third-body perturbations (Sun, Moon, Jupiter), solar radiation
       pressure, atmospheric drag, and thrust.

    4. **Orbit geometry** -- Conversions between Keplerian elements and
       Cartesian state, vis-viva, orbital period, specific energy and
       angular momentum, sphere of influence radius.

    5. **Transfer design** -- Lambert solver (universal variable
       formulation), Hohmann transfer, bi-elliptic transfer.

All vectors are in SI units (m, m/s, s) and expressed in the ECI (J2000)
frame unless noted otherwise.

References
----------
    [1] Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed.
    [2] Bate, Mueller & White, "Fundamentals of Astrodynamics", Dover.
    [3] Curtis, "Orbital Mechanics for Engineering Students", 4th ed.
    [4] Battin, "An Introduction to the Mathematics and Methods of
        Astrodynamics", AIAA, 1999.
    [5] Dormand & Prince, "A family of embedded Runge-Kutta formulae",
        J. Comp. Appl. Math., 1980.

===============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Callable, Optional

from core.constants import (
    EARTH_MU,
    EARTH_J2,
    EARTH_J3,
    EARTH_EQUATORIAL_RADIUS,
    EARTH_RADIUS,
    SUN_MU,
    MOON_MU,
    JUPITER_MU,
    TWO_PI,
    PI,
    SOLAR_PRESSURE_AT_1AU,
    AU,
    EARTH_SEA_LEVEL_DENSITY,
    EARTH_SCALE_HEIGHT,
)


# =============================================================================
# ORBITAL STATE DATACLASS
# =============================================================================

@dataclass
class OrbitalState:
    """
    Snapshot of a spacecraft's translational state at a single instant.

    Attributes
    ----------
    position : np.ndarray
        3-element position vector (m) in the specified reference frame.
    velocity : np.ndarray
        3-element velocity vector (m/s) in the specified reference frame.
    time : float
        Mission elapsed time (s) at which this state is valid.
    frame : str
        Reference frame label (default 'ECI').  Used for bookkeeping;
        the numerical values are assumed to be consistent with this frame.
    """
    position: np.ndarray
    velocity: np.ndarray
    time: float
    frame: str = 'ECI'

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)

    @property
    def r_mag(self) -> float:
        """Magnitude of the position vector (m)."""
        return float(np.linalg.norm(self.position))

    @property
    def v_mag(self) -> float:
        """Magnitude of the velocity vector (m/s)."""
        return float(np.linalg.norm(self.velocity))

    def copy(self) -> 'OrbitalState':
        """Return a deep copy of this state."""
        return OrbitalState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            time=self.time,
            frame=self.frame,
        )


# =============================================================================
# ORBITAL MECHANICS ENGINE
# =============================================================================

class OrbitalMechanics:
    """
    Core orbital mechanics engine providing propagation, force modelling,
    coordinate conversions, and transfer orbit design.

    All methods are implemented as instance methods (rather than static)
    so that the engine can cache configuration (default mu, default body
    parameters) while remaining stateless with respect to the trajectory.
    """

    def __init__(self, mu: float = EARTH_MU) -> None:
        """
        Parameters
        ----------
        mu : float
            Default gravitational parameter (m^3/s^2).  Used when a
            method's *mu* argument is not explicitly provided.
        """
        self.mu = mu

    # =====================================================================
    # NUMERICAL PROPAGATORS
    # =====================================================================

    def propagate_rk4(
        self,
        state: OrbitalState,
        dt: float,
        accel_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    ) -> OrbitalState:
        """
        Advance the orbital state by one step using the classical
        4th-order Runge-Kutta method.

        The equations of motion are:

            dr/dt = v
            dv/dt = a(r, v, t)

        where a is the total acceleration returned by *accel_func*.

        RK4 evaluates the derivative at four points within the step:

            k1 = f(t_n,             y_n)
            k2 = f(t_n + dt/2,      y_n + dt/2 * k1)
            k3 = f(t_n + dt/2,      y_n + dt/2 * k2)
            k4 = f(t_n + dt,        y_n + dt * k3)

            y_{n+1} = y_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        The local truncation error is O(dt^5), giving a global error
        of O(dt^4).

        Parameters
        ----------
        state : OrbitalState
            Current state (position, velocity, time).
        dt : float
            Time step in seconds.
        accel_func : callable
            Acceleration function with signature
            ``(position, velocity, time) -> acceleration``.
            Must return a 3-element ndarray (m/s^2).

        Returns
        -------
        OrbitalState
            The propagated state at time t + dt.
        """
        r = state.position.copy()
        v = state.velocity.copy()
        t = state.time

        # Stage 1
        a1 = accel_func(r, v, t)
        kr1 = v
        kv1 = a1

        # Stage 2
        r2 = r + 0.5 * dt * kr1
        v2 = v + 0.5 * dt * kv1
        a2 = accel_func(r2, v2, t + 0.5 * dt)
        kr2 = v2
        kv2 = a2

        # Stage 3
        r3 = r + 0.5 * dt * kr2
        v3 = v + 0.5 * dt * kv2
        a3 = accel_func(r3, v3, t + 0.5 * dt)
        kr3 = v3
        kv3 = a3

        # Stage 4
        r4 = r + dt * kr3
        v4 = v + dt * kv3
        a4 = accel_func(r4, v4, t + dt)
        kr4 = v4
        kv4 = a4

        # Weighted combination
        r_new = r + (dt / 6.0) * (kr1 + 2.0 * kr2 + 2.0 * kr3 + kr4)
        v_new = v + (dt / 6.0) * (kv1 + 2.0 * kv2 + 2.0 * kv3 + kv4)

        return OrbitalState(
            position=r_new,
            velocity=v_new,
            time=t + dt,
            frame=state.frame,
        )

    def propagate_rkdp(
        self,
        state: OrbitalState,
        dt: float,
        accel_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        tol: float = 1e-10,
    ) -> Tuple[OrbitalState, float]:
        """
        Advance the orbital state using the Dormand-Prince adaptive
        RK method (RKDP / RK45).

        This is an embedded pair of 4th/5th order Runge-Kutta methods
        that shares function evaluations.  The 5th-order solution is
        used to advance the state, and the 4th-order solution provides
        an error estimate for step-size control.

        The Dormand-Prince method uses a 7-stage Butcher tableau (FSAL
        -- First Same As Last), but only 6 new evaluations per step
        because the 7th stage reuses the first evaluation of the next
        step.

        Butcher tableau coefficients are from Dormand & Prince (1980).

        Step-size adjustment:
            If the error exceeds *tol*, the step is rejected and the
            recommended step size is returned (caller should retry).
            If the error is below *tol*, the step is accepted and the
            recommended next step size is:

                dt_new = 0.9 * dt * (tol / error)^(1/5)

        Parameters
        ----------
        state : OrbitalState
            Current state.
        dt : float
            Attempted time step in seconds.
        accel_func : callable
            ``(position, velocity, time) -> acceleration``.
        tol : float
            Absolute error tolerance (m for position, m/s for velocity).

        Returns
        -------
        new_state : OrbitalState
            The propagated state (5th-order solution).
        error_estimate : float
            L2 norm of the difference between the 4th- and 5th-order
            solutions.  If this exceeds *tol*, the caller should reduce
            *dt* and retry.
        """
        r = state.position.copy()
        v = state.velocity.copy()
        t = state.time

        # -- Dormand-Prince Butcher tableau (7 stages) --
        # Node fractions c_i
        c2, c3, c4, c5, c6 = 1.0/5, 3.0/10, 4.0/5, 8.0/9, 1.0

        # a_ij coefficients (lower-triangular)
        a21 = 1.0/5
        a31, a32 = 3.0/40, 9.0/40
        a41, a42, a43 = 44.0/45, -56.0/15, 32.0/9
        a51, a52, a53, a54 = 19372.0/6561, -25360.0/2187, 64448.0/6561, -212.0/729
        a61, a62, a63, a64, a65 = 9017.0/3168, -355.0/33, 46732.0/5247, 49.0/176, -5103.0/18656

        # 5th-order weights b_i
        b1 = 35.0/384
        b3 = 500.0/1113
        b4 = 125.0/192
        b5 = -2187.0/6784
        b6 = 11.0/84
        # b2 = 0, b7 = 0 for the 5th-order solution

        # 4th-order weights b*_i (for error estimation)
        bs1 = 5179.0/57600
        bs3 = 7571.0/16695
        bs4 = 393.0/640
        bs5 = -92097.0/339200
        bs6 = 187.0/2100
        bs7 = 1.0/40

        # -- Stage 1 --
        a1 = accel_func(r, v, t)
        kr1 = v
        kv1 = a1

        # -- Stage 2 --
        r2 = r + dt * a21 * kr1
        v2 = v + dt * a21 * kv1
        a2 = accel_func(r2, v2, t + c2 * dt)
        kr2 = v2
        kv2 = a2

        # -- Stage 3 --
        r3 = r + dt * (a31 * kr1 + a32 * kr2)
        v3 = v + dt * (a31 * kv1 + a32 * kv2)
        a3 = accel_func(r3, v3, t + c3 * dt)
        kr3 = v3
        kv3 = a3

        # -- Stage 4 --
        r4 = r + dt * (a41 * kr1 + a42 * kr2 + a43 * kr3)
        v4 = v + dt * (a41 * kv1 + a42 * kv2 + a43 * kv3)
        a4 = accel_func(r4, v4, t + c4 * dt)
        kr4 = v4
        kv4 = a4

        # -- Stage 5 --
        r5 = r + dt * (a51 * kr1 + a52 * kr2 + a53 * kr3 + a54 * kr4)
        v5 = v + dt * (a51 * kv1 + a52 * kv2 + a53 * kv3 + a54 * kv4)
        a5 = accel_func(r5, v5, t + c5 * dt)
        kr5 = v5
        kv5 = a5

        # -- Stage 6 --
        r6 = r + dt * (a61 * kr1 + a62 * kr2 + a63 * kr3 + a64 * kr4 + a65 * kr5)
        v6 = v + dt * (a61 * kv1 + a62 * kv2 + a63 * kv3 + a64 * kv4 + a65 * kv5)
        a6 = accel_func(r6, v6, t + c6 * dt)
        kr6 = v6
        kv6 = a6

        # -- 5th-order solution --
        r_new = r + dt * (b1 * kr1 + b3 * kr3 + b4 * kr4 + b5 * kr5 + b6 * kr6)
        v_new = v + dt * (b1 * kv1 + b3 * kv3 + b4 * kv4 + b5 * kv5 + b6 * kv6)

        # -- Stage 7 (for error estimate; FSAL property) --
        a7 = accel_func(r_new, v_new, t + dt)
        kr7 = v_new
        kv7 = a7

        # -- 4th-order solution (for error estimation) --
        r_star = r + dt * (bs1 * kr1 + bs3 * kr3 + bs4 * kr4 + bs5 * kr5 + bs6 * kr6 + bs7 * kr7)
        v_star = v + dt * (bs1 * kv1 + bs3 * kv3 + bs4 * kv4 + bs5 * kv5 + bs6 * kv6 + bs7 * kv7)

        # -- Error estimate --
        err_r = np.linalg.norm(r_new - r_star)
        err_v = np.linalg.norm(v_new - v_star)
        error = max(err_r, err_v)

        new_state = OrbitalState(
            position=r_new,
            velocity=v_new,
            time=t + dt,
            frame=state.frame,
        )

        return new_state, float(error)

    # =====================================================================
    # ACCELERATION MODELS
    # =====================================================================

    @staticmethod
    def two_body_acceleration(pos: np.ndarray, mu: float) -> np.ndarray:
        """
        Compute the two-body (Keplerian) gravitational acceleration.

        The central force is:

            a = -mu / |r|^3 * r

        This is the dominant term for any spacecraft orbiting a single
        body.  All perturbations (J2, third-body, SRP, drag) are small
        corrections to this Keplerian baseline.

        Parameters
        ----------
        pos : np.ndarray
            3-element position vector (m) from the central body.
        mu : float
            Gravitational parameter of the central body (m^3/s^2).

        Returns
        -------
        np.ndarray
            3-element acceleration vector (m/s^2).
        """
        r = np.asarray(pos, dtype=np.float64)
        r_mag = np.linalg.norm(r)
        if r_mag < 1.0:
            return np.zeros(3)
        return -mu / (r_mag ** 3) * r

    @staticmethod
    def j2_acceleration(
        pos: np.ndarray, mu: float, J2: float, R_body: float
    ) -> np.ndarray:
        """
        Compute the J2 zonal harmonic perturbation acceleration.

        The J2 term arises from the oblateness (equatorial bulge) of
        the central body.  It is the largest gravitational perturbation
        for Earth-orbiting satellites, causing:
            - Secular regression of the ascending node (RAAN drift)
            - Secular advance of the argument of periapsis
            - Short-period oscillations in all elements

        The acceleration components are (Vallado, Eq. 8-25):

            factor = (3/2) * J2 * (R/r)^2 * (mu/r^2)

            a_x = -factor * (x/r) * (1 - 5*z^2/r^2)
            a_y = -factor * (y/r) * (1 - 5*z^2/r^2)
            a_z = -factor * (z/r) * (3 - 5*z^2/r^2)

        Parameters
        ----------
        pos : np.ndarray
            3-element position vector (m).
        mu : float
            Gravitational parameter (m^3/s^2).
        J2 : float
            J2 zonal harmonic coefficient (dimensionless).
        R_body : float
            Equatorial radius of the central body (m).

        Returns
        -------
        np.ndarray
            3-element J2 perturbation acceleration (m/s^2).
        """
        r = np.asarray(pos, dtype=np.float64)
        x, y, z = r[0], r[1], r[2]
        r_mag = np.linalg.norm(r)

        if r_mag < 1.0:
            return np.zeros(3)

        r2 = r_mag * r_mag
        r5 = r_mag ** 5
        z2_over_r2 = (z / r_mag) ** 2

        factor = 1.5 * J2 * mu * R_body ** 2 / r5

        ax = -factor * x * (1.0 - 5.0 * z2_over_r2)
        ay = -factor * y * (1.0 - 5.0 * z2_over_r2)
        az = -factor * z * (3.0 - 5.0 * z2_over_r2)

        return np.array([ax, ay, az], dtype=np.float64)

    @staticmethod
    def j3_acceleration(
        pos: np.ndarray, mu: float, J3: float, R_body: float
    ) -> np.ndarray:
        """
        Compute the J3 zonal harmonic perturbation acceleration.

        The J3 term captures the north-south asymmetry (pear shape) of
        the central body.  It is roughly 1000x smaller than J2 for
        Earth but can matter for high-precision orbit determination and
        for frozen-orbit design.

        The acceleration components are (Montenbruck & Gill, Eq. 3.28,
        simplified form):

            factor = (1/2) * J3 * mu * R^3 / r^7

            a_x = factor * x * (5 * z_over_r * (7*z^2/r^2 - 3))
            a_y = factor * y * (5 * z_over_r * (7*z^2/r^2 - 3))
            a_z = factor * z * (35*z^4/r^4 - 30*z^2/r^2 + 3) * (r/z)

        where the last expression is reformulated to avoid division by z.

        Parameters
        ----------
        pos : np.ndarray
            3-element position vector (m).
        mu : float
            Gravitational parameter (m^3/s^2).
        J3 : float
            J3 zonal harmonic coefficient (dimensionless).
        R_body : float
            Equatorial radius of the central body (m).

        Returns
        -------
        np.ndarray
            3-element J3 perturbation acceleration (m/s^2).
        """
        r = np.asarray(pos, dtype=np.float64)
        x, y, z = r[0], r[1], r[2]
        r_mag = np.linalg.norm(r)

        if r_mag < 1.0:
            return np.zeros(3)

        r2 = r_mag * r_mag
        z_over_r = z / r_mag
        z2_over_r2 = z_over_r ** 2

        factor = 0.5 * J3 * mu * R_body ** 3 / (r_mag ** 7)

        term_xy = 5.0 * z_over_r * (7.0 * z2_over_r2 - 3.0)
        term_z = 35.0 * z2_over_r2 * z2_over_r2 - 30.0 * z2_over_r2 + 3.0

        ax = factor * x * term_xy
        ay = factor * y * term_xy
        az = factor * r_mag * term_z  # factor already has r^-7

        return np.array([ax, ay, az], dtype=np.float64)

    def compute_total_acceleration(
        self,
        state: OrbitalState,
        spacecraft_obj,
        env_models: dict,
        thrust_vec: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the total acceleration on the spacecraft from all force
        models.

        Sums the following contributions:
            1. Central-body gravity (two-body)
            2. J2 perturbation
            3. J3 perturbation
            4. Third-body gravity (Sun, Moon, Jupiter)
            5. Solar radiation pressure (SRP)
            6. Atmospheric drag
            7. Thrust

        Each term is computed only if the corresponding model is present
        in *env_models*.  The total force is divided by the spacecraft
        mass to yield acceleration.

        Parameters
        ----------
        state : OrbitalState
            Current translational state.
        spacecraft_obj : object
            Spacecraft object with get_mass(), solar_panel_area,
            reflectivity attributes.
        env_models : dict
            Dictionary of environment models, keyed by name.  Expected
            keys (all optional):
                'gravity'     : GravityField instance
                'atmosphere'  : ExponentialAtmosphere instance
                'srp'         : SolarRadiationPressure instance
                'third_body'  : ThirdBodyPerturbation instance
                'sun_pos'     : np.ndarray (3,) Sun position in ECI (m)
                'moon_pos'    : np.ndarray (3,) Moon position in ECI (m)
                'jupiter_pos' : np.ndarray (3,) Jupiter position in ECI (m)
                'drag_cd'     : float, drag coefficient (default 2.2)
                'drag_area'   : float, drag reference area (m^2)
        thrust_vec : np.ndarray
            3-element thrust force vector in ECI (N).  Zero for coasting.

        Returns
        -------
        np.ndarray
            3-element total acceleration vector (m/s^2).
        """
        pos = state.position
        vel = state.velocity
        mass = spacecraft_obj.get_mass()
        if mass < 1e-6:
            return np.zeros(3)

        a_total = np.zeros(3, dtype=np.float64)

        # --- 1. Central-body gravity ---
        a_total += self.two_body_acceleration(pos, self.mu)

        # --- 2. J2 perturbation ---
        a_total += self.j2_acceleration(
            pos, self.mu, EARTH_J2, EARTH_EQUATORIAL_RADIUS
        )

        # --- 3. J3 perturbation ---
        a_total += self.j3_acceleration(
            pos, self.mu, EARTH_J3, EARTH_EQUATORIAL_RADIUS
        )

        # --- 4. Third-body perturbations ---
        third_body = env_models.get('third_body')
        if third_body is not None:
            sun_pos = env_models.get('sun_pos')
            if sun_pos is not None:
                a_total += third_body.acceleration(pos, sun_pos, SUN_MU)

            moon_pos = env_models.get('moon_pos')
            if moon_pos is not None:
                a_total += third_body.acceleration(pos, moon_pos, MOON_MU)

            jupiter_pos = env_models.get('jupiter_pos')
            if jupiter_pos is not None:
                a_total += third_body.acceleration(pos, jupiter_pos, JUPITER_MU)

        # --- 5. Solar radiation pressure ---
        srp = env_models.get('srp')
        sun_pos = env_models.get('sun_pos')
        if srp is not None and sun_pos is not None:
            pos_sun_relative = pos - sun_pos
            a_total += srp.acceleration(
                pos_sun_relative,
                spacecraft_obj.solar_panel_area,
                spacecraft_obj.reflectivity,
                mass,
            )

        # --- 6. Atmospheric drag ---
        atmo = env_models.get('atmosphere')
        if atmo is not None:
            r_mag = np.linalg.norm(pos)
            altitude = r_mag - EARTH_RADIUS
            rho = atmo.get_density(altitude)
            if rho > 0.0:
                v_mag = np.linalg.norm(vel)
                if v_mag > 0.0:
                    cd = env_models.get('drag_cd', 2.2)
                    area = env_models.get('drag_area',
                                          spacecraft_obj.solar_panel_area)
                    # Drag acceleration: a_drag = -0.5 * rho * v^2 * Cd * A / m * v_hat
                    v_hat = vel / v_mag
                    drag_mag = 0.5 * rho * v_mag * v_mag * cd * area / mass
                    a_total -= drag_mag * v_hat

        # --- 7. Thrust ---
        thrust = np.asarray(thrust_vec, dtype=np.float64)
        a_total += thrust / mass

        return a_total

    # =====================================================================
    # KEPLERIAN <-> CARTESIAN CONVERSIONS
    # =====================================================================

    @staticmethod
    def keplerian_to_cartesian(
        a: float, e: float, i: float,
        RAAN: float, omega: float, nu: float,
        mu: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert classical Keplerian orbital elements to Cartesian
        position and velocity vectors.

        The procedure is:
            1. Compute the position and velocity in the perifocal (PQW)
               frame using the orbit equation and angular momentum.
            2. Rotate from PQW to ECI using the 3-1-3 Euler rotation
               defined by (RAAN, inclination, argument of periapsis).

        Perifocal frame quantities:

            p = a * (1 - e^2)                        (semi-latus rectum)
            r = p / (1 + e*cos(nu))                  (orbital radius)
            r_pqw = r * [cos(nu), sin(nu), 0]
            v_pqw = sqrt(mu/p) * [-sin(nu), e+cos(nu), 0]

        The rotation matrix R from PQW to ECI is:

            R = Rz(-RAAN) * Rx(-i) * Rz(-omega)

        Parameters
        ----------
        a : float
            Semi-major axis (m).  Negative for hyperbolic orbits.
        e : float
            Eccentricity (0 = circular, 0 < e < 1 = elliptic,
            e = 1 = parabolic, e > 1 = hyperbolic).
        i : float
            Inclination (rad).
        RAAN : float
            Right Ascension of the Ascending Node (rad).
        omega : float
            Argument of periapsis (rad).
        nu : float
            True anomaly (rad).
        mu : float
            Gravitational parameter (m^3/s^2).

        Returns
        -------
        r_eci : np.ndarray
            3-element ECI position vector (m).
        v_eci : np.ndarray
            3-element ECI velocity vector (m/s).

        References
        ----------
        Vallado (2013), Algorithm 10.
        """
        # Semi-latus rectum
        p = a * (1.0 - e * e)
        if abs(p) < 1e-10:
            raise ValueError("Semi-latus rectum is near zero; degenerate orbit.")

        # Orbital radius
        r_mag = p / (1.0 + e * np.cos(nu))

        # Position and velocity in perifocal frame
        cos_nu = np.cos(nu)
        sin_nu = np.sin(nu)

        r_pqw = r_mag * np.array([cos_nu, sin_nu, 0.0], dtype=np.float64)
        v_pqw = np.sqrt(mu / p) * np.array([-sin_nu, e + cos_nu, 0.0],
                                            dtype=np.float64)

        # Rotation matrix from PQW to ECI (3-1-3 sequence)
        cos_O = np.cos(RAAN)
        sin_O = np.sin(RAAN)
        cos_i = np.cos(i)
        sin_i = np.sin(i)
        cos_w = np.cos(omega)
        sin_w = np.sin(omega)

        R = np.array([
            [cos_O * cos_w - sin_O * sin_w * cos_i,
             -cos_O * sin_w - sin_O * cos_w * cos_i,
             sin_O * sin_i],
            [sin_O * cos_w + cos_O * sin_w * cos_i,
             -sin_O * sin_w + cos_O * cos_w * cos_i,
             -cos_O * sin_i],
            [sin_w * sin_i,
             cos_w * sin_i,
             cos_i],
        ], dtype=np.float64)

        r_eci = R @ r_pqw
        v_eci = R @ v_pqw

        return r_eci, v_eci

    @staticmethod
    def cartesian_to_keplerian(
        r_vec: np.ndarray, v_vec: np.ndarray, mu: float
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Convert Cartesian state vectors to classical Keplerian elements.

        The algorithm computes:
            h = r x v                       (angular momentum)
            n = z_hat x h                   (ascending node vector)
            e_vec = (v x h)/mu - r/|r|      (eccentricity vector)
            e = |e_vec|
            a = -mu / (2*E)  where E = v^2/2 - mu/r  (specific energy)
            i = arccos(h_z / |h|)           (inclination)
            RAAN = arctan2(n_y, n_x)        (right ascension of ascending node)
            omega = signed angle from n to e_vec
            nu = signed angle from e_vec to r

        Edge cases:
            - Circular orbit (e ~ 0): omega undefined, set to 0.
            - Equatorial orbit (i ~ 0): RAAN undefined, set to 0.
            - Circular equatorial: both omega and RAAN set to 0;
              nu measured from x-axis.

        Parameters
        ----------
        r_vec : np.ndarray
            3-element position vector (m).
        v_vec : np.ndarray
            3-element velocity vector (m/s).
        mu : float
            Gravitational parameter (m^3/s^2).

        Returns
        -------
        tuple of (a, e, i, RAAN, omega, nu)
            a : semi-major axis (m)
            e : eccentricity
            i : inclination (rad) [0, pi]
            RAAN : right ascension of ascending node (rad) [0, 2*pi]
            omega : argument of periapsis (rad) [0, 2*pi]
            nu : true anomaly (rad) [0, 2*pi]

        References
        ----------
        Vallado (2013), Algorithm 9.
        """
        r = np.asarray(r_vec, dtype=np.float64)
        v = np.asarray(v_vec, dtype=np.float64)

        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)

        # Angular momentum
        h = np.cross(r, v)
        h_mag = np.linalg.norm(h)

        # Node vector (z_hat x h)
        z_hat = np.array([0.0, 0.0, 1.0])
        n = np.cross(z_hat, h)
        n_mag = np.linalg.norm(n)

        # Eccentricity vector
        e_vec = (np.cross(v, h) / mu) - (r / r_mag)
        e = np.linalg.norm(e_vec)

        # Specific mechanical energy -> semi-major axis
        energy = 0.5 * v_mag * v_mag - mu / r_mag
        if abs(energy) < 1e-20:
            # Parabolic (energy ~ 0): a is undefined; use a very large value
            a = 1e20
        else:
            a = -mu / (2.0 * energy)

        # Inclination
        inc = np.arccos(np.clip(h[2] / h_mag, -1.0, 1.0))

        # Right Ascension of Ascending Node
        if n_mag > 1e-12:
            raan = np.arctan2(n[1], n[0]) % TWO_PI
        else:
            # Equatorial orbit: RAAN undefined
            raan = 0.0

        # Argument of periapsis
        if e > 1e-12 and n_mag > 1e-12:
            cos_omega = np.dot(n, e_vec) / (n_mag * e)
            cos_omega = np.clip(cos_omega, -1.0, 1.0)
            omega = np.arccos(cos_omega)
            if e_vec[2] < 0.0:
                omega = TWO_PI - omega
        elif e > 1e-12:
            # Equatorial: measure omega from x-axis
            omega = np.arctan2(e_vec[1], e_vec[0]) % TWO_PI
        else:
            omega = 0.0

        # True anomaly
        if e > 1e-12:
            cos_nu = np.dot(e_vec, r) / (e * r_mag)
            cos_nu = np.clip(cos_nu, -1.0, 1.0)
            nu = np.arccos(cos_nu)
            if np.dot(r, v) < 0.0:
                nu = TWO_PI - nu
        elif n_mag > 1e-12:
            # Circular: measure nu from ascending node
            cos_nu = np.dot(n, r) / (n_mag * r_mag)
            cos_nu = np.clip(cos_nu, -1.0, 1.0)
            nu = np.arccos(cos_nu)
            if r[2] < 0.0:
                nu = TWO_PI - nu
        else:
            # Circular equatorial: measure from x-axis
            nu = np.arctan2(r[1], r[0]) % TWO_PI

        return (a, e, inc, raan, omega, nu)

    # =====================================================================
    # LAMBERT SOLVER (Universal Variable Formulation)
    # =====================================================================

    @staticmethod
    def lambert_solver(
        r1: np.ndarray, r2: np.ndarray, tof: float, mu: float,
        direction: str = 'prograde',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve Lambert's problem using the universal variable method.

        Given two position vectors r1, r2 and the time of flight (tof)
        between them, find the initial and final velocity vectors v1, v2
        for the transfer orbit.

        Lambert's problem is fundamental to interplanetary mission
        design: given a departure point, an arrival point, and a transit
        time, what are the required departure and arrival velocities?

        The universal variable method (Bate, Mueller & White; Vallado)
        works for elliptic, parabolic, and hyperbolic transfers by
        parameterising the trajectory with a universal variable z that
        relates to the transfer angle and Stumpff functions.

        Algorithm:
            1. Compute the transfer angle delta_nu from r1 and r2.
            2. Compute A = sin(delta_nu) * sqrt(r1*r2 / (1-cos(delta_nu)))
            3. Iteratively solve for z using Newton-Raphson on the
               time-of-flight equation involving Stumpff functions C(z)
               and S(z).
            4. Compute y(z) = r1 + r2 + A*(z*S(z)-1)/sqrt(C(z))
            5. Compute the Lagrange coefficients f, g, g_dot.
            6. v1 = (r2 - f*r1) / g
               v2 = (g_dot*r2 - r1) / g

        Stumpff functions:
            C(z) = (1 - cos(sqrt(z))) / z          for z > 0
                 = (cosh(sqrt(-z)) - 1) / (-z)     for z < 0
                 = 1/2                              for z = 0
            S(z) = (sqrt(z) - sin(sqrt(z))) / z^(3/2)  for z > 0
                 = (sinh(sqrt(-z)) - sqrt(-z)) / (-z)^(3/2) for z < 0
                 = 1/6                              for z = 0

        Parameters
        ----------
        r1 : np.ndarray
            3-element initial position vector (m).
        r2 : np.ndarray
            3-element final position vector (m).
        tof : float
            Time of flight between r1 and r2 (s).  Must be positive.
        mu : float
            Gravitational parameter (m^3/s^2).
        direction : str
            'prograde' (short way, delta_nu < pi) or
            'retrograde' (long way, delta_nu > pi).

        Returns
        -------
        v1 : np.ndarray
            3-element initial velocity vector (m/s).
        v2 : np.ndarray
            3-element final velocity vector (m/s).

        Raises
        ------
        RuntimeError
            If the Newton iteration fails to converge.

        References
        ----------
        Vallado (2013), Algorithm 58.
        Curtis (2020), Algorithm 5.2.
        """
        r1_vec = np.asarray(r1, dtype=np.float64)
        r2_vec = np.asarray(r2, dtype=np.float64)
        r1_mag = np.linalg.norm(r1_vec)
        r2_mag = np.linalg.norm(r2_vec)

        # --- Stumpff functions ---
        def stumpff_C(z):
            if z > 1e-6:
                sz = np.sqrt(z)
                return (1.0 - np.cos(sz)) / z
            elif z < -1e-6:
                sz = np.sqrt(-z)
                return (np.cosh(sz) - 1.0) / (-z)
            else:
                return 0.5

        def stumpff_S(z):
            if z > 1e-6:
                sz = np.sqrt(z)
                return (sz - np.sin(sz)) / (z * sz)
            elif z < -1e-6:
                sz = np.sqrt(-z)
                return (np.sinh(sz) - sz) / ((-z) * sz)
            else:
                return 1.0 / 6.0

        # --- Transfer angle ---
        cos_dnu = np.dot(r1_vec, r2_vec) / (r1_mag * r2_mag)
        cos_dnu = np.clip(cos_dnu, -1.0, 1.0)

        # Cross product to determine the sense of the transfer
        cross = np.cross(r1_vec, r2_vec)

        if direction == 'prograde':
            if cross[2] < 0.0:
                dnu = TWO_PI - np.arccos(cos_dnu)
            else:
                dnu = np.arccos(cos_dnu)
        else:
            if cross[2] >= 0.0:
                dnu = TWO_PI - np.arccos(cos_dnu)
            else:
                dnu = np.arccos(cos_dnu)

        # --- Auxiliary quantity A ---
        sin_dnu = np.sin(dnu)
        if abs(sin_dnu) < 1e-14:
            raise RuntimeError(
                "Lambert solver: degenerate geometry (transfer angle "
                "is 0 or pi). Cannot solve."
            )
        A = sin_dnu * np.sqrt(r1_mag * r2_mag / (1.0 - cos_dnu))

        # --- Newton-Raphson iteration for z ---
        z = 0.0  # initial guess (parabolic)
        max_iter = 200
        tol = 1e-10

        for iteration in range(max_iter):
            Cz = stumpff_C(z)
            Sz = stumpff_S(z)

            sqrt_Cz = np.sqrt(Cz) if Cz > 0 else 1e-12

            y = r1_mag + r2_mag + A * (z * Sz - 1.0) / sqrt_Cz

            if y < 0.0:
                # Adjust z to keep y positive
                z = z + 0.1
                continue

            sqrt_y = np.sqrt(y)
            x = sqrt_y / np.sqrt(Cz) if Cz > 1e-14 else sqrt_y * 1e7

            # Time of flight for current z
            tof_z = (x ** 3 * Sz + A * sqrt_y) / np.sqrt(mu)

            # Derivative of tof with respect to z (for Newton step)
            if abs(z) > 1e-6:
                dtof_dz = (
                    (x ** 3 * (Sz - 3.0 * Sz * Cz / (2.0 * Cz)
                     + 1.0 / (2.0 * z) * (1.0 - z * Sz / Cz))
                     + (3.0 * Sz * sqrt_y) / (2.0 * Cz) * A)
                    / (2.0 * np.sqrt(mu))
                )
                # Simplified derivative (more numerically stable):
                # Use finite difference as fallback
                eps = 1e-7
                Cz_p = stumpff_C(z + eps)
                Sz_p = stumpff_S(z + eps)
                sqrt_Cz_p = np.sqrt(Cz_p) if Cz_p > 0 else 1e-12
                y_p = r1_mag + r2_mag + A * ((z + eps) * Sz_p - 1.0) / sqrt_Cz_p
                if y_p > 0:
                    sqrt_y_p = np.sqrt(y_p)
                    x_p = sqrt_y_p / np.sqrt(Cz_p) if Cz_p > 1e-14 else sqrt_y_p * 1e7
                    tof_p = (x_p ** 3 * Sz_p + A * sqrt_y_p) / np.sqrt(mu)
                    dtof_dz = (tof_p - tof_z) / eps
                else:
                    dtof_dz = 1.0
            else:
                # Near z = 0, use finite difference
                eps = 1e-7
                Cz_p = stumpff_C(z + eps)
                Sz_p = stumpff_S(z + eps)
                sqrt_Cz_p = np.sqrt(Cz_p) if Cz_p > 0 else 1e-12
                y_p = r1_mag + r2_mag + A * ((z + eps) * Sz_p - 1.0) / sqrt_Cz_p
                if y_p > 0:
                    sqrt_y_p = np.sqrt(y_p)
                    x_p = sqrt_y_p / np.sqrt(Cz_p) if Cz_p > 1e-14 else sqrt_y_p * 1e7
                    tof_p = (x_p ** 3 * Sz_p + A * sqrt_y_p) / np.sqrt(mu)
                    dtof_dz = (tof_p - tof_z) / eps
                else:
                    dtof_dz = 1.0

            # Newton-Raphson update
            if abs(dtof_dz) < 1e-20:
                break
            z_new = z + (tof - tof_z) / dtof_dz

            if abs(z_new - z) < tol:
                z = z_new
                break
            z = z_new
        else:
            raise RuntimeError(
                f"Lambert solver did not converge in {max_iter} iterations. "
                f"Final z = {z:.6e}, tof_error = {tof_z - tof:.6e} s"
            )

        # --- Recompute final Stumpff values ---
        Cz = stumpff_C(z)
        Sz = stumpff_S(z)
        sqrt_Cz = np.sqrt(Cz) if Cz > 0 else 1e-12
        y = r1_mag + r2_mag + A * (z * Sz - 1.0) / sqrt_Cz

        # --- Lagrange coefficients ---
        f = 1.0 - y / r1_mag
        g_dot = 1.0 - y / r2_mag
        g = A * np.sqrt(y / mu)

        # --- Velocity vectors ---
        v1_vec = (r2_vec - f * r1_vec) / g
        v2_vec = (g_dot * r2_vec - r1_vec) / g

        return v1_vec, v2_vec

    # =====================================================================
    # TRANSFER ORBIT DESIGN
    # =====================================================================

    @staticmethod
    def hohmann_transfer(
        r1: float, r2: float, mu: float
    ) -> Tuple[float, float, float]:
        """
        Compute the Hohmann transfer parameters between two circular
        coplanar orbits.

        The Hohmann transfer is the minimum-energy two-impulse transfer
        between coplanar circular orbits.  It uses a half-ellipse whose
        periapsis is at r1 and apoapsis is at r2 (or vice versa).

        The semi-major axis of the transfer ellipse is:
            a_t = (r1 + r2) / 2

        Delta-V at departure (tangential burn at r1):
            dv1 = v_transfer_periapsis - v_circular(r1)
                = sqrt(mu * (2/r1 - 1/a_t)) - sqrt(mu/r1)

        Delta-V at arrival (tangential burn at r2):
            dv2 = v_circular(r2) - v_transfer_apoapsis
                = sqrt(mu/r2) - sqrt(mu * (2/r2 - 1/a_t))

        Time of flight (half the transfer ellipse period):
            tof = pi * sqrt(a_t^3 / mu)

        Parameters
        ----------
        r1 : float
            Radius of the initial circular orbit (m).
        r2 : float
            Radius of the final circular orbit (m).
        mu : float
            Gravitational parameter (m^3/s^2).

        Returns
        -------
        delta_v1 : float
            Magnitude of the first impulse (m/s).
        delta_v2 : float
            Magnitude of the second impulse (m/s).
        tof : float
            Time of flight for the transfer (s).
        """
        a_t = (r1 + r2) / 2.0

        v_circ_1 = np.sqrt(mu / r1)
        v_circ_2 = np.sqrt(mu / r2)

        v_transfer_1 = np.sqrt(mu * (2.0 / r1 - 1.0 / a_t))
        v_transfer_2 = np.sqrt(mu * (2.0 / r2 - 1.0 / a_t))

        delta_v1 = abs(v_transfer_1 - v_circ_1)
        delta_v2 = abs(v_circ_2 - v_transfer_2)

        tof = PI * np.sqrt(a_t ** 3 / mu)

        return delta_v1, delta_v2, tof

    @staticmethod
    def bi_elliptic_transfer(
        r1: float, r2: float, r_intermediate: float, mu: float
    ) -> Tuple[float, float, float, float]:
        """
        Compute the bi-elliptic transfer parameters between two
        circular coplanar orbits via an intermediate apoapsis.

        A bi-elliptic transfer uses three impulses and two transfer
        ellipses.  It can be more fuel-efficient than a Hohmann transfer
        when r2/r1 > ~11.94 (the exact crossover depends on the
        intermediate radius).

        Transfer geometry:
            Ellipse 1: periapsis at r1, apoapsis at r_int
            Ellipse 2: periapsis at r2, apoapsis at r_int

        Semi-major axes:
            a1 = (r1 + r_int) / 2
            a2 = (r2 + r_int) / 2

        Impulses:
            dv1 = v_transfer1_peri - v_circ(r1)
            dv2 = v_transfer2_apo - v_transfer1_apo   (at r_int)
            dv3 = v_circ(r2) - v_transfer2_peri

        Total time = half-period of ellipse 1 + half-period of ellipse 2

        Parameters
        ----------
        r1 : float
            Radius of the initial circular orbit (m).
        r2 : float
            Radius of the final circular orbit (m).
        r_intermediate : float
            Apoapsis radius of the intermediate transfer ellipse (m).
            Must be >= max(r1, r2).
        mu : float
            Gravitational parameter (m^3/s^2).

        Returns
        -------
        dv1 : float
            First impulse magnitude (m/s).
        dv2 : float
            Second impulse magnitude at intermediate apoapsis (m/s).
        dv3 : float
            Third impulse magnitude (m/s).
        tof : float
            Total time of flight (s).
        """
        a1 = (r1 + r_intermediate) / 2.0
        a2 = (r2 + r_intermediate) / 2.0

        # Velocities at key points
        v_circ_1 = np.sqrt(mu / r1)
        v_circ_2 = np.sqrt(mu / r2)

        v_t1_peri = np.sqrt(mu * (2.0 / r1 - 1.0 / a1))
        v_t1_apo = np.sqrt(mu * (2.0 / r_intermediate - 1.0 / a1))

        v_t2_apo = np.sqrt(mu * (2.0 / r_intermediate - 1.0 / a2))
        v_t2_peri = np.sqrt(mu * (2.0 / r2 - 1.0 / a2))

        dv1 = abs(v_t1_peri - v_circ_1)
        dv2 = abs(v_t2_apo - v_t1_apo)
        dv3 = abs(v_circ_2 - v_t2_peri)

        tof = PI * (np.sqrt(a1 ** 3 / mu) + np.sqrt(a2 ** 3 / mu))

        return dv1, dv2, dv3, tof

    # =====================================================================
    # ORBIT UTILITY FUNCTIONS
    # =====================================================================

    @staticmethod
    def vis_viva(r: float, a: float, mu: float) -> float:
        """
        Compute orbital velocity from the vis-viva equation.

        The vis-viva equation relates the speed of an orbiting body to
        its distance from the central body and the orbit's semi-major
        axis:

            v = sqrt( mu * (2/r - 1/a) )

        This is valid for all conic sections:
            - Elliptic (a > 0): v decreases with r, minimum at apoapsis
            - Parabolic (a -> inf): v = sqrt(2*mu/r)
            - Hyperbolic (a < 0): v > escape velocity at all r

        Parameters
        ----------
        r : float
            Distance from the central body (m).
        a : float
            Semi-major axis (m).  Negative for hyperbolic orbits.
        mu : float
            Gravitational parameter (m^3/s^2).

        Returns
        -------
        float
            Orbital speed (m/s).
        """
        return np.sqrt(mu * (2.0 / r - 1.0 / a))

    @staticmethod
    def orbital_period(a: float, mu: float) -> float:
        """
        Compute the orbital period from the semi-major axis.

        Kepler's third law:

            T = 2*pi * sqrt(a^3 / mu)

        Only valid for closed (elliptic) orbits (a > 0).

        Parameters
        ----------
        a : float
            Semi-major axis (m).  Must be positive.
        mu : float
            Gravitational parameter (m^3/s^2).

        Returns
        -------
        float
            Orbital period (s).

        Raises
        ------
        ValueError
            If a <= 0 (open orbit has no finite period).
        """
        if a <= 0:
            raise ValueError(
                f"Orbital period is undefined for a <= 0 (got a = {a:.4e} m). "
                "Open (hyperbolic/parabolic) orbits have infinite period."
            )
        return TWO_PI * np.sqrt(a ** 3 / mu)

    @staticmethod
    def sphere_of_influence(
        r_body: float, m_body: float, m_central: float
    ) -> float:
        """
        Compute the sphere of influence (SOI) radius for a body
        orbiting a more massive central body.

        The SOI is the region around a secondary body where its
        gravitational influence dominates over the central body.  It
        is defined by Laplace's formula:

            r_SOI = r_body * (m_body / m_central)^(2/5)

        where r_body is the orbital radius of the secondary body about
        the central body.

        Example: Earth's SOI around the Sun is ~9.24e8 m.

        Parameters
        ----------
        r_body : float
            Orbital radius of the secondary body (m).
        m_body : float
            Mass of the secondary body (kg).
        m_central : float
            Mass of the central body (kg).

        Returns
        -------
        float
            SOI radius (m).
        """
        return r_body * (m_body / m_central) ** (2.0 / 5.0)

    @staticmethod
    def specific_energy(r: float, v: float, mu: float) -> float:
        """
        Compute the specific mechanical energy (energy per unit mass).

        The specific energy is a constant of the two-body problem:

            E = v^2/2 - mu/r

        It determines the orbit type:
            E < 0 : elliptic (bound)
            E = 0 : parabolic (escape at zero residual speed)
            E > 0 : hyperbolic (escape with excess speed)

        Parameters
        ----------
        r : float
            Distance from the central body (m).
        v : float
            Orbital speed (m/s).
        mu : float
            Gravitational parameter (m^3/s^2).

        Returns
        -------
        float
            Specific energy (J/kg = m^2/s^2).
        """
        return 0.5 * v * v - mu / r

    @staticmethod
    def specific_angular_momentum(
        r_vec: np.ndarray, v_vec: np.ndarray
    ) -> np.ndarray:
        """
        Compute the specific angular momentum vector.

        The angular momentum per unit mass is:

            h = r x v

        It is a constant of the two-body problem and defines the
        orbital plane: h is perpendicular to the orbit, and its
        magnitude determines the semi-latus rectum p = h^2 / mu.

        Parameters
        ----------
        r_vec : np.ndarray
            3-element position vector (m).
        v_vec : np.ndarray
            3-element velocity vector (m/s).

        Returns
        -------
        np.ndarray
            3-element specific angular momentum vector (m^2/s).
        """
        return np.cross(
            np.asarray(r_vec, dtype=np.float64),
            np.asarray(v_vec, dtype=np.float64),
        )
