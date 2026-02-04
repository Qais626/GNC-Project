"""
===============================================================================
GNC PROJECT - Orbital Mechanics Test Suite
===============================================================================
Tests for orbital mechanics computations: circular orbit velocity, Keplerian-
Cartesian round-trip, Hohmann transfer, vis-viva, Lambert solver, RK4 energy
conservation, J2 regression (RAAN drift), and escape velocity.

Constants and helper functions are drawn from the project's core.constants
and guidance.maneuver_planner modules.
===============================================================================
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from numpy.testing import assert_allclose

from core.constants import (
    EARTH_MU, EARTH_RADIUS, EARTH_J2, EARTH_EQUATORIAL_RADIUS,
    LEO_ALTITUDE, LEO_RADIUS, LEO_VELOCITY,
)
from guidance.maneuver_planner import ManeuverPlanner
from guidance.trajectory_opt import TrajectoryOptimizer
from dynamics.environment import GravityField


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def planner():
    """Return a ManeuverPlanner instance."""
    return ManeuverPlanner()


@pytest.fixture
def optimizer():
    """Return a TrajectoryOptimizer instance."""
    return TrajectoryOptimizer()


@pytest.fixture
def earth_gravity():
    """Return an Earth gravity field model with J2."""
    return GravityField.earth()


# =============================================================================
# Helper functions
# =============================================================================

def keplerian_to_cartesian(a, e, i, raan, omega, nu, mu):
    """
    Convert Keplerian elements to Cartesian state (position, velocity).

    Parameters
    ----------
    a : semi-major axis (m)
    e : eccentricity
    i : inclination (rad)
    raan : right ascension of ascending node (rad)
    omega : argument of periapsis (rad)
    nu : true anomaly (rad)
    mu : gravitational parameter (m^3/s^2)

    Returns
    -------
    r : position vector (3,) in ECI (m)
    v : velocity vector (3,) in ECI (m/s)
    """
    p = a * (1 - e ** 2)
    r_mag = p / (1 + e * np.cos(nu))

    # Position and velocity in the perifocal frame
    r_pf = r_mag * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    # Rotation matrix from perifocal to ECI
    cos_raan, sin_raan = np.cos(raan), np.sin(raan)
    cos_omega, sin_omega = np.cos(omega), np.sin(omega)
    cos_i, sin_i = np.cos(i), np.sin(i)

    R = np.array([
        [cos_raan * cos_omega - sin_raan * sin_omega * cos_i,
         -cos_raan * sin_omega - sin_raan * cos_omega * cos_i,
         sin_raan * sin_i],
        [sin_raan * cos_omega + cos_raan * sin_omega * cos_i,
         -sin_raan * sin_omega + cos_raan * cos_omega * cos_i,
         -cos_raan * sin_i],
        [sin_omega * sin_i,
         cos_omega * sin_i,
         cos_i],
    ])

    r_eci = R @ r_pf
    v_eci = R @ v_pf
    return r_eci, v_eci


def cartesian_to_keplerian(r, v, mu):
    """
    Convert Cartesian state to classical Keplerian elements.

    Returns
    -------
    (a, e, i, raan, omega, nu) all in SI and radians.
    """
    r_vec = np.asarray(r, dtype=float)
    v_vec = np.asarray(v, dtype=float)
    r_mag = np.linalg.norm(r_vec)
    v_mag = np.linalg.norm(v_vec)

    # Specific angular momentum
    h = np.cross(r_vec, v_vec)
    h_mag = np.linalg.norm(h)

    # Node vector
    k_hat = np.array([0.0, 0.0, 1.0])
    n = np.cross(k_hat, h)
    n_mag = np.linalg.norm(n)

    # Eccentricity vector
    e_vec = ((v_mag ** 2 - mu / r_mag) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e = np.linalg.norm(e_vec)

    # Specific energy -> semi-major axis
    energy = v_mag ** 2 / 2.0 - mu / r_mag
    if abs(e - 1.0) > 1e-10:
        a = -mu / (2.0 * energy)
    else:
        a = np.inf  # parabolic

    # Inclination
    inc = np.arccos(np.clip(h[2] / h_mag, -1, 1))

    # RAAN
    if n_mag > 1e-12:
        raan = np.arccos(np.clip(n[0] / n_mag, -1, 1))
        if n[1] < 0:
            raan = 2.0 * np.pi - raan
    else:
        raan = 0.0

    # Argument of periapsis
    if n_mag > 1e-12 and e > 1e-12:
        omega = np.arccos(np.clip(np.dot(n, e_vec) / (n_mag * e), -1, 1))
        if e_vec[2] < 0:
            omega = 2.0 * np.pi - omega
    else:
        omega = 0.0

    # True anomaly
    if e > 1e-12:
        nu = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r_mag), -1, 1))
        if np.dot(r_vec, v_vec) < 0:
            nu = 2.0 * np.pi - nu
    else:
        nu = 0.0

    return a, e, inc, raan, omega, nu


def rk4_step(f, t, y, dt):
    """Single RK4 integration step."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# =============================================================================
# Test: Circular orbit velocity
# =============================================================================

class TestCircularOrbitVelocity:
    """Tests for circular orbit velocity: v = sqrt(mu/r)."""

    def test_circular_orbit_velocity(self):
        """Circular velocity at LEO altitude matches analytical formula."""
        r = EARTH_RADIUS + LEO_ALTITUDE
        v_analytical = np.sqrt(EARTH_MU / r)
        assert_allclose(v_analytical, LEO_VELOCITY, rtol=1e-10)

    @pytest.mark.parametrize("altitude_km", [200, 400, 800, 2000, 35786])
    def test_circular_velocity_various_altitudes(self, altitude_km):
        """Circular velocity at various altitudes satisfies v = sqrt(mu/r)."""
        r = EARTH_RADIUS + altitude_km * 1e3
        v = np.sqrt(EARTH_MU / r)
        # Verify via vis-viva for circular orbit (a = r)
        v_vis_viva = np.sqrt(EARTH_MU * (2.0 / r - 1.0 / r))
        assert_allclose(v, v_vis_viva, rtol=1e-14)


# =============================================================================
# Test: Keplerian-Cartesian round-trip
# =============================================================================

class TestKeplerianCartesianRoundTrip:
    """Tests for converting between Keplerian elements and Cartesian state."""

    def test_keplerian_cartesian_roundtrip(self):
        """Convert Keplerian -> Cartesian -> Keplerian and check values match."""
        a_orig = EARTH_RADIUS + 500e3  # 500 km altitude
        e_orig = 0.01
        i_orig = np.radians(51.6)  # ISS inclination
        raan_orig = np.radians(30.0)
        omega_orig = np.radians(45.0)
        nu_orig = np.radians(60.0)

        r, v = keplerian_to_cartesian(a_orig, e_orig, i_orig, raan_orig,
                                       omega_orig, nu_orig, EARTH_MU)
        a, e, inc, raan, omega, nu = cartesian_to_keplerian(r, v, EARTH_MU)

        assert_allclose(a, a_orig, rtol=1e-10)
        assert_allclose(e, e_orig, atol=1e-12)
        assert_allclose(inc, i_orig, atol=1e-12)
        assert_allclose(raan, raan_orig, atol=1e-10)
        assert_allclose(omega, omega_orig, atol=1e-10)
        assert_allclose(nu, nu_orig, atol=1e-10)

    @pytest.mark.parametrize("e_val", [0.0, 0.1, 0.5, 0.9])
    def test_roundtrip_various_eccentricities(self, e_val):
        """Round-trip for various eccentricities."""
        a = EARTH_RADIUS + 1000e3
        # For e=0, omega and nu are degenerate -- skip tight comparison on those
        r, v = keplerian_to_cartesian(a, e_val, np.radians(28.5), 0.0, 0.0,
                                       np.radians(90.0), EARTH_MU)
        a2, e2, _, _, _, _ = cartesian_to_keplerian(r, v, EARTH_MU)
        assert_allclose(a2, a, rtol=1e-10)
        assert_allclose(e2, e_val, atol=1e-12)


# =============================================================================
# Test: Hohmann transfer
# =============================================================================

class TestHohmannTransfer:
    """Tests for Hohmann transfer delta-V computation."""

    def test_hohmann_transfer(self, planner):
        """Hohmann transfer from LEO to GEO: check known delta-V values."""
        r_leo = EARTH_RADIUS + 200e3
        r_geo = 42164e3  # GEO radius in meters

        dv1, dv2 = planner.hohmann_transfer(r_leo, r_geo, EARTH_MU)

        # Known values: approximately 2.46 km/s for first burn, 1.48 km/s for second
        assert 2300.0 < dv1 < 2600.0, f"dv1 = {dv1:.1f} m/s out of expected range"
        assert 1400.0 < dv2 < 1600.0, f"dv2 = {dv2:.1f} m/s out of expected range"

        # Total should be ~3.94 km/s
        total_dv = dv1 + dv2
        assert 3800.0 < total_dv < 4100.0, f"total dv = {total_dv:.1f} m/s"

    def test_hohmann_same_orbit(self, planner):
        """Hohmann transfer with r1 = r2 should give zero delta-V."""
        r = EARTH_RADIUS + 400e3
        dv1, dv2 = planner.hohmann_transfer(r, r, EARTH_MU)
        assert_allclose(dv1, 0.0, atol=1e-10)
        assert_allclose(dv2, 0.0, atol=1e-10)


# =============================================================================
# Test: Vis-viva equation
# =============================================================================

class TestVisViva:
    """Tests for the vis-viva equation: v^2 = mu * (2/r - 1/a)."""

    def test_vis_viva(self):
        """Check periapsis and apoapsis velocities of a known elliptical orbit."""
        r_peri = EARTH_RADIUS + 200e3
        r_apo = EARTH_RADIUS + 35786e3  # Hohmann transfer orbit to GEO
        a = (r_peri + r_apo) / 2.0

        v_peri = np.sqrt(EARTH_MU * (2.0 / r_peri - 1.0 / a))
        v_apo = np.sqrt(EARTH_MU * (2.0 / r_apo - 1.0 / a))

        # Periapsis velocity should be higher than apoapsis
        assert v_peri > v_apo

        # Check against circular velocity at periapsis (transfer is faster)
        v_circ_peri = np.sqrt(EARTH_MU / r_peri)
        assert v_peri > v_circ_peri

        # Specific energy should be the same at both points
        energy_peri = v_peri ** 2 / 2.0 - EARTH_MU / r_peri
        energy_apo = v_apo ** 2 / 2.0 - EARTH_MU / r_apo
        assert_allclose(energy_peri, energy_apo, rtol=1e-12)

    @pytest.mark.parametrize("r_peri_alt_km,r_apo_alt_km", [
        (200, 400),
        (200, 2000),
        (500, 35786),
    ])
    def test_vis_viva_energy_consistency(self, r_peri_alt_km, r_apo_alt_km):
        """Specific energy at periapsis and apoapsis must be equal."""
        r_peri = EARTH_RADIUS + r_peri_alt_km * 1e3
        r_apo = EARTH_RADIUS + r_apo_alt_km * 1e3
        a = (r_peri + r_apo) / 2.0

        v_peri = np.sqrt(EARTH_MU * (2.0 / r_peri - 1.0 / a))
        v_apo = np.sqrt(EARTH_MU * (2.0 / r_apo - 1.0 / a))

        e_peri = v_peri ** 2 / 2.0 - EARTH_MU / r_peri
        e_apo = v_apo ** 2 / 2.0 - EARTH_MU / r_apo
        assert_allclose(e_peri, e_apo, rtol=1e-12)


# =============================================================================
# Test: Lambert solver
# =============================================================================

class TestLambertSolver:
    """Tests for the Lambert problem solver."""

    def test_lambert_solver(self, optimizer):
        """Verify Lambert solver with a known Hohmann-like transfer."""
        r = EARTH_RADIUS + 200e3
        r1 = np.array([r, 0.0, 0.0])
        # Half-orbit transfer to the opposite side
        r2 = np.array([-r, 0.0, 0.0])

        # Time of flight for a circular orbit half-period
        T = np.pi * np.sqrt(r ** 3 / EARTH_MU)

        v1, v2 = optimizer.lambert_arc(r1, r2, T, EARTH_MU, prograde=True)

        # For a circular orbit, |v1| should equal circular velocity
        v_circ = np.sqrt(EARTH_MU / r)
        assert_allclose(np.linalg.norm(v1), v_circ, rtol=0.01)

    def test_lambert_raises_for_zero_tof(self, optimizer):
        """Lambert solver should raise ValueError for zero time of flight."""
        r1 = np.array([7000e3, 0.0, 0.0])
        r2 = np.array([0.0, 7000e3, 0.0])
        with pytest.raises(ValueError):
            optimizer.lambert_arc(r1, r2, 0.0, EARTH_MU)


# =============================================================================
# Test: RK4 energy conservation
# =============================================================================

class TestRK4EnergyConservation:
    """Tests for energy conservation during numerical orbit propagation."""

    def test_rk4_energy_conservation(self):
        """Propagate a circular orbit with RK4 and check energy is conserved."""
        r0 = EARTH_RADIUS + 400e3
        v0_mag = np.sqrt(EARTH_MU / r0)

        state = np.array([r0, 0.0, 0.0, 0.0, v0_mag, 0.0])

        def two_body_eom(t, y):
            r_vec = y[0:3]
            v_vec = y[3:6]
            r_mag = np.linalg.norm(r_vec)
            a = -EARTH_MU / r_mag ** 3 * r_vec
            return np.concatenate([v_vec, a])

        # Compute initial energy
        r_mag_init = np.linalg.norm(state[0:3])
        v_mag_init = np.linalg.norm(state[3:6])
        energy_init = v_mag_init ** 2 / 2.0 - EARTH_MU / r_mag_init

        # Propagate for one full orbit
        T_orbit = 2.0 * np.pi * np.sqrt(r0 ** 3 / EARTH_MU)
        dt = 10.0  # 10-second time step
        n_steps = int(T_orbit / dt)

        y = state.copy()
        for step in range(n_steps):
            y = rk4_step(two_body_eom, step * dt, y, dt)

        # Final energy
        r_mag_final = np.linalg.norm(y[0:3])
        v_mag_final = np.linalg.norm(y[3:6])
        energy_final = v_mag_final ** 2 / 2.0 - EARTH_MU / r_mag_final

        # Energy should be conserved within a tight tolerance for RK4
        assert_allclose(energy_final, energy_init, rtol=1e-8)


# =============================================================================
# Test: J2 RAAN regression
# =============================================================================

class TestJ2Regression:
    """Tests for J2-induced RAAN drift."""

    def test_j2_regression(self):
        """
        J2 causes RAAN to drift. For a prograde orbit (i < 90 deg), the
        drift should be negative (westward regression). Check sign and
        order of magnitude.

        Analytical secular rate:
            dOmega/dt = -3/2 * n * J2 * (R_e/p)^2 * cos(i)

        where n = sqrt(mu/a^3), p = a*(1-e^2).
        """
        a = EARTH_RADIUS + 400e3  # 400 km altitude
        e = 0.001
        i = np.radians(51.6)  # ISS inclination

        p = a * (1 - e ** 2)
        n = np.sqrt(EARTH_MU / a ** 3)

        # Analytical secular RAAN drift rate
        dOmega_dt = (-3.0 / 2.0) * n * EARTH_J2 * (
            EARTH_EQUATORIAL_RADIUS / p
        ) ** 2 * np.cos(i)

        # RAAN drift should be negative for prograde orbit
        assert dOmega_dt < 0, "RAAN drift should be negative (westward) for prograde orbit"

        # Convert to degrees per day
        dOmega_deg_per_day = np.degrees(dOmega_dt) * 86400.0

        # For ISS-like orbit, RAAN regression is about -5 deg/day
        assert -10.0 < dOmega_deg_per_day < -2.0, (
            f"RAAN drift = {dOmega_deg_per_day:.2f} deg/day, "
            "expected between -10 and -2 for ISS-like orbit"
        )


# =============================================================================
# Test: Escape velocity
# =============================================================================

class TestEscapeVelocity:
    """Tests for escape velocity: v_escape = sqrt(2*mu/r)."""

    def test_escape_velocity(self):
        """Compare escape velocity with analytical formula at Earth surface."""
        r = EARTH_RADIUS
        v_escape_analytical = np.sqrt(2.0 * EARTH_MU / r)

        # Should be about 11.2 km/s
        assert 11000.0 < v_escape_analytical < 11300.0

        # Also verify via energy: at escape, total energy = 0
        # E = v^2/2 - mu/r = 0 => v = sqrt(2*mu/r)
        energy = v_escape_analytical ** 2 / 2.0 - EARTH_MU / r
        assert_allclose(energy, 0.0, atol=1e-4)

    @pytest.mark.parametrize("altitude_km", [0, 200, 400, 35786])
    def test_escape_velocity_various_altitudes(self, altitude_km):
        """Escape velocity at various altitudes satisfies v = sqrt(2*mu/r)."""
        r = EARTH_RADIUS + altitude_km * 1e3
        v_esc = np.sqrt(2.0 * EARTH_MU / r)
        # Escape velocity should be sqrt(2) times circular velocity
        v_circ = np.sqrt(EARTH_MU / r)
        assert_allclose(v_esc, np.sqrt(2) * v_circ, rtol=1e-14)
