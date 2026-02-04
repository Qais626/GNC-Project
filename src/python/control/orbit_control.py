"""
Orbit Control System
====================

Plans and executes orbital maneuvers (delta-V burns) for a multi-phase
space mission.  Supported maneuver types:

  - Hohmann transfer (coplanar circular-to-circular)
  - Orbit insertion (lunar or Jovian capture)
  - Plane change (inclination adjustment)
  - Escape burn (hyperbolic departure)
  - Station-keeping (small correction burns)

The controller models *finite* burns rather than impulsive approximations.
Gravity losses during extended burns are estimated and compensated so that
the achieved delta-V matches the mission plan.

Propellant consumption follows the Tsiolkovsky rocket equation:

    delta_m = m_0 * (1 - exp(-dv / (g0 * Isp)))

Thrust is assumed constant during a burn; mass decreases linearly with
propellant flow rate.

References
----------
  Bate, Mueller, White: "Fundamentals of Astrodynamics", Dover, 1971.
  Curtis, H.D.: "Orbital Mechanics for Engineering Students", 4th ed.
"""

import numpy as np
from dataclasses import dataclass, field
from core.constants import (
    MU_EARTH,
    MU_MOON,
    MU_JUPITER,
    G0,
    DEG2RAD,
    RAD2DEG,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BurnProfile:
    """
    Describes a single finite-duration thrust arc.

    Attributes
    ----------
    start_time : float
        Mission elapsed time at ignition [s].
    duration : float
        Commanded burn duration [s].
    direction_eci : np.ndarray
        Unit vector giving the thrust direction in the ECI frame.
    delta_v_mag : float
        Desired impulsive delta-V magnitude [m/s] (before gravity-loss
        correction).
    thrust_level : float
        Engine thrust [N].
    isp : float
        Specific impulse [s].
    propellant_mass : float
        Propellant required for this burn [kg].
    gravity_loss_fraction : float
        Estimated fraction of delta-V lost to gravity during the finite
        burn (0..1).
    """
    start_time: float = 0.0
    duration: float = 0.0
    direction_eci: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    delta_v_mag: float = 0.0
    thrust_level: float = 0.0
    isp: float = 0.0
    propellant_mass: float = 0.0
    gravity_loss_fraction: float = 0.0


# ---------------------------------------------------------------------------
# Orbit Controller
# ---------------------------------------------------------------------------

class OrbitController:
    """
    Plans and executes orbital maneuvers.

    The controller operates in two phases for every burn:

      1. **Planning** -- compute the required delta-V, burn duration,
         propellant budget, and gravity-loss margin.
      2. **Execution** -- step through the burn at the simulation timestep,
         applying thrust forces and decrementing propellant mass.

    All velocities are in [m/s], distances in [m], masses in [kg], and
    angles in [rad] unless otherwise noted.
    """

    def __init__(self):
        # Book-keeping for the active burn
        self._active_burn = None          # type: BurnProfile | None
        self._burn_elapsed = 0.0          # elapsed time since ignition [s]

    # ------------------------------------------------------------------
    # Burn planning
    # ------------------------------------------------------------------

    def plan_burn(self, delta_v_vec, spacecraft_mass, thrust, isp,
                  start_time=0.0):
        """
        Plan a finite-duration burn to achieve a given delta-V vector.

        The burn duration is derived from the Tsiolkovsky rocket equation:

            dm = m0 * (1 - exp(-|dv| / (g0 * Isp)))
            t_burn = dm * g0 * Isp / F

        where F is thrust and g0 is standard gravity.

        Parameters
        ----------
        delta_v_vec : ndarray (3,)
            Desired delta-V vector in ECI [m/s].
        spacecraft_mass : float
            Total wet mass at burn start [kg].
        thrust : float
            Engine thrust level [N].
        isp : float
            Specific impulse [s].
        start_time : float
            Mission elapsed time at ignition [s].

        Returns
        -------
        burn : BurnProfile
            Fully populated burn profile.
        """
        dv_vec = np.asarray(delta_v_vec, dtype=float)
        dv_mag = np.linalg.norm(dv_vec)

        # Thrust direction (unit vector)
        if dv_mag > 1e-12:
            direction = dv_vec / dv_mag
        else:
            direction = np.array([1.0, 0.0, 0.0])

        # Propellant mass from Tsiolkovsky:
        #   m_prop = m0 * (1 - exp(-dv / (g0 * Isp)))
        exhaust_velocity = G0 * isp   # effective exhaust velocity [m/s]
        mass_ratio = np.exp(-dv_mag / exhaust_velocity)
        propellant_mass = spacecraft_mass * (1.0 - mass_ratio)

        # Burn duration:
        #   m_dot = F / (g0 * Isp)
        #   t_burn = m_prop / m_dot = m_prop * g0 * Isp / F
        if thrust > 0:
            burn_duration = propellant_mass * exhaust_velocity / thrust
        else:
            burn_duration = 0.0

        burn = BurnProfile(
            start_time=start_time,
            duration=burn_duration,
            direction_eci=direction,
            delta_v_mag=dv_mag,
            thrust_level=thrust,
            isp=isp,
            propellant_mass=propellant_mass,
            gravity_loss_fraction=0.0,       # filled in by caller if needed
        )

        self._active_burn = burn
        self._burn_elapsed = 0.0
        return burn

    # ------------------------------------------------------------------
    # Burn execution
    # ------------------------------------------------------------------

    def execute_burn_step(self, spacecraft_mass, burn_profile, dt):
        """
        Execute one timestep of a finite burn.

        Thrust is applied in the commanded direction.  Propellant mass is
        decremented according to the mass flow rate:

            m_dot = F / (g0 * Isp)

        Parameters
        ----------
        spacecraft_mass : float
            Current wet mass [kg].
        burn_profile : BurnProfile
            Active burn profile.
        dt : float
            Simulation timestep [s].

        Returns
        -------
        force_eci : ndarray (3,)
            Thrust force vector in ECI [N].
        mass_consumed : float
            Propellant consumed during this step [kg].
        burn_complete : bool
            True if the burn has finished.
        """
        if self._burn_elapsed >= burn_profile.duration:
            return np.zeros(3), 0.0, True

        # Remaining burn time may be less than dt
        dt_actual = min(dt, burn_profile.duration - self._burn_elapsed)

        # Force vector
        force_eci = burn_profile.thrust_level * burn_profile.direction_eci

        # Mass consumed this step
        exhaust_velocity = G0 * burn_profile.isp
        m_dot = burn_profile.thrust_level / exhaust_velocity  # [kg/s]
        mass_consumed = m_dot * dt_actual

        # Safety: do not consume more than available mass
        mass_consumed = min(mass_consumed, spacecraft_mass * 0.95)

        self._burn_elapsed += dt_actual
        burn_complete = self._burn_elapsed >= burn_profile.duration

        return force_eci, mass_consumed, burn_complete

    # ------------------------------------------------------------------
    # Gravity-loss estimation
    # ------------------------------------------------------------------

    def compute_finite_burn_loss(self, dv_impulsive, thrust, mass, isp,
                                 g_local, v_orbit):
        """
        Estimate the gravity loss for a finite-duration burn.

        During a non-impulsive burn the spacecraft continues to fall in the
        gravity field, so the achieved delta-V is less than the ideal
        (impulsive) value.  A first-order approximation is:

            dv_loss ~ (g_local * t_burn^2) / (2 * t_burn)
                    = g_local * t_burn / 2

        Fractional loss:
            loss_frac ~ dv_loss / dv_impulsive

        The caller should add dv_loss to the planned delta-V so that the
        net effect matches the impulsive maneuver.

        Parameters
        ----------
        dv_impulsive : float
            Ideal impulsive delta-V [m/s].
        thrust : float
            Engine thrust [N].
        mass : float
            Spacecraft mass at ignition [kg].
        isp : float
            Specific impulse [s].
        g_local : float
            Local gravitational acceleration [m/s^2].
        v_orbit : float
            Current orbital velocity [m/s] (used for higher-order loss if
            desired; kept here for interface consistency).

        Returns
        -------
        dv_actual_needed : float
            Total delta-V including gravity loss compensation [m/s].
        loss_fraction : float
            Fraction of delta-V lost to gravity (0..1).
        """
        # Burn duration estimate (simplified, constant mass)
        exhaust_velocity = G0 * isp
        mass_ratio = np.exp(-dv_impulsive / exhaust_velocity)
        prop_mass = mass * (1.0 - mass_ratio)
        if thrust > 0:
            t_burn = prop_mass * exhaust_velocity / thrust
        else:
            return dv_impulsive, 0.0

        # First-order gravity loss
        dv_loss = g_local * t_burn / 2.0

        # Second-order correction using orbital velocity (small for LEO)
        # dv_loss_2nd ~ dv_impulsive * t_burn * g_local / (2 * v_orbit)
        if v_orbit > 1.0:
            dv_loss += dv_impulsive * t_burn * g_local / (2.0 * v_orbit)

        dv_actual = dv_impulsive + dv_loss
        loss_fraction = dv_loss / dv_impulsive if dv_impulsive > 0 else 0.0

        return dv_actual, loss_fraction

    # ------------------------------------------------------------------
    # Maneuver delta-V computations
    # ------------------------------------------------------------------

    def compute_inclination_change(self, v_orbit, delta_inc_rad):
        """
        Delta-V for a pure inclination change at the ascending/descending
        node.

            dv = 2 * v * sin(delta_i / 2)

        This is the most expensive in-plane vs. out-of-plane comparison;
        combined maneuvers at apoapsis are more efficient for large changes.

        Parameters
        ----------
        v_orbit : float
            Orbital velocity at the maneuver point [m/s].
        delta_inc_rad : float
            Inclination change [rad].

        Returns
        -------
        dv : float
            Required delta-V [m/s].
        """
        return 2.0 * v_orbit * np.sin(abs(delta_inc_rad) / 2.0)

    def compute_lunar_orbit_insertion(self, v_approach, v_circular_target):
        """
        Delta-V for lunar orbit insertion (LOI).

        The spacecraft arrives on a hyperbolic approach trajectory and must
        decelerate into a circular orbit around the Moon:

            dv = |v_approach - v_circular|

        A more precise model would account for the geometry of the
        hyperbolic excess velocity, but for preliminary design the
        difference of speeds is a good first estimate.

        Parameters
        ----------
        v_approach : float
            Hyperbolic approach speed at periselene [m/s].
        v_circular_target : float
            Target circular orbit speed [m/s].

        Returns
        -------
        dv : float
            Insertion delta-V [m/s].
        """
        return abs(v_approach - v_circular_target)

    def compute_jupiter_orbit_insertion(self, v_approach, v_circular_target):
        """
        Delta-V for Jupiter orbit insertion (JOI).

        Same velocity-matching formula as LOI.  Jupiter's deep gravity well
        means v_approach is very large, leading to substantial delta-V
        requirements.

        Parameters
        ----------
        v_approach : float
            Hyperbolic approach speed at perijove [m/s].
        v_circular_target : float
            Target circular orbit speed around Jupiter [m/s].

        Returns
        -------
        dv : float
            Insertion delta-V [m/s].
        """
        return abs(v_approach - v_circular_target)

    def compute_escape_burn(self, v_circular, v_infinity_needed, mu, r):
        """
        Delta-V for a hyperbolic escape from a circular parking orbit.

        From the vis-viva equation at the parking orbit radius r:

            v_escape = sqrt(v_inf^2 + 2*mu/r)

        The delta-V is the difference between the escape speed and the
        circular orbit speed:

            dv = v_escape - v_circular

        Parameters
        ----------
        v_circular : float
            Circular orbit speed [m/s].
        v_infinity_needed : float
            Required hyperbolic excess velocity [m/s].
        mu : float
            Gravitational parameter of the central body [m^3/s^2].
        r : float
            Parking orbit radius [m].

        Returns
        -------
        dv : float
            Escape burn delta-V [m/s].
        """
        # Speed on the escape hyperbola at radius r
        v_escape = np.sqrt(v_infinity_needed**2 + 2.0 * mu / r)
        dv = v_escape - v_circular
        return dv

    def compute_hohmann(self, r1, r2, mu):
        """
        Compute the two delta-V impulses for a coplanar Hohmann transfer
        between two circular orbits.

        Impulse 1 (at r1):
            dv1 = sqrt(mu/r1) * (sqrt(2*r2/(r1+r2)) - 1)

        Impulse 2 (at r2):
            dv2 = sqrt(mu/r2) * (1 - sqrt(2*r1/(r1+r2)))

        Transfer orbit semi-major axis:
            a_transfer = (r1 + r2) / 2

        Transfer time:
            t_transfer = pi * sqrt(a_transfer^3 / mu)

        Parameters
        ----------
        r1 : float
            Radius of the initial circular orbit [m].
        r2 : float
            Radius of the final circular orbit [m].
        mu : float
            Gravitational parameter [m^3/s^2].

        Returns
        -------
        dv1 : float
            First impulse delta-V [m/s] (prograde at r1).
        dv2 : float
            Second impulse delta-V [m/s] (prograde at r2).
        transfer_time : float
            Half-period of the transfer ellipse [s].
        """
        # Circular velocities
        v1 = np.sqrt(mu / r1)
        v2 = np.sqrt(mu / r2)

        # Transfer ellipse semi-major axis
        a_transfer = (r1 + r2) / 2.0

        # Velocities on the transfer ellipse at periapsis (r1) and apoapsis (r2)
        v_transfer_peri = np.sqrt(mu * (2.0 / r1 - 1.0 / a_transfer))
        v_transfer_apo = np.sqrt(mu * (2.0 / r2 - 1.0 / a_transfer))

        # Delta-V at each impulse point
        dv1 = v_transfer_peri - v1
        dv2 = v2 - v_transfer_apo

        # Transfer time (half the orbital period of the transfer ellipse)
        transfer_time = np.pi * np.sqrt(a_transfer**3 / mu)

        return dv1, dv2, transfer_time

    # ------------------------------------------------------------------
    # Status query
    # ------------------------------------------------------------------

    def is_burn_complete(self, burn_profile, elapsed_time=None):
        """
        Check whether a burn has been completed.

        Parameters
        ----------
        burn_profile : BurnProfile
            The burn to check.
        elapsed_time : float or None
            If provided, the total mission elapsed time [s].  The burn is
            complete when elapsed_time >= start_time + duration.  If None,
            uses the internal burn elapsed counter.

        Returns
        -------
        complete : bool
        """
        if elapsed_time is not None:
            return elapsed_time >= (burn_profile.start_time
                                    + burn_profile.duration)
        return self._burn_elapsed >= burn_profile.duration

    def get_burn_progress(self):
        """
        Return the fraction of the current burn that has been executed.

        Returns
        -------
        fraction : float
            0.0 (not started) to 1.0 (complete).
        """
        if self._active_burn is None or self._active_burn.duration <= 0:
            return 0.0
        return min(self._burn_elapsed / self._active_burn.duration, 1.0)
