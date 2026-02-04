"""
===============================================================================
GNC PROJECT - Maneuver Planner
===============================================================================
Delta-V computation and maneuver planning for all orbital transfers in the
Earth-Moon-Jupiter round-trip mission.

This module provides classical orbital mechanics maneuver calculations
including Hohmann transfers, bi-elliptic transfers, plane changes, orbit
insertion burns, escape maneuvers, and gravity assists. All methods include
detailed docstrings with the governing equations.

Sign conventions and units:
    - All distances in meters
    - All velocities in m/s
    - All masses in kg
    - All angles in radians
    - Gravitational parameters (mu) in m^3/s^2
    - Specific impulse (Isp) in seconds
    - g0 = 9.80665 m/s^2 (standard gravity)
===============================================================================
"""

import logging
from typing import Tuple

import numpy as np

from core.constants import (
    EARTH_MU,
    EARTH_RADIUS,
    JUPITER_MU,
    JUPITER_RADIUS,
    MOON_MU,
    MOON_RADIUS,
)

logger = logging.getLogger(__name__)

# Standard gravitational acceleration (m/s^2), used in the rocket equation
G0 = 9.80665


class ManeuverPlanner:
    """
    Computes delta-V and propellant requirements for all mission maneuvers.

    Each method implements a classical orbital mechanics formula and returns
    the required delta-V. The planner is stateless: all inputs are passed
    as arguments and results are returned directly.

    Typical usage:
        planner = ManeuverPlanner()
        dv1, dv2 = planner.hohmann_transfer(r_leo, r_geo, EARTH_MU)
        mp = planner.propellant_mass(dv1 + dv2, isp=320, dry_mass=5000)
    """

    # -------------------------------------------------------------------------
    # Hohmann Transfer
    # -------------------------------------------------------------------------

    def hohmann_transfer(
        self,
        r1: float,
        r2: float,
        mu: float,
    ) -> Tuple[float, float]:
        """
        Compute the delta-V for a two-impulse Hohmann transfer between
        coplanar circular orbits.

        The Hohmann transfer is the minimum-energy two-impulse transfer
        between two coplanar circular orbits. It uses an elliptical transfer
        orbit tangent to both the initial and final orbits.

        Equations:
            Transfer orbit semi-major axis:
                a_t = (r1 + r2) / 2

            Velocity on initial circular orbit:
                v_c1 = sqrt(mu / r1)

            Velocity at periapsis of transfer orbit:
                v_t1 = sqrt(mu * (2/r1 - 1/a_t))

            Velocity at apoapsis of transfer orbit:
                v_t2 = sqrt(mu * (2/r2 - 1/a_t))

            Velocity on final circular orbit:
                v_c2 = sqrt(mu / r2)

            First burn (at r1):
                dv1 = |v_t1 - v_c1|

            Second burn (at r2):
                dv2 = |v_c2 - v_t2|

        Args:
            r1: Radius of initial circular orbit (m).
            r2: Radius of final circular orbit (m).
            mu: Gravitational parameter of central body (m^3/s^2).

        Returns:
            (dv1, dv2): Delta-V magnitudes for the two burns (m/s).
                        dv1 is at departure, dv2 is at arrival.
        """
        a_transfer = (r1 + r2) / 2.0

        v_circ_1 = np.sqrt(mu / r1)
        v_circ_2 = np.sqrt(mu / r2)

        v_transfer_periapsis = np.sqrt(mu * (2.0 / r1 - 1.0 / a_transfer))
        v_transfer_apoapsis = np.sqrt(mu * (2.0 / r2 - 1.0 / a_transfer))

        dv1 = abs(v_transfer_periapsis - v_circ_1)
        dv2 = abs(v_circ_2 - v_transfer_apoapsis)

        logger.debug(
            "Hohmann transfer: r1=%.0f m, r2=%.0f m, dv1=%.1f m/s, dv2=%.1f m/s",
            r1, r2, dv1, dv2,
        )
        return dv1, dv2

    # -------------------------------------------------------------------------
    # Bi-Elliptic Transfer
    # -------------------------------------------------------------------------

    def bi_elliptic_transfer(
        self,
        r1: float,
        r2: float,
        r_apoapsis: float,
        mu: float,
    ) -> Tuple[float, float, float]:
        """
        Compute the delta-V for a three-impulse bi-elliptic transfer.

        A bi-elliptic transfer uses an intermediate apoapsis r_apoapsis that
        is higher than both r1 and r2. This can be more fuel-efficient than
        a Hohmann transfer when r2/r1 > 11.94.

        Equations:
            First transfer ellipse (r1 to r_apoapsis):
                a_1 = (r1 + r_apoapsis) / 2
                v_1_circ = sqrt(mu / r1)
                v_1_trans = sqrt(mu * (2/r1 - 1/a_1))
                dv1 = |v_1_trans - v_1_circ|

            At apoapsis (transition between ellipses):
                a_2 = (r2 + r_apoapsis) / 2
                v_apo_1 = sqrt(mu * (2/r_apoapsis - 1/a_1))
                v_apo_2 = sqrt(mu * (2/r_apoapsis - 1/a_2))
                dv2 = |v_apo_2 - v_apo_1|

            Final circularization at r2:
                v_2_trans = sqrt(mu * (2/r2 - 1/a_2))
                v_2_circ = sqrt(mu / r2)
                dv3 = |v_2_circ - v_2_trans|

        Args:
            r1: Radius of initial circular orbit (m).
            r2: Radius of final circular orbit (m).
            r_apoapsis: Intermediate apoapsis radius (m). Must be >= max(r1, r2).
            mu: Gravitational parameter (m^3/s^2).

        Returns:
            (dv1, dv2, dv3): Delta-V magnitudes for the three burns (m/s).

        Raises:
            ValueError: If r_apoapsis < max(r1, r2).
        """
        if r_apoapsis < max(r1, r2):
            raise ValueError(
                f"r_apoapsis ({r_apoapsis:.0f} m) must be >= max(r1, r2) = "
                f"{max(r1, r2):.0f} m"
            )

        # First transfer ellipse: r1 -> r_apoapsis
        a1 = (r1 + r_apoapsis) / 2.0
        v_circ_1 = np.sqrt(mu / r1)
        v_trans_1_peri = np.sqrt(mu * (2.0 / r1 - 1.0 / a1))
        dv1 = abs(v_trans_1_peri - v_circ_1)

        # At apoapsis: transition from first ellipse to second
        a2 = (r2 + r_apoapsis) / 2.0
        v_apo_1 = np.sqrt(mu * (2.0 / r_apoapsis - 1.0 / a1))
        v_apo_2 = np.sqrt(mu * (2.0 / r_apoapsis - 1.0 / a2))
        dv2 = abs(v_apo_2 - v_apo_1)

        # Circularize at r2
        v_trans_2_peri = np.sqrt(mu * (2.0 / r2 - 1.0 / a2))
        v_circ_2 = np.sqrt(mu / r2)
        dv3 = abs(v_circ_2 - v_trans_2_peri)

        logger.debug(
            "Bi-elliptic transfer: r1=%.0f, r2=%.0f, r_apo=%.0f -> "
            "dv1=%.1f, dv2=%.1f, dv3=%.1f m/s",
            r1, r2, r_apoapsis, dv1, dv2, dv3,
        )
        return dv1, dv2, dv3

    # -------------------------------------------------------------------------
    # Plane Change
    # -------------------------------------------------------------------------

    def plane_change(
        self,
        velocity: float,
        delta_inclination: float,
    ) -> float:
        """
        Compute the delta-V for a simple plane change maneuver.

        A pure plane change rotates the orbital plane by an angle
        delta_inclination without changing the orbit size or shape.
        The maneuver is most efficient when performed at the lowest
        velocity point (apoapsis for elliptic orbits).

        Equation:
            dv = 2 * v * sin(delta_i / 2)

        This comes from the vector triangle: the velocity magnitude stays
        the same, but the direction changes by delta_i. The delta-V is the
        base of the isoceles triangle with sides of length v and apex angle
        delta_i.

        Args:
            velocity: Orbital velocity at the maneuver point (m/s).
            delta_inclination: Desired inclination change (radians).

        Returns:
            Delta-V magnitude for the plane change (m/s).
        """
        dv = 2.0 * velocity * np.sin(abs(delta_inclination) / 2.0)
        logger.debug(
            "Plane change: v=%.1f m/s, di=%.2f deg -> dv=%.1f m/s",
            velocity, np.degrees(delta_inclination), dv,
        )
        return dv

    # -------------------------------------------------------------------------
    # Combined Plane Change and Transfer
    # -------------------------------------------------------------------------

    def combined_plane_change_and_transfer(
        self,
        r1: float,
        r2: float,
        inc_change: float,
        mu: float,
    ) -> float:
        """
        Compute the delta-V for a combined plane change and orbit transfer.

        Rather than performing a separate plane change and Hohmann transfer,
        it is more efficient to combine them. The plane change is incorporated
        into one of the Hohmann burns (typically the one at higher altitude
        where the velocity is lower, making the plane change cheaper).

        Equation (combined at the second burn):
            a_t = (r1 + r2) / 2

            dv1 = v_t1 - v_c1    (tangential burn at r1, no plane change)

            dv2 = sqrt(v_c2^2 + v_t2^2 - 2*v_c2*v_t2*cos(delta_i))

            where v_c2 and v_t2 are the circular and transfer velocities at r2.

        This uses the cosine rule for vector subtraction when the velocity
        vectors are not parallel due to the inclination change.

        Args:
            r1: Initial orbit radius (m).
            r2: Final orbit radius (m).
            inc_change: Inclination change (radians).
            mu: Gravitational parameter (m^3/s^2).

        Returns:
            Total delta-V for the combined maneuver (m/s).
        """
        a_transfer = (r1 + r2) / 2.0

        v_circ_1 = np.sqrt(mu / r1)
        v_circ_2 = np.sqrt(mu / r2)
        v_trans_peri = np.sqrt(mu * (2.0 / r1 - 1.0 / a_transfer))
        v_trans_apo = np.sqrt(mu * (2.0 / r2 - 1.0 / a_transfer))

        # First burn: tangential at periapsis (no plane change component)
        dv1 = abs(v_trans_peri - v_circ_1)

        # Second burn: combined circularization + plane change at apoapsis
        # Using the law of cosines for velocity vector triangle
        dv2 = np.sqrt(
            v_circ_2**2 + v_trans_apo**2
            - 2.0 * v_circ_2 * v_trans_apo * np.cos(inc_change)
        )

        total_dv = dv1 + dv2
        logger.debug(
            "Combined transfer + plane change: dv1=%.1f, dv2=%.1f, total=%.1f m/s",
            dv1, dv2, total_dv,
        )
        return total_dv

    # -------------------------------------------------------------------------
    # Lunar Orbit Insertion (LOI)
    # -------------------------------------------------------------------------

    def lunar_orbit_insertion(
        self,
        v_approach: float,
        target_altitude: float,
        mu_moon: float = MOON_MU,
    ) -> float:
        """
        Compute the delta-V for Lunar Orbit Insertion (LOI).

        The spacecraft approaches the Moon on a hyperbolic trajectory. At
        periapsis, a retrograde burn decelerates it into the desired circular
        orbit.

        Equations:
            Target orbit radius:
                r_orbit = R_moon + target_altitude

            Hyperbolic excess velocity (v_infinity):
                The approach velocity v_approach is given in the Moon-centered
                frame. For a flyby, v_infinity = v_approach.

            Velocity at periapsis on the hyperbolic approach:
                v_peri_hyp = sqrt(v_approach^2 + 2 * mu_moon / r_orbit)

                This follows from the vis-viva equation for a hyperbolic orbit:
                v^2 = v_inf^2 + 2*mu/r

            Circular orbit velocity at target altitude:
                v_circ = sqrt(mu_moon / r_orbit)

            LOI delta-V (retrograde burn):
                dv_loi = v_peri_hyp - v_circ

        Args:
            v_approach: Hyperbolic approach velocity relative to Moon (m/s).
            target_altitude: Desired circular orbit altitude above surface (m).
            mu_moon: Lunar gravitational parameter (m^3/s^2).

        Returns:
            LOI delta-V magnitude (m/s).
        """
        r_orbit = MOON_RADIUS + target_altitude

        v_peri_hyperbolic = np.sqrt(v_approach**2 + 2.0 * mu_moon / r_orbit)
        v_circular = np.sqrt(mu_moon / r_orbit)

        dv_loi = v_peri_hyperbolic - v_circular

        logger.debug(
            "LOI: v_approach=%.1f m/s, alt=%.0f m, dv=%.1f m/s",
            v_approach, target_altitude, dv_loi,
        )
        return dv_loi

    # -------------------------------------------------------------------------
    # Jupiter Orbit Insertion (JOI)
    # -------------------------------------------------------------------------

    def jupiter_orbit_insertion(
        self,
        v_approach: float,
        target_altitude: float,
        mu_jupiter: float = JUPITER_MU,
    ) -> float:
        """
        Compute the delta-V for Jupiter Orbit Insertion (JOI).

        The physics are identical to LOI but at Jupiter scale. Because
        Jupiter's gravity well is much deeper, the hyperbolic approach
        velocity at periapsis is very high, and aerobraking is not feasible
        due to the extreme radiation environment.

        Equations:
            r_orbit = R_jupiter + target_altitude

            v_peri_hyp = sqrt(v_approach^2 + 2 * mu_jupiter / r_orbit)

            v_circ = sqrt(mu_jupiter / r_orbit)

            dv_joi = v_peri_hyp - v_circ

        Note: For a capture into a highly elliptical orbit (like Juno),
        a smaller delta-V is needed since v_capture < v_circular. This
        function computes the delta-V for circular orbit capture.

        Args:
            v_approach: Hyperbolic approach velocity relative to Jupiter (m/s).
            target_altitude: Desired circular orbit altitude above cloud tops (m).
            mu_jupiter: Jupiter gravitational parameter (m^3/s^2).

        Returns:
            JOI delta-V magnitude (m/s).
        """
        r_orbit = JUPITER_RADIUS + target_altitude

        v_peri_hyperbolic = np.sqrt(v_approach**2 + 2.0 * mu_jupiter / r_orbit)
        v_circular = np.sqrt(mu_jupiter / r_orbit)

        dv_joi = v_peri_hyperbolic - v_circular

        logger.debug(
            "JOI: v_approach=%.1f m/s, alt=%.0f m, dv=%.1f m/s",
            v_approach, target_altitude, dv_joi,
        )
        return dv_joi

    # -------------------------------------------------------------------------
    # Escape Maneuver
    # -------------------------------------------------------------------------

    def escape_maneuver(
        self,
        r_orbit: float,
        mu_body: float,
        v_infinity: float,
    ) -> float:
        """
        Compute the delta-V to escape from a circular orbit with a desired
        hyperbolic excess velocity.

        The spacecraft is initially in a circular orbit at radius r_orbit.
        A prograde burn at that point places it on a hyperbolic escape
        trajectory with asymptotic velocity v_infinity.

        Equations:
            Circular orbit velocity:
                v_circ = sqrt(mu / r_orbit)

            Required velocity at periapsis of escape hyperbola:
                v_escape = sqrt(v_infinity^2 + 2 * mu / r_orbit)

                This comes from energy conservation: the specific orbital
                energy of the hyperbola is:
                    E = v_infinity^2 / 2 = v^2/2 - mu/r

                Solving for v at r = r_orbit gives the equation above.

            Escape delta-V:
                dv = v_escape - v_circ

        This is the Oberth effect in action: burning at low altitude (deep
        in the gravity well) converts the burn more efficiently into
        hyperbolic excess velocity.

        Args:
            r_orbit: Radius of the circular departure orbit (m).
            mu_body: Gravitational parameter of the body being escaped (m^3/s^2).
            v_infinity: Desired hyperbolic excess velocity (m/s).

        Returns:
            Escape delta-V magnitude (m/s).
        """
        v_circ = np.sqrt(mu_body / r_orbit)
        v_periapsis = np.sqrt(v_infinity**2 + 2.0 * mu_body / r_orbit)

        dv = v_periapsis - v_circ

        logger.debug(
            "Escape: r_orbit=%.0f m, v_inf=%.1f m/s, dv=%.1f m/s",
            r_orbit, v_infinity, dv,
        )
        return dv

    # -------------------------------------------------------------------------
    # Powered Gravity Assist
    # -------------------------------------------------------------------------

    def powered_gravity_assist(
        self,
        v_in: np.ndarray,
        v_out: np.ndarray,
        periapsis: float,
        mu_body: float,
    ) -> float:
        """
        Compute the delta-V for a powered gravity assist (flyby with burn).

        In a powered gravity assist, the spacecraft performs an unpowered
        hyperbolic flyby that naturally bends the trajectory, plus a
        propulsive burn at periapsis to achieve the desired exit velocity.

        The flyby itself is free (no fuel cost) and provides a velocity
        change through gravitational deflection. The powered component adds
        additional delta-V at periapsis where the Oberth effect maximizes
        efficiency.

        Equations:
            Incoming v_infinity:
                v_inf_in = |v_in|

            Outgoing v_infinity:
                v_inf_out = |v_out|

            Velocity at periapsis for the incoming hyperbola:
                v_peri_in = sqrt(v_inf_in^2 + 2 * mu / r_periapsis)

            Velocity at periapsis for the outgoing hyperbola:
                v_peri_out = sqrt(v_inf_out^2 + 2 * mu / r_periapsis)

            Required delta-V at periapsis:
                dv = |v_peri_out - v_peri_in|

            Note: If v_inf_in == v_inf_out, a pure (unpowered) gravity assist
            is sufficient and dv = 0.

        Args:
            v_in: Incoming velocity vector relative to the body (m/s), shape (3,).
            v_out: Desired outgoing velocity vector relative to body (m/s), shape (3,).
            periapsis: Closest approach distance from body center (m).
            mu_body: Gravitational parameter of the flyby body (m^3/s^2).

        Returns:
            Delta-V magnitude at periapsis (m/s). Returns 0 if the natural
            flyby deflection is sufficient.
        """
        v_in = np.asarray(v_in, dtype=np.float64)
        v_out = np.asarray(v_out, dtype=np.float64)

        v_inf_in = np.linalg.norm(v_in)
        v_inf_out = np.linalg.norm(v_out)

        # Velocities at periapsis on respective hyperbolas
        v_peri_in = np.sqrt(v_inf_in**2 + 2.0 * mu_body / periapsis)
        v_peri_out = np.sqrt(v_inf_out**2 + 2.0 * mu_body / periapsis)

        # The delta-V is the difference in periapsis speeds
        # (both are tangent to the orbit at periapsis for coplanar case)
        dv = abs(v_peri_out - v_peri_in)

        logger.debug(
            "Powered gravity assist: v_inf_in=%.1f, v_inf_out=%.1f, "
            "r_peri=%.0f m, dv=%.1f m/s",
            v_inf_in, v_inf_out, periapsis, dv,
        )
        return dv

    # -------------------------------------------------------------------------
    # Finite Burn Loss
    # -------------------------------------------------------------------------

    def compute_finite_burn_loss(
        self,
        dv_impulsive: float,
        thrust: float,
        mass: float,
        isp: float,
    ) -> float:
        """
        Estimate the actual delta-V required when accounting for gravity
        losses during a finite-duration burn.

        Impulsive maneuver calculations assume instantaneous velocity changes.
        Real burns take finite time, during which gravity pulls the spacecraft
        off the optimal trajectory. This "gravity loss" increases the required
        delta-V.

        Equations:
            Burn time (from rocket equation rearranged):
                m_dot = thrust / (Isp * g0)
                m_propellant = m0 * (1 - exp(-dv_impulsive / (Isp * g0)))
                t_burn = m_propellant / m_dot

            Gravity loss approximation:
                For a burn of duration t_burn at acceleration a = thrust/mass,
                the gravity loss is approximately:

                dv_gravity_loss ~ (g_local^2 * t_burn^2) / (24 * v_orbit)

                where g_local is the local gravitational acceleration.

                A simpler first-order estimate:
                dv_gravity_loss ~ g_local * t_burn * (1 - cos(alpha)) / 2

                For small thrust-to-weight ratios (long burns), the loss can
                be significant (several percent of delta-V).

            Finite burn delta-V:
                dv_actual = dv_impulsive + dv_gravity_loss

        A simplified model is used here: the gravity loss fraction scales
        with the burn duration relative to the orbital period.

        Args:
            dv_impulsive: Ideal (impulsive) delta-V (m/s).
            thrust: Engine thrust (N).
            mass: Initial spacecraft mass (kg).
            isp: Specific impulse (s).

        Returns:
            Actual delta-V including gravity losses (m/s).
        """
        if thrust <= 0.0 or mass <= 0.0 or isp <= 0.0:
            raise ValueError("Thrust, mass, and Isp must all be positive.")

        exhaust_velocity = isp * G0
        mass_flow_rate = thrust / exhaust_velocity

        # Propellant mass for the impulsive maneuver
        m_prop = mass * (1.0 - np.exp(-dv_impulsive / exhaust_velocity))

        # Burn duration
        t_burn = m_prop / mass_flow_rate

        # Initial acceleration
        a0 = thrust / mass

        # Gravity loss estimate: for a spacecraft in orbit, the local
        # gravitational acceleration is approximately equal to the centripetal
        # acceleration a_grav ~ v^2/r. The gravity loss is proportional to
        # the square of the burn time and inversely proportional to the
        # initial acceleration.
        #
        # Simplified model: dv_loss ~ dv_impulsive * (dv_impulsive / (2 * a0 * t_burn))
        # This captures the essential physics: longer burns (lower T/W) have
        # higher losses.
        if a0 * t_burn > 0:
            loss_fraction = dv_impulsive / (2.0 * a0 * t_burn)
            # Clamp the loss to a reasonable range (0-20%)
            loss_fraction = min(loss_fraction, 0.20)
        else:
            loss_fraction = 0.0

        dv_actual = dv_impulsive * (1.0 + loss_fraction)

        logger.debug(
            "Finite burn loss: dv_imp=%.1f m/s, t_burn=%.1f s, "
            "loss=%.1f%%, dv_actual=%.1f m/s",
            dv_impulsive, t_burn, loss_fraction * 100, dv_actual,
        )
        return dv_actual

    # -------------------------------------------------------------------------
    # Tsiolkovsky Rocket Equation
    # -------------------------------------------------------------------------

    def tsiolkovsky_delta_v(
        self,
        isp: float,
        m0: float,
        mf: float,
    ) -> float:
        """
        Compute the delta-V from the Tsiolkovsky rocket equation.

        The rocket equation relates the change in velocity to the exhaust
        velocity and the mass ratio:

            dv = v_e * ln(m0 / mf)

        where:
            v_e = Isp * g0     is the effective exhaust velocity
            m0                 is the initial mass (with propellant)
            mf                 is the final mass (after burn, dry mass + payload)
            ln                 is the natural logarithm

        This is the fundamental equation of rocketry, derived from
        conservation of momentum applied to a variable-mass system.

        Args:
            isp: Specific impulse (seconds).
            m0: Initial (wet) mass (kg). Must be > mf.
            mf: Final (dry) mass (kg). Must be > 0.

        Returns:
            Delta-V in m/s.

        Raises:
            ValueError: If m0 <= mf or mf <= 0.
        """
        if mf <= 0.0:
            raise ValueError(f"Final mass must be positive, got {mf}")
        if m0 <= mf:
            raise ValueError(
                f"Initial mass ({m0} kg) must exceed final mass ({mf} kg)"
            )

        v_exhaust = isp * G0
        dv = v_exhaust * np.log(m0 / mf)

        logger.debug(
            "Tsiolkovsky: Isp=%.0f s, m0=%.0f kg, mf=%.0f kg -> dv=%.1f m/s",
            isp, m0, mf, dv,
        )
        return dv

    # -------------------------------------------------------------------------
    # Propellant Mass
    # -------------------------------------------------------------------------

    def propellant_mass(
        self,
        dv: float,
        isp: float,
        dry_mass: float,
    ) -> float:
        """
        Compute the required propellant mass for a given delta-V.

        Rearranging the Tsiolkovsky equation:

            dv = v_e * ln(m0 / mf)
            m0 / mf = exp(dv / v_e)
            m0 = mf * exp(dv / v_e)
            m_propellant = m0 - mf = mf * (exp(dv / v_e) - 1)

        where:
            v_e = Isp * g0
            mf = dry_mass (mass after all propellant is expended)

        Args:
            dv: Required delta-V (m/s).
            isp: Specific impulse (seconds).
            dry_mass: Mass of spacecraft without propellant (kg).

        Returns:
            Required propellant mass (kg).
        """
        if isp <= 0.0:
            raise ValueError(f"Isp must be positive, got {isp}")
        if dry_mass <= 0.0:
            raise ValueError(f"Dry mass must be positive, got {dry_mass}")

        v_exhaust = isp * G0
        mass_ratio = np.exp(dv / v_exhaust)
        m_propellant = dry_mass * (mass_ratio - 1.0)

        logger.debug(
            "Propellant: dv=%.1f m/s, Isp=%.0f s, m_dry=%.0f kg -> m_prop=%.0f kg",
            dv, isp, dry_mass, m_propellant,
        )
        return m_propellant

    # -------------------------------------------------------------------------
    # Mission-Specific Convenience Methods
    # -------------------------------------------------------------------------

    def compute_full_mission_budget(
        self,
        parking_orbit_alt: float = 200e3,
        lunar_orbit_alt: float = 100e3,
        lunar_inc_change: float = np.radians(15.0),
        jupiter_orbit_alt: float = 1e9,
        v_approach_moon: float = 800.0,
        v_approach_jupiter: float = 5500.0,
        v_inf_earth_return: float = 8000.0,
    ) -> dict:
        """
        Compute an approximate delta-V budget for the entire mission.

        This provides a quick estimate using simplified assumptions
        (circular orbits, impulsive burns, coplanar where possible).

        Args:
            parking_orbit_alt: Earth parking orbit altitude (m).
            lunar_orbit_alt: Lunar orbit altitude (m).
            lunar_inc_change: Lunar orbit inclination change (rad).
            jupiter_orbit_alt: Jupiter orbit altitude (m).
            v_approach_moon: Lunar approach velocity (m/s).
            v_approach_jupiter: Jupiter approach velocity (m/s).
            v_inf_earth_return: Earth return hyperbolic excess (m/s).

        Returns:
            Dict with itemized delta-V budget and total.
        """
        r_park = EARTH_RADIUS + parking_orbit_alt

        budget = {}

        # TLI: escape Earth with v_infinity toward Moon
        budget["tli"] = self.escape_maneuver(r_park, EARTH_MU, v_approach_moon)

        # LOI: capture into lunar orbit
        budget["loi"] = self.lunar_orbit_insertion(
            v_approach_moon, lunar_orbit_alt, MOON_MU
        )

        # Lunar plane change
        v_lunar_orbit = np.sqrt(MOON_MU / (MOON_RADIUS + lunar_orbit_alt))
        budget["lunar_plane_change"] = self.plane_change(
            v_lunar_orbit, lunar_inc_change
        )

        # Lunar escape toward Jupiter
        budget["lunar_escape"] = self.escape_maneuver(
            MOON_RADIUS + lunar_orbit_alt, MOON_MU, 1200.0
        )

        # JOI: capture into Jupiter orbit
        budget["joi"] = self.jupiter_orbit_insertion(
            v_approach_jupiter, jupiter_orbit_alt, JUPITER_MU
        )

        # Jupiter escape for return
        budget["jupiter_escape"] = self.escape_maneuver(
            JUPITER_RADIUS + jupiter_orbit_alt, JUPITER_MU, v_inf_earth_return
        )

        # Earth reentry: typically no burn (aerobraking), but include margin
        budget["earth_reentry_margin"] = 50.0  # m/s for trajectory corrections

        # Mid-course corrections (statistical estimate)
        budget["midcourse_corrections"] = 100.0  # m/s total

        budget["total"] = sum(budget.values())

        logger.info(
            "Mission delta-V budget: total = %.0f m/s", budget["total"]
        )
        return budget

    def __repr__(self) -> str:
        return "ManeuverPlanner()"
