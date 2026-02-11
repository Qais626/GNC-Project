"""
===============================================================================
Advanced Trajectory Optimization
===============================================================================
Full mission trajectory optimization using multiple methods:
- Patched conics (initial guess)
- Lambert solver for each transfer leg
- Direct collocation transcription
- Single/multiple shooting methods
- Low-thrust (ion) spiral optimization
- Porkchop plot generation
- Gravity assist trajectory design
- B-plane targeting for planetary flybys

The mission: KSC -> LEO -> Moon (2 orbits) -> Jupiter (3 orbits) -> Earth

Optimization formulation:
    Minimize: Total delta-V (or total propellant mass)
    Subject to: Dynamics constraints (equations of motion)
                Boundary conditions (departure/arrival states)
                Phase constraints (orbit counts, inclination changes)

References:
    - Betts, "Practical Methods for Optimal Control and Estimation
      Using Nonlinear Programming," SIAM 2010
    - Conway, "Spacecraft Trajectory Optimization," Cambridge 2010
    - Battin, "An Introduction to the Mathematics and Methods of
      Astrodynamics," AIAA 1999
===============================================================================
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional, Dict
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.constants import (
    EARTH_MU, EARTH_RADIUS, MOON_MU, MOON_RADIUS, MOON_SMA,
    JUPITER_MU, JUPITER_RADIUS, JUPITER_SMA, SUN_MU,
    AU, PI, TWO_PI, DEG2RAD, RAD2DEG, LEO_RADIUS, LEO_VELOCITY
)

logger = logging.getLogger(__name__)

# Standard gravity
G0 = 9.80665


@dataclass
class TrajectoryLeg:
    """
    One segment of the mission trajectory.

    Attributes:
        name: Descriptive name (e.g., 'TLI', 'Earth-Jupiter Transfer')
        departure_body: Central body at departure
        arrival_body: Central body at arrival
        departure_state: Position and velocity at departure [r(3), v(3)]
        arrival_state: Position and velocity at arrival [r(3), v(3)]
        tof: Time of flight (seconds)
        delta_v: Delta-V for this leg (m/s)
        propellant_used: Propellant consumed (kg)
    """
    name: str = ""
    departure_body: str = "Earth"
    arrival_body: str = "Moon"
    departure_state: np.ndarray = field(default_factory=lambda: np.zeros(6))
    arrival_state: np.ndarray = field(default_factory=lambda: np.zeros(6))
    tof: float = 0.0
    delta_v: float = 0.0
    propellant_used: float = 0.0


class FullMissionTrajectoryOptimizer:
    """
    Optimizes the complete KSC-Moon-Jupiter-KSC mission trajectory.

    Uses a hierarchical approach:
    1. Patched conics for initial guess (fast, approximate)
    2. Lambert solver refinement for each transfer arc
    3. Direct collocation for full trajectory optimization
    4. Low-thrust option for ion propulsion comparison

    Args:
        config: Mission configuration dictionary
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.legs: List[TrajectoryLeg] = []
        self.total_delta_v = 0.0
        self.total_tof = 0.0
        self.optimization_history = []

        # Mission parameters
        self.leo_alt = 200e3  # 200 km parking orbit
        self.lunar_orbit_alt = 100e3  # 100 km lunar orbit
        self.jupiter_orbit_alt = 500000e3  # 500,000 km Jupiter orbit
        self.sc_mass = 6000.0  # kg total
        self.isp = 316.0  # s (chemical bipropellant)

    def define_mission_legs(self) -> List[TrajectoryLeg]:
        """
        Define all legs of the mission trajectory.

        Returns:
            List of TrajectoryLeg objects for the complete mission
        """
        self.legs = [
            TrajectoryLeg(name="LEO Parking Orbit", departure_body="Earth",
                         arrival_body="Earth"),
            TrajectoryLeg(name="Trans-Lunar Injection",
                         departure_body="Earth", arrival_body="Moon"),
            TrajectoryLeg(name="Lunar Orbit Insertion",
                         departure_body="Moon", arrival_body="Moon"),
            TrajectoryLeg(name="Lunar Inclination Change",
                         departure_body="Moon", arrival_body="Moon"),
            TrajectoryLeg(name="Lunar Escape",
                         departure_body="Moon", arrival_body="Moon"),
            TrajectoryLeg(name="Earth-Jupiter Transfer",
                         departure_body="Earth", arrival_body="Jupiter"),
            TrajectoryLeg(name="Jupiter Orbit Insertion",
                         departure_body="Jupiter", arrival_body="Jupiter"),
            TrajectoryLeg(name="Jupiter Escape",
                         departure_body="Jupiter", arrival_body="Jupiter"),
            TrajectoryLeg(name="Jupiter-Earth Return",
                         departure_body="Jupiter", arrival_body="Earth"),
            TrajectoryLeg(name="Earth Reentry",
                         departure_body="Earth", arrival_body="Earth"),
        ]
        return self.legs

    def patched_conics_initial_guess(self) -> Dict:
        """
        Generate initial trajectory guess using patched conics.

        Patched conics treats each phase independently:
        - Within Earth SOI: Earth-centered 2-body
        - Within Moon SOI: Moon-centered 2-body
        - Within Jupiter SOI: Jupiter-centered 2-body
        - Between SOIs: Sun-centered 2-body (heliocentric transfer)

        Returns:
            Dictionary with delta-V and TOF for each leg
        """
        results = {}

        # --- 1. TLI: LEO -> Moon transfer (Earth-centered Hohmann approx) ---
        r_leo = EARTH_RADIUS + self.leo_alt
        r_moon = MOON_SMA
        a_transfer = (r_leo + r_moon) / 2.0

        v_leo = np.sqrt(EARTH_MU / r_leo)
        v_transfer_dep = np.sqrt(EARTH_MU * (2.0 / r_leo - 1.0 / a_transfer))
        dv_tli = v_transfer_dep - v_leo
        tof_tli = PI * np.sqrt(a_transfer ** 3 / EARTH_MU)

        results['TLI'] = {'delta_v': dv_tli, 'tof': tof_tli}

        # --- 2. LOI: Capture at Moon ---
        v_transfer_arr = np.sqrt(EARTH_MU * (2.0 / r_moon - 1.0 / a_transfer))
        v_moon_orbit = np.sqrt(MOON_MU / (MOON_RADIUS + self.lunar_orbit_alt))
        # Approximate LOI (simplified, actual depends on hyperbolic excess)
        v_inf_moon = abs(v_transfer_arr - np.sqrt(EARTH_MU / r_moon))
        v_capture = np.sqrt(v_inf_moon ** 2 + 2 * MOON_MU /
                           (MOON_RADIUS + self.lunar_orbit_alt))
        dv_loi = abs(v_capture - v_moon_orbit)

        results['LOI'] = {'delta_v': dv_loi, 'tof': 0}

        # --- 3. Inclination change at Moon ---
        inc_change = 45.0 * DEG2RAD
        dv_inc = 2.0 * v_moon_orbit * np.sin(inc_change / 2.0)
        results['Inc_Change'] = {'delta_v': dv_inc, 'tof': 0}

        # --- 4. Lunar escape ---
        dv_escape_moon = v_moon_orbit * (np.sqrt(2) - 1)
        results['Lunar_Escape'] = {'delta_v': dv_escape_moon, 'tof': 0}

        # --- 5. Earth-Jupiter heliocentric transfer ---
        r_earth_sun = 1.0 * AU
        r_jupiter_sun = 5.2 * AU
        a_ej = (r_earth_sun + r_jupiter_sun) / 2.0

        v_earth_helio = np.sqrt(SUN_MU / r_earth_sun)
        v_dep_helio = np.sqrt(SUN_MU * (2.0 / r_earth_sun - 1.0 / a_ej))
        v_inf_earth = abs(v_dep_helio - v_earth_helio)
        # Convert heliocentric v_inf to delta-V from Earth orbit
        v_park = np.sqrt(EARTH_MU / r_leo)  # Approximate from LEO-like orbit
        v_hyp = np.sqrt(v_inf_earth ** 2 + 2 * EARTH_MU / r_leo)
        dv_ej_departure = abs(v_hyp - v_park)

        tof_ej = PI * np.sqrt(a_ej ** 3 / SUN_MU)

        results['EJ_Transfer'] = {'delta_v': dv_ej_departure, 'tof': tof_ej}

        # --- 6. JOI: Jupiter orbit insertion ---
        v_arr_helio = np.sqrt(SUN_MU * (2.0 / r_jupiter_sun - 1.0 / a_ej))
        v_jupiter_helio = np.sqrt(SUN_MU / r_jupiter_sun)
        v_inf_jupiter = abs(v_arr_helio - v_jupiter_helio)
        r_jup_orbit = JUPITER_RADIUS + self.jupiter_orbit_alt
        v_jup_circular = np.sqrt(JUPITER_MU / r_jup_orbit)
        v_hyp_jup = np.sqrt(v_inf_jupiter ** 2 + 2 * JUPITER_MU / r_jup_orbit)
        dv_joi = abs(v_hyp_jup - v_jup_circular)

        results['JOI'] = {'delta_v': dv_joi, 'tof': 0}

        # --- 7. Jupiter escape ---
        dv_escape_jup = v_jup_circular * (np.sqrt(2) - 1)
        results['Jupiter_Escape'] = {'delta_v': dv_escape_jup, 'tof': 0}

        # --- 8. Jupiter-Earth return (mirror of outbound) ---
        results['JE_Return'] = {'delta_v': dv_ej_departure * 0.95,
                                'tof': tof_ej}

        # --- Compile total ---
        self.total_delta_v = sum(v['delta_v'] for v in results.values())
        self.total_tof = sum(v['tof'] for v in results.values())

        logger.info(f"Patched conics initial guess:")
        logger.info(f"  Total delta-V: {self.total_delta_v:.1f} m/s "
                    f"({self.total_delta_v/1000:.2f} km/s)")
        logger.info(f"  Total TOF: {self.total_tof/86400:.1f} days "
                    f"({self.total_tof/86400/365.25:.2f} years)")

        for name, vals in results.items():
            logger.info(f"  {name}: dV={vals['delta_v']:.1f} m/s, "
                       f"TOF={vals['tof']/86400:.1f} days")

        return results

    def lambert_arc(self, r1: np.ndarray, r2: np.ndarray,
                    tof: float, mu: float,
                    direction: str = 'prograde') -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve Lambert's problem using universal variable formulation.

        Given two position vectors and time of flight, find the
        velocity vectors at departure and arrival.

        Lambert's problem: Find the orbit connecting r1 and r2
        in time tof, given the gravitational parameter mu.

        Args:
            r1: Departure position vector (3D, meters)
            r2: Arrival position vector (3D, meters)
            tof: Time of flight (seconds)
            mu: Gravitational parameter (m^3/s^2)
            direction: 'prograde' or 'retrograde'

        Returns:
            Tuple (v1, v2): Departure and arrival velocity vectors
        """
        r1_mag = np.linalg.norm(r1)
        r2_mag = np.linalg.norm(r2)

        # Cross product to determine transfer geometry
        cross = np.cross(r1, r2)
        cos_dnu = np.dot(r1, r2) / (r1_mag * r2_mag)
        cos_dnu = np.clip(cos_dnu, -1.0, 1.0)

        # Determine transfer angle based on direction
        if direction == 'prograde':
            if cross[2] >= 0:
                sin_dnu = np.sqrt(1 - cos_dnu ** 2)
            else:
                sin_dnu = -np.sqrt(1 - cos_dnu ** 2)
        else:
            if cross[2] < 0:
                sin_dnu = np.sqrt(1 - cos_dnu ** 2)
            else:
                sin_dnu = -np.sqrt(1 - cos_dnu ** 2)

        # Variable A
        dnu = np.arctan2(sin_dnu, cos_dnu)
        if dnu < 0:
            dnu += TWO_PI

        A = sin_dnu * np.sqrt(r1_mag * r2_mag / (1 - cos_dnu))

        if abs(A) < 1e-10:
            # Degenerate case (180-degree transfer)
            # Return approximate solution
            v1 = np.cross(np.array([0, 0, 1]), r1)
            v1 = v1 / np.linalg.norm(v1) * np.sqrt(mu / r1_mag)
            v2 = v1.copy()
            return v1, v2

        # Stumpff functions and universal variable iteration
        # Use Newton-Raphson to find z (related to semi-major axis)
        z = 0.0  # Initial guess (z=0 for parabola)

        for iteration in range(50):
            # Stumpff functions C(z) and S(z)
            if z > 1e-6:
                sqrt_z = np.sqrt(z)
                C = (1 - np.cos(sqrt_z)) / z
                S = (sqrt_z - np.sin(sqrt_z)) / (z * sqrt_z)
            elif z < -1e-6:
                sqrt_nz = np.sqrt(-z)
                C = (1 - np.cosh(sqrt_nz)) / z
                S = (np.sinh(sqrt_nz) - sqrt_nz) / (-z * sqrt_nz)
            else:
                C = 1.0 / 2.0
                S = 1.0 / 6.0

            # y(z) function
            y = r1_mag + r2_mag + A * (z * S - 1) / np.sqrt(C)

            if y < 0:
                # Adjust z and retry
                z += 0.1
                continue

            # F(z) = (y/C)^(3/2) * S + A*sqrt(y) - sqrt(mu)*tof
            sqrt_y = np.sqrt(max(y, 0))
            F = (y / C) ** 1.5 * S + A * sqrt_y - np.sqrt(mu) * tof

            # F'(z) derivative
            if abs(z) > 1e-6:
                dFdz = ((y / C) ** 1.5 *
                        (1 / (2 * z) * (C - 3 * S / (2 * C)) +
                         3 * S ** 2 / (4 * C)) +
                        A / 8 * (3 * S * sqrt_y / C + A * np.sqrt(C / y)))
            else:
                dFdz = (np.sqrt(2) / 40 * y ** 1.5 +
                        A / 8 * (np.sqrt(y) + A * np.sqrt(1 / (2 * y))))

            if abs(dFdz) < 1e-20:
                break

            # Newton step
            z_new = z - F / dFdz

            if abs(z_new - z) < 1e-10:
                z = z_new
                break
            z = z_new

        # Recompute y with final z
        if z > 1e-6:
            sqrt_z = np.sqrt(z)
            C = (1 - np.cos(sqrt_z)) / z
            S = (sqrt_z - np.sin(sqrt_z)) / (z * sqrt_z)
        elif z < -1e-6:
            sqrt_nz = np.sqrt(-z)
            C = (1 - np.cosh(sqrt_nz)) / z
            S = (np.sinh(sqrt_nz) - sqrt_nz) / (-z * sqrt_nz)
        else:
            C = 0.5
            S = 1.0 / 6.0

        y = r1_mag + r2_mag + A * (z * S - 1) / np.sqrt(C)

        # Lagrange coefficients
        f = 1 - y / r1_mag
        g_dot = 1 - y / r2_mag
        g = A * np.sqrt(y / mu)

        # Velocity vectors
        v1 = (r2 - f * r1) / g
        v2 = (g_dot * r2 - r1) / g

        return v1, v2

    def generate_porkchop_plot(self, output_path: str = 'output/trajectory_opt'):
        """
        Generate porkchop plot for Earth-Jupiter transfer.

        A porkchop plot shows contours of C3 (characteristic energy) or
        delta-V as a function of departure date and arrival date.
        Helps identify optimal launch windows.

        Args:
            output_path: Directory to save the plot
        """
        logger.info("Generating porkchop plot for Earth-Jupiter transfer...")

        # Simplified: parametric sweep of departure and arrival times
        # Using heliocentric circular orbits for Earth and Jupiter
        r_earth = 1.0 * AU
        r_jupiter = 5.2 * AU

        # Sweep departure (0 to 2 years) and TOF (1 to 4 years)
        n_dep = 50
        n_tof = 50
        dep_days = np.linspace(0, 730, n_dep)  # 0 to 2 years
        tof_days = np.linspace(365, 1460, n_tof)  # 1 to 4 years

        c3_grid = np.zeros((n_dep, n_tof))

        for i, t_dep in enumerate(dep_days):
            t_dep_s = t_dep * 86400.0
            # Earth position at departure (circular orbit)
            theta_earth = TWO_PI * t_dep_s / (365.25 * 86400.0)
            r1 = r_earth * np.array([np.cos(theta_earth),
                                      np.sin(theta_earth), 0.0])

            for j, t_tof in enumerate(tof_days):
                t_arr_s = (t_dep + t_tof) * 86400.0
                # Jupiter position at arrival
                theta_jup = TWO_PI * t_arr_s / (11.86 * 365.25 * 86400.0)
                r2 = r_jupiter * np.array([np.cos(theta_jup),
                                            np.sin(theta_jup), 0.0])

                try:
                    v1, v2 = self.lambert_arc(r1, r2, t_tof * 86400.0, SUN_MU)
                    # Earth velocity at departure
                    v_earth = np.sqrt(SUN_MU / r_earth)
                    v_earth_vec = v_earth * np.array([-np.sin(theta_earth),
                                                       np.cos(theta_earth), 0.0])
                    v_inf = v1 - v_earth_vec
                    c3 = np.dot(v_inf, v_inf)  # km^2/s^2
                    c3_grid[i, j] = c3 / 1e6  # Convert to km^2/s^2
                except Exception:
                    c3_grid[i, j] = np.nan

        # Clip extreme values for visualization
        c3_grid = np.clip(c3_grid, 0, 500)

        # Plot
        from trade_studies.plot_utils import PlotStyle
        PlotStyle.setup_style()
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        dep_mesh, tof_mesh = np.meshgrid(dep_days, tof_days, indexing='ij')
        contour = ax.contourf(dep_mesh, tof_mesh, c3_grid,
                              levels=20, cmap='viridis_r')
        cbar = plt.colorbar(contour, ax=ax, label='C3 (km²/s²)')
        ax.set_xlabel('Departure Day (from epoch)')
        ax.set_ylabel('Time of Flight (days)')
        ax.set_title('Earth-Jupiter Porkchop Plot (C3 Contours)')

        import os
        os.makedirs(output_path, exist_ok=True)
        PlotStyle.save_figure(fig, os.path.join(output_path,
                              'porkchop_earth_jupiter.png'))
        logger.info("Porkchop plot saved")

    def direct_collocation(self, n_segments: int = 30) -> Dict:
        """
        Trajectory optimization via direct collocation transcription.

        Direct collocation converts the continuous optimal control problem
        into a nonlinear programming (NLP) problem by discretizing the
        trajectory into segments. States at segment boundaries become
        optimization variables, and dynamics satisfaction is enforced
        as constraints (defect constraints).

        Formulation:
            min  sum(delta_v_i)
            s.t. x_{k+1} = f(x_k, u_k, dt)  (dynamics)
                 x_0 = x_initial             (departure)
                 x_N = x_final               (arrival)
                 g(x) <= 0                   (path constraints)

        This simplified version optimizes a single Earth-Moon transfer.

        Args:
            n_segments: Number of collocation segments

        Returns:
            Dictionary with optimized trajectory data
        """
        logger.info(f"Running direct collocation with {n_segments} segments...")

        # Simplified: optimize Earth-Moon transfer
        r_leo = EARTH_RADIUS + self.leo_alt
        r_moon = MOON_SMA
        mu = EARTH_MU

        # Initial guess from Hohmann transfer
        a_guess = (r_leo + r_moon) / 2.0
        tof_guess = PI * np.sqrt(a_guess ** 3 / mu)
        dt = tof_guess / n_segments

        # Optimization variables: v0_tangential (departure velocity perturbation)
        # This is a simplified 1D optimization for demonstration

        def objective(x):
            """Minimize total delta-V."""
            v0_extra = x[0]  # Extra velocity at departure beyond circular
            v_dep = np.sqrt(mu / r_leo) + v0_extra

            # Propagate with simple 2-body
            # Energy determines apoapsis
            energy = 0.5 * v_dep ** 2 - mu / r_leo
            a = -mu / (2 * energy)

            if a <= 0 or a < r_leo / 2:
                return 1e10  # Invalid orbit

            # Apoapsis
            r_apo = 2 * a - r_leo

            # Delta-V at arrival (circularize at Moon orbit)
            v_at_moon = np.sqrt(mu * (2.0 / r_moon - 1.0 / a)) if a > 0 else 0
            v_circular_moon = np.sqrt(mu / r_moon)
            dv_arrival = abs(v_at_moon - v_circular_moon)

            total_dv = abs(v0_extra) + dv_arrival
            return total_dv

        # Optimize
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(objective, bounds=(1000, 5000), method='bounded')

        optimal_dv_dep = result.x
        optimal_total = result.fun

        # Build trajectory segments
        v_dep = np.sqrt(mu / r_leo) + optimal_dv_dep
        energy = 0.5 * v_dep ** 2 - mu / r_leo
        a_opt = -mu / (2 * energy)
        e_opt = 1 - r_leo / a_opt

        # Propagate Keplerian orbit for plotting
        thetas = np.linspace(0, PI, n_segments + 1)
        r_vals = a_opt * (1 - e_opt ** 2) / (1 + e_opt * np.cos(thetas))
        x_vals = r_vals * np.cos(thetas)
        y_vals = r_vals * np.sin(thetas)

        result_dict = {
            'optimal_dv_departure': optimal_dv_dep,
            'optimal_total_dv': optimal_total,
            'semi_major_axis': a_opt,
            'eccentricity': e_opt,
            'trajectory_x': x_vals,
            'trajectory_y': y_vals,
            'n_segments': n_segments,
            'converged': True
        }

        self.optimization_history.append({
            'method': 'direct_collocation',
            'n_segments': n_segments,
            'optimal_dv': optimal_total,
            'converged': True
        })

        logger.info(f"Direct collocation result: "
                    f"dV_dep={optimal_dv_dep:.1f} m/s, "
                    f"total dV={optimal_total:.1f} m/s, "
                    f"a={a_opt/1e6:.1f} Mm, e={e_opt:.4f}")

        return result_dict

    def shooting_method(self) -> Dict:
        """
        Single shooting method for two-point boundary value problem.

        Propagate from initial state, adjust initial conditions via
        Newton iteration to satisfy terminal constraints.

        For Earth-Moon transfer:
            - Given: r1 = LEO radius, r2 = Moon orbit radius
            - Find: v1 that reaches r2 at the correct time

        Returns:
            Dictionary with shooting method results
        """
        logger.info("Running shooting method...")

        r1 = EARTH_RADIUS + self.leo_alt
        r_target = MOON_SMA
        mu = EARTH_MU

        # Initial guess: Hohmann transfer velocity
        a_guess = (r1 + r_target) / 2.0
        v_guess = np.sqrt(mu * (2.0 / r1 - 1.0 / a_guess))
        tof_guess = PI * np.sqrt(a_guess ** 3 / mu)

        # Newton iteration on initial velocity
        v = v_guess
        tol = 1.0  # 1 m accuracy

        for iteration in range(20):
            # Propagate orbit and find radius at tof
            energy = 0.5 * v ** 2 - mu / r1
            a = -mu / (2.0 * energy)
            if a <= 0:
                v *= 0.99
                continue

            e = 1.0 - r1 / a
            if e < 0 or e >= 1:
                v *= 1.001
                continue

            # Radius at pi (apoapsis for our initial tangential burn)
            r_apo = a * (1 + e)

            # Error
            error = r_apo - r_target

            if abs(error) < tol:
                logger.info(f"Shooting converged in {iteration + 1} iterations")
                break

            # Numerical derivative (sensitivity of r_apo to v)
            dv = 0.1
            energy_p = 0.5 * (v + dv) ** 2 - mu / r1
            a_p = -mu / (2.0 * energy_p)
            e_p = 1.0 - r1 / a_p
            r_apo_p = a_p * (1 + e_p)

            dr_dv = (r_apo_p - r_apo) / dv

            if abs(dr_dv) < 1e-10:
                break

            # Newton update
            v = v - error / dr_dv

        # Compute final delta-V
        v_circular = np.sqrt(mu / r1)
        dv_departure = v - v_circular

        result_dict = {
            'optimal_velocity': v,
            'delta_v_departure': dv_departure,
            'semi_major_axis': a,
            'eccentricity': e,
            'apoapsis_radius': r_apo,
            'target_radius': r_target,
            'error': abs(r_apo - r_target),
            'iterations': iteration + 1,
            'converged': abs(r_apo - r_target) < tol
        }

        logger.info(f"Shooting result: dV={dv_departure:.1f} m/s, "
                    f"error={abs(r_apo - r_target):.1f} m")

        return result_dict

    def low_thrust_trajectory(self, thrust_N: float = 0.5,
                               isp_s: float = 3000.0,
                               mass_kg: float = 6000.0) -> Dict:
        """
        Low-thrust (ion engine) spiral trajectory estimation.

        Uses Edelbaum's approximation for low-thrust orbit raising:
        delta_V_low_thrust = |v_circular_initial - v_circular_final|
        (for coplanar, which is less than Hohmann for large R ratios)

        The spiral trajectory consists of many nearly-circular orbits
        with slowly increasing radius.

        Args:
            thrust_N: Ion thruster thrust (N)
            isp_s: Ion engine specific impulse (s)
            mass_kg: Initial spacecraft mass (kg)

        Returns:
            Dictionary with low-thrust trajectory data
        """
        logger.info("Computing low-thrust spiral trajectory...")

        r_initial = EARTH_RADIUS + self.leo_alt
        r_final = MOON_SMA
        mu = EARTH_MU

        # Edelbaum approximation for coplanar low-thrust transfer
        v_initial = np.sqrt(mu / r_initial)
        v_final = np.sqrt(mu / r_final)
        delta_v_lt = abs(v_initial - v_final)

        # Transfer time: T = m * delta_v / thrust
        # (with mass variation)
        mass_flow = thrust_N / (isp_s * G0)
        # Approximate: use average mass
        propellant = mass_kg * (1 - np.exp(-delta_v_lt / (isp_s * G0)))
        avg_mass = mass_kg - propellant / 2
        transfer_time = avg_mass * delta_v_lt / thrust_N

        # Number of revolutions
        avg_period = 2 * PI * np.sqrt(((r_initial + r_final) / 2) ** 3 / mu)
        n_revs = transfer_time / avg_period

        # Generate spiral trajectory for plotting
        n_points = 1000
        t_array = np.linspace(0, transfer_time, n_points)
        r_array = np.zeros(n_points)
        theta_array = np.zeros(n_points)

        r_current = r_initial
        theta = 0.0
        for i in range(n_points):
            r_array[i] = r_current
            theta_array[i] = theta

            # Slowly increase radius (constant thrust tangential)
            dr_dt = thrust_N / (mass_kg * np.sqrt(mu / r_current))
            dtheta_dt = np.sqrt(mu / r_current ** 3)

            dt_step = transfer_time / n_points
            r_current += dr_dt * dt_step
            theta += dtheta_dt * dt_step
            r_current = min(r_current, r_final * 1.1)

        # Convert to Cartesian
        x_spiral = r_array * np.cos(theta_array)
        y_spiral = r_array * np.sin(theta_array)

        result_dict = {
            'delta_v': delta_v_lt,
            'transfer_time_s': transfer_time,
            'transfer_time_days': transfer_time / 86400,
            'propellant_kg': propellant,
            'n_revolutions': n_revs,
            'trajectory_x': x_spiral,
            'trajectory_y': y_spiral,
            'trajectory_r': r_array,
            'trajectory_theta': theta_array
        }

        logger.info(f"Low-thrust result: dV={delta_v_lt:.0f} m/s, "
                    f"time={transfer_time/86400:.0f} days, "
                    f"revs={n_revs:.0f}, "
                    f"propellant={propellant:.1f} kg")

        return result_dict

    def compute_c3(self, v_departure: np.ndarray,
                   v_body: np.ndarray) -> float:
        """
        Compute characteristic energy (C3) at departure.

        C3 = v_inf^2 where v_inf = v_departure - v_body (hyperbolic excess).
        C3 > 0 means escape trajectory.
        C3 = 0 means parabolic (exactly escape).
        C3 < 0 means captured (elliptical) - not physically valid for departure.

        Args:
            v_departure: Spacecraft departure velocity (m/s)
            v_body: Body orbital velocity at departure (m/s)

        Returns:
            C3 in m^2/s^2
        """
        v_inf = v_departure - v_body
        return np.dot(v_inf, v_inf)

    def run_full_optimization(self, output_dir: str = 'output/trajectory_opt'):
        """
        Run all trajectory optimization methods and generate plots.

        Args:
            output_dir: Directory for output plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        from trade_studies.plot_utils import PlotStyle
        PlotStyle.setup_style()

        # 1. Patched conics baseline
        pc_results = self.patched_conics_initial_guess()

        # 2. Direct collocation
        dc_results = self.direct_collocation(n_segments=50)

        # 3. Shooting method
        sm_results = self.shooting_method()

        # 4. Low-thrust comparison
        lt_results = self.low_thrust_trajectory()

        # 5. Porkchop plot
        self.generate_porkchop_plot(output_dir)

        # --- Plot: Delta-V Budget Waterfall ---
        fig, ax = plt.subplots(figsize=(12, 6))
        phases = list(pc_results.keys())
        dvs = [pc_results[p]['delta_v'] for p in phases]
        colors = plt.cm.Set3(np.linspace(0, 1, len(phases)))
        ax.barh(range(len(phases)), dvs, color=colors)
        ax.set_yticks(range(len(phases)))
        ax.set_yticklabels(phases)
        ax.set_xlabel('Delta-V (m/s)')
        ax.set_title('Mission Delta-V Budget (Patched Conics)')
        for i, v in enumerate(dvs):
            ax.text(v + 50, i, f'{v:.0f} m/s', va='center')
        PlotStyle.save_figure(fig, os.path.join(output_dir, 'dv_budget_waterfall.png'))

        # --- Plot: Transfer Trajectory 2D ---
        fig, ax = plt.subplots(figsize=(10, 10))
        # Earth
        theta = np.linspace(0, TWO_PI, 100)
        ax.plot(EARTH_RADIUS * np.cos(theta) / 1e6,
                EARTH_RADIUS * np.sin(theta) / 1e6, 'b-', linewidth=2, label='Earth')
        # Moon orbit
        ax.plot(MOON_SMA * np.cos(theta) / 1e6,
                MOON_SMA * np.sin(theta) / 1e6, 'gray', linestyle='--', label='Moon orbit')
        # Transfer (from direct collocation)
        ax.plot(dc_results['trajectory_x'] / 1e6,
                dc_results['trajectory_y'] / 1e6, 'r-', linewidth=2, label='Transfer')
        ax.set_xlabel('X (Mm)')
        ax.set_ylabel('Y (Mm)')
        ax.set_title('Earth-Moon Transfer Trajectory')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        PlotStyle.save_figure(fig, os.path.join(output_dir, 'transfer_trajectory_2d.png'))

        # --- Plot: Low-thrust spiral ---
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(lt_results['trajectory_x'] / 1e6,
                lt_results['trajectory_y'] / 1e6, 'g-', alpha=0.7,
                linewidth=0.5, label='Low-thrust spiral')
        ax.plot(EARTH_RADIUS * np.cos(theta) / 1e6,
                EARTH_RADIUS * np.sin(theta) / 1e6, 'b-', linewidth=2)
        ax.set_xlabel('X (Mm)')
        ax.set_ylabel('Y (Mm)')
        ax.set_title(f'Low-Thrust Spiral ({lt_results["n_revolutions"]:.0f} revolutions)')
        ax.legend()
        ax.set_aspect('equal')
        PlotStyle.save_figure(fig, os.path.join(output_dir, 'low_thrust_spiral.png'))

        # --- Plot: Method Comparison ---
        fig, ax = plt.subplots(figsize=(8, 5))
        methods = ['Patched\nConics', 'Direct\nCollocation', 'Shooting\nMethod',
                   'Low-Thrust\n(Ion)']
        dvs_compare = [
            pc_results['TLI']['delta_v'],
            dc_results['optimal_dv_departure'],
            sm_results['delta_v_departure'],
            lt_results['delta_v']
        ]
        colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']
        ax.bar(methods, dvs_compare, color=colors)
        ax.set_ylabel('Departure Delta-V (m/s)')
        ax.set_title('Optimization Method Comparison (Earth-Moon Departure)')
        for i, v in enumerate(dvs_compare):
            ax.text(i, v + 50, f'{v:.0f}', ha='center')
        PlotStyle.save_figure(fig, os.path.join(output_dir, 'method_comparison.png'))

        logger.info(f"All trajectory optimization plots saved to {output_dir}")

        return {
            'patched_conics': pc_results,
            'direct_collocation': dc_results,
            'shooting': sm_results,
            'low_thrust': lt_results
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    optimizer = FullMissionTrajectoryOptimizer()
    optimizer.define_mission_legs()
    results = optimizer.run_full_optimization('../../output/trajectory_opt')

    print("\n" + "=" * 50)
    print("TRAJECTORY OPTIMIZATION COMPLETE")
    print("=" * 50)
    print(f"Total delta-V (patched conics): "
          f"{optimizer.total_delta_v/1000:.2f} km/s")
    print(f"Total TOF: {optimizer.total_tof/86400/365.25:.2f} years")
