"""
===============================================================================
GNC PROJECT - Trajectory Optimization
===============================================================================
Trajectory optimization for multi-body missions using dynamic programming and
Lambert solvers. This module computes optimal transfer trajectories between
Earth, Moon, and Jupiter.

Background
----------
Lambert's Problem:
    Given two position vectors r1 and r2, and a time of flight (tof), find the
    orbit connecting them. This is the fundamental building block of trajectory
    design. The solution gives the initial and final velocity vectors, from
    which the required delta-V can be computed.

    We use the *universal variable* formulation, which handles elliptic,
    parabolic, and hyperbolic transfers in a single unified algorithm. The
    universal variable z is related to the semi-major axis:
        z > 0  =>  elliptic
        z = 0  =>  parabolic
        z < 0  =>  hyperbolic

Dynamic Programming Approach:
    For multi-phase trajectories, we discretize the state space at each
    decision point (departure date, arrival date, flyby altitude, etc.) and
    use Bellman's principle of optimality to find the globally optimal
    sequence of maneuvers that minimizes total delta-V.

    cost(i) = min over j { stage_cost(i, j) + cost(j) }

    This avoids the combinatorial explosion of brute-force search while still
    exploring the full trade space.
===============================================================================
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from core.constants import (
    EARTH_MU,
    EARTH_RADIUS,
    JUPITER_MU,
    JUPITER_SMA,
    MOON_MU,
    MOON_SMA,
    SUN_MU,
)

logger = logging.getLogger(__name__)


# =============================================================================
# STUMPFF FUNCTIONS (for universal variable Lambert solver)
# =============================================================================

def _stumpff_c2(psi: float) -> float:
    """
    Stumpff function c2(psi).

    c2(psi) = (1 - cos(sqrt(psi))) / psi        if psi > 0   (elliptic)
            = (cosh(sqrt(-psi)) - 1) / (-psi)    if psi < 0   (hyperbolic)
            = 1/2                                 if psi = 0   (parabolic)
    """
    if abs(psi) < 1e-12:
        return 1.0 / 2.0
    elif psi > 0.0:
        sqrt_psi = np.sqrt(psi)
        return (1.0 - np.cos(sqrt_psi)) / psi
    else:
        sqrt_neg_psi = np.sqrt(-psi)
        return (np.cosh(sqrt_neg_psi) - 1.0) / (-psi)


def _stumpff_c3(psi: float) -> float:
    """
    Stumpff function c3(psi).

    c3(psi) = (sqrt(psi) - sin(sqrt(psi))) / psi^(3/2)       if psi > 0
            = (sinh(sqrt(-psi)) - sqrt(-psi)) / (-psi)^(3/2)  if psi < 0
            = 1/6                                              if psi = 0
    """
    if abs(psi) < 1e-12:
        return 1.0 / 6.0
    elif psi > 0.0:
        sqrt_psi = np.sqrt(psi)
        return (sqrt_psi - np.sin(sqrt_psi)) / (psi * sqrt_psi)
    else:
        sqrt_neg_psi = np.sqrt(-psi)
        return (np.sinh(sqrt_neg_psi) - sqrt_neg_psi) / ((-psi) * sqrt_neg_psi)


# =============================================================================
# TRAJECTORY OPTIMIZER CLASS
# =============================================================================

class TrajectoryOptimizer:
    """
    Multi-phase trajectory optimizer using Lambert solvers and dynamic
    programming.

    This class provides methods to optimize each leg of the
    Earth -> Moon -> Jupiter -> Earth mission, as well as general-purpose
    tools like porkchop plot generation and dynamic-programming-based
    trajectory search.

    The general workflow:
        1. Define departure and arrival constraints for each leg.
        2. Use lambert_arc() to solve the two-point boundary value problem.
        3. Use porkchop_plot_data() to survey the trade space.
        4. Use dp_trajectory_search() to chain multiple legs optimally.
        5. Use compute_total_delta_v() to evaluate the final trajectory.

    Attributes:
        max_lambert_iterations: Maximum Newton iterations for Lambert solver.
        lambert_tolerance:      Convergence tolerance for time of flight (s).
    """

    def __init__(
        self,
        max_lambert_iterations: int = 50,
        lambert_tolerance: float = 1e-8,
    ) -> None:
        self.max_lambert_iterations = max_lambert_iterations
        self.lambert_tolerance = lambert_tolerance

    # -------------------------------------------------------------------------
    # Lambert's Problem Solver (Universal Variable Formulation)
    # -------------------------------------------------------------------------

    def lambert_arc(
        self,
        r1: np.ndarray,
        r2: np.ndarray,
        tof: float,
        mu: float,
        prograde: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve Lambert's problem using the universal variable formulation.

        Given two position vectors and a time of flight, determine the
        initial and final velocity vectors of the connecting conic arc.

        The Algorithm:
        1. Compute the chord length and geometry.
        2. Use the universal variable z (related to reciprocal semi-major axis)
           and iterate with Newton's method until the time-of-flight equation
           is satisfied:
               tof = (z * S(z) + A * sqrt(y(z))) / sqrt(mu) * (1/sqrt(y(z)))^3
           where S and C are Stumpff functions.
        3. Extract the velocity vectors from the converged z value.

        Args:
            r1: Initial position vector (3,) in meters.
            r2: Final position vector (3,) in meters.
            tof: Time of flight in seconds (must be > 0).
            mu: Gravitational parameter of central body (m^3/s^2).
            prograde: If True, assume prograde (short-way) transfer.
                      If False, use retrograde (long-way).

        Returns:
            (v1, v2): Tuple of initial and final velocity vectors (m/s).

        Raises:
            ValueError: If tof <= 0 or solver fails to converge.
        """
        if tof <= 0.0:
            raise ValueError(f"Time of flight must be positive, got {tof}")

        r1 = np.asarray(r1, dtype=np.float64)
        r2 = np.asarray(r2, dtype=np.float64)

        r1_mag = np.linalg.norm(r1)
        r2_mag = np.linalg.norm(r2)

        # Cross product to determine transfer geometry
        cross = np.cross(r1, r2)
        cos_dtheta = np.dot(r1, r2) / (r1_mag * r2_mag)
        cos_dtheta = np.clip(cos_dtheta, -1.0, 1.0)

        # Determine transfer angle based on prograde/retrograde
        if prograde:
            if cross[2] < 0.0:
                sin_dtheta = -np.sqrt(1.0 - cos_dtheta**2)
            else:
                sin_dtheta = np.sqrt(1.0 - cos_dtheta**2)
        else:
            if cross[2] >= 0.0:
                sin_dtheta = -np.sqrt(1.0 - cos_dtheta**2)
            else:
                sin_dtheta = np.sqrt(1.0 - cos_dtheta**2)

        # Auxiliary variable A
        A = sin_dtheta * np.sqrt(r1_mag * r2_mag / (1.0 - cos_dtheta))

        if abs(A) < 1e-14:
            raise ValueError(
                "Lambert solver: degenerate geometry (A ~ 0). "
                "Positions may be collinear."
            )

        # --- Newton iteration on universal variable z ---
        # y(z) = r1 + r2 + A * (z * S(z) - 1) / sqrt(C(z))
        # F(z) = [y(z)/C(z)]^(3/2) * S(z) + A * sqrt(y(z)) - sqrt(mu) * tof
        # We seek F(z) = 0.

        def _y(z: float) -> float:
            c2 = _stumpff_c2(z)
            if abs(c2) < 1e-30:
                return float("inf")
            return r1_mag + r2_mag + A * (z * _stumpff_c3(z) - 1.0) / np.sqrt(c2)

        def _F(z: float) -> float:
            y = _y(z)
            if y < 0.0:
                return float("inf")
            c2 = _stumpff_c2(z)
            c3 = _stumpff_c3(z)
            return (
                (y / c2) ** 1.5 * c3
                + A * np.sqrt(y)
                - np.sqrt(mu) * tof
            )

        def _dFdz(z: float) -> float:
            y = _y(z)
            if y < 0.0:
                return 1.0
            c2 = _stumpff_c2(z)
            c3 = _stumpff_c3(z)
            if abs(z) < 1e-12:
                term1 = (np.sqrt(2.0) / 40.0) * y**1.5
                term2 = (A / 8.0) * (np.sqrt(y) + A * np.sqrt(1.0 / (2.0 * y)))
            else:
                term1 = ((y / c2) ** 1.5) * (
                    1.0 / (2.0 * z) * (c2 - 3.0 * c3 / (2.0 * c2))
                    + 3.0 * c3**2 / (4.0 * c2)
                )
                term2 = (A / 8.0) * (
                    3.0 * c3 * np.sqrt(y) / c2 + A * np.sqrt(c2 / y)
                )
            return term1 + term2

        # Initial guess for z: start near zero (parabolic)
        z = 0.0

        for iteration in range(self.max_lambert_iterations):
            F_val = _F(z)
            if abs(F_val) < self.lambert_tolerance:
                break
            dF_val = _dFdz(z)
            if abs(dF_val) < 1e-30:
                z += 0.1  # nudge if derivative vanishes
                continue
            z_new = z - F_val / dF_val

            # Ensure y stays positive
            while _y(z_new) < 0.0:
                z_new = (z + z_new) / 2.0

            z = z_new
        else:
            logger.warning(
                "Lambert solver did not converge in %d iterations (F=%.3e).",
                self.max_lambert_iterations,
                _F(z),
            )

        # --- Extract velocity vectors ---
        y = _y(z)
        c2 = _stumpff_c2(z)

        f = 1.0 - y / r1_mag
        g_dot = 1.0 - y / r2_mag
        g = A * np.sqrt(y / mu)

        v1 = (r2 - f * r1) / g
        v2 = (g_dot * r2 - r1) / g

        return v1, v2

    # -------------------------------------------------------------------------
    # Mission Leg Optimizers
    # -------------------------------------------------------------------------

    def optimize_earth_to_moon(
        self,
        departure_state: dict,
        arrival_constraints: dict,
    ) -> dict:
        """
        Find the optimal Trans-Lunar Injection (TLI) burn.

        Uses a Lambert arc from the departure orbit to a target point near
        the Moon, then optimizes the departure true anomaly and time of
        flight to minimize delta-V.

        The TLI burn transforms a circular parking orbit into a lunar transfer
        trajectory. The delta-V is:
            dv_tli = |v_lambert_departure - v_circular_parking|

        Args:
            departure_state: Dict with keys:
                - 'position': (3,) departure position (m)
                - 'velocity': (3,) departure velocity (m/s)
                - 'orbit_radius': Parking orbit radius (m)
            arrival_constraints: Dict with keys:
                - 'target_position': (3,) Moon position at arrival (m)
                - 'target_altitude': Desired periselene altitude (m)
                - 'tof_range': (min_s, max_s) allowable time of flight

        Returns:
            Dict with:
                - 'delta_v': TLI delta-V vector (m/s)
                - 'delta_v_magnitude': Scalar delta-V (m/s)
                - 'v_departure': Required departure velocity (m/s)
                - 'tof_optimal': Optimal time of flight (s)
                - 'v_arrival': Velocity at Moon arrival (m/s)
        """
        r1 = np.asarray(departure_state["position"], dtype=np.float64)
        v_park = np.asarray(departure_state["velocity"], dtype=np.float64)
        r2 = np.asarray(arrival_constraints["target_position"], dtype=np.float64)
        tof_min, tof_max = arrival_constraints["tof_range"]

        def _cost(tof: float) -> float:
            """Cost function: magnitude of TLI delta-V for a given tof."""
            try:
                v1, _ = self.lambert_arc(r1, r2, tof, EARTH_MU)
                return np.linalg.norm(v1 - v_park)
            except (ValueError, RuntimeWarning):
                return 1e12

        # Optimize time of flight using bounded scalar optimization
        result = minimize_scalar(
            _cost,
            bounds=(tof_min, tof_max),
            method="bounded",
            options={"xatol": 1.0},  # 1 second tolerance
        )

        tof_opt = result.x
        v1_opt, v2_opt = self.lambert_arc(r1, r2, tof_opt, EARTH_MU)
        dv = v1_opt - v_park

        return {
            "delta_v": dv,
            "delta_v_magnitude": np.linalg.norm(dv),
            "v_departure": v1_opt,
            "tof_optimal": tof_opt,
            "v_arrival": v2_opt,
        }

    def optimize_moon_to_jupiter(
        self,
        departure_state: dict,
        arrival_constraints: dict,
    ) -> dict:
        """
        Find the optimal transfer orbit from the Moon (via Earth escape) to
        Jupiter.

        This is a heliocentric Lambert problem. The spacecraft departs from
        the Earth-Moon system with some hyperbolic excess velocity and arrives
        at Jupiter. We optimize the departure date and time of flight to
        minimize the total delta-V (Earth escape + Jupiter capture).

        Args:
            departure_state: Dict with keys:
                - 'position_helio': (3,) heliocentric position at departure (m)
                - 'velocity_helio': (3,) heliocentric velocity at departure (m/s)
            arrival_constraints: Dict with keys:
                - 'target_position_helio': (3,) Jupiter heliocentric position (m)
                - 'tof_range': (min_s, max_s) time of flight bounds
                - 'capture_orbit_radius': Target Jupiter orbit radius (m)

        Returns:
            Dict with transfer orbit parameters and delta-V budget.
        """
        r1 = np.asarray(departure_state["position_helio"], dtype=np.float64)
        v_dep = np.asarray(departure_state["velocity_helio"], dtype=np.float64)
        r2 = np.asarray(arrival_constraints["target_position_helio"], dtype=np.float64)
        tof_min, tof_max = arrival_constraints["tof_range"]
        r_capture = arrival_constraints.get(
            "capture_orbit_radius", JUPITER_SMA * 0.001
        )

        def _total_cost(tof: float) -> float:
            """
            Total cost = departure delta-V + arrival delta-V.

            Departure dv: difference between Lambert solution v1 and current
                          heliocentric velocity.
            Arrival dv:   difference between Lambert v2 and Jupiter's
                          heliocentric velocity (approximated as circular).
            """
            try:
                v1, v2 = self.lambert_arc(r1, r2, tof, SUN_MU)

                # Departure delta-V (escape from Earth-Moon system)
                dv_depart = np.linalg.norm(v1 - v_dep)

                # Jupiter's approximate heliocentric velocity (circular orbit)
                r2_mag = np.linalg.norm(r2)
                v_jupiter_mag = np.sqrt(SUN_MU / r2_mag)
                # Rough direction: perpendicular to r2 in orbital plane
                r2_hat = r2 / r2_mag
                v_jup_dir = np.array([-r2_hat[1], r2_hat[0], 0.0])
                v_jup_dir /= np.linalg.norm(v_jup_dir)
                v_jupiter = v_jup_dir * v_jupiter_mag

                v_inf_arrival = np.linalg.norm(v2 - v_jupiter)

                # JOI delta-V from hyperbolic approach to capture orbit
                v_peri = np.sqrt(v_inf_arrival**2 + 2.0 * JUPITER_MU / r_capture)
                v_circ = np.sqrt(JUPITER_MU / r_capture)
                dv_capture = v_peri - v_circ

                return dv_depart + dv_capture
            except (ValueError, RuntimeWarning):
                return 1e12

        result = minimize_scalar(
            _total_cost,
            bounds=(tof_min, tof_max),
            method="bounded",
            options={"xatol": 3600.0},  # 1 hour tolerance
        )

        tof_opt = result.x
        v1_opt, v2_opt = self.lambert_arc(r1, r2, tof_opt, SUN_MU)

        return {
            "tof_optimal": tof_opt,
            "v_departure_helio": v1_opt,
            "v_arrival_helio": v2_opt,
            "delta_v_departure": np.linalg.norm(v1_opt - v_dep),
            "total_delta_v": result.fun,
        }

    def optimize_jupiter_to_earth(
        self,
        departure_state: dict,
        arrival_constraints: dict,
    ) -> dict:
        """
        Find the optimal return trajectory from Jupiter to Earth.

        Similar to the outbound leg, this is a heliocentric Lambert problem.
        We optimize for minimum total delta-V, including Jupiter escape and
        Earth arrival conditions.

        Args:
            departure_state: Dict with keys:
                - 'position_helio': (3,) heliocentric position at Jupiter (m)
                - 'velocity_helio': (3,) heliocentric velocity (m/s)
                - 'orbit_radius_jupiter': Current Jupiter orbit radius (m)
            arrival_constraints: Dict with keys:
                - 'target_position_helio': (3,) Earth heliocentric position (m)
                - 'tof_range': (min_s, max_s) time of flight bounds
                - 'entry_velocity_max': Maximum Earth entry speed (m/s)

        Returns:
            Dict with return trajectory parameters and delta-V budget.
        """
        r1 = np.asarray(departure_state["position_helio"], dtype=np.float64)
        v_dep = np.asarray(departure_state["velocity_helio"], dtype=np.float64)
        r2 = np.asarray(arrival_constraints["target_position_helio"], dtype=np.float64)
        tof_min, tof_max = arrival_constraints["tof_range"]
        r_jup_orbit = departure_state.get("orbit_radius_jupiter", 1e9)
        v_entry_max = arrival_constraints.get("entry_velocity_max", 15000.0)

        def _return_cost(tof: float) -> float:
            try:
                v1, v2 = self.lambert_arc(r1, r2, tof, SUN_MU)

                # Jupiter escape delta-V
                v_inf_depart = np.linalg.norm(v1 - v_dep)
                v_peri = np.sqrt(v_inf_depart**2 + 2.0 * JUPITER_MU / r_jup_orbit)
                v_circ = np.sqrt(JUPITER_MU / r_jup_orbit)
                dv_escape = v_peri - v_circ

                # Earth arrival v_infinity
                r2_mag = np.linalg.norm(r2)
                v_earth_mag = np.sqrt(SUN_MU / r2_mag)
                r2_hat = r2 / r2_mag
                v_earth_dir = np.array([-r2_hat[1], r2_hat[0], 0.0])
                v_earth_dir /= np.linalg.norm(v_earth_dir)
                v_earth = v_earth_dir * v_earth_mag
                v_inf_arr = np.linalg.norm(v2 - v_earth)

                # Penalty for exceeding entry velocity limit
                v_entry = np.sqrt(v_inf_arr**2 + 2.0 * EARTH_MU / EARTH_RADIUS)
                penalty = max(0.0, v_entry - v_entry_max) * 10.0

                return dv_escape + penalty
            except (ValueError, RuntimeWarning):
                return 1e12

        result = minimize_scalar(
            _return_cost,
            bounds=(tof_min, tof_max),
            method="bounded",
            options={"xatol": 3600.0},
        )

        tof_opt = result.x
        v1_opt, v2_opt = self.lambert_arc(r1, r2, tof_opt, SUN_MU)

        return {
            "tof_optimal": tof_opt,
            "v_departure_helio": v1_opt,
            "v_arrival_helio": v2_opt,
            "delta_v_escape": result.fun,
            "total_cost": result.fun,
        }

    # -------------------------------------------------------------------------
    # Porkchop Plot Data
    # -------------------------------------------------------------------------

    def porkchop_plot_data(
        self,
        body1_ephemeris: Callable[[float], np.ndarray],
        body2_ephemeris: Callable[[float], np.ndarray],
        departure_dates: np.ndarray,
        arrival_dates: np.ndarray,
        mu: float = SUN_MU,
    ) -> dict:
        """
        Generate porkchop plot data: a grid of delta-V values as a function
        of departure date and arrival date.

        A porkchop plot is the standard tool in astrodynamics for visualizing
        the launch window trade space. Each cell in the grid represents a
        Lambert arc connecting the departure body at one epoch to the arrival
        body at another. The color/contour value is typically the total
        delta-V (departure + arrival).

        The characteristic "porkchop" shape arises from the geometry of
        planetary alignments and Keplerian transfer orbits.

        Args:
            body1_ephemeris: Callable(t) -> (3,) position of departure body at
                            epoch t (seconds from reference epoch).
            body2_ephemeris: Callable(t) -> (3,) position of arrival body at
                            epoch t.
            departure_dates: 1-D array of departure epochs (s).
            arrival_dates:   1-D array of arrival epochs (s).
            mu: Gravitational parameter of central body (default: Sun).

        Returns:
            Dict with:
                - 'departure_dates': The input departure date array
                - 'arrival_dates':   The input arrival date array
                - 'delta_v_grid':    2-D array (n_dep x n_arr) of total dv (m/s)
                - 'c3_grid':         2-D array of departure C3 (km^2/s^2)
                - 'tof_grid':        2-D array of time-of-flight (s)
                - 'v_inf_arr_grid':  2-D array of arrival v_infinity (m/s)
        """
        n_dep = len(departure_dates)
        n_arr = len(arrival_dates)

        delta_v_grid = np.full((n_dep, n_arr), np.nan)
        c3_grid = np.full((n_dep, n_arr), np.nan)
        tof_grid = np.full((n_dep, n_arr), np.nan)
        v_inf_arr_grid = np.full((n_dep, n_arr), np.nan)

        for i, t_dep in enumerate(departure_dates):
            r1 = body1_ephemeris(t_dep)
            # Approximate body1 velocity via finite difference
            dt_fd = 3600.0  # 1 hour step for finite difference
            r1_plus = body1_ephemeris(t_dep + dt_fd)
            v_body1 = (r1_plus - r1) / dt_fd

            for j, t_arr in enumerate(arrival_dates):
                tof = t_arr - t_dep
                if tof <= 0.0:
                    continue

                tof_grid[i, j] = tof
                r2 = body2_ephemeris(t_arr)

                # Approximate body2 velocity
                r2_plus = body2_ephemeris(t_arr + dt_fd)
                v_body2 = (r2_plus - r2) / dt_fd

                try:
                    v1, v2 = self.lambert_arc(r1, r2, tof, mu)

                    # Departure delta-V and C3
                    v_inf_dep = v1 - v_body1
                    v_inf_dep_mag = np.linalg.norm(v_inf_dep)
                    c3 = v_inf_dep_mag**2  # m^2/s^2

                    # Arrival v_infinity
                    v_inf_arr = v2 - v_body2
                    v_inf_arr_mag = np.linalg.norm(v_inf_arr)

                    # Total delta-V (simplified: departure + arrival)
                    dv_total = v_inf_dep_mag + v_inf_arr_mag

                    delta_v_grid[i, j] = dv_total
                    c3_grid[i, j] = c3 / 1e6  # convert to km^2/s^2
                    v_inf_arr_grid[i, j] = v_inf_arr_mag

                except (ValueError, RuntimeWarning):
                    # Solver failed for this combination; leave as NaN
                    pass

        return {
            "departure_dates": departure_dates,
            "arrival_dates": arrival_dates,
            "delta_v_grid": delta_v_grid,
            "c3_grid": c3_grid,
            "tof_grid": tof_grid,
            "v_inf_arr_grid": v_inf_arr_grid,
        }

    # -------------------------------------------------------------------------
    # Dynamic Programming Trajectory Search
    # -------------------------------------------------------------------------

    def dp_trajectory_search(
        self,
        state_grid: List[np.ndarray],
        cost_func: Callable[[np.ndarray, np.ndarray, int], float],
        dynamics: Callable[[np.ndarray, int], np.ndarray],
        num_stages: Optional[int] = None,
    ) -> dict:
        """
        Multi-stage trajectory optimization via dynamic programming.

        Discretizes the state space at each decision point (stage) and uses
        backward induction to find the globally optimal sequence of states
        that minimizes total cost.

        Bellman recursion (backward pass):
            J*(x_k) = min over x_{k+1} in X_{k+1}
                       { L(x_k, x_{k+1}, k) + J*(x_{k+1}) }

        where:
            J*(x_k)            = cost-to-go from state x_k at stage k
            L(x_k, x_{k+1}, k) = stage cost of transitioning from x_k to x_{k+1}
            X_{k+1}            = discretized set of feasible states at stage k+1

        The forward pass then reconstructs the optimal trajectory by following
        the policy computed in the backward pass.

        Args:
            state_grid: List of N+1 arrays, where state_grid[k] is an array
                        of shape (n_k, state_dim) representing the discretized
                        states at stage k.
            cost_func:  Callable(x_k, x_{k+1}, k) -> float.
                        Returns the cost of transitioning from state x_k to
                        state x_{k+1} at stage k. Return np.inf for infeasible.
            dynamics:   Callable(x_k, k) -> x_predicted.
                        Propagates state forward one stage (used for
                        feasibility checks, not directly in the DP).
            num_stages: Number of stages (default: len(state_grid) - 1).

        Returns:
            Dict with:
                - 'optimal_cost':       Total minimum cost
                - 'optimal_trajectory': List of optimal state vectors
                - 'stage_costs':        List of individual stage costs
                - 'cost_to_go':         List of cost-to-go arrays at each stage
        """
        if num_stages is None:
            num_stages = len(state_grid) - 1

        # --- Backward pass: compute cost-to-go ---
        # cost_to_go[k][i] = optimal cost from state_grid[k][i] to the end
        cost_to_go: List[np.ndarray] = [np.array([]) for _ in range(num_stages + 1)]
        policy: List[np.ndarray] = [np.array([], dtype=int) for _ in range(num_stages)]

        # Terminal cost: zero at the last stage
        n_final = len(state_grid[num_stages])
        cost_to_go[num_stages] = np.zeros(n_final)

        for k in range(num_stages - 1, -1, -1):
            n_k = len(state_grid[k])
            n_next = len(state_grid[k + 1])
            cost_to_go[k] = np.full(n_k, np.inf)
            policy[k] = np.zeros(n_k, dtype=int)

            for i in range(n_k):
                best_cost = np.inf
                best_j = 0
                x_k = state_grid[k][i]

                for j in range(n_next):
                    x_next = state_grid[k + 1][j]
                    stage_cost = cost_func(x_k, x_next, k)
                    total = stage_cost + cost_to_go[k + 1][j]

                    if total < best_cost:
                        best_cost = total
                        best_j = j

                cost_to_go[k][i] = best_cost
                policy[k][i] = best_j

        # --- Forward pass: reconstruct optimal trajectory ---
        # Start from the best initial state
        optimal_trajectory = []
        stage_costs = []
        best_start = int(np.argmin(cost_to_go[0]))
        current_idx = best_start

        for k in range(num_stages):
            x_k = state_grid[k][current_idx]
            next_idx = int(policy[k][current_idx])
            x_next = state_grid[k + 1][next_idx]

            optimal_trajectory.append(x_k)
            stage_costs.append(cost_func(x_k, x_next, k))
            current_idx = next_idx

        # Append final state
        optimal_trajectory.append(state_grid[num_stages][current_idx])

        return {
            "optimal_cost": cost_to_go[0][best_start],
            "optimal_trajectory": optimal_trajectory,
            "stage_costs": stage_costs,
            "cost_to_go": cost_to_go,
        }

    # -------------------------------------------------------------------------
    # Delta-V Accounting
    # -------------------------------------------------------------------------

    def compute_total_delta_v(
        self,
        trajectory: List[dict],
    ) -> dict:
        """
        Sum all delta-V contributions for a complete multi-leg trajectory.

        Each entry in the trajectory list is a dict with at least:
            - 'name':            Descriptive name of the maneuver
            - 'delta_v_vector':  (3,) delta-V vector in m/s
        OR
            - 'delta_v_magnitude': Scalar delta-V in m/s

        Args:
            trajectory: List of maneuver dictionaries.

        Returns:
            Dict with:
                - 'total_delta_v':  Total delta-V magnitude (m/s)
                - 'maneuver_list':  List of (name, dv_magnitude) tuples
                - 'budget_margin':  Percentage margin if budget is specified
        """
        total_dv = 0.0
        maneuver_list = []

        for maneuver in trajectory:
            name = maneuver.get("name", "unnamed")
            if "delta_v_vector" in maneuver:
                dv_mag = np.linalg.norm(maneuver["delta_v_vector"])
            elif "delta_v_magnitude" in maneuver:
                dv_mag = maneuver["delta_v_magnitude"]
            else:
                logger.warning("Maneuver '%s' has no delta-V data.", name)
                dv_mag = 0.0

            total_dv += dv_mag
            maneuver_list.append((name, dv_mag))

        return {
            "total_delta_v": total_dv,
            "maneuver_list": maneuver_list,
        }

    def __repr__(self) -> str:
        return (
            f"TrajectoryOptimizer("
            f"max_iter={self.max_lambert_iterations}, "
            f"tol={self.lambert_tolerance:.1e})"
        )
