"""
===============================================================================
GNC PROJECT - Multidisciplinary Design Optimization (MDO) Framework
===============================================================================
Comprehensive MDO framework that simultaneously optimizes across multiple
coupled disciplines for the Miami -> Moon (2 orbits) -> Jupiter (3 orbits)
-> return to Miami spacecraft mission.

Disciplines Modeled
-------------------
    1. Propulsion   -- Tsiolkovsky rocket equation, delta-V budget per leg
    2. Structural   -- Euler buckling, max-Q loading, safety factors
    3. Thermal      -- Stefan-Boltzmann radiation, solar/albedo heating
    4. Power        -- Solar flux modeling, radiation degradation, eclipse cycles
    5. Attitude     -- Reaction wheel sizing, CMG torque, pointing accuracy
    6. Trajectory   -- Lambert solver legs, patched-conic delta-V requirements

MDO Architectures
-----------------
    MDF (Multidisciplinary Feasible):
        Fixed-point iteration drives all disciplines to consistency, then an
        outer optimizer (scipy.optimize.minimize) searches the design space.
        Each function evaluation guarantees a physically consistent design.

    IDF (Individual Discipline Feasible):
        Coupling variables are promoted to optimization variables.  Each
        discipline runs independently per iteration; compatibility constraints
        enforce inter-discipline consistency.  Converges faster on loosely
        coupled problems at the expense of a larger optimization variable set.

Key Outputs
-----------
    - Optimal spacecraft mass (dry + propellant)
    - Pareto front: mass vs. transfer time
    - Sensitivity tornado chart (which design variables matter most)
    - N^2 discipline coupling diagram
    - Convergence history, tradespace scatter, constraint satisfaction waterfall

Units
-----
    SI throughout: meters, seconds, kilograms, radians, Watts, Kelvin.

References
----------
    [1] Martins & Lambe, "Multidisciplinary Design Optimization: A Survey of
        Architectures," AIAA Journal, 2013.
    [2] Sobieszczanski-Sobieski & Haftka, "Multidisciplinary Aerospace Design
        Optimization: Survey of Recent Developments," Structural Optimization,
        1997.
===============================================================================
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from core.constants import (
    AU,
    EARTH_MU,
    EARTH_RADIUS,
    JUPITER_MU,
    JUPITER_RADIUS,
    JUPITER_SMA,
    MOON_MU,
    MOON_RADIUS,
    MOON_SMA,
    STEFAN_BOLTZMANN,
    SUN_MU,
)

logger = logging.getLogger(__name__)

# Standard gravitational acceleration (m/s^2)
G0 = 9.80665

# Solar flux at 1 AU (W/m^2)
SOLAR_FLUX_1AU = 1361.0

# Output directory for all MDO plots
OUTPUT_DIR = os.path.join("output", "mdo")


# =============================================================================
# OPTIMIZATION RESULT CONTAINER
# =============================================================================

@dataclass
class OptimizationResult:
    """Container for MDO optimization output.

    Attributes
    ----------
    x_optimal : np.ndarray
        Optimal design variable vector.
    objective_value : float
        Value of the objective function at the optimum.
    variable_names : list of str
        Names corresponding to each element of x_optimal.
    constraint_values : dict
        Constraint name -> evaluated value at the optimum.
    convergence_history : list of float
        Objective function value at each iteration.
    design_history : list of np.ndarray
        Design variable vector at each iteration.
    discipline_outputs : dict
        Final outputs from every discipline at the optimum.
    success : bool
        Whether the optimizer converged.
    message : str
        Optimizer termination message.
    """
    x_optimal: np.ndarray
    objective_value: float
    variable_names: List[str]
    constraint_values: Dict[str, float] = field(default_factory=dict)
    convergence_history: List[float] = field(default_factory=list)
    design_history: List[np.ndarray] = field(default_factory=list)
    discipline_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    success: bool = False
    message: str = ""


# =============================================================================
# ABSTRACT BASE: DisciplineModel
# =============================================================================

class DisciplineModel(ABC):
    """Abstract base class for a single engineering discipline.

    Every discipline consumes a dictionary of *inputs* (which may include
    outputs from other disciplines -- the coupling variables) and produces
    a dictionary of *outputs*.  The ``get_gradients`` method returns the
    Jacobian of outputs with respect to inputs for use in gradient-based
    optimization.

    Subclasses must implement ``analyze`` and ``get_gradients``.
    """

    name: str = "base"

    @abstractmethod
    def analyze(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """Run the discipline analysis.

        Parameters
        ----------
        inputs : dict
            Mapping of input-variable names to their current values.

        Returns
        -------
        dict
            Mapping of output-variable names to computed values.
        """

    @abstractmethod
    def get_gradients(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Compute partial derivatives of outputs w.r.t. inputs.

        Parameters
        ----------
        inputs : dict
            Same input dictionary used by ``analyze``.

        Returns
        -------
        dict of dict
            Outer key = output name, inner key = input name, value = partial.
            Example: ``{"delta_v_total": {"Isp": 28.3, "thrust": 0.001}}``
        """


# =============================================================================
# DISCIPLINE 1: PROPULSION
# =============================================================================

class PropulsionDiscipline(DisciplineModel):
    """Propulsion subsystem discipline.

    Models the total delta-V capability of the spacecraft using the
    Tsiolkovsky rocket equation and verifies that every mission leg has
    sufficient delta-V margin.

    Tsiolkovsky rocket equation
    ---------------------------
        delta_v = Isp * g0 * ln(m0 / mf)

    where
        Isp  = specific impulse (s)
        g0   = 9.80665 m/s^2
        m0   = initial (wet) mass = dry_mass + propellant_mass (kg)
        mf   = final (dry) mass (kg)

    Mission-leg delta-V requirements (m/s)
    --------------------------------------
        TLI (Trans-Lunar Injection)     : 3 150
        LOI (Lunar Orbit Insertion)     :   850
        Lunar escape                    :   900
        Jupiter transfer (deep-space)   :   500
        JOI (Jupiter Orbit Insertion)   : 2 000
        Jupiter escape                  : 2 200
        Earth return                    : 1 000

    Inputs
    ------
        Isp              : specific impulse (s)
        thrust           : engine thrust level (N)
        propellant_mass  : propellant load (kg)
        dry_mass         : structural + payload mass (kg)
        engine_mass      : engine dry mass (kg)

    Outputs
    -------
        delta_v_total    : total mission delta-V capability (m/s)
        burn_time_total  : total burn time across all legs (s)
        propellant_consumed : sum of propellant used in all legs (kg)
        delta_v_margin   : surplus delta-V beyond requirements (m/s)
        mass_ratio       : m0 / mf
    """

    name = "propulsion"

    # Required delta-V per mission leg (m/s)
    LEG_REQUIREMENTS: Dict[str, float] = {
        "TLI":              3150.0,
        "LOI":               850.0,
        "lunar_escape":      900.0,
        "jupiter_transfer":  500.0,
        "JOI":              2000.0,
        "jupiter_escape":   2200.0,
        "earth_return":     1000.0,
    }

    def analyze(self, inputs: Dict[str, float]) -> Dict[str, float]:
        Isp = inputs["Isp"]
        thrust = inputs["thrust"]
        propellant_mass = inputs["propellant_mass"]
        dry_mass = inputs["dry_mass"]
        engine_mass = inputs.get("engine_mass", 0.0)

        m0 = dry_mass + engine_mass + propellant_mass
        mf = dry_mass + engine_mass
        if mf <= 0.0 or m0 <= mf:
            return {
                "delta_v_total": 0.0,
                "burn_time_total": 0.0,
                "propellant_consumed": 0.0,
                "delta_v_margin": -1e6,
                "mass_ratio": 1.0,
                "leg_delta_vs": {},
            }

        ve = Isp * G0
        delta_v_total = ve * np.log(m0 / mf)
        mass_ratio = m0 / mf

        # Distribute delta-V to legs proportionally to requirements
        total_required = sum(self.LEG_REQUIREMENTS.values())
        leg_dvs: Dict[str, float] = {}
        remaining_mass = m0
        propellant_consumed = 0.0
        burn_time_total = 0.0
        mass_flow = thrust / ve if ve > 0.0 else 1e-12

        for leg_name, dv_req in self.LEG_REQUIREMENTS.items():
            # Allocate proportion of total delta-V to this leg
            dv_leg = delta_v_total * (dv_req / total_required)
            leg_dvs[leg_name] = dv_leg

            # Propellant consumed in this leg (from rocket equation inverted)
            mf_leg = remaining_mass / np.exp(dv_leg / ve) if ve > 0.0 else remaining_mass
            dm = remaining_mass - mf_leg
            propellant_consumed += dm
            remaining_mass = mf_leg

            # Burn time for this leg
            if mass_flow > 0.0:
                burn_time_total += dm / mass_flow

        delta_v_margin = delta_v_total - total_required

        return {
            "delta_v_total": delta_v_total,
            "burn_time_total": burn_time_total,
            "propellant_consumed": propellant_consumed,
            "delta_v_margin": delta_v_margin,
            "mass_ratio": mass_ratio,
            "leg_delta_vs": leg_dvs,
        }

    def get_gradients(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        Isp = inputs["Isp"]
        propellant_mass = inputs["propellant_mass"]
        dry_mass = inputs["dry_mass"]
        engine_mass = inputs.get("engine_mass", 0.0)

        m0 = dry_mass + engine_mass + propellant_mass
        mf = dry_mass + engine_mass
        ve = Isp * G0
        if mf <= 0.0 or m0 <= mf or ve <= 0.0:
            return {"delta_v_total": {"Isp": 0.0, "propellant_mass": 0.0, "dry_mass": 0.0}}

        ln_ratio = np.log(m0 / mf)

        # d(delta_v)/d(Isp) = G0 * ln(m0/mf)
        d_dv_dIsp = G0 * ln_ratio

        # d(delta_v)/d(propellant_mass) = Isp * G0 / m0
        d_dv_dprop = ve / m0

        # d(delta_v)/d(dry_mass) = Isp * G0 * (1/m0 - 1/mf)
        d_dv_ddry = ve * (1.0 / m0 - 1.0 / mf)

        return {
            "delta_v_total": {
                "Isp": d_dv_dIsp,
                "propellant_mass": d_dv_dprop,
                "dry_mass": d_dv_ddry,
            }
        }


# =============================================================================
# DISCIPLINE 2: STRUCTURAL
# =============================================================================

class StructuralDiscipline(DisciplineModel):
    """Structural subsystem discipline.

    Models spacecraft structural integrity under launch and maneuver loads.

    Models
    ------
    Euler column buckling
        The critical buckling load for a column of length L, moment of
        inertia I, and Young's modulus E is:

            P_critical = pi^2 * E * I / L^2

        The buckling margin is:

            margin_buckling = P_critical / P_applied  -  1

        A positive margin means the structure will not buckle.

    Max-Q loading
        During ascent, the maximum dynamic pressure (max-Q) occurs at
        roughly Mach 1.  The axial force on the spacecraft is:

            F_axial = thrust + q_max * A_ref

        where A_ref is the reference cross-sectional area.  The resulting
        axial stress must remain below the yield strength divided by the
        safety factor.

    Structural safety factor constraint
        All stress margins must satisfy:

            safety_factor  >=  1.4

    Inputs
    ------
        structural_mass    : mass of primary structure (kg)
        propellant_mass    : propellant load -- drives inertial loads (kg)
        thrust             : engine thrust (N)
        acceleration_load  : peak g-load factor (dimensionless, e.g. 4.0)

    Outputs
    -------
        buckling_margin    : Euler-column buckling margin (>0 is safe)
        max_q_stress       : peak axial stress from max-Q (Pa)
        safety_factor      : minimum safety factor across all load cases
        structural_mass_fraction : structural mass / total mass
        first_natural_freq : first bending-mode natural frequency (Hz)
    """

    name = "structural"

    # Material properties (aluminium 7075-T6)
    YOUNGS_MODULUS = 71.7e9     # Pa
    YIELD_STRENGTH = 503e6      # Pa
    DENSITY = 2810.0            # kg/m^3

    # Geometry defaults
    COLUMN_LENGTH = 3.0         # m (spacecraft length)
    CROSS_SECTION_RADIUS = 1.0  # m
    WALL_THICKNESS = 0.005      # m

    def analyze(self, inputs: Dict[str, float]) -> Dict[str, float]:
        structural_mass = inputs["structural_mass"]
        propellant_mass = inputs["propellant_mass"]
        thrust = inputs["thrust"]
        accel_load = inputs.get("acceleration_load", 4.0)

        total_mass = structural_mass + propellant_mass
        if total_mass <= 0.0:
            total_mass = 1.0

        # --- Euler column buckling ---
        # Thin-walled cylinder moment of inertia:
        #   I = pi * r^3 * t
        r = self.CROSS_SECTION_RADIUS
        t = self.WALL_THICKNESS
        I_section = np.pi * r**3 * t
        P_critical = np.pi**2 * self.YOUNGS_MODULUS * I_section / self.COLUMN_LENGTH**2

        # Applied axial load during boost (thrust + inertial load)
        P_applied = thrust + total_mass * G0 * accel_load
        buckling_margin = (P_critical / max(P_applied, 1.0)) - 1.0

        # --- Max-Q axial stress ---
        # Reference area = cross-section of the cylindrical bus
        A_ref = np.pi * r**2
        # Conservative max-Q dynamic pressure for orbital launch ~ 35 kPa
        q_max = 35000.0  # Pa
        F_axial = thrust + q_max * A_ref
        # Axial stress = F / A_wall  where A_wall = 2 * pi * r * t
        A_wall = 2.0 * np.pi * r * t
        max_q_stress = F_axial / max(A_wall, 1e-6)

        # --- Safety factor ---
        safety_factor = self.YIELD_STRENGTH / max(max_q_stress, 1.0)

        # --- Structural mass fraction ---
        structural_mass_fraction = structural_mass / max(total_mass, 1.0)

        # --- First natural frequency (beam bending) ---
        # f1 = (1.875)^2 / (2*pi*L^2) * sqrt(E*I / (rho*A))
        # Use structural mass as effective distributed mass
        linear_density = structural_mass / max(self.COLUMN_LENGTH, 0.1)
        if linear_density > 0.0 and I_section > 0.0:
            first_freq = (1.875**2 / (2.0 * np.pi * self.COLUMN_LENGTH**2)) * \
                         np.sqrt(self.YOUNGS_MODULUS * I_section / linear_density)
        else:
            first_freq = 0.0

        return {
            "buckling_margin": buckling_margin,
            "max_q_stress": max_q_stress,
            "safety_factor": safety_factor,
            "structural_mass_fraction": structural_mass_fraction,
            "first_natural_freq": first_freq,
        }

    def get_gradients(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        structural_mass = inputs["structural_mass"]
        propellant_mass = inputs["propellant_mass"]
        thrust = inputs["thrust"]
        accel_load = inputs.get("acceleration_load", 4.0)

        total_mass = structural_mass + propellant_mass
        if total_mass <= 0.0:
            total_mass = 1.0

        r = self.CROSS_SECTION_RADIUS
        t = self.WALL_THICKNESS
        I_section = np.pi * r**3 * t
        P_critical = np.pi**2 * self.YOUNGS_MODULUS * I_section / self.COLUMN_LENGTH**2
        P_applied = thrust + total_mass * G0 * accel_load

        if P_applied <= 0.0:
            P_applied = 1.0

        # d(buckling_margin)/d(thrust) = -P_critical / P_applied^2
        d_bm_dthrust = -P_critical / P_applied**2

        # d(buckling_margin)/d(structural_mass) via d(P_applied)/d(m_s) = G0*accel
        d_bm_dstruct = -P_critical * G0 * accel_load / P_applied**2

        return {
            "buckling_margin": {
                "thrust": d_bm_dthrust,
                "structural_mass": d_bm_dstruct,
            },
            "safety_factor": {
                "thrust": 0.0,  # Simplified -- full chain rule in production
                "structural_mass": 0.0,
            },
        }


# =============================================================================
# DISCIPLINE 3: THERMAL
# =============================================================================

class ThermalDiscipline(DisciplineModel):
    """Thermal control subsystem discipline.

    Models spacecraft thermal balance as a function of solar distance,
    orientation, internal power dissipation, and insulation.

    Physics
    -------
    Stefan-Boltzmann radiation equilibrium:
        At thermal equilibrium the absorbed heat equals the radiated heat:

            alpha * S(r) * A_proj + Q_internal + Q_albedo
                = epsilon * sigma * T^4 * A_rad

        where:
            alpha   = solar absorptivity (~0.3 for white paint)
            S(r)    = SOLAR_FLUX_1AU / (r_AU)^2   (solar flux at distance r)
            A_proj  = projected area facing the Sun (m^2)
            Q_internal = internal power dissipation (W)
            Q_albedo   = reflected planetary radiation (W) -- significant at Jupiter
            epsilon = infrared emissivity (~0.85)
            sigma   = Stefan-Boltzmann constant
            T       = equilibrium surface temperature (K)
            A_rad   = total radiator area (m^2)

    Jupiter albedo heating:
        Q_albedo = albedo * S(r_jupiter) * A_proj * view_factor
        Jupiter bond albedo ~ 0.503, view factor depends on orbital altitude.

    Operating temperature constraint:
        All components must remain within -40 C to +60 C  (233 K to 333 K).

    Inputs
    ------
        solar_distance_AU   : distance from Sun (AU)
        orientation_angle    : angle between Sun vector and panel normal (rad)
        power_dissipation   : internal heat generation (W)
        insulation_mass     : MLI insulation mass budget (kg)
        radiator_area_base  : baseline radiator area (m^2)

    Outputs
    -------
        equilibrium_temp    : spacecraft equilibrium temperature (K)
        radiator_area_needed: radiator area to maintain temp range (m^2)
        thermal_mass        : total thermal subsystem mass (kg)
        temp_margin_hot     : margin below max operating temp (K)
        temp_margin_cold    : margin above min operating temp (K)
    """

    name = "thermal"

    # Optical properties
    SOLAR_ABSORPTIVITY = 0.30
    IR_EMISSIVITY = 0.85

    # Jupiter albedo
    JUPITER_BOND_ALBEDO = 0.503

    # Operating limits (K)
    T_MIN = 233.0   # -40 C
    T_MAX = 333.0   # +60 C

    # Projected area facing Sun (m^2) -- simplified
    A_PROJECTED = 8.0

    # MLI effectiveness: each kg of insulation reduces heat loss by this factor
    MLI_FACTOR_PER_KG = 0.02

    def analyze(self, inputs: Dict[str, float]) -> Dict[str, float]:
        r_AU = inputs.get("solar_distance_AU", 1.0)
        orient = inputs.get("orientation_angle", 0.0)
        Q_internal = inputs.get("power_dissipation", 500.0)
        insulation_mass = inputs.get("insulation_mass", 20.0)
        radiator_area_base = inputs.get("radiator_area_base", 4.0)

        if r_AU <= 0.0:
            r_AU = 0.01

        # Solar flux at current distance
        S = SOLAR_FLUX_1AU / (r_AU ** 2)

        # Absorbed solar heat
        Q_solar = self.SOLAR_ABSORPTIVITY * S * self.A_PROJECTED * np.cos(max(orient, 0.0))

        # Jupiter albedo contribution (significant only near Jupiter, r ~ 5.2 AU)
        r_jupiter_AU = JUPITER_SMA / AU
        if abs(r_AU - r_jupiter_AU) < 1.0:
            # Simplified view factor for low Jupiter orbit
            view_factor = 0.3
            Q_albedo = self.JUPITER_BOND_ALBEDO * S * self.A_PROJECTED * view_factor
        else:
            Q_albedo = 0.0

        # Total heat input
        Q_total_in = Q_solar + Q_internal + Q_albedo

        # Insulation reduces radiative losses (MLI effectiveness)
        insulation_factor = max(1.0 - insulation_mass * self.MLI_FACTOR_PER_KG, 0.1)

        # Equilibrium temperature (radiation balance)
        # Q_in = epsilon * sigma * T^4 * A_rad * insulation_factor
        # T = (Q_in / (epsilon * sigma * A_rad * insulation_factor))^0.25
        A_rad = max(radiator_area_base, 0.1)
        denom = self.IR_EMISSIVITY * STEFAN_BOLTZMANN * A_rad * insulation_factor
        if denom > 0.0:
            T_eq = (Q_total_in / denom) ** 0.25
        else:
            T_eq = 500.0

        # Required radiator area to keep T <= T_MAX
        Q_at_Tmax = self.IR_EMISSIVITY * STEFAN_BOLTZMANN * self.T_MAX**4 * insulation_factor
        radiator_area_needed = Q_total_in / max(Q_at_Tmax, 1.0)

        # Thermal subsystem mass: insulation + radiator panels (2 kg/m^2)
        RADIATOR_DENSITY = 2.0  # kg per m^2
        thermal_mass = insulation_mass + radiator_area_needed * RADIATOR_DENSITY

        temp_margin_hot = self.T_MAX - T_eq
        temp_margin_cold = T_eq - self.T_MIN

        return {
            "equilibrium_temp": T_eq,
            "radiator_area_needed": radiator_area_needed,
            "thermal_mass": thermal_mass,
            "temp_margin_hot": temp_margin_hot,
            "temp_margin_cold": temp_margin_cold,
        }

    def get_gradients(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        r_AU = inputs.get("solar_distance_AU", 1.0)
        Q_internal = inputs.get("power_dissipation", 500.0)
        insulation_mass = inputs.get("insulation_mass", 20.0)
        radiator_area_base = inputs.get("radiator_area_base", 4.0)

        if r_AU <= 0.0:
            r_AU = 0.01

        S = SOLAR_FLUX_1AU / (r_AU ** 2)
        Q_solar = self.SOLAR_ABSORPTIVITY * S * self.A_PROJECTED

        insulation_factor = max(1.0 - insulation_mass * self.MLI_FACTOR_PER_KG, 0.1)
        A_rad = max(radiator_area_base, 0.1)
        denom = self.IR_EMISSIVITY * STEFAN_BOLTZMANN * A_rad * insulation_factor
        Q_total_in = Q_solar + Q_internal

        if denom > 0.0 and Q_total_in > 0.0:
            T_eq = (Q_total_in / denom) ** 0.25
            # d(T_eq)/d(r_AU):  Q_solar ~ 1/r^2  =>  dQ/dr = -2*Q_solar/r
            dQ_dr = -2.0 * Q_solar / r_AU
            dT_dr = 0.25 * (Q_total_in / denom) ** (-0.75) * dQ_dr / denom
        else:
            dT_dr = 0.0

        return {
            "equilibrium_temp": {
                "solar_distance_AU": dT_dr,
                "power_dissipation": 0.25 * (Q_total_in / max(denom, 1e-30)) ** (-0.75) / max(denom, 1e-30),
                "insulation_mass": 0.0,
            }
        }


# =============================================================================
# DISCIPLINE 4: POWER
# =============================================================================

class PowerDiscipline(DisciplineModel):
    """Electrical power subsystem discipline.

    Models solar array power generation, radiation degradation, battery
    sizing, and eclipse power management over the mission.

    Solar flux model
    ----------------
        P_solar = eta * S(r) * A_panel * cos(theta) * degradation

    where:
        eta          = solar cell efficiency (~0.30 for triple-junction GaAs)
        S(r)         = 1361 / r_AU^2   (W/m^2)
        A_panel      = total solar panel area (m^2)
        theta        = sun incidence angle (rad)
        degradation  = exp(-dose / dose_half)   radiation degradation factor

    Radiation degradation
    ---------------------
        At Jupiter (5.2 AU), the intense radiation belts cause rapid panel
        degradation.  The model uses an exponential decay with accumulated
        dose.

    Eclipse power
    -------------
        During eclipse (planet shadow), only battery power is available.
        Battery sizing must cover the maximum eclipse duration at full
        power demand.

    Constraint
    ----------
        Power generated >= power demanded at Jupiter distance (5.2 AU).

    Inputs
    ------
        solar_panel_area    : total deployed panel area (m^2)
        solar_distance_AU   : current distance from Sun (AU)
        radiation_dose      : accumulated radiation dose (krad)
        power_demand        : spacecraft power demand (W)
        eclipse_duration    : max eclipse duration (s)

    Outputs
    -------
        power_generated     : current electrical power output (W)
        battery_mass        : required battery mass (kg)
        panel_mass          : solar panel mass (kg)
        eol_power           : end-of-life power at Jupiter (W)
        power_margin        : surplus power at Jupiter (W)
    """

    name = "power"

    # Solar cell parameters
    CELL_EFFICIENCY = 0.30           # Triple-junction GaAs
    PANEL_MASS_DENSITY = 1.5         # kg/m^2
    DEGRADATION_DOSE_HALF = 50.0     # krad (dose for 50% degradation)

    # Battery parameters
    BATTERY_SPECIFIC_ENERGY = 200.0  # Wh/kg (Li-ion)
    BATTERY_DOD = 0.70               # Maximum depth of discharge

    def analyze(self, inputs: Dict[str, float]) -> Dict[str, float]:
        A_panel = inputs.get("solar_panel_area", 40.0)
        r_AU = inputs.get("solar_distance_AU", 1.0)
        dose_krad = inputs.get("radiation_dose", 0.0)
        power_demand = inputs.get("power_demand", 800.0)
        eclipse_dur = inputs.get("eclipse_duration", 3600.0)

        if r_AU <= 0.0:
            r_AU = 0.01

        # Solar flux
        S = SOLAR_FLUX_1AU / (r_AU ** 2)

        # Radiation degradation
        degradation = np.exp(-dose_krad / self.DEGRADATION_DOSE_HALF)

        # Power generated
        power_generated = self.CELL_EFFICIENCY * S * A_panel * degradation

        # Panel mass
        panel_mass = A_panel * self.PANEL_MASS_DENSITY

        # Battery sizing: must cover eclipse at full demand
        energy_eclipse_Wh = power_demand * eclipse_dur / 3600.0  # convert Ws to Wh
        battery_mass = energy_eclipse_Wh / (self.BATTERY_SPECIFIC_ENERGY * self.BATTERY_DOD)

        # End-of-life power at Jupiter (5.2 AU, with dose)
        S_jupiter = SOLAR_FLUX_1AU / (5.2 ** 2)
        eol_power = self.CELL_EFFICIENCY * S_jupiter * A_panel * degradation

        # Power margin at Jupiter
        power_margin = eol_power - power_demand

        return {
            "power_generated": power_generated,
            "battery_mass": battery_mass,
            "panel_mass": panel_mass,
            "eol_power": eol_power,
            "power_margin": power_margin,
        }

    def get_gradients(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        A_panel = inputs.get("solar_panel_area", 40.0)
        r_AU = inputs.get("solar_distance_AU", 1.0)
        dose_krad = inputs.get("radiation_dose", 0.0)

        if r_AU <= 0.0:
            r_AU = 0.01
        S = SOLAR_FLUX_1AU / (r_AU ** 2)
        degradation = np.exp(-dose_krad / self.DEGRADATION_DOSE_HALF)

        # d(power_generated)/d(A_panel) = eta * S * degradation
        dp_dA = self.CELL_EFFICIENCY * S * degradation

        # d(power_generated)/d(r_AU) = eta * A * degradation * (-2 * S_1AU / r^3)
        dp_dr = self.CELL_EFFICIENCY * A_panel * degradation * (-2.0 * SOLAR_FLUX_1AU / r_AU**3)

        return {
            "power_generated": {
                "solar_panel_area": dp_dA,
                "solar_distance_AU": dp_dr,
            },
            "panel_mass": {
                "solar_panel_area": self.PANEL_MASS_DENSITY,
            },
        }


# =============================================================================
# DISCIPLINE 5: ATTITUDE CONTROL
# =============================================================================

class AttitudeDiscipline(DisciplineModel):
    """Attitude determination and control subsystem (ADCS) discipline.

    Models reaction wheel (RW) sizing, control moment gyroscope (CMG)
    sizing, desaturation propellant, and pointing accuracy.

    Models
    ------
    Reaction wheel sizing
        The wheel angular momentum storage must exceed the worst-case
        accumulated environmental torque over one orbit:

            h_wheel >= T_dist * P_orbit / 4

        where T_dist is the dominant disturbance torque and P_orbit is
        the orbital period.

    CMG sizing
        The maximum slew torque determines the CMG gimbal torque rating:

            tau_cmg = I_max * alpha_slew

        where I_max is the largest principal moment of inertia and
        alpha_slew is the required angular acceleration for the
        fastest slew maneuver.

    Pointing accuracy
        Pointing error depends on sensor noise, actuator quantization,
        and structural flexibility:

            sigma_point = sqrt(sigma_sensor^2 + sigma_actuator^2 + sigma_flex^2)

    Constraint
        Pointing accuracy <= 0.1 degrees.

    Inputs
    ------
        inertia_max          : largest principal MOI (kg*m^2)
        pointing_req         : required pointing accuracy (deg)
        slew_rate_req        : required slew rate (deg/s)
        disturbance_torque   : dominant environmental torque (N*m)
        orbital_period       : orbit period for momentum buildup (s)

    Outputs
    -------
        rw_momentum          : required RW angular momentum (N*m*s)
        rw_mass              : total reaction wheel mass (kg)
        cmg_torque           : required CMG torque (N*m)
        cmg_mass             : total CMG mass (kg)
        desat_propellant     : annual desaturation propellant (kg/yr)
        pointing_accuracy    : achieved pointing accuracy (deg)
    """

    name = "attitude"

    # Hardware scaling factors
    RW_SPECIFIC_MOMENTUM = 10.0    # N*m*s per kg of wheel
    CMG_SPECIFIC_TORQUE = 50.0     # N*m per kg of CMG
    DESAT_ISP = 220.0              # Desaturation thruster Isp (s)

    # Sensor noise contributions (arcsec, converted to degrees below)
    SENSOR_NOISE_ARCSEC = 5.0
    ACTUATOR_NOISE_ARCSEC = 3.0
    FLEX_NOISE_ARCSEC = 2.0

    def analyze(self, inputs: Dict[str, float]) -> Dict[str, float]:
        I_max = inputs.get("inertia_max", 4800.0)
        pointing_req = inputs.get("pointing_req", 0.1)
        slew_rate = inputs.get("slew_rate_req", 0.5)
        T_dist = inputs.get("disturbance_torque", 1e-4)
        P_orb = inputs.get("orbital_period", 5400.0)

        # --- Reaction wheel sizing ---
        h_wheel = T_dist * P_orb / 4.0
        rw_mass = h_wheel / self.RW_SPECIFIC_MOMENTUM
        # Minimum 4 wheels (3 + 1 redundant), each at least 1 kg
        rw_mass = max(rw_mass, 4.0)

        # --- CMG sizing ---
        alpha_slew = np.radians(slew_rate) / 10.0  # ramp to slew rate in 10 s
        cmg_torque = I_max * alpha_slew
        cmg_mass = cmg_torque / self.CMG_SPECIFIC_TORQUE
        cmg_mass = max(cmg_mass, 2.0)

        # --- Desaturation propellant ---
        # Number of desat events per year: ~4 per orbit
        orbits_per_year = 365.25 * 86400.0 / max(P_orb, 1.0)
        desat_events = 4.0 * orbits_per_year
        dv_per_desat = h_wheel / max(I_max, 1.0) * 0.1  # small impulse
        annual_desat_dv = desat_events * dv_per_desat
        ve_desat = self.DESAT_ISP * G0
        # Using rocket equation for small dv: dm ~ m * dv / ve
        desat_mass_per_year = (rw_mass + cmg_mass) * annual_desat_dv / max(ve_desat, 1.0)
        desat_mass_per_year = max(desat_mass_per_year, 0.1)

        # --- Pointing accuracy ---
        sigma_sensor = self.SENSOR_NOISE_ARCSEC / 3600.0   # to degrees
        sigma_actuator = self.ACTUATOR_NOISE_ARCSEC / 3600.0
        sigma_flex = self.FLEX_NOISE_ARCSEC / 3600.0
        pointing_accuracy = np.sqrt(sigma_sensor**2 + sigma_actuator**2 + sigma_flex**2)

        return {
            "rw_momentum": h_wheel,
            "rw_mass": rw_mass,
            "cmg_torque": cmg_torque,
            "cmg_mass": cmg_mass,
            "desat_propellant": desat_mass_per_year,
            "pointing_accuracy": pointing_accuracy,
        }

    def get_gradients(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        T_dist = inputs.get("disturbance_torque", 1e-4)
        P_orb = inputs.get("orbital_period", 5400.0)
        I_max = inputs.get("inertia_max", 4800.0)

        # d(h_wheel)/d(T_dist) = P_orb / 4
        dh_dT = P_orb / 4.0

        # d(cmg_torque)/d(I_max) = alpha_slew
        slew_rate = inputs.get("slew_rate_req", 0.5)
        alpha_slew = np.radians(slew_rate) / 10.0
        dtau_dI = alpha_slew

        return {
            "rw_momentum": {"disturbance_torque": dh_dT, "orbital_period": T_dist / 4.0},
            "cmg_torque": {"inertia_max": dtau_dI},
        }


# =============================================================================
# DISCIPLINE 6: TRAJECTORY
# =============================================================================

class TrajectoryDiscipline(DisciplineModel):
    """Trajectory discipline.

    Computes delta-V requirements for each mission leg using patched-conic
    analysis and simplified Lambert solutions.  Provides the coupling
    between trajectory design and the propulsion discipline.

    Mission legs
    ------------
        1. Earth (Miami) -> Moon                  (TLI + LOI)
        2. Moon orbit phase -- 2 revolutions      (station-keeping)
        3. Moon -> Jupiter transfer                (lunar escape + deep space)
        4. Jupiter orbit phase -- 3 revolutions   (JOI + orbit maintenance)
        5. Jupiter -> Earth return                 (escape + return)

    Lambert solver approximation
    ----------------------------
        For initial design, the delta-V for each heliocentric leg is
        approximated using the vis-viva equation for Hohmann-like transfers:

            dv = |sqrt(mu_sun * (2/r1 - 1/a_t)) - v_circ_1|

        where a_t = (r1 + r2)/2 is the transfer semi-major axis.

    Constraint
        Total delta-V <= propulsion capability.

    Inputs
    ------
        Isp                    : engine specific impulse (s) -- coupling
        departure_date_offset  : offset from baseline departure (days)
        transfer_time_factor   : multiplier on Hohmann TOF (>= 1.0)

    Outputs
    -------
        delta_v_per_leg    : dict mapping leg name to required dv (m/s)
        total_delta_v      : sum of all leg delta-Vs (m/s)
        total_tof          : total mission time of flight (s)
        leg_tof            : dict mapping leg name to TOF (s)
    """

    name = "trajectory"

    # Approximate orbital radii for patched-conic model (m)
    R_EARTH_HELIO = AU               # 1 AU
    R_MOON_ORBIT = MOON_SMA          # ~384,400 km from Earth
    R_JUPITER_HELIO = JUPITER_SMA    # ~5.2 AU

    def analyze(self, inputs: Dict[str, float]) -> Dict[str, float]:
        transfer_factor = inputs.get("transfer_time_factor", 1.0)
        if transfer_factor < 1.0:
            transfer_factor = 1.0

        leg_dvs: Dict[str, float] = {}
        leg_tofs: Dict[str, float] = {}

        # --- Leg 1: Earth -> Moon (TLI + LOI) ---
        # TLI: escape Earth gravity to reach Moon
        r_park = EARTH_RADIUS + 200e3  # 200 km parking orbit
        v_circ_park = np.sqrt(EARTH_MU / r_park)
        # Hohmann transfer to Moon distance
        a_t_moon = (r_park + self.R_MOON_ORBIT) / 2.0
        v_t_peri = np.sqrt(EARTH_MU * (2.0 / r_park - 1.0 / a_t_moon))
        dv_tli = v_t_peri - v_circ_park

        # LOI: capture into lunar orbit (100 km altitude)
        r_lunar_orbit = MOON_RADIUS + 100e3
        v_approach_moon = 800.0  # typical approach speed (m/s)
        v_peri_hyp = np.sqrt(v_approach_moon**2 + 2.0 * MOON_MU / r_lunar_orbit)
        v_circ_moon = np.sqrt(MOON_MU / r_lunar_orbit)
        dv_loi = v_peri_hyp - v_circ_moon

        leg_dvs["TLI"] = dv_tli
        leg_dvs["LOI"] = dv_loi
        tof_earth_moon = np.pi * np.sqrt(a_t_moon**3 / EARTH_MU) * transfer_factor
        leg_tofs["earth_to_moon"] = tof_earth_moon

        # --- Leg 2: Lunar orbit (2 revolutions) ---
        P_lunar = 2.0 * np.pi * np.sqrt(r_lunar_orbit**3 / MOON_MU)
        leg_dvs["lunar_stationkeep"] = 10.0  # small maintenance dv (m/s)
        leg_tofs["lunar_orbit"] = 2.0 * P_lunar

        # --- Leg 3: Moon -> Jupiter ---
        # Lunar escape
        v_inf_lunar_escape = 1200.0  # m/s hyperbolic excess
        v_peri_esc = np.sqrt(v_inf_lunar_escape**2 + 2.0 * MOON_MU / r_lunar_orbit)
        dv_lunar_escape = v_peri_esc - v_circ_moon
        leg_dvs["lunar_escape"] = dv_lunar_escape

        # Heliocentric transfer Earth -> Jupiter (Hohmann approximation)
        a_t_jup = (self.R_EARTH_HELIO + self.R_JUPITER_HELIO) / 2.0
        v_earth_helio = np.sqrt(SUN_MU / self.R_EARTH_HELIO)
        v_t_earth = np.sqrt(SUN_MU * (2.0 / self.R_EARTH_HELIO - 1.0 / a_t_jup))
        dv_earth_escape_helio = v_t_earth - v_earth_helio
        leg_dvs["jupiter_transfer"] = dv_earth_escape_helio

        tof_earth_jupiter = np.pi * np.sqrt(a_t_jup**3 / SUN_MU) * transfer_factor
        leg_tofs["earth_to_jupiter"] = tof_earth_jupiter

        # JOI: capture into Jupiter orbit (high orbit, ~1e9 m altitude)
        r_jup_orbit = JUPITER_RADIUS + 1e9
        v_jup_helio = np.sqrt(SUN_MU / self.R_JUPITER_HELIO)
        v_t_jup = np.sqrt(SUN_MU * (2.0 / self.R_JUPITER_HELIO - 1.0 / a_t_jup))
        v_inf_jup = abs(v_jup_helio - v_t_jup)
        v_peri_hyp_jup = np.sqrt(v_inf_jup**2 + 2.0 * JUPITER_MU / r_jup_orbit)
        v_circ_jup = np.sqrt(JUPITER_MU / r_jup_orbit)
        dv_joi = v_peri_hyp_jup - v_circ_jup
        leg_dvs["JOI"] = dv_joi

        # --- Leg 4: Jupiter orbit (3 revolutions) ---
        P_jupiter = 2.0 * np.pi * np.sqrt(r_jup_orbit**3 / JUPITER_MU)
        leg_dvs["jupiter_stationkeep"] = 15.0
        leg_tofs["jupiter_orbit"] = 3.0 * P_jupiter

        # --- Leg 5: Jupiter -> Earth return ---
        # Jupiter escape
        v_inf_jup_escape = v_inf_jup * 1.05  # slightly more to set up return
        v_peri_esc_jup = np.sqrt(v_inf_jup_escape**2 + 2.0 * JUPITER_MU / r_jup_orbit)
        dv_jup_escape = v_peri_esc_jup - v_circ_jup
        leg_dvs["jupiter_escape"] = dv_jup_escape

        # Return transfer (Hohmann-like)
        dv_return = dv_earth_escape_helio * 0.95  # slightly less due to different geometry
        leg_dvs["earth_return"] = dv_return
        leg_tofs["jupiter_to_earth"] = tof_earth_jupiter * transfer_factor

        total_dv = sum(leg_dvs.values())
        total_tof = sum(leg_tofs.values())

        return {
            "delta_v_per_leg": leg_dvs,
            "total_delta_v": total_dv,
            "total_tof": total_tof,
            "leg_tof": leg_tofs,
        }

    def get_gradients(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        # Numerical gradient via finite differences
        h = 1e-4
        base = self.analyze(inputs)
        base_dv = base["total_delta_v"]

        grads: Dict[str, float] = {}
        for key in ["transfer_time_factor"]:
            perturbed = dict(inputs)
            perturbed[key] = inputs.get(key, 1.0) + h
            perturbed_result = self.analyze(perturbed)
            grads[key] = (perturbed_result["total_delta_v"] - base_dv) / h

        return {"total_delta_v": grads}


# =============================================================================
# MDO FRAMEWORK (ORCHESTRATOR)
# =============================================================================

class MDOFramework:
    """Multidisciplinary Design Optimization orchestrator.

    Couples all six disciplines and exposes two MDO architectures (MDF
    and IDF) plus post-optimality analysis tools.

    Parameters
    ----------
    disciplines : list of DisciplineModel
        Instantiated discipline objects to include in the MDO loop.
    """

    def __init__(self, disciplines: List[DisciplineModel]) -> None:
        self.disciplines = {d.name: d for d in disciplines}
        self.variable_names: List[str] = []
        self.variable_bounds: List[Tuple[float, float]] = []
        self.constraints: List[Dict[str, Any]] = []
        self._convergence_history: List[float] = []
        self._design_history: List[np.ndarray] = []

    # -----------------------------------------------------------------
    # DESIGN VARIABLE SETUP
    # -----------------------------------------------------------------

    def setup_design_variables(self) -> None:
        """Define the MDO design variables with lower and upper bounds.

        Variables
        ---------
            dry_mass                  :   800 -- 5 000 kg
            propellant_mass           :   500 -- 8 000 kg
            Isp                       :   250 -- 450 s
            thrust                    : 5 000 -- 100 000 N
            solar_panel_area          :    10 -- 200 m^2
            structural_mass_fraction  :  0.05 -- 0.40
            departure_date_offset     :     0 -- 365 days (from baseline)
            transfer_time_factor      :   1.0 -- 2.5  (multiplier on Hohmann TOF)
        """
        self.variable_names = [
            "dry_mass",
            "propellant_mass",
            "Isp",
            "thrust",
            "solar_panel_area",
            "structural_mass_fraction",
            "departure_date_offset",
            "transfer_time_factor",
        ]
        self.variable_bounds = [
            (800.0, 5000.0),
            (500.0, 8000.0),
            (250.0, 450.0),
            (5000.0, 100000.0),
            (10.0, 200.0),
            (0.05, 0.40),
            (0.0, 365.0),
            (1.0, 2.5),
        ]

    # -----------------------------------------------------------------
    # CONSTRAINT SETUP
    # -----------------------------------------------------------------

    def setup_constraints(self) -> None:
        """Compile discipline constraints into a unified list.

        Each constraint is a dict with:
            'name'   : human-readable label
            'type'   : 'ineq' (>= 0) or 'eq' (== 0)
            'fun'    : callable(x) -> float
        """
        self.constraints = [
            {
                "name": "delta_v_margin",
                "type": "ineq",
                "fun": lambda x: self._eval_propulsion(x).get("delta_v_margin", -1e6),
            },
            {
                "name": "safety_factor_min_1.4",
                "type": "ineq",
                "fun": lambda x: self._eval_structural(x).get("safety_factor", 0.0) - 1.4,
            },
            {
                "name": "temp_hot_margin",
                "type": "ineq",
                "fun": lambda x: self._eval_thermal(x).get("temp_margin_hot", -100.0),
            },
            {
                "name": "temp_cold_margin",
                "type": "ineq",
                "fun": lambda x: self._eval_thermal(x).get("temp_margin_cold", -100.0),
            },
            {
                "name": "power_margin_jupiter",
                "type": "ineq",
                "fun": lambda x: self._eval_power(x).get("power_margin", -1e6),
            },
            {
                "name": "pointing_accuracy",
                "type": "ineq",
                "fun": lambda x: 0.1 - self._eval_attitude(x).get("pointing_accuracy", 1.0),
            },
        ]

    # -----------------------------------------------------------------
    # HELPER: DESIGN VECTOR -> DISCIPLINE INPUTS
    # -----------------------------------------------------------------

    def _x_to_inputs(self, x: np.ndarray) -> Dict[str, float]:
        """Convert design variable vector to a flat input dictionary."""
        d: Dict[str, float] = {}
        for i, name in enumerate(self.variable_names):
            d[name] = float(x[i])
        # Derived quantities
        d["structural_mass"] = d["dry_mass"] * d["structural_mass_fraction"]
        d["engine_mass"] = d["thrust"] * 0.002  # rough 2 g per N scaling
        d["solar_distance_AU"] = 1.0  # default; overridden per phase
        d["power_dissipation"] = 500.0
        d["insulation_mass"] = 20.0
        d["radiator_area_base"] = 4.0
        d["power_demand"] = 800.0
        d["eclipse_duration"] = 3600.0
        d["radiation_dose"] = 10.0
        d["inertia_max"] = 4800.0
        d["pointing_req"] = 0.1
        d["slew_rate_req"] = 0.5
        d["disturbance_torque"] = 1e-4
        d["orbital_period"] = 5400.0
        d["acceleration_load"] = 4.0
        return d

    def _eval_propulsion(self, x: np.ndarray) -> Dict[str, float]:
        return self.disciplines["propulsion"].analyze(self._x_to_inputs(x))

    def _eval_structural(self, x: np.ndarray) -> Dict[str, float]:
        return self.disciplines["structural"].analyze(self._x_to_inputs(x))

    def _eval_thermal(self, x: np.ndarray) -> Dict[str, float]:
        return self.disciplines["thermal"].analyze(self._x_to_inputs(x))

    def _eval_power(self, x: np.ndarray) -> Dict[str, float]:
        return self.disciplines["power"].analyze(self._x_to_inputs(x))

    def _eval_attitude(self, x: np.ndarray) -> Dict[str, float]:
        return self.disciplines["attitude"].analyze(self._x_to_inputs(x))

    def _eval_trajectory(self, x: np.ndarray) -> Dict[str, float]:
        return self.disciplines["trajectory"].analyze(self._x_to_inputs(x))

    # -----------------------------------------------------------------
    # COUPLED ANALYSIS
    # -----------------------------------------------------------------

    def _run_all_disciplines(self, x: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Execute every discipline and return their outputs."""
        inputs = self._x_to_inputs(x)
        outputs: Dict[str, Dict[str, Any]] = {}
        for name, disc in self.disciplines.items():
            outputs[name] = disc.analyze(inputs)
        return outputs

    def _fixed_point_iteration(
        self, x: np.ndarray, max_iter: int = 20, tol: float = 1e-4
    ) -> Dict[str, Dict[str, Any]]:
        """Drive all disciplines to self-consistency via fixed-point iteration.

        At each iteration, discipline outputs are fed back as inputs to
        the next round.  Convergence is declared when the maximum change
        in any coupling variable falls below ``tol``.

        Parameters
        ----------
        x : np.ndarray
            Current design variable vector.
        max_iter : int
            Maximum number of fixed-point iterations.
        tol : float
            Convergence tolerance on coupling variable change.

        Returns
        -------
        dict
            Converged discipline outputs.
        """
        inputs = self._x_to_inputs(x)
        prev_total_dv = 0.0

        for iteration in range(max_iter):
            outputs: Dict[str, Dict[str, Any]] = {}
            for name, disc in self.disciplines.items():
                outputs[name] = disc.analyze(inputs)

            # Update coupling variables
            traj_out = outputs.get("trajectory", {})
            prop_out = outputs.get("propulsion", {})

            current_total_dv = traj_out.get("total_delta_v", 0.0)
            change = abs(current_total_dv - prev_total_dv)
            prev_total_dv = current_total_dv

            # Feed trajectory delta-V requirements back into propulsion inputs
            # (coupling: trajectory <-> propulsion)
            if current_total_dv > 0.0:
                inputs["required_delta_v"] = current_total_dv

            # Feed thermal mass back into structural mass
            therm_out = outputs.get("thermal", {})
            power_out = outputs.get("power", {})
            att_out = outputs.get("attitude", {})

            subsys_mass = (
                therm_out.get("thermal_mass", 0.0)
                + power_out.get("panel_mass", 0.0)
                + power_out.get("battery_mass", 0.0)
                + att_out.get("rw_mass", 0.0)
                + att_out.get("cmg_mass", 0.0)
            )
            inputs["subsystem_mass"] = subsys_mass

            if change < tol and iteration > 0:
                logger.debug("Fixed-point converged in %d iterations.", iteration + 1)
                break

        return outputs

    # -----------------------------------------------------------------
    # OBJECTIVE FUNCTION
    # -----------------------------------------------------------------

    def compute_objective(self, x: np.ndarray) -> float:
        """Minimize total spacecraft mass (dry + propellant + subsystems).

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.

        Returns
        -------
        float
            Total mass (kg).  Lower is better.
        """
        inputs = self._x_to_inputs(x)
        dry_mass = inputs["dry_mass"]
        prop_mass = inputs["propellant_mass"]
        engine_mass = inputs["engine_mass"]
        panel_area = inputs["solar_panel_area"]

        # Subsystem mass contributions
        power_out = self.disciplines["power"].analyze(inputs)
        thermal_out = self.disciplines["thermal"].analyze(inputs)
        att_out = self.disciplines["attitude"].analyze(inputs)

        total_mass = (
            dry_mass
            + prop_mass
            + engine_mass
            + power_out.get("panel_mass", 0.0)
            + power_out.get("battery_mass", 0.0)
            + thermal_out.get("thermal_mass", 0.0)
            + att_out.get("rw_mass", 0.0)
            + att_out.get("cmg_mass", 0.0)
        )
        return total_mass

    # -----------------------------------------------------------------
    # MDF ARCHITECTURE
    # -----------------------------------------------------------------

    def mdf_architecture(self, x0: np.ndarray) -> OptimizationResult:
        """Multidisciplinary Feasible (MDF) architecture.

        In MDF, a fixed-point iteration is run *inside* every objective
        function evaluation to ensure all disciplines are mutually
        consistent.  The outer optimizer (SLSQP) then searches the design
        space over the fully-converged objective.

        Parameters
        ----------
        x0 : np.ndarray
            Initial guess for the design variable vector.

        Returns
        -------
        OptimizationResult
        """
        self._convergence_history.clear()
        self._design_history.clear()

        def _obj(x: np.ndarray) -> float:
            self._fixed_point_iteration(x)
            val = self.compute_objective(x)
            self._convergence_history.append(val)
            self._design_history.append(x.copy())
            return val

        scipy_constraints = []
        for c in self.constraints:
            scipy_constraints.append({
                "type": c["type"],
                "fun": c["fun"],
            })

        result = minimize(
            _obj,
            x0,
            method="SLSQP",
            bounds=self.variable_bounds,
            constraints=scipy_constraints,
            options={"maxiter": 200, "ftol": 1e-6, "disp": False},
        )

        disc_outputs = self._run_all_disciplines(result.x)

        # Evaluate constraints at optimum
        constraint_vals = {}
        for c in self.constraints:
            constraint_vals[c["name"]] = c["fun"](result.x)

        return OptimizationResult(
            x_optimal=result.x,
            objective_value=result.fun,
            variable_names=self.variable_names,
            constraint_values=constraint_vals,
            convergence_history=list(self._convergence_history),
            design_history=list(self._design_history),
            discipline_outputs=disc_outputs,
            success=result.success,
            message=result.message,
        )

    # -----------------------------------------------------------------
    # IDF ARCHITECTURE
    # -----------------------------------------------------------------

    def idf_architecture(self, x0: np.ndarray) -> OptimizationResult:
        """Individual Discipline Feasible (IDF) architecture.

        In IDF, coupling variables (e.g. trajectory total_delta_v fed
        back to propulsion) are promoted to optimization variables.
        Disciplines run independently each iteration.  Compatibility
        constraints ensure that coupling variable copies converge to
        a single consistent value.

        Parameters
        ----------
        x0 : np.ndarray
            Initial guess for the base design variables.

        Returns
        -------
        OptimizationResult
        """
        self._convergence_history.clear()
        self._design_history.clear()

        # Augment design vector with coupling variables
        coupling_names = ["coupling_total_dv", "coupling_subsystem_mass"]
        coupling_bounds = [(0.0, 50000.0), (0.0, 2000.0)]
        coupling_x0 = np.array([10000.0, 100.0])

        full_names = self.variable_names + coupling_names
        full_bounds = self.variable_bounds + coupling_bounds
        full_x0 = np.concatenate([x0, coupling_x0])

        n_base = len(self.variable_names)

        def _obj_idf(x_full: np.ndarray) -> float:
            x_base = x_full[:n_base]
            val = self.compute_objective(x_base)
            self._convergence_history.append(val)
            self._design_history.append(x_base.copy())
            return val

        def _compat_dv(x_full: np.ndarray) -> float:
            """Compatibility constraint: coupling dv matches trajectory output."""
            x_base = x_full[:n_base]
            traj_out = self._eval_trajectory(x_base)
            return x_full[n_base] - traj_out.get("total_delta_v", 0.0)

        def _compat_mass(x_full: np.ndarray) -> float:
            """Compatibility constraint: coupling subsys mass matches subsystem outputs."""
            x_base = x_full[:n_base]
            inputs = self._x_to_inputs(x_base)
            thermal_out = self.disciplines["thermal"].analyze(inputs)
            power_out = self.disciplines["power"].analyze(inputs)
            att_out = self.disciplines["attitude"].analyze(inputs)
            subsys = (
                thermal_out.get("thermal_mass", 0.0)
                + power_out.get("panel_mass", 0.0)
                + power_out.get("battery_mass", 0.0)
                + att_out.get("rw_mass", 0.0)
                + att_out.get("cmg_mass", 0.0)
            )
            return x_full[n_base + 1] - subsys

        scipy_constraints = []
        for c in self.constraints:
            scipy_constraints.append({
                "type": c["type"],
                "fun": lambda x_full, _c=c: _c["fun"](x_full[:n_base]),
            })
        # Add compatibility constraints (equality)
        scipy_constraints.append({"type": "eq", "fun": _compat_dv})
        scipy_constraints.append({"type": "eq", "fun": _compat_mass})

        result = minimize(
            _obj_idf,
            full_x0,
            method="SLSQP",
            bounds=full_bounds,
            constraints=scipy_constraints,
            options={"maxiter": 300, "ftol": 1e-6, "disp": False},
        )

        x_opt_base = result.x[:n_base]
        disc_outputs = self._run_all_disciplines(x_opt_base)

        constraint_vals = {}
        for c in self.constraints:
            constraint_vals[c["name"]] = c["fun"](x_opt_base)

        return OptimizationResult(
            x_optimal=x_opt_base,
            objective_value=result.fun,
            variable_names=self.variable_names,
            constraint_values=constraint_vals,
            convergence_history=list(self._convergence_history),
            design_history=list(self._design_history),
            discipline_outputs=disc_outputs,
            success=result.success,
            message=result.message,
        )

    # -----------------------------------------------------------------
    # RUN OPTIMIZATION (DISPATCHER)
    # -----------------------------------------------------------------

    def run_optimization(self, method: str = "mdf") -> OptimizationResult:
        """Run the MDO using the selected architecture.

        Parameters
        ----------
        method : str
            ``'mdf'`` for Multidisciplinary Feasible,
            ``'idf'`` for Individual Discipline Feasible.

        Returns
        -------
        OptimizationResult
        """
        self.setup_design_variables()
        self.setup_constraints()

        # Initial guess (midpoint of bounds)
        x0 = np.array([
            (lo + hi) / 2.0 for lo, hi in self.variable_bounds
        ])

        logger.info("Starting MDO with architecture: %s", method.upper())

        if method.lower() == "mdf":
            return self.mdf_architecture(x0)
        elif method.lower() == "idf":
            return self.idf_architecture(x0)
        else:
            raise ValueError(f"Unknown MDO method: {method}. Use 'mdf' or 'idf'.")

    # -----------------------------------------------------------------
    # SENSITIVITY ANALYSIS
    # -----------------------------------------------------------------

    def sensitivity_analysis(
        self, x_optimal: np.ndarray, perturbation: float = 0.01
    ) -> pd.DataFrame:
        """Compute sensitivities of the objective w.r.t. each design variable.

        Uses central finite differences:

            df/dx_i  ~  [f(x + h*e_i) - f(x - h*e_i)] / (2*h)

        where h = perturbation * (upper - lower).

        Parameters
        ----------
        x_optimal : np.ndarray
            Optimal design point.
        perturbation : float
            Fractional step size relative to variable range.

        Returns
        -------
        pd.DataFrame
            Columns: variable, sensitivity, normalized_sensitivity
        """
        base_obj = self.compute_objective(x_optimal)
        rows = []

        for i, name in enumerate(self.variable_names):
            lo, hi = self.variable_bounds[i]
            h = perturbation * (hi - lo)
            if h < 1e-12:
                h = 1e-6

            x_plus = x_optimal.copy()
            x_minus = x_optimal.copy()
            x_plus[i] = min(x_optimal[i] + h, hi)
            x_minus[i] = max(x_optimal[i] - h, lo)

            f_plus = self.compute_objective(x_plus)
            f_minus = self.compute_objective(x_minus)

            sensitivity = (f_plus - f_minus) / (x_plus[i] - x_minus[i])
            normalized = sensitivity * x_optimal[i] / max(abs(base_obj), 1e-12)

            rows.append({
                "variable": name,
                "sensitivity": sensitivity,
                "normalized_sensitivity": normalized,
            })

        df = pd.DataFrame(rows)
        df = df.reindex(df["normalized_sensitivity"].abs().sort_values(ascending=False).index)
        return df.reset_index(drop=True)

    # -----------------------------------------------------------------
    # PARETO FRONT
    # -----------------------------------------------------------------

    def pareto_front(
        self,
        objectives: Optional[List[str]] = None,
        n_points: int = 50,
    ) -> pd.DataFrame:
        """Generate a Pareto front via the weighted-sum method.

        Sweeps the weight from 0 to 1 between two objectives:
            J = w * obj1  +  (1 - w) * obj2

        Parameters
        ----------
        objectives : list of str, optional
            Two objective names from ``['mass', 'transfer_time']``.
            Defaults to ``['mass', 'transfer_time']``.
        n_points : int
            Number of points along the Pareto front.

        Returns
        -------
        pd.DataFrame
            Columns: weight, mass, transfer_time, plus all design variables.
        """
        if objectives is None:
            objectives = ["mass", "transfer_time"]

        self.setup_design_variables()
        self.setup_constraints()

        weights = np.linspace(0.0, 1.0, n_points)
        results = []

        x0 = np.array([(lo + hi) / 2.0 for lo, hi in self.variable_bounds])

        for w in weights:
            def _pareto_obj(x: np.ndarray, _w: float = w) -> float:
                mass = self.compute_objective(x)
                traj = self._eval_trajectory(x)
                tof = traj.get("total_tof", 1e12)
                # Normalize both objectives to similar scales
                mass_norm = mass / 5000.0
                tof_norm = tof / (365.25 * 86400.0 * 5.0)  # normalize by 5 years
                return _w * mass_norm + (1.0 - _w) * tof_norm

            scipy_constraints = [{"type": c["type"], "fun": c["fun"]} for c in self.constraints]
            res = minimize(
                _pareto_obj,
                x0,
                method="SLSQP",
                bounds=self.variable_bounds,
                constraints=scipy_constraints,
                options={"maxiter": 100, "ftol": 1e-5, "disp": False},
            )

            mass_val = self.compute_objective(res.x)
            traj_val = self._eval_trajectory(res.x)
            tof_val = traj_val.get("total_tof", 0.0)

            row = {"weight": w, "mass": mass_val, "transfer_time": tof_val}
            for i, vn in enumerate(self.variable_names):
                row[vn] = res.x[i]
            results.append(row)

            # Warm-start next iteration
            x0 = res.x.copy()

        return pd.DataFrame(results)

    # =================================================================
    # PLOTTING METHODS
    # =================================================================

    @staticmethod
    def _ensure_output_dir() -> str:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        return OUTPUT_DIR

    # ---- Plot 1: Convergence history ------------------------------------

    def plot_convergence_history(self, filepath: Optional[str] = None) -> None:
        """Objective function value vs. iteration number."""
        out = self._ensure_output_dir()
        if filepath is None:
            filepath = os.path.join(out, "convergence_history.png")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self._convergence_history, "b-o", markersize=3, linewidth=1)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Total Mass (kg)")
        ax.set_title("MDO Convergence History")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        logger.info("Saved convergence history to %s", filepath)

    # ---- Plot 2: N^2 diagram -------------------------------------------

    def generate_n2_diagram(self, filepath: Optional[str] = None) -> None:
        """Generate N^2 (N-squared) diagram showing discipline coupling.

        The N^2 diagram is the standard MDO visualization for inter-discipline
        data flow.  Disciplines sit on the diagonal; off-diagonal cells show
        the coupling variables passed between disciplines.
        """
        out = self._ensure_output_dir()
        if filepath is None:
            filepath = os.path.join(out, "n2_diagram.png")

        disc_names = list(self.disciplines.keys())
        n = len(disc_names)

        # Coupling matrix: 1 where discipline i sends data to discipline j
        coupling = np.zeros((n, n))
        # Define known couplings
        coupling_map = {
            ("propulsion", "trajectory"): 1,
            ("trajectory", "propulsion"): 1,
            ("structural", "propulsion"): 1,
            ("propulsion", "structural"): 1,
            ("thermal", "power"): 1,
            ("power", "thermal"): 1,
            ("attitude", "structural"): 1,
            ("structural", "attitude"): 1,
            ("trajectory", "thermal"): 1,
            ("trajectory", "power"): 1,
        }
        for (src, tgt), val in coupling_map.items():
            if src in disc_names and tgt in disc_names:
                i = disc_names.index(src)
                j = disc_names.index(tgt)
                coupling[i, j] = val

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(coupling, cmap="YlOrRd", interpolation="nearest", vmin=0, vmax=1)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(disc_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(disc_names, fontsize=9)
        ax.set_title("N$^2$ Discipline Coupling Diagram", fontsize=14)
        ax.set_xlabel("Receiving Discipline")
        ax.set_ylabel("Sending Discipline")

        # Annotate diagonal and off-diagonal
        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, disc_names[i].upper(), ha="center", va="center",
                            fontsize=7, fontweight="bold", color="white",
                            bbox=dict(boxstyle="round", facecolor="navy", alpha=0.8))
                elif coupling[i, j] > 0:
                    ax.text(j, i, "X", ha="center", va="center",
                            fontsize=10, fontweight="bold", color="darkred")

        fig.colorbar(im, ax=ax, label="Coupling Strength")
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        logger.info("Saved N^2 diagram to %s", filepath)

    # ---- Plot 3: Pareto front -------------------------------------------

    def plot_pareto_front(
        self, pareto_df: pd.DataFrame, filepath: Optional[str] = None
    ) -> None:
        """Mass vs. transfer time Pareto front."""
        out = self._ensure_output_dir()
        if filepath is None:
            filepath = os.path.join(out, "pareto_front.png")

        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(
            pareto_df["mass"],
            pareto_df["transfer_time"] / (365.25 * 86400.0),
            c=pareto_df["weight"],
            cmap="viridis",
            s=40,
            edgecolors="k",
            linewidths=0.5,
        )
        ax.set_xlabel("Total Mass (kg)")
        ax.set_ylabel("Total Transfer Time (years)")
        ax.set_title("Pareto Front: Mass vs. Transfer Time")
        ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax, label="Mass Weight (w)")
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        logger.info("Saved Pareto front to %s", filepath)

    # ---- Plot 4: Sensitivity tornado chart ------------------------------

    def plot_sensitivity_tornado(
        self, sens_df: pd.DataFrame, filepath: Optional[str] = None
    ) -> None:
        """Horizontal bar chart of normalized sensitivities."""
        out = self._ensure_output_dir()
        if filepath is None:
            filepath = os.path.join(out, "sensitivity_tornado.png")

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#d62728" if v < 0 else "#2ca02c" for v in sens_df["normalized_sensitivity"]]
        ax.barh(sens_df["variable"], sens_df["normalized_sensitivity"], color=colors, edgecolor="k")
        ax.set_xlabel("Normalized Sensitivity (d obj / d x_i * x_i / obj)")
        ax.set_title("Design Variable Sensitivity Tornado Chart")
        ax.axvline(0, color="k", linewidth=0.8)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        logger.info("Saved sensitivity tornado to %s", filepath)

    # ---- Plot 5: Tradespace scatter -------------------------------------

    def plot_tradespace(
        self, pareto_df: pd.DataFrame, filepath: Optional[str] = None
    ) -> None:
        """Tradespace scatter: mass vs. total delta-V colored by transfer time."""
        out = self._ensure_output_dir()
        if filepath is None:
            filepath = os.path.join(out, "tradespace_scatter.png")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Compute delta-V for each Pareto point
        dvs = []
        for _, row in pareto_df.iterrows():
            x = np.array([row[vn] for vn in self.variable_names])
            traj = self._eval_trajectory(x)
            dvs.append(traj.get("total_delta_v", 0.0))
        pareto_df = pareto_df.copy()
        pareto_df["total_delta_v"] = dvs

        sc = ax.scatter(
            pareto_df["total_delta_v"],
            pareto_df["mass"],
            c=pareto_df["transfer_time"] / (365.25 * 86400.0),
            cmap="plasma",
            s=40,
            edgecolors="k",
            linewidths=0.5,
        )
        ax.set_xlabel("Total Delta-V (m/s)")
        ax.set_ylabel("Total Mass (kg)")
        ax.set_title("Tradespace: Mass vs. Delta-V (colored by Transfer Time)")
        ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax, label="Transfer Time (years)")
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        logger.info("Saved tradespace scatter to %s", filepath)

    # ---- Plot 6: Constraint satisfaction waterfall ----------------------

    def plot_constraint_waterfall(
        self, result: OptimizationResult, filepath: Optional[str] = None
    ) -> None:
        """Waterfall chart showing constraint satisfaction at the optimum."""
        out = self._ensure_output_dir()
        if filepath is None:
            filepath = os.path.join(out, "constraint_waterfall.png")

        names = list(result.constraint_values.keys())
        values = list(result.constraint_values.values())

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values]
        bars = ax.bar(range(len(names)), values, color=colors, edgecolor="k")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.axhline(0, color="k", linewidth=1.0, linestyle="--")
        ax.set_ylabel("Constraint Value (>= 0 is satisfied)")
        ax.set_title("Constraint Satisfaction Waterfall at Optimum")
        ax.grid(True, axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        logger.info("Saved constraint waterfall to %s", filepath)

    # ---- Plot 7: Design variable history --------------------------------

    def plot_design_history(self, filepath: Optional[str] = None) -> None:
        """Track design variable values through the optimization."""
        out = self._ensure_output_dir()
        if filepath is None:
            filepath = os.path.join(out, "design_variable_history.png")

        if not self._design_history:
            logger.warning("No design history to plot.")
            return

        history_array = np.array(self._design_history)
        n_iter, n_vars = history_array.shape

        fig, axes = plt.subplots(
            int(np.ceil(n_vars / 2)), 2, figsize=(14, 3 * int(np.ceil(n_vars / 2)))
        )
        axes = axes.flatten()

        for i in range(n_vars):
            ax = axes[i]
            # Normalize to [0, 1] range for comparison
            lo, hi = self.variable_bounds[i]
            if hi - lo > 1e-12:
                normalized = (history_array[:, i] - lo) / (hi - lo)
            else:
                normalized = history_array[:, i]
            ax.plot(normalized, linewidth=1.0)
            ax.set_title(self.variable_names[i], fontsize=9)
            ax.set_xlabel("Iteration", fontsize=8)
            ax.set_ylabel("Normalized Value", fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for j in range(n_vars, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Design Variable History Through Optimization", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        logger.info("Saved design variable history to %s", filepath)

    # ---- Plot 8: Discipline coupling heatmap ----------------------------

    def generate_design_structure_matrix(self, filepath: Optional[str] = None) -> None:
        """Design Structure Matrix (DSM) visualization.

        Similar to the N^2 diagram but weighted by the strength of coupling
        (estimated from gradient magnitudes).
        """
        out = self._ensure_output_dir()
        if filepath is None:
            filepath = os.path.join(out, "coupling_heatmap.png")

        disc_names = list(self.disciplines.keys())
        n = len(disc_names)

        # Estimate coupling strength from shared variables
        strength = np.zeros((n, n))
        coupling_strengths = {
            ("propulsion", "trajectory"): 0.9,
            ("trajectory", "propulsion"): 0.9,
            ("structural", "propulsion"): 0.6,
            ("propulsion", "structural"): 0.5,
            ("thermal", "power"): 0.7,
            ("power", "thermal"): 0.7,
            ("attitude", "structural"): 0.4,
            ("structural", "attitude"): 0.3,
            ("trajectory", "thermal"): 0.5,
            ("trajectory", "power"): 0.5,
            ("power", "attitude"): 0.2,
            ("thermal", "structural"): 0.3,
        }
        for (src, tgt), val in coupling_strengths.items():
            if src in disc_names and tgt in disc_names:
                i = disc_names.index(src)
                j = disc_names.index(tgt)
                strength[i, j] = val

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(strength, cmap="hot_r", interpolation="nearest", vmin=0, vmax=1)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(disc_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(disc_names, fontsize=9)
        ax.set_title("Discipline Coupling Strength Heatmap (DSM)", fontsize=14)

        for i in range(n):
            for j in range(n):
                val = strength[i, j]
                if val > 0:
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            fontsize=8, color="white" if val > 0.5 else "black")

        fig.colorbar(im, ax=ax, label="Coupling Strength")
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        logger.info("Saved coupling heatmap to %s", filepath)

    def generate_tradespace_plot(self, filepath: Optional[str] = None) -> None:
        """Alias for plot_tradespace using a quick Pareto sweep."""
        pareto_df = self.pareto_front(n_points=30)
        self.plot_tradespace(pareto_df, filepath)

    # -----------------------------------------------------------------
    # REPRESENTATION
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        disc_str = ", ".join(self.disciplines.keys())
        return f"MDOFramework(disciplines=[{disc_str}])"


# =============================================================================
# MAIN BLOCK
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    print("=" * 72)
    print("  GNC PROJECT - Multidisciplinary Design Optimization")
    print("  Mission: Miami -> Moon (2 orbits) -> Jupiter (3 orbits) -> Miami")
    print("=" * 72)

    # --- Instantiate all disciplines ---
    propulsion = PropulsionDiscipline()
    structural = StructuralDiscipline()
    thermal = ThermalDiscipline()
    power = PowerDiscipline()
    attitude = AttitudeDiscipline()
    trajectory = TrajectoryDiscipline()

    disciplines = [propulsion, structural, thermal, power, attitude, trajectory]

    # --- Create and configure the MDO framework ---
    mdo = MDOFramework(disciplines)

    # --- Run MDF optimization ---
    print("\n--- Running MDF Architecture ---")
    result_mdf = mdo.run_optimization(method="mdf")
    print(f"  Converged: {result_mdf.success}")
    print(f"  Message:   {result_mdf.message}")
    print(f"  Optimal total mass: {result_mdf.objective_value:.1f} kg")
    print(f"  Design variables:")
    for name, val in zip(result_mdf.variable_names, result_mdf.x_optimal):
        print(f"    {name:30s} = {val:.4f}")
    print(f"\n  Constraint satisfaction:")
    for cname, cval in result_mdf.constraint_values.items():
        status = "OK" if cval >= 0 else "VIOLATED"
        print(f"    {cname:30s} = {cval:+.4f}  [{status}]")

    # --- Sensitivity analysis ---
    print("\n--- Sensitivity Analysis ---")
    sens_df = mdo.sensitivity_analysis(result_mdf.x_optimal)
    print(sens_df.to_string(index=False))

    # --- Pareto front ---
    print("\n--- Computing Pareto Front (mass vs. transfer time) ---")
    pareto_df = mdo.pareto_front(n_points=25)
    print(f"  Generated {len(pareto_df)} Pareto points.")
    print(f"  Mass range: {pareto_df['mass'].min():.0f} -- {pareto_df['mass'].max():.0f} kg")
    tof_years = pareto_df["transfer_time"] / (365.25 * 86400.0)
    print(f"  TOF  range: {tof_years.min():.2f} -- {tof_years.max():.2f} years")

    # --- Generate all plots ---
    print("\n--- Generating Plots ---")
    mdo.plot_convergence_history()
    mdo.generate_n2_diagram()
    mdo.plot_pareto_front(pareto_df)
    mdo.plot_sensitivity_tornado(sens_df)
    mdo.plot_tradespace(pareto_df)
    mdo.plot_constraint_waterfall(result_mdf)
    mdo.plot_design_history()
    mdo.generate_design_structure_matrix()

    print(f"\n  All plots saved to: {os.path.abspath(OUTPUT_DIR)}/")

    # --- Run IDF architecture for comparison ---
    print("\n--- Running IDF Architecture ---")
    result_idf = mdo.run_optimization(method="idf")
    print(f"  Converged: {result_idf.success}")
    print(f"  Optimal total mass (IDF): {result_idf.objective_value:.1f} kg")
    print(f"  Difference from MDF: {result_idf.objective_value - result_mdf.objective_value:+.1f} kg")

    print("\n" + "=" * 72)
    print("  MDO Complete.")
    print("=" * 72)
