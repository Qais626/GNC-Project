"""
===============================================================================
GNC PROJECT - Spacecraft Model
===============================================================================
Object-oriented spacecraft model that tracks every dynamically relevant
property of the vehicle over the course of a mission:

    - Dry mass, propellant mass, total mass
    - 3x3 inertia tensor (updated as propellant is consumed)
    - Centre-of-gravity (CG) offset from the geometric centre
    - Structural flex modes (frequency, damping, amplitude)
    - Solar-panel and electronics radiation degradation
    - YAML-based configuration for easy scenario changes

Conventions
-----------
    - Body frame:  +X = forward (along thrust axis)
                   +Y = starboard
                   +Z = nadir (for Earth-pointing mode)
    - SI units throughout (m, s, kg, rad).

Usage
-----
    config = yaml.safe_load(open("spacecraft.yaml"))
    sc = Spacecraft(config)
    sc.consume_propellant(50.0)
    I = sc.get_inertia()
===============================================================================
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Any, Dict, List


# ============================================================================
#  FLEX MODE DATACLASS
# ============================================================================

@dataclass
class FlexMode:
    """
    A single structural flex mode of the spacecraft.

    Parameters
    ----------
    frequency : float
        Natural frequency of the mode (Hz).
    damping : float
        Damping ratio (dimensionless, typically 0.001 -- 0.05).
    amplitude : float
        Peak amplitude of the mode shape at the sensor location (m).
    phase : float
        Initial phase offset (rad).
    """
    frequency: float        # Hz
    damping: float          # dimensionless zeta
    amplitude: float        # m (peak displacement)
    phase: float = 0.0      # rad


# ============================================================================
#  SPACECRAFT CLASS
# ============================================================================

class Spacecraft:
    """
    Full state model of the spacecraft.

    The constructor accepts a *config* dictionary (typically loaded from a
    YAML file) with the following expected keys::

        spacecraft:
            dry_mass: 2500.0          # kg
            propellant_mass: 1500.0   # kg
            dimensions: [3.0, 2.0, 2.0]  # length, width, height (m)
            solar_panel_area: 40.0    # m^2
            reflectivity: 0.3         # bulk reflectivity [0,1]
            inertia:                   # 3x3 tensor (kg*m^2) -- upper-triangular
                Ixx: 3500.0
                Iyy: 4200.0
                Izz: 4800.0
                Ixy: -50.0
                Ixz:  20.0
                Iyz: -10.0
            cg_offset: [0.0, 0.0, 0.0]  # nominal CG offset from geometric centre (m)
            flex_modes:
                - frequency: 0.5
                  damping: 0.01
                  amplitude: 0.002
                  phase: 0.0
                - frequency: 1.2
                  damping: 0.02
                  amplitude: 0.001
                  phase: 0.0
            mass_uncertainty: 0.02     # fractional 1-sigma (e.g. 2 %)

    Missing keys fall back to sensible defaults.

    Parameters
    ----------
    config : dict
        Configuration dictionary (see format above).
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def __init__(self, config: Dict[str, Any]) -> None:
        sc = config.get("spacecraft", config)  # allow top-level or nested

        # --- Mass ---
        self.dry_mass: float = float(sc.get("dry_mass", 2500.0))
        self.propellant_mass: float = float(sc.get("propellant_mass", 1500.0))

        # --- Dimensions ---
        dims = sc.get("dimensions", [3.0, 2.0, 2.0])
        self.length: float = float(dims[0])
        self.width: float = float(dims[1])
        self.height: float = float(dims[2])

        # --- Solar panel area and reflectivity ---
        self.solar_panel_area: float = float(sc.get("solar_panel_area", 40.0))
        self.reflectivity: float = float(sc.get("reflectivity", 0.3))

        # --- Inertia tensor ---
        inertia_cfg = sc.get("inertia", {})
        Ixx = float(inertia_cfg.get("Ixx", 3500.0))
        Iyy = float(inertia_cfg.get("Iyy", 4200.0))
        Izz = float(inertia_cfg.get("Izz", 4800.0))
        Ixy = float(inertia_cfg.get("Ixy", 0.0))
        Ixz = float(inertia_cfg.get("Ixz", 0.0))
        Iyz = float(inertia_cfg.get("Iyz", 0.0))

        # Store the dry inertia (propellant contribution is added dynamically)
        self._dry_inertia = np.array([
            [Ixx, Ixy, Ixz],
            [Ixy, Iyy, Iyz],
            [Ixz, Iyz, Izz],
        ], dtype=np.float64)

        # --- Centre-of-gravity offset ---
        cg = sc.get("cg_offset", [0.0, 0.0, 0.0])
        self._nominal_cg_offset = np.array(cg, dtype=np.float64)

        # --- Structural flex modes ---
        self.flex_modes: List[FlexMode] = []
        for fm_cfg in sc.get("flex_modes", []):
            self.flex_modes.append(FlexMode(
                frequency=float(fm_cfg.get("frequency", 1.0)),
                damping=float(fm_cfg.get("damping", 0.01)),
                amplitude=float(fm_cfg.get("amplitude", 0.001)),
                phase=float(fm_cfg.get("phase", 0.0)),
            ))

        # --- Mass uncertainty ---
        self.mass_uncertainty: float = float(sc.get("mass_uncertainty", 0.02))

        # --- Radiation damage state ---
        self.cumulative_dose: float = 0.0          # rad
        self.panel_degradation_factor: float = 1.0
        self.electronics_degradation_factor: float = 1.0

        # Dose-response constants (same model as RadiationEnvironment)
        self._panel_dose_half: float = 50_000.0       # 50 krad
        self._electronics_dose_half: float = 150_000.0  # 150 krad

        # --- Propellant tank geometry (simplified: spherical tank on +X axis) ---
        # Used to compute inertia contribution of propellant.
        self._tank_offset = np.array([0.5, 0.0, 0.0])  # metres from CG
        self._tank_radius: float = 0.8                  # metres (nominal)

    # ================================================================== #
    #  Mass properties
    # ================================================================== #

    def get_mass(self) -> float:
        """
        Total spacecraft mass (dry + remaining propellant).

        Returns
        -------
        float
            Mass in kg.
        """
        return self.dry_mass + self.propellant_mass

    # ------------------------------------------------------------------ #
    def consume_propellant(self, dm: float) -> float:
        """
        Remove *dm* kg of propellant.  Clamps to zero -- never goes negative.

        Parameters
        ----------
        dm : float
            Mass of propellant to consume (kg).  Must be >= 0.

        Returns
        -------
        float
            Actual mass consumed (may be less than *dm* if the tank runs dry).

        Raises
        ------
        ValueError
            If dm < 0.
        """
        if dm < 0.0:
            raise ValueError("dm must be non-negative.")

        actual = min(dm, self.propellant_mass)
        self.propellant_mass -= actual
        return actual

    # ================================================================== #
    #  Inertia tensor
    # ================================================================== #

    def get_inertia(self) -> NDArray:
        """
        Current 3x3 inertia tensor about the body-frame origin, accounting
        for propellant mass using the parallel-axis theorem.

        The propellant is modelled as a uniform-density sphere located at
        ``_tank_offset`` from the geometric centre.  As propellant is consumed
        the sphere shrinks in radius while keeping its centre fixed.

        Returns
        -------
        ndarray, shape (3, 3)
            Inertia tensor in kg*m^2.
        """
        I_total = self._dry_inertia.copy()

        if self.propellant_mass > 0.0:
            m_p = self.propellant_mass

            # Approximate propellant as a solid sphere whose mass shrinks
            # with consumption.  Radius scales as (m_p / m_p_initial)^(1/3)
            # but we just use a fixed "effective" radius for the moment of
            # inertia since the tank geometry is simplified.
            r_eff = self._tank_radius  # constant for simplicity

            # Moment of inertia of a solid sphere about its own centre:
            #   I_sphere = (2/5) * m * r^2
            I_sphere = 0.4 * m_p * r_eff ** 2

            # Parallel-axis theorem to shift to body origin:
            #   I_total += I_sphere_cm * E + m_p * (d^T d * E - d d^T)
            d = self._tank_offset
            d_sq = np.dot(d, d)
            I_total += I_sphere * np.eye(3)
            I_total += m_p * (d_sq * np.eye(3) - np.outer(d, d))

        return I_total

    # ================================================================== #
    #  Centre of gravity
    # ================================================================== #

    def get_cg_offset(self) -> NDArray:
        """
        CG offset from the geometric centre of the spacecraft, accounting
        for propellant location.

        As propellant is consumed the CG shifts towards the dry-mass CG.

        Returns
        -------
        ndarray, shape (3,)
            CG offset in metres (body frame).
        """
        m_total = self.get_mass()
        if m_total < 1e-12:
            return np.zeros(3)

        # Weighted average of dry-mass CG (at nominal offset) and
        # propellant CG (at tank centre)
        cg = (
            self.dry_mass * self._nominal_cg_offset
            + self.propellant_mass * self._tank_offset
        ) / m_total

        return cg

    # ================================================================== #
    #  Structural flex
    # ================================================================== #

    def get_flex_displacement(self, time: float) -> NDArray:
        """
        Net structural flex displacement at the current time.

        Each flex mode contributes a sinusoidal displacement along the body
        Z-axis (bending mode approximation), damped exponentially:

            x_i(t) = A_i * exp(-zeta_i * omega_i * t)
                        * sin(omega_d_i * t + phi_i)

        where
            omega_i = 2*pi*f_i           (natural frequency)
            omega_d = omega_i * sqrt(1 - zeta^2)  (damped frequency)

        Parameters
        ----------
        time : float
            Mission elapsed time (s).

        Returns
        -------
        ndarray, shape (3,)
            Displacement vector in the body frame (m).
            Flex is modelled as bending about the Z-axis.
        """
        displacement = np.zeros(3)

        for mode in self.flex_modes:
            omega_n = 2.0 * np.pi * mode.frequency
            zeta = mode.damping

            # Damped natural frequency
            omega_d = omega_n * np.sqrt(max(1.0 - zeta ** 2, 0.0))

            # Exponential envelope
            envelope = np.exp(-zeta * omega_n * time)

            # Oscillatory part
            oscillation = np.sin(omega_d * time + mode.phase)

            # Contribution along body Z  (bending mode)
            displacement[2] += mode.amplitude * envelope * oscillation

        return displacement

    # ================================================================== #
    #  Radiation degradation
    # ================================================================== #

    def degrade_from_radiation(self, dose: float) -> None:
        """
        Apply a radiation dose to the spacecraft and update degradation
        factors for solar panels and electronics.

        The model uses exponential decay:

            factor = exp(-cumulative_dose / dose_half)

        Parameters
        ----------
        dose : float
            Additional absorbed dose (rad).
        """
        if dose < 0.0:
            raise ValueError("Radiation dose must be non-negative.")

        self.cumulative_dose += dose

        self.panel_degradation_factor = float(
            np.exp(-self.cumulative_dose / self._panel_dose_half)
        )
        self.electronics_degradation_factor = float(
            np.exp(-self.cumulative_dose / self._electronics_dose_half)
        )

    # ================================================================== #
    #  Effective solar-panel output
    # ================================================================== #

    @property
    def effective_panel_area(self) -> float:
        """
        Solar panel area derated by radiation degradation.

        Returns
        -------
        float
            Effective area in m^2.
        """
        return self.solar_panel_area * self.panel_degradation_factor

    # ================================================================== #
    #  Representation
    # ================================================================== #

    def __repr__(self) -> str:
        return (
            f"Spacecraft(mass={self.get_mass():.1f} kg, "
            f"propellant={self.propellant_mass:.1f} kg, "
            f"panel_deg={self.panel_degradation_factor:.4f}, "
            f"elec_deg={self.electronics_degradation_factor:.4f})"
        )
