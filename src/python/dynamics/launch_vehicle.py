"""
===============================================================================
Launch Vehicle Model
===============================================================================
Two-stage rocket with fairing for launch from Miami, FL to LEO parking orbit.
Models: thrust variation with altitude (Isp interpolation), Mach-dependent
drag (transonic drag rise), aerodynamic forces, gravity turn guidance,
staging events, heating rate estimation.

Physics modeled:
    - Thrust: Varies with altitude as Isp transitions from sea-level to vacuum
    - Drag: Cd(Mach) with transonic drag rise at Mach 0.8-1.2
    - Heating: Sutton-Graves stagnation point heating: q = k * sqrt(rho/r_n) * v^3
    - Staging: Instantaneous mass drop, configuration change
    - Gravity turn: Pitch program following gravity vector after initial vertical
===============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

from core.constants import (
    EARTH_MU, EARTH_RADIUS, EARTH_SEA_LEVEL_DENSITY,
    EARTH_SCALE_HEIGHT, EARTH_ROTATION_RATE, DEG2RAD, RAD2DEG
)

# Standard gravity for Isp calculations
G0 = 9.80665  # m/s^2


@dataclass
class RocketStage:
    """
    Single rocket stage with propulsion and structural parameters.

    Attributes:
        name: Stage identifier
        dry_mass: Structure mass without propellant (kg)
        propellant_mass: Initial propellant mass (kg)
        thrust: Vacuum thrust (N)
        isp_sl: Sea-level specific impulse (s)
        isp_vac: Vacuum specific impulse (s)
        burn_time: Maximum burn time at full thrust (s)
        diameter: Stage diameter for drag reference area (m)
        cd_0: Zero angle-of-attack drag coefficient (subsonic)
        structural_factor: Structural mass fraction = dry/(dry+prop)
    """
    name: str = "Stage"
    dry_mass: float = 5000.0
    propellant_mass: float = 50000.0
    thrust: float = 1000000.0
    isp_sl: float = 280.0
    isp_vac: float = 320.0
    burn_time: float = 200.0
    diameter: float = 5.0
    cd_0: float = 0.3
    structural_factor: float = 0.06

    @property
    def reference_area(self) -> float:
        """Aerodynamic reference area (circular cross-section) in m^2."""
        return np.pi * (self.diameter / 2.0) ** 2

    @property
    def mass_flow_rate(self) -> float:
        """Propellant mass flow rate at full vacuum thrust (kg/s)."""
        return self.thrust / (self.isp_vac * G0)

    @property
    def total_mass(self) -> float:
        """Total stage mass = dry + propellant (kg)."""
        return self.dry_mass + self.propellant_mass


class LaunchVehicle:
    """
    Two-stage launch vehicle model for ascent from Miami to LEO.

    Simulates the full ascent trajectory including:
    - Two-stage propulsion with altitude-dependent Isp
    - Mach-dependent aerodynamic drag with transonic drag rise
    - Fairing jettison above the sensible atmosphere
    - Gravity turn steering law
    - Dynamic pressure and heating rate monitoring

    Args:
        config: Dictionary from mission_config.yaml 'launch_vehicle' section
    """

    def __init__(self, config: dict):
        # Build stages from config
        stages_cfg = config.get('stages', [])
        self.stages = []
        for s_cfg in stages_cfg:
            stage = RocketStage(
                name=s_cfg.get('name', 'Stage'),
                dry_mass=s_cfg.get('dry_mass_kg', 5000.0),
                propellant_mass=s_cfg.get('propellant_mass_kg', 50000.0),
                thrust=s_cfg.get('thrust_N', 1000000.0),
                isp_sl=s_cfg.get('isp_sl_s', 280.0),
                isp_vac=s_cfg.get('isp_vac_s', 320.0),
                burn_time=s_cfg.get('burn_time_s', 200.0),
                diameter=s_cfg.get('diameter_m', 5.0),
                cd_0=s_cfg.get('cd_0', 0.3),
                structural_factor=s_cfg.get('structural_factor', 0.06)
            )
            self.stages.append(stage)

        # Fairing
        self.fairing_mass = config.get('fairing', {}).get('mass_kg', 2000.0)
        self.fairing_jettison_alt = config.get('fairing', {}).get(
            'jettison_altitude_m', 110000.0)
        self.fairing_jettisoned = False

        # State tracking
        self.current_stage_index = 0
        self.propellant_consumed = [0.0] * len(self.stages)
        self.stages_separated = [False] * len(self.stages)
        self.total_burn_time = [0.0] * len(self.stages)

    def get_current_stage(self) -> Optional[RocketStage]:
        """Return the currently active stage, or None if all spent."""
        if self.current_stage_index < len(self.stages):
            return self.stages[self.current_stage_index]
        return None

    def get_isp(self, altitude: float) -> float:
        """
        Compute effective specific impulse at given altitude.

        Isp varies between sea-level and vacuum values based on
        atmospheric pressure ratio p/p0 = exp(-h/H).

        Args:
            altitude: Height above sea level (m)

        Returns:
            Effective Isp in seconds
        """
        stage = self.get_current_stage()
        if stage is None:
            return 0.0

        # Atmospheric pressure ratio (exponential model)
        pressure_ratio = np.exp(-altitude / EARTH_SCALE_HEIGHT)

        # Linear interpolation between sea-level and vacuum Isp
        # At sea level (p/p0=1): Isp = Isp_sl
        # In vacuum (p/p0=0): Isp = Isp_vac
        isp = stage.isp_vac - (stage.isp_vac - stage.isp_sl) * pressure_ratio
        return isp

    def get_thrust(self, altitude: float) -> float:
        """
        Compute thrust at given altitude.

        Thrust increases slightly with altitude due to nozzle expansion
        in lower ambient pressure. Approximation: T = T_vac - (T_vac - T_sl) * (p/p0)
        where T_sl = T_vac * Isp_sl / Isp_vac.

        Args:
            altitude: Height above sea level (m)

        Returns:
            Thrust in Newtons
        """
        stage = self.get_current_stage()
        if stage is None:
            return 0.0

        # Check if stage has fuel remaining
        remaining = stage.propellant_mass - self.propellant_consumed[self.current_stage_index]
        if remaining <= 0.0:
            return 0.0

        pressure_ratio = np.exp(-altitude / EARTH_SCALE_HEIGHT)
        thrust_sl = stage.thrust * (stage.isp_sl / stage.isp_vac)
        thrust = stage.thrust - (stage.thrust - thrust_sl) * pressure_ratio
        return thrust

    def get_mass(self) -> float:
        """
        Compute current total vehicle mass.

        Total = sum of remaining stages (dry + remaining prop) + fairing + payload.

        Returns:
            Current mass in kg
        """
        total = 0.0

        for i in range(self.current_stage_index, len(self.stages)):
            if not self.stages_separated[i]:
                stage = self.stages[i]
                remaining_prop = max(0.0, stage.propellant_mass -
                                     self.propellant_consumed[i])
                total += stage.dry_mass + remaining_prop

        if not self.fairing_jettisoned:
            total += self.fairing_mass

        return total

    def get_drag_coefficient(self, mach: float) -> float:
        """
        Mach-dependent drag coefficient with transonic drag rise.

        The drag coefficient follows a characteristic curve:
        - Subsonic (M < 0.8): Cd = Cd_0 (constant)
        - Transonic (0.8 < M < 1.2): Cd rises sharply (drag divergence)
        - Supersonic (M > 1.2): Cd decreases gradually

        The transonic drag rise is modeled as a Gaussian bump centered at M=1.

        Args:
            mach: Mach number (dimensionless)

        Returns:
            Drag coefficient (dimensionless)
        """
        stage = self.get_current_stage()
        if stage is None:
            return 0.3

        cd_0 = stage.cd_0

        if mach < 0.8:
            # Subsonic: constant drag
            return cd_0
        elif mach < 1.2:
            # Transonic drag rise: Gaussian bump
            # Peak at M=1.0, width ~0.2
            cd_bump = 0.7 * cd_0 * np.exp(-((mach - 1.0) / 0.15) ** 2)
            return cd_0 + cd_bump
        else:
            # Supersonic: Prandtl-Glauert decrease
            # Cd ~ Cd_0 * (1 + 0.3/sqrt(M^2 - 1))
            cd_super = cd_0 * (1.0 + 0.3 / np.sqrt(mach ** 2 - 1.0 + 0.01))
            return cd_super

    def get_speed_of_sound(self, altitude: float) -> float:
        """
        Approximate speed of sound at altitude.

        Uses simplified US Standard Atmosphere temperature profile:
        - Troposphere (0-11 km): T decreases linearly (lapse rate -6.5 K/km)
        - Stratosphere (11-20 km): T constant at 216.65 K
        - Above 20 km: T gradually increases

        Speed of sound: a = sqrt(gamma * R * T) where gamma=1.4, R=287 J/(kg*K)

        Args:
            altitude: Height above sea level (m)

        Returns:
            Speed of sound in m/s
        """
        gamma = 1.4
        R_air = 287.058  # J/(kg*K)

        if altitude < 11000.0:
            # Troposphere: T = 288.15 - 6.5e-3 * h
            T = 288.15 - 6.5e-3 * altitude
        elif altitude < 20000.0:
            # Tropopause: T = 216.65 K (isothermal)
            T = 216.65
        elif altitude < 47000.0:
            # Stratosphere: slight increase
            T = 216.65 + 1.0e-3 * (altitude - 20000.0)
        else:
            # Upper atmosphere approximation
            T = max(180.0, 270.65 - 2.0e-3 * (altitude - 47000.0))

        T = max(T, 150.0)  # Floor temperature
        return np.sqrt(gamma * R_air * T)

    def get_atmospheric_density(self, altitude: float) -> float:
        """
        Atmospheric density using exponential model.

        rho(h) = rho_0 * exp(-h / H)

        Args:
            altitude: Height above sea level (m)

        Returns:
            Atmospheric density in kg/m^3
        """
        if altitude < 0:
            return EARTH_SEA_LEVEL_DENSITY
        return EARTH_SEA_LEVEL_DENSITY * np.exp(-altitude / EARTH_SCALE_HEIGHT)

    def get_drag(self, velocity_mag: float, altitude: float,
                 angle_of_attack: float = 0.0) -> float:
        """
        Compute aerodynamic drag force magnitude.

        D = 0.5 * rho * V^2 * Cd(M) * A_ref

        Args:
            velocity_mag: Speed relative to atmosphere (m/s)
            altitude: Height above sea level (m)
            angle_of_attack: Angle of attack (rad), increases Cd

        Returns:
            Drag force in Newtons
        """
        rho = self.get_atmospheric_density(altitude)
        if rho < 1e-15 or velocity_mag < 0.1:
            return 0.0

        a = self.get_speed_of_sound(altitude)
        mach = velocity_mag / max(a, 1.0)

        cd = self.get_drag_coefficient(mach)
        # Angle of attack effect: Cd increases with AoA^2
        cd += 2.0 * np.sin(angle_of_attack) ** 2

        stage = self.get_current_stage()
        area = stage.reference_area if stage else 20.0

        drag = 0.5 * rho * velocity_mag ** 2 * cd * area
        return drag

    def get_aero_forces(self, velocity_vec: np.ndarray, altitude: float,
                        angle_of_attack: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute aerodynamic drag and lift force vectors.

        Drag opposes velocity direction. Lift is perpendicular to velocity
        in the pitch plane, proportional to AoA.

        Args:
            velocity_vec: Velocity vector (3D, m/s)
            altitude: Height above sea level (m)
            angle_of_attack: AoA in radians

        Returns:
            Tuple of (drag_vector, lift_vector) in Newtons
        """
        v_mag = np.linalg.norm(velocity_vec)
        if v_mag < 0.1:
            return np.zeros(3), np.zeros(3)

        v_hat = velocity_vec / v_mag
        drag_mag = self.get_drag(v_mag, altitude, angle_of_attack)
        drag_vec = -drag_mag * v_hat

        # Simplified lift (perpendicular to velocity in vertical plane)
        # L = 0.5 * rho * v^2 * Cl * A, where Cl ~ 2*pi*alpha (thin airfoil)
        rho = self.get_atmospheric_density(altitude)
        stage = self.get_current_stage()
        area = stage.reference_area if stage else 20.0
        cl = 2.0 * np.pi * angle_of_attack  # Thin airfoil approximation
        lift_mag = 0.5 * rho * v_mag ** 2 * cl * area

        # Lift direction: perpendicular to velocity, in vertical plane
        up = np.array([0.0, 0.0, 1.0])
        lift_dir = np.cross(v_hat, np.cross(up, v_hat))
        lift_norm = np.linalg.norm(lift_dir)
        if lift_norm > 1e-10:
            lift_dir /= lift_norm
        lift_vec = lift_mag * lift_dir

        return drag_vec, lift_vec

    def get_dynamic_pressure(self, velocity_mag: float, altitude: float) -> float:
        """
        Dynamic pressure q = 0.5 * rho * V^2.

        Used to monitor structural loading (max-Q).

        Args:
            velocity_mag: Speed in m/s
            altitude: Height in m

        Returns:
            Dynamic pressure in Pascals (N/m^2)
        """
        rho = self.get_atmospheric_density(altitude)
        return 0.5 * rho * velocity_mag ** 2

    def get_heating_rate(self, velocity_mag: float, altitude: float) -> float:
        """
        Stagnation point convective heating rate (Sutton-Graves model).

        q_dot = k * sqrt(rho / r_nose) * V^3

        where k ~ 1.7415e-4 (for Earth atmosphere) and r_nose is the
        nose radius. This gives heating rate in W/m^2.

        Args:
            velocity_mag: Speed in m/s
            altitude: Height in m

        Returns:
            Heating rate in W/m^2
        """
        rho = self.get_atmospheric_density(altitude)
        if rho < 1e-15 or velocity_mag < 100.0:
            return 0.0

        k_sg = 1.7415e-4  # Sutton-Graves constant for air
        r_nose = 0.5  # Assumed nose radius (m)

        q_dot = k_sg * np.sqrt(rho / r_nose) * velocity_mag ** 3
        return q_dot

    def consume_propellant(self, dt: float, altitude: float) -> float:
        """
        Consume propellant for given time step.

        Args:
            dt: Time step (s)
            altitude: Current altitude for Isp computation (m)

        Returns:
            Mass consumed (kg)
        """
        stage = self.get_current_stage()
        if stage is None:
            return 0.0

        isp = self.get_isp(altitude)
        thrust = self.get_thrust(altitude)
        if thrust <= 0.0 or isp <= 0.0:
            return 0.0

        mass_flow = thrust / (isp * G0)
        dm = mass_flow * dt

        # Don't consume more than remaining
        remaining = stage.propellant_mass - self.propellant_consumed[self.current_stage_index]
        dm = min(dm, remaining)

        self.propellant_consumed[self.current_stage_index] += dm
        self.total_burn_time[self.current_stage_index] += dt
        return dm

    def is_stage_burnout(self) -> bool:
        """Check if current stage propellant is depleted."""
        stage = self.get_current_stage()
        if stage is None:
            return True
        remaining = stage.propellant_mass - self.propellant_consumed[self.current_stage_index]
        return remaining <= 0.0

    def stage_separation(self) -> float:
        """
        Separate current spent stage.

        Returns:
            Mass of separated stage (kg)
        """
        if self.current_stage_index >= len(self.stages):
            return 0.0

        stage = self.stages[self.current_stage_index]
        separated_mass = stage.dry_mass  # Dry mass of spent stage
        self.stages_separated[self.current_stage_index] = True
        self.current_stage_index += 1
        return separated_mass

    def jettison_fairing(self) -> float:
        """
        Jettison payload fairing.

        Returns:
            Mass of jettisoned fairing (kg)
        """
        if self.fairing_jettisoned:
            return 0.0
        self.fairing_jettisoned = True
        return self.fairing_mass

    def should_jettison_fairing(self, altitude: float) -> bool:
        """Check if altitude is above fairing jettison threshold."""
        return (not self.fairing_jettisoned and
                altitude >= self.fairing_jettison_alt)

    def gravity_turn_guidance(self, altitude: float, velocity_vec: np.ndarray,
                              position_vec: np.ndarray,
                              time_since_launch: float) -> float:
        """
        Gravity turn steering law for ascent trajectory.

        Phase 1 (t < 10s): Vertical ascent (pitch = 90 deg from horizontal)
        Phase 2 (10s < t < 30s): Pitch kickover initiation (small tilt)
        Phase 3 (t > 30s): Follow velocity vector (gravity turn)

        Args:
            altitude: Current altitude (m)
            velocity_vec: Inertial velocity vector (3D, m/s)
            position_vec: Position vector (3D, m)
            time_since_launch: Time since liftoff (s)

        Returns:
            Commanded pitch angle in radians (from local horizontal)
        """
        if time_since_launch < 10.0:
            # Vertical ascent
            return np.pi / 2.0

        elif time_since_launch < 30.0:
            # Pitch kickover: linearly decrease from 90 to ~85 degrees
            t_frac = (time_since_launch - 10.0) / 20.0
            return np.pi / 2.0 - t_frac * (5.0 * DEG2RAD)

        else:
            # Gravity turn: follow velocity vector
            v_mag = np.linalg.norm(velocity_vec)
            if v_mag < 1.0:
                return np.pi / 2.0

            # Pitch = angle of velocity from local horizontal
            r_hat = position_vec / np.linalg.norm(position_vec)
            v_radial = np.dot(velocity_vec, r_hat)
            v_tangential = np.sqrt(max(0, v_mag ** 2 - v_radial ** 2))

            pitch = np.arctan2(v_radial, v_tangential)
            return pitch

    def get_thrust_direction(self, pitch: float, position_vec: np.ndarray,
                             velocity_vec: np.ndarray) -> np.ndarray:
        """
        Compute unit thrust direction vector from pitch angle.

        The thrust direction is in the plane defined by the position
        (radial) and velocity (along-track) vectors.

        Args:
            pitch: Pitch angle from local horizontal (rad)
            position_vec: Position vector (m)
            velocity_vec: Velocity vector (m/s)

        Returns:
            Unit thrust direction vector (3D)
        """
        r_hat = position_vec / np.linalg.norm(position_vec)

        # Along-track direction (velocity component perpendicular to radial)
        v_proj = velocity_vec - np.dot(velocity_vec, r_hat) * r_hat
        v_proj_mag = np.linalg.norm(v_proj)

        if v_proj_mag > 0.1:
            t_hat = v_proj / v_proj_mag
        else:
            # Arbitrary tangential direction at launch
            t_hat = np.array([0.0, 1.0, 0.0])
            t_hat = t_hat - np.dot(t_hat, r_hat) * r_hat
            t_hat /= np.linalg.norm(t_hat)

        # Thrust direction in radial-tangential plane
        thrust_dir = np.sin(pitch) * r_hat + np.cos(pitch) * t_hat
        return thrust_dir / np.linalg.norm(thrust_dir)
