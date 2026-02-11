"""
===============================================================================
GNC PROJECT - Simulation Engine
===============================================================================
Central orchestrator for the GNC mission simulation.  Ties together dynamics,
guidance, navigation, and control in a time-stepped loop.  Logs all telemetry
to a pandas DataFrame for post-run analysis.

The core simulation loop follows the classical GNC pipeline executed at every
time step:

    1. GUIDANCE   -- Determine the current mission phase and desired targets.
    2. NAVIGATION -- Read sensors, fuse measurements through the EKF.
    3. CONTROL    -- Compute actuator commands from navigation estimates.
    4. DYNAMICS   -- Propagate the physical state (position, velocity,
                     attitude, angular velocity, mass).
    5. LOGGING    -- Record telemetry for post-analysis.

An adaptive time-step scheduler selects fine dt during powered burns and
critical phases (reentry, landing) and coarse dt during unpowered coast arcs.

References
----------
    [1] Wertz, "Space Mission Engineering: The New SMAD", Ch. 18-19.
    [2] Markley & Crassidis, "Fundamentals of Spacecraft Attitude
        Determination and Control", Springer, 2014.
===============================================================================
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Any, Dict, List, Optional

from core.constants import (
    EARTH_RADIUS,
    EARTH_ROTATION_RATE,
    EARTH_MU,
    KSC_LATITUDE,
    KSC_LONGITUDE,
    RAD2DEG,
)
from core.quaternion import Quaternion

logger = logging.getLogger(__name__)


class SimulationEngine:
    """
    Central orchestrator for the GNC mission simulation.

    Manages the time-stepped integration loop, subsystem dispatch, state
    propagation, and telemetry recording.  Requires a configuration dictionary
    and a dictionary of initialised subsystem objects (mission_planner,
    navigation, control, dynamics, etc.).

    Parameters
    ----------
    config : dict
        Simulation configuration containing at minimum:
            - 'dt'           : float -- nominal time step (s)
            - 'dt_fine'      : float -- fine time step for burns (s)
            - 'dt_coarse'    : float -- coarse time step for coast (s)
            - 'max_time'     : float -- absolute time limit (s)
            - 'vehicle_mass' : float -- initial wet mass (kg)
            - 'fuel_mass'    : float -- initial propellant mass (kg)
    subsystems : dict
        Keyed subsystem instances.  Expected keys:
            - 'mission_planner' : MissionPlanner instance
            - 'navigation'      : Navigation filter (optional)
            - 'control'         : Attitude / orbit controller (optional)
            - 'dynamics'        : Dynamics propagator (optional)
            - 'sensors'         : Sensor model suite (optional)

    Attributes
    ----------
    state : dict
        The full spacecraft state vector as a flat dictionary:
            position    : np.ndarray (3,) -- ECI position (m)
            velocity    : np.ndarray (3,) -- ECI velocity (m/s)
            attitude    : Quaternion       -- body-to-ECI rotation
            omega       : np.ndarray (3,) -- body angular velocity (rad/s)
            mass        : float            -- current mass (kg)
            fuel        : float            -- remaining propellant (kg)
    current_time : float
        Simulation elapsed time (s).
    telemetry : list of dict
        Raw telemetry records, converted to DataFrame on request.
    """

    # Phases considered "critical" -- use fine time step.
    _FINE_DT_PHASES = {
        'STAGE1_ASCENT', 'STAGE2_ASCENT', 'STAGE_SEPARATION',
        'TLI_BURN', 'LUNAR_ORBIT_INSERTION', 'LUNAR_INCLINATION_CHANGE',
        'LUNAR_ESCAPE', 'JUPITER_ORBIT_INSERTION', 'JUPITER_ESCAPE',
        'EARTH_REENTRY', 'DESCENT_LANDING',
    }

    def __init__(self, config: Dict[str, Any], subsystems: Dict[str, Any]) -> None:
        self.config = config
        self.subsystems = subsystems

        # Time-stepping parameters
        self.dt: float = config.get('dt', 1.0)
        self._dt_fine: float = config.get('dt_fine', 0.1)
        self._dt_coarse: float = config.get('dt_coarse', 60.0)

        self.current_time: float = 0.0
        self._max_time: float = config.get('max_time', 1e9)

        # Spacecraft state -- populated by initialize()
        self.state: Dict[str, Any] = {}

        # Telemetry accumulator
        self.telemetry: List[Dict[str, Any]] = []

        # Bookkeeping
        self._total_delta_v: float = 0.0
        self._max_pointing_error: float = 0.0
        self._phases_completed: List[str] = []
        self._wall_start: Optional[float] = None

        logger.info("SimulationEngine created.  dt=%.3f s", self.dt)

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def initialize(self) -> None:
        """
        Set the initial spacecraft state at the launch site.

        Position is computed from the Kennedy Space Center launch site
        coordinates in the Earth-Centered Inertial (ECI) frame (assumes epoch
        alignment so that GMST ~ KSC longitude at t = 0 for simplicity).  Velocity includes
        the Earth surface rotation contribution.

        After this call, the state dictionary is fully populated and ready
        for the first simulation step.
        """
        # Launch site in ECI (simplified: GMST offset = 0 at epoch)
        lat = KSC_LATITUDE
        lon = KSC_LONGITUDE
        r = EARTH_RADIUS

        # ECI position from geodetic (spherical approximation)
        pos_x = r * np.cos(lat) * np.cos(lon)
        pos_y = r * np.cos(lat) * np.sin(lon)
        pos_z = r * np.sin(lat)
        position = np.array([pos_x, pos_y, pos_z], dtype=np.float64)

        # Surface velocity due to Earth rotation (v = omega x r)
        omega_earth = np.array([0.0, 0.0, EARTH_ROTATION_RATE], dtype=np.float64)
        velocity = np.cross(omega_earth, position)

        # Identity attitude (body aligned with ECI at launch)
        attitude = Quaternion.identity()

        # Zero angular velocity at launch
        omega = np.zeros(3, dtype=np.float64)

        # Mass
        vehicle_mass = self.config.get('vehicle_mass', 50000.0)
        fuel_mass = self.config.get('fuel_mass', 35000.0)

        self.state = {
            'position': position,
            'velocity': velocity,
            'attitude': attitude,
            'omega': omega,
            'mass': vehicle_mass,
            'fuel': fuel_mass,
            'phase': 'PRE_LAUNCH',
            'altitude_m': 0.0,
            'velocity_m_s': np.linalg.norm(velocity),
            'time_s': 0.0,
            'thrust': np.zeros(3, dtype=np.float64),
            'control_torque': np.zeros(3, dtype=np.float64),
            'pointing_error_deg': 0.0,
            'commanded': False,
            'propellant_depleted': False,
            'body_distance_m': 0.0,
            'eccentricity': 0.0,
            'orbital_energy': 0.0,
            'burn_complete': False,
        }

        self.current_time = 0.0
        self.telemetry.clear()
        self._total_delta_v = 0.0
        self._max_pointing_error = 0.0
        self._phases_completed = ['PRE_LAUNCH']

        logger.info(
            "Simulation initialized.  Position: [%.0f, %.0f, %.0f] m  "
            "Mass: %.0f kg  Fuel: %.0f kg",
            pos_x, pos_y, pos_z, vehicle_mass, fuel_mass,
        )

    # =========================================================================
    # CORE SIMULATION STEP
    # =========================================================================

    def step(self, dt: Optional[float] = None) -> None:
        """
        Execute one simulation time step.

        This method implements the full GNC loop:

            1. GUIDANCE   -- Query mission planner for current phase and
                             check for phase transitions.
            2. NAVIGATION -- Read simulated sensors, run EKF predict/update.
            3. CONTROL    -- Compute attitude torque and thrust commands.
            4. DYNAMICS   -- Propagate state with RK4 integration.
            5. LOGGING    -- Record telemetry snapshot.

        Parameters
        ----------
        dt : float, optional
            Override time step.  If None, uses the adaptive scheduler.
        """
        if dt is None:
            dt = self._get_adaptive_dt()

        # ------------------------------------------------------------------
        # 1. GUIDANCE: Query mission phase and check transitions
        # ------------------------------------------------------------------
        mission_planner = self.subsystems.get('mission_planner')
        if mission_planner is not None:
            phase = mission_planner.get_current_phase()
            self.state['phase'] = phase.name if hasattr(phase, 'name') else str(phase)

            # Supply the current state for transition evaluation
            self.state['time_s'] = self.current_time
            old_phase = self.state['phase']
            mission_planner.check_transition(self.state)
            new_phase = mission_planner.get_current_phase()
            new_phase_name = new_phase.name if hasattr(new_phase, 'name') else str(new_phase)

            if new_phase_name != old_phase:
                logger.info(
                    "Phase transition at t=%.1f s: %s -> %s",
                    self.current_time, old_phase, new_phase_name,
                )
                self._phases_completed.append(new_phase_name)

            self.state['phase'] = new_phase_name

        # ------------------------------------------------------------------
        # 2. NAVIGATION: Sensor fusion and state estimation
        # ------------------------------------------------------------------
        nav = self.subsystems.get('navigation')
        sensors = self.subsystems.get('sensors')

        if sensors is not None and nav is not None:
            # Generate sensor measurements from true state + noise
            measurements = sensors.get_measurements(self.state, self.current_time)
            # Run EKF predict + update cycle
            nav.predict(dt)
            nav.update(measurements)
            nav_estimate = nav.get_state_estimate()
        else:
            # No nav subsystem -- use true state directly
            nav_estimate = self.state.copy()

        # ------------------------------------------------------------------
        # 3. CONTROL: Compute torque and thrust commands
        # ------------------------------------------------------------------
        control = self.subsystems.get('control')
        control_torque = np.zeros(3, dtype=np.float64)
        thrust_vector = np.zeros(3, dtype=np.float64)

        if control is not None:
            # Determine target attitude based on current phase
            target_attitude = self._get_target_attitude()

            # Compute attitude control torque
            control_output = control.compute(
                nav_estimate, target_attitude, dt
            )
            if isinstance(control_output, dict):
                control_torque = control_output.get(
                    'torque', np.zeros(3, dtype=np.float64)
                )
                thrust_vector = control_output.get(
                    'thrust', np.zeros(3, dtype=np.float64)
                )
            else:
                control_torque = np.asarray(control_output, dtype=np.float64)

        self.state['control_torque'] = control_torque
        self.state['thrust'] = thrust_vector

        # ------------------------------------------------------------------
        # 4. DYNAMICS: Propagate state forward by dt
        # ------------------------------------------------------------------
        self._propagate_state(dt, thrust_vector, control_torque)

        # ------------------------------------------------------------------
        # 5. LOGGING: Record telemetry
        # ------------------------------------------------------------------
        self._update_derived_quantities()
        self._log_telemetry()
        self.current_time += dt

    # =========================================================================
    # STATE PROPAGATION
    # =========================================================================

    def _propagate_state(
        self,
        dt: float,
        thrust: np.ndarray,
        torque: np.ndarray,
    ) -> None:
        """
        Propagate the spacecraft state using 4th-order Runge-Kutta.

        Integrates translational dynamics (position, velocity) and rotational
        dynamics (attitude quaternion, angular velocity).  Updates mass based
        on propellant consumption from thrust.

        Parameters
        ----------
        dt : float
            Time step (s).
        thrust : np.ndarray
            Thrust vector in body frame (N).
        torque : np.ndarray
            Control torque vector in body frame (N*m).
        """
        dynamics = self.subsystems.get('dynamics')

        if dynamics is not None:
            # Delegate to the full dynamics propagator
            new_state = dynamics.propagate(self.state, dt, thrust, torque)
            self.state.update(new_state)
            return

        # ---- Fallback: simplified two-body + Euler integration ----

        pos = self.state['position'].copy()
        vel = self.state['velocity'].copy()
        att = self.state['attitude']
        omega = self.state['omega'].copy()
        mass = self.state['mass']
        fuel = self.state['fuel']

        # -- Translational dynamics (RK4) --
        def accel(r: np.ndarray, v: np.ndarray) -> np.ndarray:
            """Compute total acceleration: gravity + thrust."""
            r_mag = np.linalg.norm(r)
            if r_mag < 1.0:
                return np.zeros(3, dtype=np.float64)

            # Two-body gravity
            a_grav = -EARTH_MU / (r_mag ** 3) * r

            # Thrust acceleration (rotate from body to inertial)
            if mass > 0.0 and np.linalg.norm(thrust) > 0.0:
                thrust_inertial = att.rotate_vector(thrust)
                a_thrust = thrust_inertial / mass
            else:
                a_thrust = np.zeros(3, dtype=np.float64)

            return a_grav + a_thrust

        # RK4 stages for translational motion
        k1_v = accel(pos, vel)
        k1_r = vel

        k2_v = accel(pos + 0.5 * dt * k1_r, vel + 0.5 * dt * k1_v)
        k2_r = vel + 0.5 * dt * k1_v

        k3_v = accel(pos + 0.5 * dt * k2_r, vel + 0.5 * dt * k2_v)
        k3_r = vel + 0.5 * dt * k2_v

        k4_v = accel(pos + dt * k3_r, vel + dt * k3_v)
        k4_r = vel + dt * k3_v

        new_pos = pos + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
        new_vel = vel + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        # -- Rotational dynamics (first-order quaternion propagation) --
        new_att = att.propagate(omega, dt)

        # Simple angular velocity update: domega/dt = I^{-1} * (torque - omega x I*omega)
        # Use a diagonal inertia approximation
        I_diag = np.array([500.0, 500.0, 200.0], dtype=np.float64)  # kg*m^2
        omega_dot = (torque - np.cross(omega, I_diag * omega)) / I_diag
        new_omega = omega + omega_dot * dt

        # -- Mass update --
        thrust_mag = np.linalg.norm(thrust)
        if thrust_mag > 0.0 and fuel > 0.0:
            Isp = self.config.get('Isp', 300.0)
            g0 = 9.80665
            mdot = thrust_mag / (Isp * g0)
            dm = mdot * dt
            dm = min(dm, fuel)  # Cannot burn more fuel than available
            new_mass = mass - dm
            new_fuel = fuel - dm

            # Accumulate delta-v
            if mass > 0.0:
                self._total_delta_v += (thrust_mag / mass) * dt
        else:
            new_mass = mass
            new_fuel = fuel

        # Write back
        self.state['position'] = new_pos
        self.state['velocity'] = new_vel
        self.state['attitude'] = new_att
        self.state['omega'] = new_omega
        self.state['mass'] = new_mass
        self.state['fuel'] = new_fuel

    # =========================================================================
    # DERIVED QUANTITIES
    # =========================================================================

    def _update_derived_quantities(self) -> None:
        """
        Recompute derived state quantities from the primary state vector.

        Updates altitude, speed, pointing error, orbital elements, and
        body-relative distances in the state dictionary.
        """
        pos = self.state['position']
        vel = self.state['velocity']
        att = self.state['attitude']

        r_mag = np.linalg.norm(pos)
        v_mag = np.linalg.norm(vel)

        self.state['altitude_m'] = max(r_mag - EARTH_RADIUS, 0.0)
        self.state['velocity_m_s'] = v_mag

        # Pointing error: angle between body +X axis and velocity vector
        if v_mag > 1.0:
            body_x_inertial = att.rotate_vector(np.array([1.0, 0.0, 0.0]))
            v_hat = vel / v_mag
            cos_angle = np.clip(np.dot(body_x_inertial, v_hat), -1.0, 1.0)
            pointing_error_deg = np.degrees(np.arccos(cos_angle))
        else:
            pointing_error_deg = 0.0

        self.state['pointing_error_deg'] = pointing_error_deg
        self._max_pointing_error = max(self._max_pointing_error, pointing_error_deg)

        # Orbital energy (specific)
        if r_mag > 0.0:
            self.state['orbital_energy'] = 0.5 * v_mag**2 - EARTH_MU / r_mag
        else:
            self.state['orbital_energy'] = 0.0

        # Body distance (default to Earth center distance)
        self.state['body_distance_m'] = r_mag

        # Eccentricity from vis-viva (approximation for the current orbit)
        if r_mag > 0.0:
            h_vec = np.cross(pos, vel)
            h_mag = np.linalg.norm(h_vec)
            e_vec = np.cross(vel, h_vec) / EARTH_MU - pos / r_mag
            self.state['eccentricity'] = np.linalg.norm(e_vec)
        else:
            self.state['eccentricity'] = 0.0

    # =========================================================================
    # TARGET ATTITUDE
    # =========================================================================

    def _get_target_attitude(self) -> Quaternion:
        """
        Determine the desired attitude quaternion for the current phase.

        Maps the phase-specific target_attitude string from the mission
        planner configuration to an actual quaternion.

        Returns
        -------
        Quaternion
            Target attitude quaternion for the current phase.
        """
        phase_name = self.state.get('phase', 'PRE_LAUNCH')

        # Simple attitude targets based on phase
        if phase_name in ('STAGE1_ASCENT', 'STAGE2_ASCENT', 'LUNAR_ESCAPE',
                          'JUPITER_ESCAPE'):
            # Point along velocity vector (prograde)
            vel = self.state['velocity']
            v_mag = np.linalg.norm(vel)
            if v_mag > 1.0:
                v_hat = vel / v_mag
                # Construct quaternion that rotates +X to v_hat
                return self._quaternion_from_direction(v_hat)

        if phase_name in ('LUNAR_ORBIT_INSERTION', 'JUPITER_ORBIT_INSERTION'):
            # Retrograde: point along negative velocity
            vel = self.state['velocity']
            v_mag = np.linalg.norm(vel)
            if v_mag > 1.0:
                v_hat = -vel / v_mag
                return self._quaternion_from_direction(v_hat)

        # Default: identity (inertial hold)
        return Quaternion.identity()

    @staticmethod
    def _quaternion_from_direction(direction: np.ndarray) -> Quaternion:
        """
        Compute the quaternion that rotates the body +X axis to align with
        the given direction vector.

        Parameters
        ----------
        direction : np.ndarray
            Target direction unit vector in the inertial frame.

        Returns
        -------
        Quaternion
            Rotation quaternion from body to inertial that aligns +X with
            direction.
        """
        direction = direction / np.linalg.norm(direction)
        x_body = np.array([1.0, 0.0, 0.0])

        # Rotation axis = cross product, angle = arccos(dot product)
        cross = np.cross(x_body, direction)
        cross_mag = np.linalg.norm(cross)
        dot = np.dot(x_body, direction)

        if cross_mag < 1e-12:
            if dot > 0:
                return Quaternion.identity()
            else:
                # 180-degree rotation about Z
                return Quaternion.from_axis_angle(
                    np.array([0.0, 0.0, 1.0]), np.pi
                )

        axis = cross / cross_mag
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        return Quaternion.from_axis_angle(axis, angle)

    # =========================================================================
    # ADAPTIVE TIME STEP
    # =========================================================================

    def _get_adaptive_dt(self) -> float:
        """
        Select the simulation time step based on the current mission phase.

        Critical phases (powered burns, reentry, landing) use a fine dt for
        accuracy; coast phases use a coarse dt for efficiency.

        Returns
        -------
        float
            Time step in seconds.
        """
        phase = self.state.get('phase', 'PRE_LAUNCH')
        if phase in self._FINE_DT_PHASES:
            return self._dt_fine
        return self._dt_coarse

    # =========================================================================
    # MISSION COMPLETION CHECK
    # =========================================================================

    def _is_mission_complete(self) -> bool:
        """
        Determine whether the simulation should terminate.

        The simulation ends when:
            - The DESCENT_LANDING phase is complete (altitude ~ 0, velocity ~ 0)
            - The maximum simulation time is exceeded
            - Fuel is exhausted during a critical burn (abort condition)

        Returns
        -------
        bool
            True if the simulation should stop.
        """
        # Time limit
        if self.current_time >= self._max_time:
            logger.warning("Simulation time limit reached: %.1f s", self._max_time)
            return True

        # Successful landing
        phase = self.state.get('phase', '')
        if phase == 'DESCENT_LANDING':
            alt = self.state.get('altitude_m', float('inf'))
            vel = self.state.get('velocity_m_s', float('inf'))
            if alt <= 1.0 and vel <= 5.0:
                logger.info("Touchdown confirmed.  Mission complete.")
                return True

        return False

    # =========================================================================
    # FULL SIMULATION RUN
    # =========================================================================

    def run(self, max_time: Optional[float] = None) -> pd.DataFrame:
        """
        Run the complete simulation through all mission phases or until
        max_time is reached.

        Parameters
        ----------
        max_time : float, optional
            Override the maximum simulation time from config.

        Returns
        -------
        pd.DataFrame
            Complete telemetry record for the run.
        """
        if max_time is not None:
            self._max_time = max_time

        self._wall_start = time.time()
        step_count = 0

        logger.info("Simulation run started.  Max time: %.1f s", self._max_time)

        while not self._is_mission_complete():
            dt = self._get_adaptive_dt()
            self.step(dt)
            step_count += 1

            # Progress reporting every 10000 steps
            if step_count % 10000 == 0:
                wall_elapsed = time.time() - self._wall_start
                logger.info(
                    "Step %d  t=%.1f s  phase=%s  wall=%.1f s",
                    step_count,
                    self.current_time,
                    self.state.get('phase', '?'),
                    wall_elapsed,
                )

        wall_total = time.time() - self._wall_start
        logger.info(
            "Simulation complete.  %d steps in %.2f s wall time.  "
            "Sim time: %.1f s",
            step_count, wall_total, self.current_time,
        )

        return self.get_telemetry()

    # =========================================================================
    # TELEMETRY
    # =========================================================================

    def _log_telemetry(self) -> None:
        """
        Append the current state as a telemetry record.

        Extracts scalar quantities from the state dictionary into a flat
        dictionary suitable for DataFrame construction.
        """
        pos = self.state['position']
        vel = self.state['velocity']
        att = self.state['attitude']
        omega = self.state['omega']
        torque = self.state.get('control_torque', np.zeros(3))
        thrust = self.state.get('thrust', np.zeros(3))

        record = {
            'time': self.current_time,
            'phase': self.state.get('phase', ''),
            'pos_x': pos[0],
            'pos_y': pos[1],
            'pos_z': pos[2],
            'vel_x': vel[0],
            'vel_y': vel[1],
            'vel_z': vel[2],
            'quat_w': att.w,
            'quat_x': att.x,
            'quat_y': att.y,
            'quat_z': att.z,
            'omega_x': omega[0],
            'omega_y': omega[1],
            'omega_z': omega[2],
            'mass': self.state['mass'],
            'fuel': self.state['fuel'],
            'pointing_error_deg': self.state.get('pointing_error_deg', 0.0),
            'control_torque_x': torque[0],
            'control_torque_y': torque[1],
            'control_torque_z': torque[2],
            'thrust_mag': np.linalg.norm(thrust),
            'altitude_m': self.state.get('altitude_m', 0.0),
            'velocity_m_s': self.state.get('velocity_m_s', 0.0),
        }
        self.telemetry.append(record)

    def get_telemetry(self) -> pd.DataFrame:
        """
        Convert the telemetry record list to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: time, phase, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
            quat_w, quat_x, quat_y, quat_z, omega_x, omega_y, omega_z,
            mass, fuel, pointing_error_deg, control_torque_x/y/z,
            thrust_mag, altitude_m, velocity_m_s.
        """
        if not self.telemetry:
            logger.warning("No telemetry recorded.")
            return pd.DataFrame()

        df = pd.DataFrame(self.telemetry)
        df.set_index('time', inplace=True)
        return df

    def save_telemetry(self, filepath: str) -> None:
        """
        Save the telemetry DataFrame to a CSV file.

        Parameters
        ----------
        filepath : str
            Output file path (e.g., 'output/telemetry.csv').
        """
        df = self.get_telemetry()
        df.to_csv(filepath)
        logger.info("Telemetry saved to %s  (%d records)", filepath, len(df))

    # =========================================================================
    # MISSION SUMMARY
    # =========================================================================

    def get_mission_summary(self) -> Dict[str, Any]:
        """
        Compile a summary of the completed (or current) mission.

        Returns
        -------
        dict
            total_delta_v      : float -- Total delta-V expended (m/s)
            fuel_consumed      : float -- Propellant consumed (kg)
            max_pointing_error : float -- Peak pointing error (deg)
            phases_completed   : list  -- List of phase name strings
            total_time         : float -- Total simulation time (s)
            final_mass         : float -- Final spacecraft mass (kg)
            final_altitude     : float -- Final altitude (m)
            final_velocity     : float -- Final inertial speed (m/s)
        """
        initial_fuel = self.config.get('fuel_mass', 35000.0)
        fuel_consumed = initial_fuel - self.state.get('fuel', 0.0)

        summary = {
            'total_delta_v': self._total_delta_v,
            'fuel_consumed': fuel_consumed,
            'max_pointing_error': self._max_pointing_error,
            'phases_completed': list(self._phases_completed),
            'total_time': self.current_time,
            'final_mass': self.state.get('mass', 0.0),
            'final_altitude': self.state.get('altitude_m', 0.0),
            'final_velocity': self.state.get('velocity_m_s', 0.0),
        }

        logger.info("Mission Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                logger.info("  %-25s: %.4f", key, value)
            else:
                logger.info("  %-25s: %s", key, value)

        return summary

    # =========================================================================
    # REPRESENTATION
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"SimulationEngine(t={self.current_time:.1f}s, "
            f"phase={self.state.get('phase', 'UNINITIALIZED')}, "
            f"records={len(self.telemetry)})"
        )
