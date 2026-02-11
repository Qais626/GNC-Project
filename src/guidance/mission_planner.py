"""
===============================================================================
GNC PROJECT - Mission Planner (Finite State Machine)
===============================================================================
Manages all mission phases from pre-launch through descent and landing via a
finite state machine. Each phase has associated guidance, navigation, and
control modes, sensor configurations, target attitudes, and transition
conditions.

The mission profile:
    1. Launch from Kennedy Space Center on a two-stage rocket
    2. Parking orbit around Earth
    3. Trans-Lunar Injection (TLI) burn
    4. Lunar operations (orbit, inclination change)
    5. Lunar escape toward Jupiter
    6. Jupiter orbit insertion and operations
    7. Jupiter escape and return to Earth
    8. Earth reentry and landing

The FSM is modeled conceptually as a directed graph where nodes are mission
phases and edges are transition conditions (altitude thresholds, velocity
thresholds, orbit counts, time elapsed).
===============================================================================
"""

import logging
from enum import IntEnum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# MISSION PHASE ENUMERATION
# =============================================================================

class MissionPhase(IntEnum):
    """
    All mission phases in chronological order. The integer values give
    a natural ordering that the FSM advances through sequentially.
    """
    PRE_LAUNCH = 0
    STAGE1_ASCENT = auto()
    STAGE_SEPARATION = auto()
    STAGE2_ASCENT = auto()
    FAIRING_JETTISON = auto()
    PARKING_ORBIT = auto()
    TLI_BURN = auto()
    LUNAR_COAST = auto()
    LUNAR_ORBIT_INSERTION = auto()
    LUNAR_ORBIT_1 = auto()
    LUNAR_INCLINATION_CHANGE = auto()
    LUNAR_ORBIT_2 = auto()
    LUNAR_ESCAPE = auto()
    EARTH_JUPITER_TRANSFER = auto()
    JUPITER_ORBIT_INSERTION = auto()
    JUPITER_ORBITS = auto()
    JUPITER_ESCAPE = auto()
    JUPITER_EARTH_RETURN = auto()
    EARTH_REENTRY = auto()
    DESCENT_LANDING = auto()


# =============================================================================
# PHASE CONFIGURATION DATABASE
# =============================================================================

# Each phase is fully described by a configuration dictionary that specifies
# guidance, navigation, and control modes along with transition conditions.

PHASE_CONFIGS: Dict[MissionPhase, dict] = {

    MissionPhase.PRE_LAUNCH: {
        "description": "Vehicle on pad, pre-launch checks and countdown",
        "guidance_mode": "hold",
        "control_mode": "ground_commanded",
        "navigation_mode": "ground_truth",
        "active_sensors": ["imu", "gps"],
        "target_attitude": "launch_azimuth",
        "transition": {
            "type": "commanded",
            "description": "Launch command issued by mission control",
        },
    },

    MissionPhase.STAGE1_ASCENT: {
        "description": "First stage powered ascent through atmosphere",
        "guidance_mode": "gravity_turn",
        "control_mode": "pitch_program",
        "navigation_mode": "inertial_gps",
        "active_sensors": ["imu", "gps", "altimeter", "accelerometer"],
        "target_attitude": "velocity_vector",
        "transition": {
            "type": "propellant_depleted",
            "altitude_min_m": 50000.0,
            "description": "Stage 1 burnout / propellant depletion",
        },
    },

    MissionPhase.STAGE_SEPARATION: {
        "description": "First stage separation and second stage ignition",
        "guidance_mode": "coast",
        "control_mode": "rcs_attitude_hold",
        "navigation_mode": "inertial",
        "active_sensors": ["imu", "accelerometer"],
        "target_attitude": "previous_hold",
        "transition": {
            "type": "time_elapsed",
            "duration_s": 5.0,
            "description": "Separation confirmed and stage 2 ignition",
        },
    },

    MissionPhase.STAGE2_ASCENT: {
        "description": "Second stage powered ascent to orbit",
        "guidance_mode": "powered_explicit_guidance",
        "control_mode": "thrust_vector_control",
        "navigation_mode": "inertial_gps",
        "active_sensors": ["imu", "gps", "star_tracker", "accelerometer"],
        "target_attitude": "peg_steering",
        "transition": {
            "type": "orbit_achieved",
            "target_altitude_m": 200000.0,
            "target_velocity_m_s": 7784.0,
            "description": "Orbital velocity and altitude achieved",
        },
    },

    MissionPhase.FAIRING_JETTISON: {
        "description": "Payload fairing jettison above sensible atmosphere",
        "guidance_mode": "coast",
        "control_mode": "rcs_attitude_hold",
        "navigation_mode": "inertial",
        "active_sensors": ["imu", "star_tracker"],
        "target_attitude": "previous_hold",
        "transition": {
            "type": "altitude_threshold",
            "altitude_min_m": 110000.0,
            "description": "Above sensible atmosphere, fairing separated",
        },
    },

    MissionPhase.PARKING_ORBIT: {
        "description": "Circular parking orbit, systems checkout, TLI window wait",
        "guidance_mode": "orbit_maintenance",
        "control_mode": "three_axis_stabilized",
        "navigation_mode": "full_nav",
        "active_sensors": ["imu", "gps", "star_tracker", "horizon_sensor"],
        "target_attitude": "local_vertical_local_horizontal",
        "transition": {
            "type": "tli_window",
            "min_orbits": 1,
            "max_wait_s": 10800.0,
            "description": "TLI injection window reached",
        },
    },

    MissionPhase.TLI_BURN: {
        "description": "Trans-Lunar Injection burn to depart Earth orbit",
        "guidance_mode": "powered_explicit_guidance",
        "control_mode": "thrust_vector_control",
        "navigation_mode": "inertial_gps",
        "active_sensors": ["imu", "gps", "star_tracker", "accelerometer"],
        "target_attitude": "tli_steering",
        "transition": {
            "type": "velocity_threshold",
            "target_v_infinity_m_s": 800.0,
            "description": "Target hyperbolic excess velocity achieved",
        },
    },

    MissionPhase.LUNAR_COAST: {
        "description": "Unpowered coast from Earth to Moon",
        "guidance_mode": "coast_midcourse_correction",
        "control_mode": "three_axis_stabilized",
        "navigation_mode": "deep_space_nav",
        "active_sensors": ["imu", "star_tracker", "dsn_ranging"],
        "target_attitude": "sun_point",
        "transition": {
            "type": "soi_entry",
            "body": "moon",
            "soi_radius_m": 6.61e7,
            "description": "Enter lunar sphere of influence",
        },
    },

    MissionPhase.LUNAR_ORBIT_INSERTION: {
        "description": "LOI burn to capture into lunar orbit",
        "guidance_mode": "powered_explicit_guidance",
        "control_mode": "thrust_vector_control",
        "navigation_mode": "lunar_nav",
        "active_sensors": ["imu", "star_tracker", "lunar_altimeter", "accelerometer"],
        "target_attitude": "retrograde",
        "transition": {
            "type": "orbit_captured",
            "max_eccentricity": 0.05,
            "target_altitude_m": 100000.0,
            "description": "Stable lunar orbit achieved",
        },
    },

    MissionPhase.LUNAR_ORBIT_1: {
        "description": "Initial lunar orbit for observation and planning",
        "guidance_mode": "orbit_maintenance",
        "control_mode": "three_axis_stabilized",
        "navigation_mode": "lunar_nav",
        "active_sensors": ["imu", "star_tracker", "lunar_altimeter", "horizon_sensor"],
        "target_attitude": "nadir_point_moon",
        "transition": {
            "type": "orbit_count",
            "required_orbits": 3,
            "description": "Completed required observation orbits",
        },
    },

    MissionPhase.LUNAR_INCLINATION_CHANGE: {
        "description": "Plane change maneuver to adjust lunar orbit inclination",
        "guidance_mode": "finite_burn_steering",
        "control_mode": "thrust_vector_control",
        "navigation_mode": "lunar_nav",
        "active_sensors": ["imu", "star_tracker", "accelerometer"],
        "target_attitude": "normal_to_orbit",
        "transition": {
            "type": "burn_complete",
            "target_inclination_change_rad": np.radians(15.0),
            "description": "Inclination change maneuver complete",
        },
    },

    MissionPhase.LUNAR_ORBIT_2: {
        "description": "Second lunar orbit phase in new orbital plane",
        "guidance_mode": "orbit_maintenance",
        "control_mode": "three_axis_stabilized",
        "navigation_mode": "lunar_nav",
        "active_sensors": ["imu", "star_tracker", "lunar_altimeter", "horizon_sensor"],
        "target_attitude": "nadir_point_moon",
        "transition": {
            "type": "orbit_count",
            "required_orbits": 2,
            "description": "Observation orbits in new plane complete",
        },
    },

    MissionPhase.LUNAR_ESCAPE: {
        "description": "Trans-Jupiter injection burn from lunar orbit",
        "guidance_mode": "powered_explicit_guidance",
        "control_mode": "thrust_vector_control",
        "navigation_mode": "lunar_nav",
        "active_sensors": ["imu", "star_tracker", "accelerometer"],
        "target_attitude": "prograde",
        "transition": {
            "type": "escape_achieved",
            "body": "moon",
            "description": "Escaped lunar sphere of influence",
        },
    },

    MissionPhase.EARTH_JUPITER_TRANSFER: {
        "description": "Heliocentric coast from Earth vicinity to Jupiter",
        "guidance_mode": "coast_midcourse_correction",
        "control_mode": "spin_stabilized",
        "navigation_mode": "deep_space_nav",
        "active_sensors": ["imu", "star_tracker", "dsn_ranging"],
        "target_attitude": "sun_point",
        "transition": {
            "type": "soi_entry",
            "body": "jupiter",
            "soi_radius_m": 4.82e10,
            "description": "Enter Jupiter sphere of influence",
        },
    },

    MissionPhase.JUPITER_ORBIT_INSERTION: {
        "description": "JOI burn to capture into Jupiter orbit",
        "guidance_mode": "powered_explicit_guidance",
        "control_mode": "thrust_vector_control",
        "navigation_mode": "deep_space_nav",
        "active_sensors": ["imu", "star_tracker", "accelerometer", "dsn_ranging"],
        "target_attitude": "retrograde",
        "transition": {
            "type": "orbit_captured",
            "max_eccentricity": 0.95,
            "description": "Captured into Jupiter orbit (highly elliptical)",
        },
    },

    MissionPhase.JUPITER_ORBITS: {
        "description": "Jupiter orbital operations and science collection",
        "guidance_mode": "orbit_maintenance",
        "control_mode": "three_axis_stabilized",
        "navigation_mode": "deep_space_nav",
        "active_sensors": ["imu", "star_tracker", "dsn_ranging", "horizon_sensor"],
        "target_attitude": "nadir_point_jupiter",
        "transition": {
            "type": "orbit_count",
            "required_orbits": 5,
            "description": "Science orbits complete, prepare for departure",
        },
    },

    MissionPhase.JUPITER_ESCAPE: {
        "description": "Escape burn to depart Jupiter for Earth return",
        "guidance_mode": "powered_explicit_guidance",
        "control_mode": "thrust_vector_control",
        "navigation_mode": "deep_space_nav",
        "active_sensors": ["imu", "star_tracker", "accelerometer", "dsn_ranging"],
        "target_attitude": "prograde",
        "transition": {
            "type": "escape_achieved",
            "body": "jupiter",
            "description": "Escaped Jupiter sphere of influence",
        },
    },

    MissionPhase.JUPITER_EARTH_RETURN: {
        "description": "Heliocentric coast from Jupiter back to Earth",
        "guidance_mode": "coast_midcourse_correction",
        "control_mode": "spin_stabilized",
        "navigation_mode": "deep_space_nav",
        "active_sensors": ["imu", "star_tracker", "dsn_ranging"],
        "target_attitude": "sun_point",
        "transition": {
            "type": "soi_entry",
            "body": "earth",
            "soi_radius_m": 9.24e8,
            "description": "Enter Earth sphere of influence for return",
        },
    },

    MissionPhase.EARTH_REENTRY: {
        "description": "Earth atmospheric reentry",
        "guidance_mode": "entry_guidance",
        "control_mode": "bank_angle_modulation",
        "navigation_mode": "inertial_gps",
        "active_sensors": ["imu", "gps", "altimeter", "accelerometer"],
        "target_attitude": "trim_angle_of_attack",
        "transition": {
            "type": "altitude_threshold",
            "altitude_max_m": 10000.0,
            "description": "Below 10 km altitude, transition to terminal descent",
        },
    },

    MissionPhase.DESCENT_LANDING: {
        "description": "Terminal descent and landing (parachute / powered)",
        "guidance_mode": "terminal_descent",
        "control_mode": "parachute_or_powered",
        "navigation_mode": "inertial_gps",
        "active_sensors": ["imu", "gps", "altimeter", "radar_altimeter"],
        "target_attitude": "vertical",
        "transition": {
            "type": "landed",
            "altitude_m": 0.0,
            "velocity_m_s": 0.0,
            "description": "Touchdown confirmed",
        },
    },
}


# =============================================================================
# MISSION PLANNER FSM
# =============================================================================

class MissionPlanner:
    """
    Finite State Machine for managing all mission phases.

    The planner tracks the current phase, evaluates transition conditions based
    on spacecraft state, and advances through the mission timeline. It provides
    the active guidance, navigation, and control mode for each phase so that
    the rest of the GNC system can configure itself appropriately.

    The phase sequence is modeled as a directed graph (conceptually using the
    MissionGraph data structure), though the primary path is a linear chain
    from PRE_LAUNCH through DESCENT_LANDING.

    Attributes:
        current_phase:          Active mission phase.
        phase_start_time:       MET when the current phase began.
        mission_elapsed_time:   Total mission elapsed time in seconds.
        orbit_count:            Number of complete orbits in current phase.
        timeline:               Ordered list of (time, phase, reason) tuples.
    """

    def __init__(self) -> None:
        self.current_phase: MissionPhase = MissionPhase.PRE_LAUNCH
        self.phase_start_time: float = 0.0
        self.mission_elapsed_time: float = 0.0
        self.orbit_count: int = 0

        # Timeline records every phase transition
        self.timeline: List[Tuple[float, MissionPhase, str]] = [
            (0.0, MissionPhase.PRE_LAUNCH, "mission_initialized"),
        ]

        # Phase-specific accumulators
        self._previous_anomaly: Optional[float] = None  # for orbit counting

        logger.info(
            "MissionPlanner initialized. Starting phase: %s",
            self.current_phase.name,
        )

    # -------------------------------------------------------------------------
    # Phase Queries
    # -------------------------------------------------------------------------

    def get_current_phase(self) -> MissionPhase:
        """Return the current mission phase."""
        return self.current_phase

    def get_phase_config(self) -> dict:
        """
        Return the full configuration dictionary for the current phase.

        The config contains guidance_mode, control_mode, navigation_mode,
        active_sensors, target_attitude, and transition conditions.
        """
        return PHASE_CONFIGS[self.current_phase]

    def get_guidance_mode(self) -> str:
        """
        Return the guidance algorithm identifier for the current phase.

        Possible values include:
            'hold', 'gravity_turn', 'coast', 'powered_explicit_guidance',
            'orbit_maintenance', 'coast_midcourse_correction',
            'finite_burn_steering', 'entry_guidance', 'terminal_descent'
        """
        return PHASE_CONFIGS[self.current_phase]["guidance_mode"]

    def get_control_mode(self) -> str:
        """
        Return the attitude control mode identifier for the current phase.

        Possible values include:
            'ground_commanded', 'pitch_program', 'rcs_attitude_hold',
            'thrust_vector_control', 'three_axis_stabilized',
            'spin_stabilized', 'bank_angle_modulation',
            'parachute_or_powered'
        """
        return PHASE_CONFIGS[self.current_phase]["control_mode"]

    def get_navigation_mode(self) -> str:
        """
        Return the navigation mode identifier for the current phase.

        This determines which sensors and filters are active. Possible values:
            'ground_truth', 'inertial_gps', 'inertial', 'full_nav',
            'deep_space_nav', 'lunar_nav'
        """
        return PHASE_CONFIGS[self.current_phase]["navigation_mode"]

    def get_mission_elapsed_time(self) -> float:
        """Return total mission elapsed time in seconds."""
        return self.mission_elapsed_time

    def get_phase_elapsed_time(self) -> float:
        """Return elapsed time since the start of the current phase."""
        return self.mission_elapsed_time - self.phase_start_time

    def get_mission_timeline(self) -> List[Tuple[float, MissionPhase, str]]:
        """
        Return the complete mission timeline.

        Returns:
            List of (time_s, phase, reason) tuples in chronological order.
        """
        return list(self.timeline)

    # -------------------------------------------------------------------------
    # Transition Logic
    # -------------------------------------------------------------------------

    def check_transition(self, state: dict) -> bool:
        """
        Evaluate whether a transition to the next phase should occur.

        Examines the current phase's transition conditions against the
        provided spacecraft state dictionary.

        Args:
            state: Dictionary containing at minimum:
                - 'altitude_m':      Current altitude above reference body (m)
                - 'velocity_m_s':    Current inertial speed (m/s)
                - 'position':        3-vector position (m) in relevant frame
                - 'true_anomaly_rad': Current true anomaly (rad), if in orbit
                - 'eccentricity':    Current orbital eccentricity
                - 'commanded':       Boolean, True if ground command issued
                - 'propellant_depleted': Boolean for stage burnout
                - 'body_distance_m': Distance to relevant body center (m)
                - 'time_s':          Current simulation time (s)

        Returns:
            True if the transition condition is satisfied and the FSM should
            advance to the next phase.
        """
        config = PHASE_CONFIGS[self.current_phase]
        transition = config["transition"]
        transition_type = transition["type"]

        # Update mission elapsed time from state
        if "time_s" in state:
            self.mission_elapsed_time = state["time_s"]

        phase_elapsed = self.get_phase_elapsed_time()

        # --- Commanded transition (PRE_LAUNCH) ---
        if transition_type == "commanded":
            return state.get("commanded", False)

        # --- Propellant depletion (STAGE1_ASCENT) ---
        if transition_type == "propellant_depleted":
            altitude = state.get("altitude_m", 0.0)
            depleted = state.get("propellant_depleted", False)
            return depleted and altitude >= transition.get("altitude_min_m", 0.0)

        # --- Time elapsed (STAGE_SEPARATION) ---
        if transition_type == "time_elapsed":
            return phase_elapsed >= transition["duration_s"]

        # --- Orbit achieved (STAGE2_ASCENT) ---
        if transition_type == "orbit_achieved":
            altitude = state.get("altitude_m", 0.0)
            velocity = state.get("velocity_m_s", 0.0)
            target_alt = transition["target_altitude_m"]
            target_vel = transition["target_velocity_m_s"]
            return altitude >= target_alt and velocity >= target_vel * 0.99

        # --- Altitude threshold (FAIRING_JETTISON, EARTH_REENTRY) ---
        if transition_type == "altitude_threshold":
            altitude = state.get("altitude_m", 0.0)
            if "altitude_min_m" in transition:
                return altitude >= transition["altitude_min_m"]
            if "altitude_max_m" in transition:
                return altitude <= transition["altitude_max_m"]

        # --- TLI window (PARKING_ORBIT) ---
        if transition_type == "tli_window":
            self._update_orbit_count(state)
            orbits_ok = self.orbit_count >= transition.get("min_orbits", 1)
            window_ready = state.get("tli_window_open", False)
            return orbits_ok and window_ready

        # --- Velocity threshold (TLI_BURN) ---
        if transition_type == "velocity_threshold":
            v_inf = state.get("v_infinity_m_s", 0.0)
            return v_inf >= transition["target_v_infinity_m_s"]

        # --- SOI entry (LUNAR_COAST, EARTH_JUPITER_TRANSFER, etc.) ---
        if transition_type == "soi_entry":
            body_dist = state.get("body_distance_m", float("inf"))
            return body_dist <= transition["soi_radius_m"]

        # --- Orbit captured (LOI, JOI) ---
        if transition_type == "orbit_captured":
            ecc = state.get("eccentricity", 1.0)
            return ecc <= transition["max_eccentricity"]

        # --- Orbit count (LUNAR_ORBIT_1, LUNAR_ORBIT_2, JUPITER_ORBITS) ---
        if transition_type == "orbit_count":
            self._update_orbit_count(state)
            return self.orbit_count >= transition["required_orbits"]

        # --- Burn complete (LUNAR_INCLINATION_CHANGE) ---
        if transition_type == "burn_complete":
            burn_done = state.get("burn_complete", False)
            return burn_done

        # --- Escape achieved (LUNAR_ESCAPE, JUPITER_ESCAPE) ---
        if transition_type == "escape_achieved":
            ecc = state.get("eccentricity", 0.0)
            energy = state.get("orbital_energy", 0.0)
            return ecc >= 1.0 or energy >= 0.0

        # --- Landed (DESCENT_LANDING) ---
        if transition_type == "landed":
            altitude = state.get("altitude_m", float("inf"))
            velocity = state.get("velocity_m_s", float("inf"))
            return altitude <= 1.0 and velocity <= 5.0

        logger.warning(
            "Unknown transition type '%s' for phase %s",
            transition_type,
            self.current_phase.name,
        )
        return False

    def advance_phase(self, reason: str = "") -> MissionPhase:
        """
        Advance the FSM to the next mission phase.

        Logs the transition, resets phase-specific counters, and returns the
        new phase.

        Args:
            reason: Human-readable explanation for why the transition occurred.

        Returns:
            The new MissionPhase after advancement.

        Raises:
            StopIteration: If the mission is already in the final phase
                           (DESCENT_LANDING).
        """
        old_phase = self.current_phase

        if old_phase == MissionPhase.DESCENT_LANDING:
            raise StopIteration("Mission complete. No further phases.")

        # Advance to the next integer phase
        new_phase = MissionPhase(old_phase.value + 1)

        self.log_phase_transition(old_phase, new_phase, self.mission_elapsed_time, reason)

        self.current_phase = new_phase
        self.phase_start_time = self.mission_elapsed_time
        self.orbit_count = 0
        self._previous_anomaly = None

        logger.info(
            "Phase transition: %s -> %s at MET=%.1f s. Reason: %s",
            old_phase.name,
            new_phase.name,
            self.mission_elapsed_time,
            reason,
        )

        return new_phase

    def update(self, state: dict) -> MissionPhase:
        """
        Main update call: check transition and advance if needed.

        This is the primary interface called every simulation step.

        Args:
            state: Spacecraft state dictionary (see check_transition for keys).

        Returns:
            The current (possibly updated) mission phase.
        """
        if self.check_transition(state):
            transition = PHASE_CONFIGS[self.current_phase]["transition"]
            reason = transition.get("description", "condition_met")
            self.advance_phase(reason=reason)
        return self.current_phase

    # -------------------------------------------------------------------------
    # Logging and Timeline
    # -------------------------------------------------------------------------

    def log_phase_transition(
        self,
        old_phase: MissionPhase,
        new_phase: MissionPhase,
        time: float,
        reason: str,
    ) -> None:
        """
        Record a phase transition in the mission timeline.

        Args:
            old_phase: Phase being departed.
            new_phase: Phase being entered.
            time:      Mission elapsed time at transition (s).
            reason:    Description of why the transition occurred.
        """
        self.timeline.append((time, new_phase, reason))
        logger.info(
            "TIMELINE [MET %.1f s]: %s -> %s (%s)",
            time,
            old_phase.name,
            new_phase.name,
            reason,
        )

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _update_orbit_count(self, state: dict) -> None:
        """
        Track orbit completions by monitoring true anomaly wrap-around.

        An orbit is counted each time the true anomaly crosses 0 (ascending
        through 2*pi -> 0). This is a simple orbit counter suitable for
        near-circular orbits.

        Args:
            state: Must contain 'true_anomaly_rad' key.
        """
        anomaly = state.get("true_anomaly_rad", None)
        if anomaly is None:
            return

        if self._previous_anomaly is not None:
            # Detect wrap-around: previous anomaly near 2*pi, current near 0
            if self._previous_anomaly > np.pi and anomaly < np.pi:
                self.orbit_count += 1
                logger.debug(
                    "Orbit %d completed in phase %s",
                    self.orbit_count,
                    self.current_phase.name,
                )

        self._previous_anomaly = anomaly

    # -------------------------------------------------------------------------
    # Summary and Display
    # -------------------------------------------------------------------------

    def get_status_summary(self) -> str:
        """Return a human-readable summary of the current mission state."""
        config = self.get_phase_config()
        return (
            f"Mission Phase: {self.current_phase.name}\n"
            f"  Description:     {config['description']}\n"
            f"  Guidance Mode:   {config['guidance_mode']}\n"
            f"  Control Mode:    {config['control_mode']}\n"
            f"  Navigation Mode: {config['navigation_mode']}\n"
            f"  Active Sensors:  {', '.join(config['active_sensors'])}\n"
            f"  Target Attitude: {config['target_attitude']}\n"
            f"  MET:             {self.mission_elapsed_time:.1f} s\n"
            f"  Phase Elapsed:   {self.get_phase_elapsed_time():.1f} s\n"
            f"  Orbits:          {self.orbit_count}\n"
        )

    def __repr__(self) -> str:
        return (
            f"MissionPlanner(phase={self.current_phase.name}, "
            f"MET={self.mission_elapsed_time:.1f}s)"
        )
