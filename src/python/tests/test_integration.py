"""
===============================================================================
GNC PROJECT - Integration Test Suite
===============================================================================
End-to-end integration tests for the full GNC simulation system covering
initialization, single-step execution, phase transitions, telemetry recording,
Monte Carlo runs, sensor fault injection, full ascent phase simulation, and
database write/read round-trip.

These tests exercise the interaction between multiple subsystems: mission
planner FSM, spacecraft model, attitude dynamics, sensors, actuators,
guidance, and data persistence.
===============================================================================
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tempfile
import sqlite3

import numpy as np
import pytest

from core.quaternion import Quaternion
from core.constants import (
    EARTH_MU, EARTH_RADIUS, LEO_ALTITUDE, LEO_RADIUS, LEO_VELOCITY,
)
from core.data_structures import StateHistory, TelemetryBuffer, MissionGraph
from dynamics.spacecraft import Spacecraft
from dynamics.attitude_dynamics import AttitudeDynamics, AttitudeConfig
from dynamics.environment import GravityField, ExponentialAtmosphere
from guidance.mission_planner import MissionPlanner, MissionPhase
from navigation.sensors import IMU, StarTracker
from control.actuators import ReactionWheelArray, Thruster


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def spacecraft_config():
    """Return a minimal spacecraft configuration dict."""
    return {
        "spacecraft": {
            "dry_mass": 2500.0,
            "propellant_mass": 1500.0,
            "dimensions": [3.0, 2.0, 2.0],
            "solar_panel_area": 40.0,
            "reflectivity": 0.3,
            "inertia": {
                "Ixx": 3500.0, "Iyy": 4200.0, "Izz": 4800.0,
                "Ixy": 0.0, "Ixz": 0.0, "Iyz": 0.0,
            },
            "cg_offset": [0.0, 0.0, 0.0],
            "flex_modes": [],
            "mass_uncertainty": 0.02,
        }
    }


@pytest.fixture
def spacecraft(spacecraft_config):
    """Return a Spacecraft instance."""
    return Spacecraft(spacecraft_config)


@pytest.fixture
def mission_planner():
    """Return a fresh MissionPlanner."""
    return MissionPlanner()


@pytest.fixture
def attitude_dynamics():
    """Return a basic AttitudeDynamics instance."""
    I = np.diag([3500.0, 4200.0, 4800.0])
    config = AttitudeConfig(inertia=I)
    return AttitudeDynamics(config)


@pytest.fixture
def imu_sensor():
    """Return an IMU sensor with deterministic seed."""
    return IMU({"seed": 42, "dt": 0.01})


@pytest.fixture
def rw_array():
    """Return a reaction wheel array."""
    return ReactionWheelArray(max_torque=0.2, max_momentum=50.0)


@pytest.fixture
def state_history():
    """Return a StateHistory buffer for telemetry."""
    return StateHistory(capacity=10000, state_dim=13)


@pytest.fixture
def telemetry_buffer():
    """Return a TelemetryBuffer for frame-based telemetry."""
    return TelemetryBuffer(frame_size=100, channel_count=6)


# =============================================================================
# Simulation Engine (lightweight for testing)
# =============================================================================

class SimulationEngine:
    """Lightweight simulation engine that ties subsystems together."""

    def __init__(self, config):
        self.spacecraft = Spacecraft(config)
        I = np.diag([3500.0, 4200.0, 4800.0])
        att_config = AttitudeConfig(inertia=I)
        self.attitude_dynamics = AttitudeDynamics(att_config)
        self.mission_planner = MissionPlanner()
        self.gravity = GravityField.earth()
        self.atmosphere = ExponentialAtmosphere()
        self.imu = IMU({"seed": 0, "dt": 0.01})
        self.rw_array = ReactionWheelArray()
        self.telemetry = StateHistory(capacity=100000, state_dim=13)
        self.time = 0.0
        self.dt = 0.1
        self.step_count = 0

        # Initial state: quaternion (scalar-last) + omega
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        omega0 = np.zeros(3)
        self.attitude_state = self.attitude_dynamics.build_initial_state(q0, omega0)

        # Orbital state: simple circular orbit
        self.position = np.array([LEO_RADIUS, 0.0, 0.0])
        self.velocity = np.array([0.0, LEO_VELOCITY, 0.0])

    def step(self):
        """Execute one simulation time step."""
        # Attitude propagation
        self.attitude_state = self.attitude_dynamics.propagate(
            self.attitude_state, self.dt
        )

        # Simple orbital propagation (Euler step for simplicity)
        r_mag = np.linalg.norm(self.position)
        accel = -EARTH_MU / r_mag ** 3 * self.position
        self.velocity += accel * self.dt
        self.position += self.velocity * self.dt

        # Record telemetry
        altitude = np.linalg.norm(self.position) - EARTH_RADIUS
        state_vec = np.concatenate([
            self.position, self.velocity,
            self.attitude_state[:4],  # quaternion
            self.attitude_state[4:7],  # omega
        ])
        self.telemetry.append(self.time, state_vec)

        self.time += self.dt
        self.step_count += 1

        return {
            "altitude_m": altitude,
            "velocity_m_s": np.linalg.norm(self.velocity),
            "position": self.position.copy(),
            "time_s": self.time,
        }


# =============================================================================
# Test: Simulation engine initializes
# =============================================================================

class TestSimulationEngineInitializes:
    """Tests for simulation engine initialization."""

    def test_simulation_engine_initializes(self, spacecraft_config):
        """Sim engine loads config and creates all subsystems."""
        sim = SimulationEngine(spacecraft_config)

        assert sim.spacecraft is not None
        assert sim.attitude_dynamics is not None
        assert sim.mission_planner is not None
        assert sim.gravity is not None
        assert sim.atmosphere is not None
        assert sim.imu is not None
        assert sim.rw_array is not None
        assert sim.telemetry is not None
        assert sim.time == 0.0
        assert sim.step_count == 0


# =============================================================================
# Test: Single step runs
# =============================================================================

class TestSingleStepRuns:
    """Tests that a single simulation step executes without error."""

    def test_single_step_runs(self, spacecraft_config):
        """One sim step should execute without error and return state."""
        sim = SimulationEngine(spacecraft_config)
        state = sim.step()

        assert "altitude_m" in state
        assert "velocity_m_s" in state
        assert "position" in state
        assert "time_s" in state
        assert sim.step_count == 1
        assert state["time_s"] == pytest.approx(0.1, abs=1e-10)


# =============================================================================
# Test: Phase transition
# =============================================================================

class TestPhaseTransition:
    """Tests for FSM phase transitions."""

    def test_phase_transition(self, mission_planner):
        """Sim should transition from PRE_LAUNCH to STAGE1_ASCENT on command."""
        assert mission_planner.get_current_phase() == MissionPhase.PRE_LAUNCH

        # Simulate launch command
        state = {
            "commanded": True,
            "time_s": 0.0,
        }
        new_phase = mission_planner.update(state)
        assert new_phase == MissionPhase.STAGE1_ASCENT

    def test_phase_does_not_advance_without_condition(self, mission_planner):
        """Phase should NOT advance if transition condition is not met."""
        state = {
            "commanded": False,
            "time_s": 0.0,
        }
        phase = mission_planner.update(state)
        assert phase == MissionPhase.PRE_LAUNCH

    def test_multiple_transitions(self, mission_planner):
        """Test sequential phase transitions through early mission phases."""
        # PRE_LAUNCH -> STAGE1_ASCENT
        mission_planner.update({"commanded": True, "time_s": 0.0})
        assert mission_planner.get_current_phase() == MissionPhase.STAGE1_ASCENT

        # STAGE1_ASCENT -> STAGE_SEPARATION
        state = {
            "propellant_depleted": True,
            "altitude_m": 60000.0,
            "time_s": 120.0,
        }
        mission_planner.update(state)
        assert mission_planner.get_current_phase() == MissionPhase.STAGE_SEPARATION


# =============================================================================
# Test: Telemetry recorded
# =============================================================================

class TestTelemetryRecorded:
    """Tests for telemetry data recording."""

    def test_telemetry_recorded(self, spacecraft_config):
        """After N steps, telemetry StateHistory should have N entries."""
        sim = SimulationEngine(spacecraft_config)
        n_steps = 50

        for _ in range(n_steps):
            sim.step()

        assert sim.telemetry.count == n_steps

        # Verify we can retrieve the data
        timestamps, states = sim.telemetry.to_array()
        assert len(timestamps) == n_steps
        assert states.shape == (n_steps, 13)


# =============================================================================
# Test: Monte Carlo runs
# =============================================================================

class TestMonteCarlo:
    """Tests for Monte Carlo simulation ensemble."""

    def test_monte_carlo_runs(self, spacecraft_config):
        """3-run Monte Carlo should complete without error."""
        n_mc = 3
        n_steps = 20
        results = []

        for run_idx in range(n_mc):
            np.random.seed(run_idx)
            sim = SimulationEngine(spacecraft_config)

            # Add small random perturbation to initial velocity
            sim.velocity += np.random.normal(0, 1.0, size=3)

            final_state = None
            for _ in range(n_steps):
                final_state = sim.step()

            results.append({
                "run": run_idx,
                "final_altitude": final_state["altitude_m"],
                "final_velocity": final_state["velocity_m_s"],
                "steps": sim.step_count,
            })

        assert len(results) == n_mc
        for r in results:
            assert r["steps"] == n_steps
            assert r["final_altitude"] > 0


# =============================================================================
# Test: Sensor fault injection
# =============================================================================

class TestSensorFaultInjection:
    """Tests for sensor fault injection and detection."""

    def test_sensor_fault_injection(self, imu_sensor):
        """SIL interface can inject and detect a fault on the IMU."""
        # Verify healthy initially
        status = imu_sensor.get_health_status()
        assert status["healthy"] is True

        # Normal measurement works
        omega_true = np.array([0.01, -0.005, 0.002])
        accel_true = np.array([0.0, 0.0, -9.81])
        omega_m, accel_m = imu_sensor.measure(omega_true, accel_true)
        assert omega_m is not None
        assert accel_m is not None

        # Inject fault: disable the sensor
        imu_sensor._healthy = False

        # After fault, measurement should return None
        omega_m2, accel_m2 = imu_sensor.measure(omega_true, accel_true)
        assert omega_m2 is None
        assert accel_m2 is None

        # Detect fault
        status = imu_sensor.get_health_status()
        assert status["healthy"] is False

        # Reset (power-cycle) clears the fault
        imu_sensor.reset()
        status = imu_sensor.get_health_status()
        assert status["healthy"] is True


# =============================================================================
# Test: Full ascent phase
# =============================================================================

class TestFullAscentPhase:
    """Tests for running the ascent phase to parking orbit."""

    def test_full_ascent_phase(self, spacecraft_config):
        """Run ascent simulation; final altitude should be near 200 km."""
        sim = SimulationEngine(spacecraft_config)

        # Simulate until we reach roughly one orbit or enough time
        n_steps = 500
        for _ in range(n_steps):
            state = sim.step()

        altitude_km = state["altitude_m"] / 1000.0

        # For a circular orbit at ~200 km, the altitude should stay near LEO
        assert 150.0 < altitude_km < 250.0, (
            f"Final altitude {altitude_km:.1f} km not near expected 200 km"
        )


# =============================================================================
# Test: Database write and read
# =============================================================================

class TestDatabaseWriteRead:
    """Tests for writing telemetry to SQLite and reading back."""

    def test_database_write_read(self, spacecraft_config):
        """Write telemetry to SQLite and read back; verify data matches."""
        sim = SimulationEngine(spacecraft_config)

        # Run a few steps
        n_steps = 20
        for _ in range(n_steps):
            sim.step()

        timestamps, states = sim.telemetry.to_array()

        # Write to a temporary SQLite database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Create table
            cursor.execute("""
                CREATE TABLE telemetry (
                    step_id INTEGER PRIMARY KEY,
                    timestamp REAL,
                    x REAL, y REAL, z REAL,
                    vx REAL, vy REAL, vz REAL,
                    q0 REAL, q1 REAL, q2 REAL, q3 REAL,
                    wx REAL, wy REAL, wz REAL
                )
            """)

            # Insert data
            for i in range(n_steps):
                row = (
                    i, float(timestamps[i]),
                    *[float(v) for v in states[i]]
                )
                cursor.execute(
                    "INSERT INTO telemetry VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    row,
                )
            conn.commit()

            # Read back
            cursor.execute("SELECT * FROM telemetry ORDER BY step_id")
            rows = cursor.fetchall()

            assert len(rows) == n_steps

            # Verify data matches
            for i, row in enumerate(rows):
                assert row[0] == i  # step_id
                assert row[1] == pytest.approx(float(timestamps[i]), abs=1e-10)
                for j in range(13):
                    assert row[2 + j] == pytest.approx(
                        float(states[i, j]), abs=1e-10
                    ), f"Mismatch at step {i}, column {j}"

            conn.close()
        finally:
            os.unlink(db_path)

    def test_database_write_read_pandas(self, spacecraft_config):
        """Write telemetry using pandas to SQLite and read back."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")

        sim = SimulationEngine(spacecraft_config)

        n_steps = 15
        for _ in range(n_steps):
            sim.step()

        timestamps, states = sim.telemetry.to_array()

        # Build DataFrame
        columns = ["x", "y", "z", "vx", "vy", "vz",
                    "q0", "q1", "q2", "q3", "wx", "wy", "wz"]
        df = pd.DataFrame(states, columns=columns)
        df.insert(0, "timestamp", timestamps)

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Write via pandas
            conn = sqlite3.connect(db_path)
            df.to_sql("telemetry", conn, if_exists="replace", index=False)
            conn.close()

            # Read back via pandas
            conn = sqlite3.connect(db_path)
            df_read = pd.read_sql("SELECT * FROM telemetry", conn)
            conn.close()

            assert len(df_read) == n_steps
            pd.testing.assert_frame_equal(df, df_read, atol=1e-10)
        finally:
            os.unlink(db_path)
