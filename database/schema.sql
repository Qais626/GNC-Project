-- =============================================================================
-- GNC MISSION TELEMETRY DATABASE SCHEMA
-- =============================================================================
-- SQLite-compatible schema for the Miami-Moon-Jupiter Round Trip mission.
-- Stores telemetry, sensor readings, control commands, maneuvers, anomalies,
-- phase transitions, and Monte Carlo analysis results.
--
-- All units follow SI convention (meters, seconds, kilograms, radians)
-- unless otherwise noted in column comments.
-- =============================================================================

PRAGMA journal_mode = WAL;          -- Write-ahead logging for concurrent reads
PRAGMA foreign_keys = ON;           -- Enforce referential integrity
PRAGMA synchronous = NORMAL;        -- Balance safety and performance

-- =============================================================================
-- MISSION INFORMATION
-- =============================================================================

CREATE TABLE IF NOT EXISTS mission_info (
    mission_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    description     TEXT,
    start_time      REAL,                       -- Mission elapsed time (s) or epoch
    end_time        REAL,
    status          TEXT    NOT NULL DEFAULT 'planned'
                    CHECK (status IN ('planned', 'active', 'completed',
                                      'aborted', 'failed'))
);

-- =============================================================================
-- TELEMETRY
-- =============================================================================
-- Core spacecraft state vector sampled at each simulation time step.
-- Positions and velocities are in the Earth-Centered Inertial (ECI) frame
-- unless the mission phase dictates another primary body (Moon/Jupiter).
-- Quaternion convention: scalar-first (w, x, y, z).

CREATE TABLE IF NOT EXISTS telemetry (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id      INTEGER NOT NULL REFERENCES mission_info(mission_id),
    timestamp       REAL    NOT NULL,           -- Mission elapsed time (s)
    phase           TEXT,                        -- Current mission phase name

    -- Position (m) in primary-body-centered inertial frame
    pos_x           REAL    NOT NULL,
    pos_y           REAL    NOT NULL,
    pos_z           REAL    NOT NULL,

    -- Velocity (m/s)
    vel_x           REAL    NOT NULL,
    vel_y           REAL    NOT NULL,
    vel_z           REAL    NOT NULL,

    -- Attitude quaternion (body-to-inertial, scalar-first)
    quat_w          REAL    NOT NULL DEFAULT 1.0,
    quat_x          REAL    NOT NULL DEFAULT 0.0,
    quat_y          REAL    NOT NULL DEFAULT 0.0,
    quat_z          REAL    NOT NULL DEFAULT 0.0,

    -- Angular velocity (rad/s, body frame)
    omega_x         REAL    NOT NULL DEFAULT 0.0,
    omega_y         REAL    NOT NULL DEFAULT 0.0,
    omega_z         REAL    NOT NULL DEFAULT 0.0,

    -- Mass properties
    mass            REAL,                       -- Current total mass (kg)
    fuel_remaining  REAL                        -- Remaining propellant (kg)
);

-- =============================================================================
-- SENSOR READINGS
-- =============================================================================
-- Raw or processed sensor measurements logged for navigation analysis.
-- The reading_json column stores a JSON blob whose structure depends on the
-- sensor_type (IMU gives {gyro_x/y/z, accel_x/y/z}, star tracker gives
-- {quat_w/x/y/z}, etc.).

CREATE TABLE IF NOT EXISTS sensor_readings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id      INTEGER NOT NULL REFERENCES mission_info(mission_id),
    timestamp       REAL    NOT NULL,           -- Mission elapsed time (s)
    sensor_type     TEXT    NOT NULL
                    CHECK (sensor_type IN ('imu', 'star_tracker', 'sun_sensor',
                                           'gps', 'deep_space_network',
                                           'magnetometer', 'horizon_sensor')),
    reading_json    TEXT    NOT NULL,            -- JSON payload
    is_valid        INTEGER NOT NULL DEFAULT 1,  -- 1 = valid, 0 = flagged bad
    snr             REAL                         -- Signal-to-noise ratio (dB)
);

-- =============================================================================
-- CONTROL COMMANDS
-- =============================================================================
-- Commands issued by the GNC control law to actuators (reaction wheels,
-- thrusters, CMGs).

CREATE TABLE IF NOT EXISTS control_commands (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id      INTEGER NOT NULL REFERENCES mission_info(mission_id),
    timestamp       REAL    NOT NULL,
    command_type    TEXT    NOT NULL
                    CHECK (command_type IN ('reaction_wheel', 'thruster',
                                            'cmg', 'rcs', 'main_engine')),

    -- Torque command (Nm, body frame)
    torque_x        REAL    NOT NULL DEFAULT 0.0,
    torque_y        REAL    NOT NULL DEFAULT 0.0,
    torque_z        REAL    NOT NULL DEFAULT 0.0,

    -- Thrust magnitude (N) -- for translational maneuvers
    thrust_magnitude REAL   NOT NULL DEFAULT 0.0
);

-- =============================================================================
-- MANEUVERS
-- =============================================================================
-- Record of each propulsive or impulsive maneuver executed during the mission.

CREATE TABLE IF NOT EXISTS maneuvers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id      INTEGER NOT NULL REFERENCES mission_info(mission_id),
    phase           TEXT    NOT NULL,
    start_time      REAL    NOT NULL,
    end_time        REAL    NOT NULL,
    delta_v         REAL    NOT NULL,            -- Achieved delta-V (m/s)
    fuel_consumed   REAL    NOT NULL,            -- Propellant used (kg)
    maneuver_type   TEXT    NOT NULL
                    CHECK (maneuver_type IN ('hohmann', 'bielliptic',
                                             'plane_change', 'phasing',
                                             'insertion', 'escape',
                                             'correction', 'deorbit',
                                             'powered_descent', 'attitude'))
);

-- =============================================================================
-- ANOMALY LOG
-- =============================================================================
-- Events flagged by the autonomous anomaly detection system.

CREATE TABLE IF NOT EXISTS anomaly_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id      INTEGER NOT NULL REFERENCES mission_info(mission_id),
    timestamp       REAL    NOT NULL,
    sensor_name     TEXT    NOT NULL,
    anomaly_score   REAL    NOT NULL,            -- 0.0 to 1.0 confidence
    is_anomaly      INTEGER NOT NULL DEFAULT 0,  -- 1 = confirmed anomaly
    description     TEXT
);

-- =============================================================================
-- PHASE TRANSITIONS
-- =============================================================================
-- Log of mission phase changes with transition rationale.

CREATE TABLE IF NOT EXISTS phase_transitions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id      INTEGER NOT NULL REFERENCES mission_info(mission_id),
    from_phase      TEXT    NOT NULL,
    to_phase        TEXT    NOT NULL,
    transition_time REAL    NOT NULL,            -- Time of transition (s)
    reason          TEXT                          -- Human-readable explanation
);

-- =============================================================================
-- MONTE CARLO RESULTS
-- =============================================================================
-- Aggregate results from each Monte Carlo dispersion run.

CREATE TABLE IF NOT EXISTS monte_carlo_results (
    run_id              INTEGER PRIMARY KEY,
    seed                INTEGER NOT NULL,
    total_delta_v       REAL    NOT NULL,        -- Total delta-V expended (m/s)
    total_fuel          REAL    NOT NULL,        -- Total fuel consumed (kg)
    max_pointing_error  REAL    NOT NULL,        -- Peak pointing error (deg)
    landing_lat         REAL,                    -- Final landing latitude (deg)
    landing_lon         REAL,                    -- Final landing longitude (deg)
    landing_error_m     REAL,                    -- Miss distance from Miami (m)
    success             INTEGER NOT NULL DEFAULT 1  -- 1 = success, 0 = failure
);


-- =============================================================================
-- INDEXES
-- =============================================================================
-- Indexes on timestamp columns for efficient range queries during analysis.

CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp
    ON telemetry(timestamp);

CREATE INDEX IF NOT EXISTS idx_telemetry_mission_time
    ON telemetry(mission_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_telemetry_phase
    ON telemetry(phase);

CREATE INDEX IF NOT EXISTS idx_sensor_readings_timestamp
    ON sensor_readings(timestamp);

CREATE INDEX IF NOT EXISTS idx_sensor_readings_mission_time
    ON sensor_readings(mission_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_sensor_readings_type
    ON sensor_readings(sensor_type);

CREATE INDEX IF NOT EXISTS idx_control_commands_timestamp
    ON control_commands(timestamp);

CREATE INDEX IF NOT EXISTS idx_control_commands_mission_time
    ON control_commands(mission_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_maneuvers_mission_phase
    ON maneuvers(mission_id, phase);

CREATE INDEX IF NOT EXISTS idx_maneuvers_time
    ON maneuvers(start_time);

CREATE INDEX IF NOT EXISTS idx_anomaly_log_timestamp
    ON anomaly_log(timestamp);

CREATE INDEX IF NOT EXISTS idx_anomaly_log_mission_time
    ON anomaly_log(mission_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_phase_transitions_time
    ON phase_transitions(transition_time);

CREATE INDEX IF NOT EXISTS idx_phase_transitions_mission
    ON phase_transitions(mission_id);


-- =============================================================================
-- VIEWS
-- =============================================================================

-- v_mission_summary: Aggregated statistics for each mission
CREATE VIEW IF NOT EXISTS v_mission_summary AS
SELECT
    mi.mission_id,
    mi.name                                         AS mission_name,
    mi.status,
    mi.start_time,
    mi.end_time,
    (mi.end_time - mi.start_time)                   AS duration_s,
    COUNT(DISTINCT t.id)                             AS telemetry_count,
    COUNT(DISTINCT m.id)                             AS maneuver_count,
    COALESCE(SUM(m.delta_v), 0.0)                   AS total_delta_v,
    COALESCE(SUM(m.fuel_consumed), 0.0)             AS total_fuel_consumed,
    MIN(t.fuel_remaining)                           AS min_fuel_remaining,
    COUNT(DISTINCT a.id)                             AS anomaly_count,
    COUNT(DISTINCT pt.id)                            AS phase_transition_count,
    COUNT(DISTINCT sr.id)                            AS sensor_reading_count,
    COUNT(DISTINCT cc.id)                            AS control_command_count
FROM
    mission_info mi
LEFT JOIN telemetry t           ON t.mission_id = mi.mission_id
LEFT JOIN maneuvers m           ON m.mission_id = mi.mission_id
LEFT JOIN anomaly_log a         ON a.mission_id = mi.mission_id
LEFT JOIN phase_transitions pt  ON pt.mission_id = mi.mission_id
LEFT JOIN sensor_readings sr    ON sr.mission_id = mi.mission_id
LEFT JOIN control_commands cc   ON cc.mission_id = mi.mission_id
GROUP BY mi.mission_id;


-- v_phase_duration: Time spent in each mission phase (from phase transitions)
CREATE VIEW IF NOT EXISTS v_phase_duration AS
SELECT
    pt1.mission_id,
    pt1.to_phase                                    AS phase_name,
    pt1.transition_time                             AS phase_start_time,
    COALESCE(
        MIN(pt2.transition_time),
        (SELECT end_time FROM mission_info
         WHERE mission_id = pt1.mission_id)
    )                                               AS phase_end_time,
    COALESCE(
        MIN(pt2.transition_time),
        (SELECT end_time FROM mission_info
         WHERE mission_id = pt1.mission_id)
    ) - pt1.transition_time                         AS duration_s,
    pt1.reason                                      AS entry_reason
FROM
    phase_transitions pt1
LEFT JOIN phase_transitions pt2
    ON  pt2.mission_id = pt1.mission_id
    AND pt2.transition_time > pt1.transition_time
GROUP BY pt1.mission_id, pt1.id
ORDER BY pt1.mission_id, pt1.transition_time;


-- =============================================================================
-- SAMPLE DATA
-- =============================================================================

INSERT INTO mission_info (name, description, start_time, end_time, status)
VALUES (
    'Miami-Moon-Jupiter Round Trip v1',
    'Primary simulation: Launch from Miami, 2 lunar orbits (equatorial + 45-deg inclined), '
    || '3 Jupiter orbits, return to Miami. Chemical bipropellant propulsion.',
    0.0,
    200000000.0,
    'planned'
);

INSERT INTO mission_info (name, description, start_time, end_time, status)
VALUES (
    'Lunar-Only Test Run',
    'Abbreviated mission: Launch from Miami, LEO parking orbit, TLI, 2 lunar orbits, '
    || 'and return. Used for navigation filter validation.',
    0.0,
    1500000.0,
    'completed'
);

INSERT INTO mission_info (name, description, start_time, end_time, status)
VALUES (
    'Monte Carlo Dispersion Campaign',
    '100-run Monte Carlo analysis with dispersed initial conditions, sensor biases, '
    || 'thrust magnitude, and Isp. Seeds 1-100.',
    0.0,
    200000000.0,
    'active'
);

INSERT INTO mission_info (name, description, start_time, end_time, status)
VALUES (
    'Ion Thruster Trade Study',
    'Full mission profile using Xenon ion thruster option. Low thrust, high Isp. '
    || 'Compares total fuel mass and trip time against chemical baseline.',
    0.0,
    500000000.0,
    'planned'
);
