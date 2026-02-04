"""
===============================================================================
GNC PROJECT - Telemetry Database Interface
===============================================================================
SQLite-backed storage for mission telemetry, sensor readings, control commands,
maneuvers, anomalies, phase transitions, and Monte Carlo results.

Uses sqlite3 for database operations and pandas for efficient bulk I/O and
analysis queries. The schema is automatically created from schema.sql when a
new database is initialized.

Usage:
    from database.telemetry_db import TelemetryDatabase

    db = TelemetryDatabase("output/data/mission.db")
    db.insert_telemetry({
        "mission_id": 1, "timestamp": 100.0, "phase": "stage1_ascent",
        "pos_x": 1e6, "pos_y": 0.0, "pos_z": 0.0,
        "vel_x": 0.0, "vel_y": 7800.0, "vel_z": 0.0,
    })
    df = db.query_telemetry(0.0, 1000.0)
    db.close()

===============================================================================
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np


# Path to the SQL schema file relative to this module
_MODULE_DIR = Path(__file__).resolve().parent
_SCHEMA_PATH = _MODULE_DIR.parent.parent.parent / "database" / "schema.sql"


class TelemetryDatabase:
    """SQLite database interface for GNC mission telemetry and analysis data.

    This class wraps a SQLite database with methods tailored for the spacecraft
    GNC simulation pipeline. It supports single-row inserts for real-time
    logging and bulk DataFrame operations for post-processing analysis.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file. Created if it does not exist.
    schema_path : str or Path, optional
        Override path to the SQL schema file. Defaults to
        ``<project_root>/database/schema.sql``.

    Examples
    --------
    >>> db = TelemetryDatabase("output/data/mission.db")
    >>> db.insert_telemetry({"mission_id": 1, "timestamp": 0.0, ...})
    >>> df = db.query_telemetry(0.0, 3600.0)
    >>> db.close()
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        schema_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.schema_path = Path(schema_path) if schema_path else _SCHEMA_PATH

        # Ensure the parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        is_new_db = not self.db_path.exists()

        self._conn = sqlite3.connect(
            str(self.db_path),
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")

        if is_new_db:
            self.create_tables()

    # =========================================================================
    # Schema Management
    # =========================================================================

    def create_tables(self) -> None:
        """Execute the SQL schema file to create all tables, indexes, and views.

        This is called automatically when the database file is first created.
        It can also be called explicitly to reset or re-apply the schema.

        Raises
        ------
        FileNotFoundError
            If the schema SQL file cannot be found.
        sqlite3.OperationalError
            If the SQL contains errors.
        """
        if not self.schema_path.exists():
            raise FileNotFoundError(
                f"Schema file not found: {self.schema_path}\n"
                f"Expected at: {_SCHEMA_PATH}"
            )

        schema_sql = self.schema_path.read_text(encoding="utf-8")
        self._conn.executescript(schema_sql)
        self._conn.commit()

    # =========================================================================
    # Insert Operations
    # =========================================================================

    def insert_telemetry(self, data: Dict[str, Any]) -> int:
        """Insert a single telemetry record.

        Parameters
        ----------
        data : dict
            Must contain at minimum: ``mission_id``, ``timestamp``,
            ``pos_x``, ``pos_y``, ``pos_z``, ``vel_x``, ``vel_y``, ``vel_z``.
            Optional fields: ``phase``, ``quat_w/x/y/z``, ``omega_x/y/z``,
            ``mass``, ``fuel_remaining``.

        Returns
        -------
        int
            The row ID of the inserted record.
        """
        columns = [
            "mission_id", "timestamp", "phase",
            "pos_x", "pos_y", "pos_z",
            "vel_x", "vel_y", "vel_z",
            "quat_w", "quat_x", "quat_y", "quat_z",
            "omega_x", "omega_y", "omega_z",
            "mass", "fuel_remaining",
        ]
        # Filter to only keys present in data
        present = [c for c in columns if c in data]
        placeholders = ", ".join(["?"] * len(present))
        col_names = ", ".join(present)
        values = [data[c] for c in present]

        cursor = self._conn.execute(
            f"INSERT INTO telemetry ({col_names}) VALUES ({placeholders})",
            values,
        )
        self._conn.commit()
        return cursor.lastrowid

    def insert_batch_telemetry(self, df: pd.DataFrame) -> int:
        """Bulk insert telemetry from a pandas DataFrame.

        This is significantly faster than row-by-row insertion for large
        datasets. Uses pandas ``to_sql`` with ``if_exists='append'`` for
        optimal performance.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns matching the telemetry table schema.
            The ``id`` column should be omitted (auto-generated).

        Returns
        -------
        int
            Number of rows inserted.
        """
        # Drop 'id' column if present -- it is auto-incremented
        if "id" in df.columns:
            df = df.drop(columns=["id"])

        rows_before = self._count_rows("telemetry")
        df.to_sql("telemetry", self._conn, if_exists="append", index=False)
        self._conn.commit()
        rows_after = self._count_rows("telemetry")
        return rows_after - rows_before

    def insert_sensor_reading(self, data: Dict[str, Any]) -> int:
        """Insert a single sensor reading.

        Parameters
        ----------
        data : dict
            Keys: ``mission_id``, ``timestamp``, ``sensor_type``,
            ``reading_json`` (str or dict), ``is_valid`` (optional, default 1),
            ``snr`` (optional).

        Returns
        -------
        int
            Row ID of the inserted record.
        """
        # Convert dict readings to JSON string
        reading = data.get("reading_json", "{}")
        if isinstance(reading, dict):
            reading = json.dumps(reading)

        cursor = self._conn.execute(
            "INSERT INTO sensor_readings "
            "(mission_id, timestamp, sensor_type, reading_json, is_valid, snr) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                data["mission_id"],
                data["timestamp"],
                data["sensor_type"],
                reading,
                data.get("is_valid", 1),
                data.get("snr"),
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def insert_control_command(self, data: Dict[str, Any]) -> int:
        """Insert a control command record.

        Parameters
        ----------
        data : dict
            Keys: ``mission_id``, ``timestamp``, ``command_type``,
            ``torque_x/y/z``, ``thrust_magnitude``.

        Returns
        -------
        int
            Row ID.
        """
        cursor = self._conn.execute(
            "INSERT INTO control_commands "
            "(mission_id, timestamp, command_type, torque_x, torque_y, torque_z, "
            "thrust_magnitude) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                data["mission_id"],
                data["timestamp"],
                data["command_type"],
                data.get("torque_x", 0.0),
                data.get("torque_y", 0.0),
                data.get("torque_z", 0.0),
                data.get("thrust_magnitude", 0.0),
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def insert_maneuver(self, data: Dict[str, Any]) -> int:
        """Log a completed maneuver.

        Parameters
        ----------
        data : dict
            Keys: ``mission_id``, ``phase``, ``start_time``, ``end_time``,
            ``delta_v``, ``fuel_consumed``, ``maneuver_type``.

        Returns
        -------
        int
            Row ID.
        """
        cursor = self._conn.execute(
            "INSERT INTO maneuvers "
            "(mission_id, phase, start_time, end_time, delta_v, fuel_consumed, "
            "maneuver_type) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                data["mission_id"],
                data["phase"],
                data["start_time"],
                data["end_time"],
                data["delta_v"],
                data["fuel_consumed"],
                data["maneuver_type"],
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def insert_anomaly(self, data: Dict[str, Any]) -> int:
        """Log an anomaly detection event.

        Parameters
        ----------
        data : dict
            Keys: ``mission_id``, ``timestamp``, ``sensor_name``,
            ``anomaly_score``, ``is_anomaly``, ``description`` (optional).

        Returns
        -------
        int
            Row ID.
        """
        cursor = self._conn.execute(
            "INSERT INTO anomaly_log "
            "(mission_id, timestamp, sensor_name, anomaly_score, is_anomaly, "
            "description) VALUES (?, ?, ?, ?, ?, ?)",
            (
                data["mission_id"],
                data["timestamp"],
                data["sensor_name"],
                data["anomaly_score"],
                data.get("is_anomaly", 0),
                data.get("description"),
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def insert_phase_transition(self, data: Dict[str, Any]) -> int:
        """Log a mission phase transition.

        Parameters
        ----------
        data : dict
            Keys: ``mission_id``, ``from_phase``, ``to_phase``,
            ``transition_time``, ``reason`` (optional).

        Returns
        -------
        int
            Row ID.
        """
        cursor = self._conn.execute(
            "INSERT INTO phase_transitions "
            "(mission_id, from_phase, to_phase, transition_time, reason) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                data["mission_id"],
                data["from_phase"],
                data["to_phase"],
                data["transition_time"],
                data.get("reason"),
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def insert_monte_carlo_result(self, data: Dict[str, Any]) -> int:
        """Insert a Monte Carlo run result.

        Parameters
        ----------
        data : dict
            Keys: ``run_id``, ``seed``, ``total_delta_v``, ``total_fuel``,
            ``max_pointing_error``, ``landing_lat``, ``landing_lon``,
            ``landing_error_m``, ``success``.

        Returns
        -------
        int
            Row ID (run_id).
        """
        cursor = self._conn.execute(
            "INSERT INTO monte_carlo_results "
            "(run_id, seed, total_delta_v, total_fuel, max_pointing_error, "
            "landing_lat, landing_lon, landing_error_m, success) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                data["run_id"],
                data["seed"],
                data["total_delta_v"],
                data["total_fuel"],
                data["max_pointing_error"],
                data.get("landing_lat"),
                data.get("landing_lon"),
                data.get("landing_error_m"),
                data.get("success", 1),
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    # =========================================================================
    # Query Operations
    # =========================================================================

    def query_telemetry(
        self,
        start_time: float,
        end_time: float,
        mission_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """Query telemetry records within a time range.

        Parameters
        ----------
        start_time : float
            Inclusive lower bound on timestamp (mission elapsed seconds).
        end_time : float
            Inclusive upper bound on timestamp.
        mission_id : int, optional
            Filter to a specific mission. If None, returns all missions.

        Returns
        -------
        pd.DataFrame
            Telemetry records sorted by timestamp.
        """
        if mission_id is not None:
            query = (
                "SELECT * FROM telemetry "
                "WHERE timestamp >= ? AND timestamp <= ? AND mission_id = ? "
                "ORDER BY timestamp"
            )
            params: tuple = (start_time, end_time, mission_id)
        else:
            query = (
                "SELECT * FROM telemetry "
                "WHERE timestamp >= ? AND timestamp <= ? "
                "ORDER BY timestamp"
            )
            params = (start_time, end_time)

        return pd.read_sql_query(query, self._conn, params=params)

    def query_phase_data(
        self,
        phase_name: str,
        mission_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get all telemetry records for a specific mission phase.

        Parameters
        ----------
        phase_name : str
            Name of the mission phase (e.g., ``"stage1_ascent"``).
        mission_id : int, optional
            Filter to a specific mission.

        Returns
        -------
        pd.DataFrame
            Telemetry records for the specified phase, sorted by timestamp.
        """
        if mission_id is not None:
            query = (
                "SELECT * FROM telemetry "
                "WHERE phase = ? AND mission_id = ? "
                "ORDER BY timestamp"
            )
            params: tuple = (phase_name, mission_id)
        else:
            query = (
                "SELECT * FROM telemetry "
                "WHERE phase = ? "
                "ORDER BY timestamp"
            )
            params = (phase_name,)

        return pd.read_sql_query(query, self._conn, params=params)

    def query_sensor_readings(
        self,
        sensor_type: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        mission_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """Query sensor readings with optional filters.

        Parameters
        ----------
        sensor_type : str, optional
            Filter by sensor type (e.g., ``"imu"``, ``"star_tracker"``).
        start_time : float, optional
            Inclusive lower time bound.
        end_time : float, optional
            Inclusive upper time bound.
        mission_id : int, optional
            Filter by mission.

        Returns
        -------
        pd.DataFrame
        """
        conditions = []
        params = []

        if sensor_type is not None:
            conditions.append("sensor_type = ?")
            params.append(sensor_type)
        if start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(end_time)
        if mission_id is not None:
            conditions.append("mission_id = ?")
            params.append(mission_id)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM sensor_readings WHERE {where_clause} ORDER BY timestamp"

        return pd.read_sql_query(query, self._conn, params=params)

    def query_maneuvers(
        self,
        mission_id: Optional[int] = None,
        phase: Optional[str] = None,
    ) -> pd.DataFrame:
        """Query maneuver records.

        Parameters
        ----------
        mission_id : int, optional
        phase : str, optional

        Returns
        -------
        pd.DataFrame
        """
        conditions = []
        params = []

        if mission_id is not None:
            conditions.append("mission_id = ?")
            params.append(mission_id)
        if phase is not None:
            conditions.append("phase = ?")
            params.append(phase)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM maneuvers WHERE {where_clause} ORDER BY start_time"

        return pd.read_sql_query(query, self._conn, params=params)

    def query_anomalies(
        self,
        mission_id: Optional[int] = None,
        only_confirmed: bool = False,
    ) -> pd.DataFrame:
        """Query anomaly log entries.

        Parameters
        ----------
        mission_id : int, optional
        only_confirmed : bool
            If True, return only rows where ``is_anomaly = 1``.

        Returns
        -------
        pd.DataFrame
        """
        conditions = []
        params = []

        if mission_id is not None:
            conditions.append("mission_id = ?")
            params.append(mission_id)
        if only_confirmed:
            conditions.append("is_anomaly = 1")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM anomaly_log WHERE {where_clause} ORDER BY timestamp"

        return pd.read_sql_query(query, self._conn, params=params)

    def get_mission_summary(self, mission_id: Optional[int] = None) -> dict:
        """Get aggregated mission statistics from the v_mission_summary view.

        Parameters
        ----------
        mission_id : int, optional
            Specific mission to summarize. If None, returns summary for
            the first (primary) mission.

        Returns
        -------
        dict
            Keys include: ``mission_name``, ``status``, ``duration_s``,
            ``telemetry_count``, ``maneuver_count``, ``total_delta_v``,
            ``total_fuel_consumed``, ``min_fuel_remaining``, ``anomaly_count``,
            ``phase_transition_count``, ``sensor_reading_count``,
            ``control_command_count``.
        """
        if mission_id is not None:
            row = self._conn.execute(
                "SELECT * FROM v_mission_summary WHERE mission_id = ?",
                (mission_id,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM v_mission_summary ORDER BY mission_id LIMIT 1"
            ).fetchone()

        if row is None:
            return {}
        return dict(row)

    def get_phase_durations(self, mission_id: Optional[int] = None) -> pd.DataFrame:
        """Get phase durations from the v_phase_duration view.

        Parameters
        ----------
        mission_id : int, optional

        Returns
        -------
        pd.DataFrame
            Columns: ``phase_name``, ``phase_start_time``, ``phase_end_time``,
            ``duration_s``, ``entry_reason``.
        """
        if mission_id is not None:
            query = "SELECT * FROM v_phase_duration WHERE mission_id = ?"
            return pd.read_sql_query(query, self._conn, params=(mission_id,))
        else:
            return pd.read_sql_query("SELECT * FROM v_phase_duration", self._conn)

    def get_monte_carlo_results(self) -> pd.DataFrame:
        """Retrieve all Monte Carlo simulation results.

        Returns
        -------
        pd.DataFrame
            All columns from ``monte_carlo_results`` table, sorted by run_id.
        """
        return pd.read_sql_query(
            "SELECT * FROM monte_carlo_results ORDER BY run_id", self._conn
        )

    # =========================================================================
    # Export Operations
    # =========================================================================

    def export_to_csv(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """Export all database tables to CSV files using pandas.

        Creates one CSV file per table in the specified output directory.

        Parameters
        ----------
        output_dir : str or Path
            Directory where CSV files will be written. Created if needed.

        Returns
        -------
        dict
            Mapping of table name to the absolute path of the exported CSV.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tables = [
            "mission_info",
            "telemetry",
            "sensor_readings",
            "control_commands",
            "maneuvers",
            "anomaly_log",
            "phase_transitions",
            "monte_carlo_results",
        ]

        exported = {}
        for table in tables:
            try:
                df = pd.read_sql_query(f"SELECT * FROM {table}", self._conn)
                csv_path = output_dir / f"{table}.csv"
                df.to_csv(csv_path, index=False)
                exported[table] = str(csv_path.resolve())
            except Exception as e:
                print(f"Warning: Could not export table '{table}': {e}")

        return exported

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _count_rows(self, table: str) -> int:
        """Return the current row count for a table."""
        cursor = self._conn.execute(f"SELECT COUNT(*) FROM {table}")
        return cursor.fetchone()[0]

    def get_table_sizes(self) -> Dict[str, int]:
        """Return row counts for all tables.

        Returns
        -------
        dict
            Mapping of table name to row count.
        """
        tables = [
            "mission_info", "telemetry", "sensor_readings",
            "control_commands", "maneuvers", "anomaly_log",
            "phase_transitions", "monte_carlo_results",
        ]
        return {t: self._count_rows(t) for t in tables}

    def execute_raw(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        """Execute an arbitrary SQL query and return results as a DataFrame.

        Parameters
        ----------
        sql : str
            SQL query string.
        params : tuple
            Query parameters for placeholder substitution.

        Returns
        -------
        pd.DataFrame
        """
        return pd.read_sql_query(sql, self._conn, params=params)

    def close(self) -> None:
        """Close the database connection.

        It is good practice to call this when the simulation is complete
        to ensure all data is flushed to disk.
        """
        if self._conn:
            self._conn.commit()
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "TelemetryDatabase":
        """Support usage as a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ensure the connection is closed on context manager exit."""
        self.close()

    def __repr__(self) -> str:
        sizes = self.get_table_sizes() if self._conn else {}
        total = sum(sizes.values())
        return f"TelemetryDatabase(path='{self.db_path}', total_rows={total})"
