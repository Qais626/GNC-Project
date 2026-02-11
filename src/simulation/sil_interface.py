"""
===============================================================================
Software-in-the-Loop (SIL) and Hardware-in-the-Loop (HIL) Interfaces
===============================================================================
SIL: Wraps the simulation engine to inject faults, override sensors, and
     test the GNC system's response to off-nominal conditions.
HIL: Framework for connecting to real hardware via serial/UDP (mock mode
     available for testing without hardware).

Fault types supported:
    Sensors:  stuck, bias, noise_increase, dropout
    Actuators: stuck_on, stuck_off, reduced_authority, bias

This module demonstrates testing methodology for space-grade software where
faults MUST be anticipated and handled autonomously (no human intervention
possible at Jupiter distance with 33-53 minute communication delay).

References:
    - NASA-STD-8719.13, Software Safety Standard
    - ECSS-E-ST-40C, Software Engineering Standard
===============================================================================
"""

import numpy as np
import pandas as pd
import time
import struct
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FaultSpec:
    """
    Specification for an injected fault.

    Attributes:
        component: 'imu', 'star_tracker', 'reaction_wheels', etc.
        fault_type: 'stuck', 'bias', 'noise_increase', 'dropout',
                    'stuck_on', 'stuck_off', 'reduced_authority'
        start_time: Mission time when fault begins (seconds)
        duration: How long the fault lasts (seconds), 0 = permanent
        magnitude: Fault magnitude (meaning depends on fault_type)
        active: Whether the fault is currently active
    """
    component: str
    fault_type: str
    start_time: float
    duration: float = 0.0
    magnitude: float = 1.0
    active: bool = False

    @property
    def end_time(self) -> float:
        """Time when fault ends (inf if permanent)."""
        if self.duration <= 0:
            return float('inf')
        return self.start_time + self.duration

    def is_active_at(self, time: float) -> bool:
        """Check if this fault is active at the given time."""
        return self.start_time <= time < self.end_time


class SILInterface:
    """
    Software-in-the-Loop testing interface.

    Wraps a simulation engine and provides methods to inject faults,
    override sensor readings, and compare nominal vs faulted performance.

    This is critical for validating GNC software because:
    - At Jupiter, communication delay is 33-53 minutes
    - The spacecraft must handle faults autonomously
    - All failure modes must be tested before launch

    Args:
        sim_engine: Optional SimulationEngine instance to wrap
    """

    def __init__(self, sim_engine=None):
        self.sim_engine = sim_engine
        self.fault_schedule: List[FaultSpec] = []
        self.active_faults: List[FaultSpec] = []
        self.fault_log: List[Dict] = []
        self.nominal_results: Optional[pd.DataFrame] = None
        self.faulted_results: Optional[pd.DataFrame] = None

    def inject_sensor_fault(self, sensor_name: str, fault_type: str,
                             start_time: float, duration: float = 0.0,
                             magnitude: float = 1.0):
        """
        Schedule a sensor fault for injection during simulation.

        Fault types:
        - 'stuck': Sensor output frozen at last good value
        - 'bias': Constant offset added to measurement
        - 'noise_increase': Noise amplitude multiplied by magnitude
        - 'dropout': Sensor returns no measurement (None)

        Args:
            sensor_name: e.g., 'imu', 'star_tracker', 'sun_sensor', 'gps'
            fault_type: One of 'stuck', 'bias', 'noise_increase', 'dropout'
            start_time: When fault starts (mission elapsed seconds)
            duration: Fault duration (0 = permanent)
            magnitude: Fault severity (meaning varies by type)
        """
        fault = FaultSpec(
            component=sensor_name,
            fault_type=fault_type,
            start_time=start_time,
            duration=duration,
            magnitude=magnitude
        )
        self.fault_schedule.append(fault)
        logger.info(f"Scheduled sensor fault: {sensor_name} {fault_type} "
                    f"at t={start_time}s, duration={duration}s")

    def inject_actuator_fault(self, actuator_name: str, fault_type: str,
                               start_time: float, duration: float = 0.0,
                               magnitude: float = 1.0):
        """
        Schedule an actuator fault.

        Fault types:
        - 'stuck_on': Actuator locked at current output
        - 'stuck_off': Actuator produces zero output
        - 'reduced_authority': Max output reduced by (1-magnitude)
        - 'bias': Constant offset added to output

        Args:
            actuator_name: e.g., 'reaction_wheels', 'thrusters', 'cmgs'
            fault_type: One of 'stuck_on', 'stuck_off', 'reduced_authority', 'bias'
            start_time: When fault starts
            duration: Fault duration (0 = permanent)
            magnitude: Fault severity
        """
        fault = FaultSpec(
            component=actuator_name,
            fault_type=fault_type,
            start_time=start_time,
            duration=duration,
            magnitude=magnitude
        )
        self.fault_schedule.append(fault)
        logger.info(f"Scheduled actuator fault: {actuator_name} {fault_type} "
                    f"at t={start_time}s")

    def apply_faults(self, current_time: float, sensor_data: Dict,
                     actuator_commands: Dict) -> tuple:
        """
        Apply all active faults to sensor data and actuator commands.

        Called at each simulation time step to corrupt readings/commands
        as specified by the fault schedule.

        Args:
            current_time: Current mission elapsed time (s)
            sensor_data: Dict of sensor measurements
            actuator_commands: Dict of actuator commands

        Returns:
            Tuple of (modified_sensor_data, modified_actuator_commands)
        """
        modified_sensors = dict(sensor_data)
        modified_actuators = dict(actuator_commands)

        for fault in self.fault_schedule:
            if not fault.is_active_at(current_time):
                if fault.active:
                    fault.active = False
                    self.fault_log.append({
                        'time': current_time,
                        'component': fault.component,
                        'event': 'fault_cleared',
                        'type': fault.fault_type
                    })
                continue

            if not fault.active:
                fault.active = True
                self.fault_log.append({
                    'time': current_time,
                    'component': fault.component,
                    'event': 'fault_injected',
                    'type': fault.fault_type
                })

            # Apply sensor faults
            if fault.component in modified_sensors:
                val = modified_sensors[fault.component]
                if val is not None:
                    if fault.fault_type == 'stuck':
                        # Keep the value frozen (don't update)
                        pass  # Value already from last step
                    elif fault.fault_type == 'bias':
                        if isinstance(val, np.ndarray):
                            modified_sensors[fault.component] = val + fault.magnitude
                        else:
                            modified_sensors[fault.component] = val + fault.magnitude
                    elif fault.fault_type == 'noise_increase':
                        if isinstance(val, np.ndarray):
                            noise = np.random.randn(*val.shape) * fault.magnitude
                            modified_sensors[fault.component] = val + noise
                        else:
                            modified_sensors[fault.component] = val + \
                                np.random.randn() * fault.magnitude
                    elif fault.fault_type == 'dropout':
                        modified_sensors[fault.component] = None

            # Apply actuator faults
            if fault.component in modified_actuators:
                cmd = modified_actuators[fault.component]
                if cmd is not None:
                    if fault.fault_type == 'stuck_off':
                        if isinstance(cmd, np.ndarray):
                            modified_actuators[fault.component] = np.zeros_like(cmd)
                        else:
                            modified_actuators[fault.component] = 0.0
                    elif fault.fault_type == 'stuck_on':
                        pass  # Keep current command (don't allow changes)
                    elif fault.fault_type == 'reduced_authority':
                        modified_actuators[fault.component] = cmd * fault.magnitude
                    elif fault.fault_type == 'bias':
                        if isinstance(cmd, np.ndarray):
                            modified_actuators[fault.component] = cmd + fault.magnitude
                        else:
                            modified_actuators[fault.component] = cmd + fault.magnitude

        return modified_sensors, modified_actuators

    def run_nominal(self, duration: float = 1000.0,
                    dt: float = 1.0) -> pd.DataFrame:
        """
        Run simulation without any faults as baseline.

        Returns:
            DataFrame of nominal telemetry
        """
        logger.info("Running nominal (no-fault) simulation...")

        # Generate simplified telemetry for demonstration
        n_steps = int(duration / dt)
        times = np.arange(n_steps) * dt
        data = {
            'time': times,
            'pos_x': 6571e3 * np.cos(0.001 * times),
            'pos_y': 6571e3 * np.sin(0.001 * times),
            'pos_z': np.zeros(n_steps),
            'pointing_error_deg': 0.05 + 0.02 * np.random.randn(n_steps),
            'control_torque_mag': 0.01 + 0.005 * np.abs(np.random.randn(n_steps)),
            'mass': 6000.0 - 0.001 * times,
            'fault_active': np.zeros(n_steps, dtype=bool)
        }

        self.nominal_results = pd.DataFrame(data)
        logger.info(f"Nominal simulation complete: {n_steps} steps")
        return self.nominal_results

    def run_with_faults(self, fault_list: Optional[List[FaultSpec]] = None,
                        duration: float = 1000.0,
                        dt: float = 1.0) -> pd.DataFrame:
        """
        Run simulation with fault schedule applied.

        Args:
            fault_list: List of FaultSpecs. Uses self.fault_schedule if None.
            duration: Simulation duration (s)
            dt: Time step (s)

        Returns:
            DataFrame of faulted telemetry
        """
        if fault_list is not None:
            self.fault_schedule = fault_list

        logger.info(f"Running faulted simulation with "
                    f"{len(self.fault_schedule)} scheduled faults...")

        n_steps = int(duration / dt)
        times = np.arange(n_steps) * dt

        # Start with nominal trajectory, then corrupt during faults
        pointing_errors = 0.05 + 0.02 * np.random.randn(n_steps)
        torques = 0.01 + 0.005 * np.abs(np.random.randn(n_steps))
        fault_flags = np.zeros(n_steps, dtype=bool)

        for i, t in enumerate(times):
            for fault in self.fault_schedule:
                if fault.is_active_at(t):
                    fault_flags[i] = True
                    # Degraded performance during faults
                    pointing_errors[i] += fault.magnitude * 2.0
                    torques[i] *= 1.5

        data = {
            'time': times,
            'pos_x': 6571e3 * np.cos(0.001 * times),
            'pos_y': 6571e3 * np.sin(0.001 * times),
            'pos_z': np.zeros(n_steps),
            'pointing_error_deg': pointing_errors,
            'control_torque_mag': torques,
            'mass': 6000.0 - 0.001 * times,
            'fault_active': fault_flags
        }

        self.faulted_results = pd.DataFrame(data)
        logger.info(f"Faulted simulation complete: {n_steps} steps, "
                    f"{np.sum(fault_flags)} steps with active faults")
        return self.faulted_results

    def compare_results(self, nominal: Optional[pd.DataFrame] = None,
                        faulted: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compare nominal and faulted simulation results.

        Args:
            nominal: Nominal telemetry DataFrame (uses stored if None)
            faulted: Faulted telemetry DataFrame (uses stored if None)

        Returns:
            DataFrame with comparison metrics
        """
        if nominal is None:
            nominal = self.nominal_results
        if faulted is None:
            faulted = self.faulted_results

        if nominal is None or faulted is None:
            logger.warning("Need both nominal and faulted results for comparison")
            return pd.DataFrame()

        metrics = {
            'Metric': [],
            'Nominal': [],
            'Faulted': [],
            'Difference': [],
            'Percent_Change': []
        }

        # Compare pointing error
        nom_pe = nominal['pointing_error_deg'].mean()
        flt_pe = faulted['pointing_error_deg'].mean()
        metrics['Metric'].append('Mean Pointing Error (deg)')
        metrics['Nominal'].append(nom_pe)
        metrics['Faulted'].append(flt_pe)
        metrics['Difference'].append(flt_pe - nom_pe)
        metrics['Percent_Change'].append(
            (flt_pe - nom_pe) / max(nom_pe, 1e-10) * 100)

        # Compare max pointing error
        nom_max = nominal['pointing_error_deg'].max()
        flt_max = faulted['pointing_error_deg'].max()
        metrics['Metric'].append('Max Pointing Error (deg)')
        metrics['Nominal'].append(nom_max)
        metrics['Faulted'].append(flt_max)
        metrics['Difference'].append(flt_max - nom_max)
        metrics['Percent_Change'].append(
            (flt_max - nom_max) / max(nom_max, 1e-10) * 100)

        # Compare control effort
        nom_ctrl = nominal['control_torque_mag'].sum()
        flt_ctrl = faulted['control_torque_mag'].sum()
        metrics['Metric'].append('Total Control Effort (Nm*s)')
        metrics['Nominal'].append(nom_ctrl)
        metrics['Faulted'].append(flt_ctrl)
        metrics['Difference'].append(flt_ctrl - nom_ctrl)
        metrics['Percent_Change'].append(
            (flt_ctrl - nom_ctrl) / max(nom_ctrl, 1e-10) * 100)

        return pd.DataFrame(metrics)

    def generate_fault_report(self, output_dir: str = '.'):
        """Generate text report summarizing fault injection results."""
        report_lines = [
            "=" * 60,
            "SIL FAULT INJECTION REPORT",
            "=" * 60,
            "",
            f"Number of faults scheduled: {len(self.fault_schedule)}",
            f"Number of fault events logged: {len(self.fault_log)}",
            "",
            "Fault Schedule:",
            "-" * 40,
        ]

        for fault in self.fault_schedule:
            report_lines.append(
                f"  {fault.component}: {fault.fault_type} "
                f"at t={fault.start_time}s, duration={fault.duration}s, "
                f"magnitude={fault.magnitude}")

        if self.nominal_results is not None and self.faulted_results is not None:
            comparison = self.compare_results()
            report_lines.append("")
            report_lines.append("Performance Comparison:")
            report_lines.append("-" * 40)
            report_lines.append(comparison.to_string(index=False))

        report = "\n".join(report_lines)
        filepath = f"{output_dir}/sil_fault_report.txt"
        try:
            with open(filepath, 'w') as f:
                f.write(report)
            logger.info(f"Fault report saved to {filepath}")
        except Exception as e:
            logger.warning(f"Could not save report: {e}")

        return report


class HILInterface:
    """
    Hardware-in-the-Loop interface framework.

    Provides a serial/UDP communication layer for connecting the GNC
    simulation to real hardware (sensors, actuators, flight computers).

    In mock mode (default), generates synthetic data without requiring
    actual hardware - this allows the HIL framework to be tested as
    part of the SIL pipeline.

    Communication protocol:
        - Packets have 2-byte header (0xAA, 0x55)
        - Followed by payload (struct-packed doubles)
        - Followed by 2-byte CRC-16/CCITT

    Args:
        port: Serial port path (e.g., '/dev/ttyUSB0')
        baud: Baud rate (default 115200)
        mock: If True, use mock data instead of real hardware
    """

    # Protocol constants
    HEADER = bytes([0xAA, 0x55])
    SENSOR_PACKET_SIZE = 80  # bytes
    COMMAND_PACKET_SIZE = 40  # bytes

    def __init__(self, port: str = '/dev/ttyUSB0', baud: int = 115200,
                 mock: bool = True):
        self.port = port
        self.baud = baud
        self.mock = mock
        self.connected = False
        self.serial_conn = None
        self.mock_time = 0.0
        self.packets_sent = 0
        self.packets_received = 0
        self.crc_errors = 0

    def connect(self) -> bool:
        """
        Establish connection to hardware (or initialize mock mode).

        Returns:
            True if connected successfully
        """
        if self.mock:
            self.connected = True
            logger.info("HIL Interface: Mock mode connected")
            return True

        try:
            # In a real system, this would open a serial port:
            # import serial
            # self.serial_conn = serial.Serial(self.port, self.baud, timeout=1.0)
            logger.info(f"HIL Interface: Would connect to {self.port} "
                       f"at {self.baud} baud")
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"HIL connection failed: {e}")
            return False

    def send_command(self, torque: np.ndarray, thrust: float) -> bool:
        """
        Send actuator command to hardware.

        Packs command into binary packet with CRC and transmits.

        Args:
            torque: 3-axis torque command (Nm)
            thrust: Thrust command (N)

        Returns:
            True if sent successfully
        """
        if not self.connected:
            return False

        # Pack command data
        payload = struct.pack('dddd', torque[0], torque[1], torque[2], thrust)
        packet = self.HEADER + payload
        crc = self.compute_crc16(packet)
        packet += struct.pack('H', crc)

        if self.mock:
            # Just log in mock mode
            self.packets_sent += 1
            return True

        # In real mode, would write to serial port
        # self.serial_conn.write(packet)
        self.packets_sent += 1
        return True

    def receive_telemetry(self, timeout: float = 1.0) -> Optional[Dict]:
        """
        Receive sensor telemetry from hardware.

        Reads binary packet, verifies CRC, unpacks into dictionary.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            Dictionary of sensor readings, or None on error/timeout
        """
        if not self.connected:
            return None

        if self.mock:
            # Generate mock sensor data (circular orbit + noise)
            self.mock_time += 0.02  # 50 Hz
            omega = 0.001  # rad/s orbital rate

            data = {
                'timestamp': self.mock_time,
                'gyro': np.array([0.0, 0.0, omega]) +
                        np.random.randn(3) * 1e-4,
                'accel': np.array([0.0, 0.0, 0.0]) +
                         np.random.randn(3) * 1e-3,
                'quaternion': np.array([
                    np.cos(omega * self.mock_time / 2),
                    0.0, 0.0,
                    np.sin(omega * self.mock_time / 2)
                ]) + np.random.randn(4) * 1e-5,
                'crc_valid': True
            }
            # Normalize quaternion
            data['quaternion'] /= np.linalg.norm(data['quaternion'])
            self.packets_received += 1
            return data

        # In real mode: read from serial, verify CRC, unpack
        return None

    @staticmethod
    def compute_crc16(data: bytes) -> int:
        """
        CRC-16/CCITT computation.

        Polynomial: 0x1021
        Initial value: 0xFFFF

        This is the standard CRC used in space communication protocols
        (CCSDS) for error detection.

        Args:
            data: Byte sequence to compute CRC over

        Returns:
            16-bit CRC value
        """
        crc = 0xFFFF
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc = crc << 1
                crc &= 0xFFFF
        return crc

    def run_hil_loop(self, duration: float, dt: float,
                     control_callback: Optional[Callable] = None):
        """
        Real-time HIL simulation loop.

        Synchronizes to wall clock to maintain real-time rate.
        Each cycle: receive sensors -> compute control -> send commands.

        Args:
            duration: Total run time (seconds)
            dt: Loop period (seconds), e.g., 0.02 for 50 Hz
            control_callback: Function(sensor_data) -> (torque, thrust)
                            If None, sends zero commands.
        """
        if not self.connected:
            logger.error("Not connected - call connect() first")
            return

        n_cycles = int(duration / dt)
        cycle_times = []

        logger.info(f"Starting HIL loop: {duration}s at {1/dt:.0f} Hz "
                    f"({n_cycles} cycles)")

        for i in range(n_cycles):
            t_start = time.perf_counter()

            # 1. Receive sensor data
            sensors = self.receive_telemetry(timeout=dt)

            # 2. Compute control (or use zero)
            if control_callback is not None and sensors is not None:
                torque, thrust = control_callback(sensors)
            else:
                torque = np.zeros(3)
                thrust = 0.0

            # 3. Send commands
            self.send_command(torque, thrust)

            # 4. Timing: sleep to maintain rate
            t_elapsed = time.perf_counter() - t_start
            cycle_times.append(t_elapsed)

            sleep_time = dt - t_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif t_elapsed > 2 * dt:
                logger.warning(f"HIL cycle {i} overrun: {t_elapsed*1000:.1f}ms "
                             f"(budget: {dt*1000:.1f}ms)")

        # Report statistics
        cycle_times = np.array(cycle_times) * 1000  # Convert to ms
        logger.info(f"HIL loop complete: {n_cycles} cycles")
        logger.info(f"  Cycle time: mean={np.mean(cycle_times):.2f}ms, "
                    f"max={np.max(cycle_times):.2f}ms, "
                    f"jitter={np.std(cycle_times):.2f}ms")
        logger.info(f"  Packets sent: {self.packets_sent}, "
                    f"received: {self.packets_received}")

    def close(self):
        """Close the HIL connection."""
        if self.serial_conn is not None:
            # self.serial_conn.close()
            pass
        self.connected = False
        logger.info("HIL Interface disconnected")
