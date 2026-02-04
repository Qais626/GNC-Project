#!/usr/bin/env python3
"""
===============================================================================
GNC MISSION SIMULATION - MAIN ENTRY POINT
===============================================================================
Miami -> Moon (2 orbits) -> Jupiter (3 orbits) -> Miami

This is the master script that orchestrates the entire GNC simulation.
It initializes all subsystems, runs the mission simulation, generates
trade studies, and produces output plots and data.

USAGE:
    python main.py                    # Full simulation
    python main.py --phase ascent     # Single phase only
    python main.py --monte-carlo 50   # Monte Carlo with 50 runs
    python main.py --trade-study      # Trade studies only
    python main.py --benchmark        # Performance benchmarks only
    python main.py --quick            # Quick test run (reduced fidelity)

OUTPUTS:
    output/plots/          - Mission trajectory and state plots
    output/trade_studies/  - Propulsion and attitude trade study plots
    output/data/           - CSV telemetry data
    output/matlab/         - Data files for MATLAB post-processing
    output/mission.db      - SQLite telemetry database

DEPENDENCIES:
    numpy, scipy, matplotlib, pandas, pyyaml
    Install: pip install numpy scipy matplotlib pandas pyyaml

===============================================================================
"""

import sys
import os
import argparse
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Path setup: ensure all project modules are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from core.constants import (
    EARTH_MU, EARTH_RADIUS, MOON_MU, MOON_SMA, JUPITER_MU,
    MIAMI_LATITUDE, MIAMI_LONGITUDE, DEG2RAD, RAD2DEG, LEO_ALTITUDE,
    LEO_RADIUS, LEO_VELOCITY, PI, TWO_PI
)
from core.quaternion import Quaternion
from core.frames import geodetic_to_ecef, ecef_to_eci, eci_to_ecef
from core.data_structures import EventPriorityQueue, StateHistory

from dynamics.environment import (
    ExponentialAtmosphere, GravityField, SolarRadiationPressure,
    RadiationEnvironment, ThirdBodyPerturbation
)
from dynamics.spacecraft import Spacecraft
from dynamics.orbital_mechanics import OrbitalMechanics, OrbitalState
from dynamics.attitude_dynamics import AttitudeDynamics, AttitudeConfig, FlexMode, MomentumExchangeDevice
from dynamics.launch_vehicle import LaunchVehicle

from guidance.mission_planner import MissionPlanner, MissionPhase
from guidance.trajectory_opt import TrajectoryOptimizer
from guidance.maneuver_planner import ManeuverPlanner

from navigation.sensors import IMU, StarTracker, SunSensor, GPSReceiver
from navigation.signal_model import SignalModel
from navigation.ekf import EKF
from navigation.ukf import UKF

from control.actuators import ReactionWheelArray, ThrusterArray
from control.attitude_control import AttitudeControlSystem
from control.orbit_control import OrbitController
from control.optimal_control import LQGController

from simulation.sim_engine import SimulationEngine
from simulation.monte_carlo import MonteCarloSim
from simulation.bayesian import BayesianEstimator
from simulation.sil_interface import SILInterface

from autonomy.anomaly_detection import SensorAnomalyDetector
from autonomy.auto_trajectory import TrajectoryCorrector, SimpleTrajectoryEnv
from autonomy.attitude_predictor import AttitudePredictor

from trade_studies.propulsion_trade import PropulsionTradeStudy
from trade_studies.attitude_trade import AttitudeTradeStudy
from trade_studies.plot_utils import PlotStyle, plot_trajectory_3d, plot_state_history

from performance.benchmarks import Benchmark
from performance.cache_optimizer import CacheAnalysis
from performance.parallel import VectorizedOps

from database.telemetry_db import TelemetryDatabase

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('output/simulation.log', mode='w')
    ]
)
logger = logging.getLogger('GNC_MAIN')


def load_config(config_path: str = None) -> dict:
    """
    Load mission configuration from YAML file.

    Args:
        config_path: Path to YAML config. Defaults to config/mission_config.yaml

    Returns:
        Dictionary of mission configuration parameters
    """
    if config_path is None:
        config_path = str(PROJECT_ROOT.parent.parent / 'config' / 'mission_config.yaml')

    logger.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Mission: {config['mission']['name']}")
    return config


def setup_output_directories():
    """Create all output directories if they don't exist."""
    dirs = [
        'output/plots', 'output/trade_studies', 'output/matlab',
        'output/data', 'output/monte_carlo', 'output/benchmarks',
        'output/autonomy'
    ]
    base = PROJECT_ROOT.parent.parent
    for d in dirs:
        (base / d).mkdir(parents=True, exist_ok=True)
    logger.info("Output directories ready")
    return str(base / 'output')


def initialize_subsystems(config: dict) -> dict:
    """
    Initialize all GNC subsystems from configuration.

    This function creates instances of every subsystem needed for the
    simulation: dynamics models, sensors, actuators, controllers, filters,
    guidance algorithms, and autonomy modules.

    Args:
        config: Mission configuration dictionary

    Returns:
        Dictionary of initialized subsystem instances
    """
    logger.info("=" * 60)
    logger.info("INITIALIZING SUBSYSTEMS")
    logger.info("=" * 60)

    subsystems = {}

    # --- Spacecraft ---
    logger.info("Creating spacecraft model...")
    subsystems['spacecraft'] = Spacecraft(config['spacecraft'])

    # --- Launch Vehicle ---
    logger.info("Creating launch vehicle model...")
    subsystems['launch_vehicle'] = LaunchVehicle(config['launch_vehicle'])

    # --- Environment Models ---
    logger.info("Creating environment models...")
    subsystems['atmosphere'] = ExponentialAtmosphere()
    subsystems['gravity'] = GravityField(
        mu=EARTH_MU, body_radius=EARTH_RADIUS,
        j2=config['celestial_bodies']['earth']['J2'],
        j3=config['celestial_bodies']['earth']['J3']
    )
    subsystems['srp'] = SolarRadiationPressure()
    subsystems['radiation'] = RadiationEnvironment()
    subsystems['third_body'] = ThirdBodyPerturbation(mu_third_body=MOON_MU)

    # --- Orbital Mechanics ---
    logger.info("Creating orbital mechanics engine...")
    subsystems['orbital_mech'] = OrbitalMechanics()

    # --- Attitude Dynamics ---
    logger.info("Creating attitude dynamics engine...")
    sc_cfg = config['spacecraft']
    inertia_cfg = sc_cfg['inertia_tensor']
    inertia_matrix = np.array([
        [inertia_cfg['Ixx'], inertia_cfg['Ixy'], inertia_cfg['Ixz']],
        [inertia_cfg['Ixy'], inertia_cfg['Iyy'], inertia_cfg['Iyz']],
        [inertia_cfg['Ixz'], inertia_cfg['Iyz'], inertia_cfg['Izz']],
    ])
    flex_freqs = sc_cfg['structural']['flex_mode_freq_hz']
    flex_damps = sc_cfg['structural']['flex_damping_ratios']
    flex_modes = [
        FlexMode(
            frequency_hz=f,
            damping_ratio=d,
            coupling_vector=np.array([0.1, 0.1, 0.05])
        )
        for f, d in zip(flex_freqs, flex_damps)
    ]
    att_config = AttitudeConfig(
        inertia=inertia_matrix,
        inertia_uncertainty=sc_cfg['structural']['inertia_uncertainty_percent'] / 100.0,
        cg_offset=np.array(sc_cfg['structural']['cg_offset_m']),
        flex_modes=flex_modes,
    )
    subsystems['attitude_dyn'] = AttitudeDynamics(att_config)

    # --- Sensors ---
    logger.info("Creating sensor models...")
    subsystems['imu'] = IMU(config['spacecraft']['sensors']['imu'])
    subsystems['star_tracker'] = StarTracker(config['spacecraft']['sensors']['star_tracker'])
    subsystems['sun_sensor'] = SunSensor(config['spacecraft']['sensors']['sun_sensor'])
    subsystems['gps'] = GPSReceiver(config['spacecraft']['sensors']['gps'])
    subsystems['signal_model'] = SignalModel()

    # --- Navigation Filters ---
    logger.info("Creating navigation filters (EKF + UKF)...")
    # Initial state: position at Miami, zero velocity, identity attitude
    x0 = np.zeros(15)
    miami_ecef = geodetic_to_ecef(MIAMI_LATITUDE, MIAMI_LONGITUDE, 0.0)
    x0[0:3] = ecef_to_eci(miami_ecef, 0.0)  # Initial ECI position

    P0 = np.diag([
        100.0, 100.0, 100.0,       # Position uncertainty (m)
        1.0, 1.0, 1.0,             # Velocity uncertainty (m/s)
        0.01, 0.01, 0.01,          # Attitude error (rad)
        1e-4, 1e-4, 1e-4,          # Gyro bias (rad/s)
        1e-3, 1e-3, 1e-3           # Accel bias (m/s^2)
    ])

    Q = np.diag([
        0.01, 0.01, 0.01,          # Position process noise
        0.001, 0.001, 0.001,       # Velocity process noise
        1e-6, 1e-6, 1e-6,          # Attitude process noise
        1e-8, 1e-8, 1e-8,          # Gyro bias drift
        1e-7, 1e-7, 1e-7           # Accel bias drift
    ])

    subsystems['ekf'] = EKF(x0=x0, P0=P0, Q=Q, dt=1.0)
    subsystems['ukf'] = UKF(
        x0=x0, P0=P0, Q=Q,
        R=np.eye(6) * 0.01,        # Default measurement noise
        alpha=1e-3, beta=2.0, kappa=0.0
    )

    # --- Actuators ---
    logger.info("Creating actuator models...")
    rw_cfg = config['spacecraft']['actuators']['reaction_wheels']
    subsystems['reaction_wheels'] = ReactionWheelArray(
        max_torque=rw_cfg['max_torque_Nm'],
        max_momentum=rw_cfg['max_momentum_Nms'],
        wheel_inertia=rw_cfg['wheel_inertia_kgm2'],
    )
    thr_cfg = config['spacecraft']['actuators']['thrusters']
    subsystems['thrusters'] = ThrusterArray(
        nominal_thrust=thr_cfg['max_thrust_N'],
        isp=config['spacecraft']['propulsion']['rcs']['isp_s'],
    )

    # --- Controllers ---
    logger.info("Creating control systems...")
    ctrl_att = config['control']['attitude']
    att_ctrl_cfg = {
        'inertia': inertia_matrix.tolist(),
        'pid_kp': ctrl_att['pid']['kp'],
        'pid_ki': ctrl_att['pid']['ki'],
        'pid_kd': ctrl_att['pid']['kd'],
        'lqr_Q_diag': ctrl_att['lqr']['Q_diag'],
        'lqr_R_diag': ctrl_att['lqr']['R_diag'],
        'smc_lambda': ctrl_att['sliding_mode']['lambda_val'],
        'smc_eta': max(ctrl_att['sliding_mode']['eta']),
        'smc_boundary_layer': ctrl_att['sliding_mode']['boundary_layer'],
    }
    subsystems['attitude_control'] = AttitudeControlSystem(att_ctrl_cfg)
    subsystems['orbit_control'] = OrbitController()

    # --- Guidance ---
    logger.info("Creating guidance system...")
    subsystems['mission_planner'] = MissionPlanner()
    subsystems['trajectory_opt'] = TrajectoryOptimizer()
    subsystems['maneuver_planner'] = ManeuverPlanner()

    # --- Autonomy ---
    logger.info("Creating autonomy modules...")
    subsystems['anomaly_detector'] = SensorAnomalyDetector(n_sensors=6)
    subsystems['trajectory_corrector'] = TrajectoryCorrector()
    subsystems['attitude_predictor'] = AttitudePredictor()

    # --- Data Structures ---
    subsystems['event_queue'] = EventPriorityQueue()
    subsystems['state_history'] = StateHistory(capacity=100000, state_dim=13)

    logger.info("All subsystems initialized successfully")
    return subsystems


def run_simulation(config: dict, output_dir: str, quick_mode: bool = False):
    """
    Run the full mission simulation.

    This is the main simulation function that:
    1. Initializes all subsystems
    2. Runs the simulation engine through all mission phases
    3. Logs telemetry to database and CSV
    4. Generates mission plots

    Args:
        config: Mission configuration
        output_dir: Path to output directory
        quick_mode: If True, use reduced fidelity for faster execution
    """
    logger.info("=" * 60)
    logger.info("STARTING MISSION SIMULATION")
    logger.info(f"Mission: {config['mission']['name']}")
    logger.info(f"Quick mode: {quick_mode}")
    logger.info("=" * 60)

    # Adjust simulation parameters for quick mode
    if quick_mode:
        config['simulation']['dt_s'] = 10.0       # Larger time step
        config['simulation']['dt_fine_s'] = 0.1
        config['simulation']['max_time_s'] = 50000.0  # Limit to ~14 hours sim time
        logger.info("Quick mode: dt=10s, max_time=50000s, reduced fidelity")

    # Initialize subsystems
    subsystems = initialize_subsystems(config)

    # Create simulation engine
    sim = SimulationEngine(config, subsystems)
    sim.initialize()

    # Run simulation
    start_time = time.time()
    logger.info("Beginning simulation loop...")

    try:
        sim.run()
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Simulation error: {e}", exc_info=True)

    elapsed = time.time() - start_time
    logger.info(f"Simulation completed in {elapsed:.1f} seconds")

    # --- Retrieve and save telemetry ---
    import pandas as pd

    telemetry = sim.get_telemetry()
    logger.info(f"Telemetry: {len(telemetry)} records")

    # Save to CSV
    csv_path = os.path.join(output_dir, 'data', 'telemetry.csv')
    telemetry.to_csv(csv_path, index=False)
    logger.info(f"Telemetry saved to {csv_path}")

    # Save to SQLite database
    db_path = os.path.join(output_dir, 'mission.db')
    db = TelemetryDatabase(db_path)
    db.insert_batch_telemetry(telemetry)
    db.close()
    logger.info(f"Telemetry saved to database: {db_path}")

    # --- Generate plots ---
    logger.info("Generating mission plots...")
    plot_dir = os.path.join(output_dir, 'plots')

    try:
        # 3D trajectory
        if 'pos_x' in telemetry.columns:
            positions = telemetry[['pos_x', 'pos_y', 'pos_z']].values
            plot_trajectory_3d(
                [positions], ['Mission Trajectory'],
                'Full Mission Trajectory',
                os.path.join(plot_dir, 'trajectory_3d.png')
            )

        # State histories
        if 'time' in telemetry.columns:
            times = telemetry['time'].values

            # Position magnitude
            if 'pos_x' in telemetry.columns:
                r_mag = np.sqrt(
                    telemetry['pos_x']**2 +
                    telemetry['pos_y']**2 +
                    telemetry['pos_z']**2
                ).values
                plot_state_history(
                    times, [r_mag], ['|r| (m)'],
                    'Orbital Radius vs Time',
                    os.path.join(plot_dir, 'radius_history.png')
                )

            # Attitude quaternion
            quat_cols = ['quat_w', 'quat_x', 'quat_y', 'quat_z']
            if all(c in telemetry.columns for c in quat_cols):
                quat_data = [telemetry[c].values for c in quat_cols]
                plot_state_history(
                    times, quat_data, ['q_w', 'q_x', 'q_y', 'q_z'],
                    'Attitude Quaternion History',
                    os.path.join(plot_dir, 'quaternion_history.png')
                )

            # Pointing error
            if 'pointing_error_deg' in telemetry.columns:
                from trade_studies.plot_utils import plot_pointing_error
                plot_pointing_error(
                    times, telemetry['pointing_error_deg'].values,
                    0.1,  # requirement: 0.1 deg
                    'Pointing Error vs Time',
                    os.path.join(plot_dir, 'pointing_error.png')
                )

        logger.info(f"Plots saved to {plot_dir}")
    except Exception as e:
        logger.warning(f"Plot generation error: {e}")

    # --- Mission Summary ---
    summary = sim.get_mission_summary()
    logger.info("=" * 60)
    logger.info("MISSION SUMMARY")
    logger.info("=" * 60)
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")

    return sim, telemetry


def run_trade_studies(config: dict, output_dir: str):
    """
    Run all trade studies and generate comparison plots.

    Trade Study 1: Propulsion type comparison
        - Chemical bipropellant, solid, ion, nuclear thermal
        - Compare delta-V, transfer time, fuel mass, trajectory shape

    Trade Study 2: Attitude pointing impact
        - How pointing errors affect trajectory and fuel usage
        - Controller comparison (PID vs LQR vs Sliding Mode)
    """
    logger.info("=" * 60)
    logger.info("RUNNING TRADE STUDIES")
    logger.info("=" * 60)

    trade_dir = os.path.join(output_dir, 'trade_studies')

    # --- Propulsion Trade Study ---
    logger.info("Running propulsion trade study...")
    raw_opts = config['propulsion_trade'].get('options', [])
    prop_options = [
        {
            'name': o['name'],
            'isp': o.get('isp_s', o.get('isp', 300.0)),
            'thrust': o.get('thrust_N', o.get('thrust', 1000.0)),
            'engine_mass': o.get('mass_kg', o.get('engine_mass', 50.0)),
            'type': o.get('type', 'chemical'),
        }
        for o in raw_opts
    ] if raw_opts else None
    prop_trade = PropulsionTradeStudy(prop_options)
    prop_trade.run_and_plot(trade_dir)
    logger.info("Propulsion trade study complete")

    # --- Attitude Trade Study ---
    logger.info("Running attitude trade study...")
    att_trade = AttitudeTradeStudy(config)
    att_trade.run_and_plot(trade_dir)
    logger.info("Attitude trade study complete")


def run_monte_carlo(config: dict, output_dir: str, num_runs: int = 100):
    """
    Run Monte Carlo simulation campaign.

    Disperses key parameters (mass, thrust, Isp, initial conditions)
    across N runs and aggregates statistics on mission success,
    fuel usage, pointing accuracy, and landing accuracy.

    Args:
        config: Mission configuration
        output_dir: Output directory
        num_runs: Number of Monte Carlo runs
    """
    logger.info("=" * 60)
    logger.info(f"RUNNING MONTE CARLO ({num_runs} runs)")
    logger.info("=" * 60)

    mc_dir = os.path.join(output_dir, 'monte_carlo')

    mc = MonteCarloSim(
        base_config=config,
        num_runs=num_runs,
        seed=config['simulation']['rng_seed']
    )

    results = mc.run_all(num_workers=4)

    # Save results
    results.to_csv(os.path.join(mc_dir, 'monte_carlo_results.csv'), index=False)

    # Generate plots
    mc.plot_dispersions(mc_dir)
    mc.plot_trajectory_fan(mc_dir)
    mc.plot_landing_scatter(mc_dir)

    # Statistics
    stats = mc.compute_statistics()
    logger.info("Monte Carlo Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    logger.info(f"Success rate: {mc.get_success_rate():.1%}")


def run_benchmarks(output_dir: str):
    """
    Run performance benchmarks demonstrating optimization techniques.

    Compares vectorized vs loop operations, cache-friendly vs unfriendly
    data layouts, and parallel vs sequential computation.
    """
    logger.info("=" * 60)
    logger.info("RUNNING PERFORMANCE BENCHMARKS")
    logger.info("=" * 60)

    bench_dir = os.path.join(output_dir, 'benchmarks')

    # Computation benchmarks
    bench = Benchmark()
    bench.run_all_benchmarks(bench_dir)
    bench.generate_report(bench_dir)

    # Cache analysis
    cache = CacheAnalysis()
    cache.run_all_demonstrations(bench_dir)

    # Vectorized operations
    vec_ops = VectorizedOps()
    logger.info("Benchmarking vectorized quaternion multiply...")
    q1 = np.random.randn(10000, 4)
    q2 = np.random.randn(10000, 4)
    # Normalize
    q1 = q1 / np.linalg.norm(q1, axis=1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=1, keepdims=True)
    result = vec_ops.batch_quaternion_multiply(q1, q2)
    logger.info(f"Batch quaternion multiply: {len(result)} operations")

    logger.info("Benchmarks complete")


def run_autonomy_demo(config: dict, output_dir: str):
    """
    Demonstrate AI/ML autonomy features:
    - Sensor anomaly detection training and testing
    - RL-based trajectory correction
    - Neural network attitude prediction
    """
    logger.info("=" * 60)
    logger.info("RUNNING AUTONOMY DEMONSTRATIONS")
    logger.info("=" * 60)

    auto_dir = os.path.join(output_dir, 'autonomy')

    # --- Anomaly Detection ---
    logger.info("Training anomaly detector...")
    detector = SensorAnomalyDetector(n_sensors=6)

    # Generate synthetic nominal data
    nominal_data = np.random.randn(1000, 6) * 0.1  # Normal sensor readings
    detector.train(nominal_data, epochs=50)

    # Test with nominal and anomalous data
    normal_reading = np.random.randn(6) * 0.1
    is_anomaly, score = detector.detect(normal_reading)
    logger.info(f"Normal reading - Anomaly: {is_anomaly}, Score: {score:.4f}")

    anomalous_reading = np.random.randn(6) * 5.0  # Way off nominal
    is_anomaly, score = detector.detect(anomalous_reading)
    logger.info(f"Anomalous reading - Anomaly: {is_anomaly}, Score: {score:.4f}")

    # --- Trajectory Correction ---
    logger.info("Training trajectory corrector (RL)...")
    corrector = TrajectoryCorrector()
    traj_env = SimpleTrajectoryEnv()
    metrics = corrector.train(env=traj_env, num_episodes=500)
    logger.info(f"Training complete. Final reward: {metrics.episode_rewards[-1]:.2f}")

    # Test correction decision
    state = traj_env.reset()
    action = corrector.choose_action(state)
    logger.info(f"Correction action for initial state: {action}")

    # --- Attitude Prediction ---
    logger.info("Training attitude predictor (neural net)...")
    predictor = AttitudePredictor(input_dim=10, prediction_horizon=10)
    X_train, y_train = predictor.generate_training_data(num_samples=5000)
    predictor.train(X_train, y_train, epochs=100)

    logger.info("Autonomy demonstrations complete")


def main():
    """
    Main entry point. Parses command line arguments and runs
    the requested simulation mode(s).
    """
    parser = argparse.ArgumentParser(
        description='GNC Mission Simulation: Miami -> Moon -> Jupiter -> Miami',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     Full simulation
  python main.py --quick             Quick test run
  python main.py --trade-study       Trade studies only
  python main.py --monte-carlo 50    Monte Carlo (50 runs)
  python main.py --benchmark         Performance benchmarks
  python main.py --autonomy          AI/ML demonstrations
  python main.py --all               Everything
        """
    )

    parser.add_argument('--config', type=str, default=None,
                        help='Path to mission config YAML')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (reduced fidelity)')
    parser.add_argument('--phase', type=str, default=None,
                        help='Run only a specific mission phase')
    parser.add_argument('--trade-study', action='store_true',
                        help='Run trade studies only')
    parser.add_argument('--monte-carlo', type=int, default=0,
                        help='Run Monte Carlo with N runs')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run performance benchmarks')
    parser.add_argument('--autonomy', action='store_true',
                        help='Run autonomy demonstrations')
    parser.add_argument('--all', action='store_true',
                        help='Run everything')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Print banner
    print("=" * 70)
    print("  GNC MISSION SIMULATION")
    print("  Miami, FL -> Moon (2 orbits) -> Jupiter (3 orbits) -> Miami, FL")
    print("=" * 70)
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Random seed: {args.seed}")
    print("=" * 70)

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Load configuration
    config = load_config(args.config)

    # Create output directories
    output_dir = setup_output_directories()

    # Determine what to run
    run_sim = not (args.trade_study or args.benchmark or args.autonomy)
    if args.all:
        run_sim = True
        args.trade_study = True
        args.monte_carlo = args.monte_carlo or 50
        args.benchmark = True
        args.autonomy = True

    mission_start = time.time()

    # --- Run requested modes ---
    if run_sim:
        sim, telemetry = run_simulation(config, output_dir, quick_mode=args.quick)

    if args.trade_study or args.all:
        run_trade_studies(config, output_dir)

    if args.monte_carlo > 0:
        run_monte_carlo(config, output_dir, num_runs=args.monte_carlo)

    if args.benchmark or args.all:
        run_benchmarks(output_dir)

    if args.autonomy or args.all:
        run_autonomy_demo(config, output_dir)

    # --- Final Summary ---
    total_time = time.time() - mission_start
    print("\n" + "=" * 70)
    print("  SIMULATION COMPLETE")
    print(f"  Total wall time: {total_time:.1f} seconds")
    print(f"  Outputs saved to: {output_dir}")
    print("=" * 70)

    # List output files
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            rel_path = os.path.relpath(os.path.join(root, f), output_dir)
            print(f"    {rel_path}")

    print("=" * 70)
    print("  Bismillah - Mission Complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
