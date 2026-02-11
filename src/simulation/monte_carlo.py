"""
===============================================================================
GNC PROJECT - Monte Carlo Simulation Framework
===============================================================================
Runs N dispersed simulations in parallel to assess mission robustness under
parameter uncertainty.  Uses multiprocessing for parallel execution and pandas
for result aggregation.

Monte Carlo analysis is the standard technique in mission design for answering
"What is the probability of mission success given uncertainties in ...?"  Each
run perturbs the simulation configuration by drawing from probability
distributions on uncertain parameters (mass, thrust, Isp, initial state,
sensor biases, etc.).  The ensemble of results is then processed to compute
success rates, percentile bounds, and worst-case scenarios.

Typical dispersions for a space mission:
    - Vehicle dry mass:         +/- 5%  (3-sigma Gaussian)
    - Thrust magnitude:         +/- 3%  (3-sigma Gaussian)
    - Specific impulse:         +/- 2%  (3-sigma Gaussian)
    - Initial attitude:         +/- 1 deg per axis (Gaussian)
    - Initial position:         +/- 100 m per axis (Gaussian)
    - Initial velocity:         +/- 0.1 m/s per axis (Gaussian)
    - Sensor biases:            scaled from nominal model

References
----------
    [1] Hanson, "Monte Carlo Techniques in Astronomy and Astrophysics",
        Annual Review, 2007.
    [2] NASA-STD-7009A, "Standard for Models and Simulations", 2023.
===============================================================================
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool
import copy
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _run_single_wrapper(args: Tuple[Dict[str, Any], int]) -> Dict[str, Any]:
    """
    Module-level wrapper for single-run execution.

    Required because multiprocessing Pool.map cannot pickle instance methods.
    This function creates a SimulationEngine, runs the simulation, and
    extracts summary metrics.

    Parameters
    ----------
    args : tuple of (config_dict, run_id)

    Returns
    -------
    dict
        Run summary including run_id, success flag, and all metrics.
    """
    config, run_id = args

    # Import here to avoid circular imports at module level
    from simulation.sim_engine import SimulationEngine

    result = {
        'run_id': run_id,
        'success': False,
        'total_delta_v': np.nan,
        'fuel_consumed': np.nan,
        'max_pointing_error': np.nan,
        'total_time': np.nan,
        'final_mass': np.nan,
        'final_altitude': np.nan,
        'final_velocity': np.nan,
        'num_phases_completed': 0,
        'landing_lat': np.nan,
        'landing_lon': np.nan,
        'error_message': '',
    }

    try:
        # Build subsystem dict (placeholder for full integration)
        subsystems = {}

        # Check if subsystem factories are provided in config
        if 'subsystem_factory' in config:
            subsystems = config['subsystem_factory'](config)

        engine = SimulationEngine(config, subsystems)
        engine.initialize()
        engine.run()

        summary = engine.get_mission_summary()
        result.update({
            'success': True,
            'total_delta_v': summary.get('total_delta_v', np.nan),
            'fuel_consumed': summary.get('fuel_consumed', np.nan),
            'max_pointing_error': summary.get('max_pointing_error', np.nan),
            'total_time': summary.get('total_time', np.nan),
            'final_mass': summary.get('final_mass', np.nan),
            'final_altitude': summary.get('final_altitude', np.nan),
            'final_velocity': summary.get('final_velocity', np.nan),
            'num_phases_completed': len(summary.get('phases_completed', [])),
        })

        # Extract landing coordinates if available
        final_pos = engine.state.get('position', np.zeros(3))
        r_mag = np.linalg.norm(final_pos)
        if r_mag > 0.0:
            result['landing_lat'] = np.degrees(np.arcsin(final_pos[2] / r_mag))
            result['landing_lon'] = np.degrees(
                np.arctan2(final_pos[1], final_pos[0])
            )

    except Exception as exc:
        result['error_message'] = str(exc)
        logger.warning("Run %d failed: %s", run_id, exc)

    return result


class MonteCarloSim:
    """
    Monte Carlo simulation framework for mission robustness assessment.

    Runs N dispersed simulations with randomized parameters drawn from
    specified distributions.  Results are aggregated into a pandas DataFrame
    for statistical analysis and visualization.

    Parameters
    ----------
    base_config : dict
        Nominal simulation configuration (passed to SimulationEngine).
        Should contain a 'monte_carlo' key with dispersion definitions::

            {
                'monte_carlo': {
                    'dispersions': {
                        'vehicle_mass':    {'mean': 50000, 'sigma_pct': 5.0},
                        'thrust':          {'mean': 100000, 'sigma_pct': 3.0},
                        'Isp':             {'mean': 300,    'sigma_pct': 2.0},
                        'attitude_sigma_deg': 1.0,
                        'position_sigma_m':   100.0,
                        'velocity_sigma_mps': 0.1,
                        'sensor_bias_scale':  1.0,
                    }
                }
            }
    num_runs : int
        Number of dispersed simulation runs.
    seed : int
        Master random seed for reproducibility.

    Attributes
    ----------
    results : pd.DataFrame or None
        Populated after run_all() completes.
    dispersed_configs : list of dict
        The dispersed configurations generated for each run.
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        num_runs: int = 100,
        seed: int = 42,
    ) -> None:
        self.base_config = base_config
        self.num_runs = num_runs
        self.seed = seed

        # Extract dispersion definitions
        mc_block = base_config.get('monte_carlo', {})
        self.dispersions = mc_block.get('dispersions', {})

        # Pre-generate all dispersed configs
        self._rng = np.random.RandomState(seed)
        self.dispersed_configs: List[Dict[str, Any]] = [
            self._disperse_config(base_config, run_id)
            for run_id in range(num_runs)
        ]

        # Results storage
        self.results: Optional[pd.DataFrame] = None

        logger.info(
            "MonteCarloSim initialized: %d runs, seed=%d, %d dispersions",
            num_runs, seed, len(self.dispersions),
        )

    # =========================================================================
    # DISPERSION GENERATION
    # =========================================================================

    def _disperse_config(
        self, config: Dict[str, Any], run_id: int
    ) -> Dict[str, Any]:
        """
        Create a dispersed copy of the simulation configuration.

        Deep-copies the base config and applies Gaussian perturbations to
        uncertain parameters based on the dispersion definitions.

        Parameters
        ----------
        config : dict
            Base configuration to perturb.
        run_id : int
            Run identifier (used for logging / tracing).

        Returns
        -------
        dict
            Dispersed configuration with 'run_id' and '_dispersion_values'
            keys added.
        """
        dispersed = copy.deepcopy(config)
        dispersion_record: Dict[str, float] = {'run_id': run_id}

        # --- Vehicle mass ---
        mass_disp = self.dispersions.get('vehicle_mass', {})
        if mass_disp:
            nominal = mass_disp.get('mean', config.get('vehicle_mass', 50000.0))
            sigma_pct = mass_disp.get('sigma_pct', 5.0)
            sigma = nominal * sigma_pct / 100.0 / 3.0  # 3-sigma bound
            dispersed_mass = self._rng.normal(nominal, sigma)
            dispersed['vehicle_mass'] = max(dispersed_mass, nominal * 0.5)
            dispersion_record['vehicle_mass'] = dispersed['vehicle_mass']

        # --- Thrust ---
        thrust_disp = self.dispersions.get('thrust', {})
        if thrust_disp:
            nominal = thrust_disp.get('mean', config.get('thrust_nominal', 100000.0))
            sigma_pct = thrust_disp.get('sigma_pct', 3.0)
            sigma = nominal * sigma_pct / 100.0 / 3.0
            dispersed_thrust = self._rng.normal(nominal, sigma)
            dispersed['thrust_nominal'] = max(dispersed_thrust, 0.0)
            dispersion_record['thrust'] = dispersed['thrust_nominal']

        # --- Specific impulse ---
        isp_disp = self.dispersions.get('Isp', {})
        if isp_disp:
            nominal = isp_disp.get('mean', config.get('Isp', 300.0))
            sigma_pct = isp_disp.get('sigma_pct', 2.0)
            sigma = nominal * sigma_pct / 100.0 / 3.0
            dispersed_isp = self._rng.normal(nominal, sigma)
            dispersed['Isp'] = max(dispersed_isp, nominal * 0.5)
            dispersion_record['Isp'] = dispersed['Isp']

        # --- Initial attitude (small-angle perturbation) ---
        att_sigma_deg = self.dispersions.get('attitude_sigma_deg', 0.0)
        if att_sigma_deg > 0.0:
            att_sigma_rad = np.radians(att_sigma_deg)
            phi = self._rng.normal(0.0, att_sigma_rad)
            theta = self._rng.normal(0.0, att_sigma_rad)
            psi = self._rng.normal(0.0, att_sigma_rad)
            dispersed['initial_attitude_euler'] = [phi, theta, psi]
            dispersion_record['att_phi_rad'] = phi
            dispersion_record['att_theta_rad'] = theta
            dispersion_record['att_psi_rad'] = psi

        # --- Initial position ---
        pos_sigma = self.dispersions.get('position_sigma_m', 0.0)
        if pos_sigma > 0.0:
            dp = self._rng.normal(0.0, pos_sigma, size=3)
            dispersed['initial_position_offset'] = dp.tolist()
            dispersion_record['pos_offset_x'] = dp[0]
            dispersion_record['pos_offset_y'] = dp[1]
            dispersion_record['pos_offset_z'] = dp[2]

        # --- Initial velocity ---
        vel_sigma = self.dispersions.get('velocity_sigma_mps', 0.0)
        if vel_sigma > 0.0:
            dv = self._rng.normal(0.0, vel_sigma, size=3)
            dispersed['initial_velocity_offset'] = dv.tolist()
            dispersion_record['vel_offset_x'] = dv[0]
            dispersion_record['vel_offset_y'] = dv[1]
            dispersion_record['vel_offset_z'] = dv[2]

        # --- Sensor biases ---
        bias_scale = self.dispersions.get('sensor_bias_scale', 1.0)
        if bias_scale > 0.0:
            # Scale all sensor bias parameters
            sensor_config = dispersed.get('sensors', {})
            for sensor_name, sensor_params in sensor_config.items():
                if 'bias' in sensor_params:
                    nominal_bias = sensor_params['bias']
                    if isinstance(nominal_bias, (list, np.ndarray)):
                        perturbed = [
                            b * self._rng.normal(1.0, 0.1 * bias_scale)
                            for b in nominal_bias
                        ]
                        sensor_params['bias'] = perturbed
                    elif isinstance(nominal_bias, (int, float)):
                        sensor_params['bias'] = (
                            nominal_bias * self._rng.normal(1.0, 0.1 * bias_scale)
                        )
            dispersed['sensors'] = sensor_config
            dispersion_record['sensor_bias_scale'] = bias_scale

        # --- Fuel mass (proportional to vehicle mass change) ---
        if 'vehicle_mass' in dispersion_record:
            mass_ratio = dispersion_record['vehicle_mass'] / config.get(
                'vehicle_mass', 50000.0
            )
            dispersed['fuel_mass'] = config.get('fuel_mass', 35000.0) * mass_ratio
            dispersion_record['fuel_mass'] = dispersed['fuel_mass']

        dispersed['run_id'] = run_id
        dispersed['_dispersion_values'] = dispersion_record

        return dispersed

    # =========================================================================
    # RUN ALL SIMULATIONS
    # =========================================================================

    def run_all(self, num_workers: int = 4) -> pd.DataFrame:
        """
        Execute all dispersed simulation runs in parallel.

        Uses a multiprocessing Pool to distribute runs across worker
        processes.  Each worker executes _run_single_wrapper, which creates
        a SimulationEngine, runs the simulation, and extracts summary metrics.

        Parameters
        ----------
        num_workers : int
            Number of parallel worker processes.  Set to 1 for sequential
            execution (useful for debugging).

        Returns
        -------
        pd.DataFrame
            Aggregated results with one row per run.  Columns include:
            run_id, success, total_delta_v, fuel_consumed,
            max_pointing_error, total_time, final_mass, final_altitude,
            final_velocity, num_phases_completed, landing_lat, landing_lon,
            error_message.
        """
        logger.info(
            "Starting Monte Carlo: %d runs on %d workers",
            self.num_runs, num_workers,
        )

        # Prepare arguments for the worker function
        args_list = [
            (cfg, cfg['run_id'])
            for cfg in self.dispersed_configs
        ]

        if num_workers <= 1:
            # Sequential execution (easier to debug)
            results_list = [_run_single_wrapper(args) for args in args_list]
        else:
            with Pool(processes=num_workers) as pool:
                results_list = pool.map(_run_single_wrapper, args_list)

        self.results = pd.DataFrame(results_list)
        self.results.set_index('run_id', inplace=True)

        n_success = self.results['success'].sum()
        logger.info(
            "Monte Carlo complete: %d/%d runs successful (%.1f%%)",
            n_success, self.num_runs, 100.0 * n_success / max(self.num_runs, 1),
        )

        return self.results

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def compute_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics for all numeric result metrics.

        Returns
        -------
        dict
            For each metric (total_delta_v, fuel_consumed, etc.):
                - mean, std, min, max
                - p01, p50, p99  (1st, 50th, 99th percentiles)
        """
        if self.results is None or self.results.empty:
            logger.warning("No results to compute statistics on.")
            return {}

        numeric_cols = self.results.select_dtypes(include=[np.number]).columns
        stats: Dict[str, Dict[str, float]] = {}

        for col in numeric_cols:
            if col == 'success':
                continue
            data = self.results[col].dropna()
            if len(data) == 0:
                continue

            stats[col] = {
                'mean': float(data.mean()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'p01': float(np.percentile(data, 1)),
                'p50': float(np.percentile(data, 50)),
                'p99': float(np.percentile(data, 99)),
            }

        return stats

    def get_success_rate(self) -> float:
        """
        Compute the fraction of runs that completed successfully.

        Returns
        -------
        float
            Success rate in [0, 1].
        """
        if self.results is None or self.results.empty:
            return 0.0
        return float(self.results['success'].mean())

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def plot_dispersions(self, output_dir: str) -> None:
        """
        Generate histograms of each dispersed parameter across all runs.

        Saves one PNG per dispersed parameter showing the distribution of
        applied dispersions.

        Parameters
        ----------
        output_dir : str
            Directory to save the histogram images.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping dispersion plots.")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Collect dispersion values from all configs
        disp_records = [
            cfg.get('_dispersion_values', {})
            for cfg in self.dispersed_configs
        ]
        disp_df = pd.DataFrame(disp_records)

        for col in disp_df.columns:
            if col == 'run_id':
                continue
            data = disp_df[col].dropna()
            if len(data) == 0:
                continue

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(data.mean(), color='red', linestyle='--',
                       label=f'Mean = {data.mean():.4f}')
            ax.axvline(data.mean() + data.std(), color='orange', linestyle=':',
                       label=f'+1 sigma = {data.std():.4f}')
            ax.axvline(data.mean() - data.std(), color='orange', linestyle=':')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            ax.set_title(f'Monte Carlo Dispersion: {col}  (N={self.num_runs})')
            ax.legend()
            ax.grid(True, alpha=0.3)

            filepath = os.path.join(output_dir, f'dispersion_{col}.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info("Saved dispersion plot: %s", filepath)

    def plot_trajectory_fan(self, output_dir: str) -> None:
        """
        Generate an overlay plot of all run trajectories in 2D projection.

        Individual runs are shown as faded lines; the nominal trajectory
        (run 0) is highlighted in bold.  Useful for visualizing dispersion
        spread in physical space.

        Parameters
        ----------
        output_dir : str
            Directory to save the plot image.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping trajectory fan plot.")
            return

        os.makedirs(output_dir, exist_ok=True)

        if self.results is None or self.results.empty:
            logger.warning("No results for trajectory fan plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # If per-run trajectory data is stored, plot it
        # Otherwise, plot summary scatter from results
        if 'final_altitude' in self.results.columns:
            successful = self.results[self.results['success']]
            failed = self.results[~self.results['success']]

            # Scatter: total_time vs final_altitude
            if not successful.empty:
                ax.scatter(
                    successful['total_time'],
                    successful['final_altitude'],
                    alpha=0.3, s=10, color='steelblue', label='Successful',
                )
            if not failed.empty:
                ax.scatter(
                    failed['total_time'],
                    failed['final_altitude'],
                    alpha=0.5, s=20, color='red', marker='x', label='Failed',
                )

            # Highlight nominal (run 0)
            if 0 in self.results.index:
                nominal = self.results.loc[0]
                ax.scatter(
                    [nominal['total_time']],
                    [nominal['final_altitude']],
                    s=100, color='gold', edgecolor='black', zorder=5,
                    label='Nominal',
                )

        ax.set_xlabel('Total Mission Time (s)')
        ax.set_ylabel('Final Altitude (m)')
        ax.set_title(f'Monte Carlo Trajectory Fan  (N={self.num_runs})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        filepath = os.path.join(output_dir, 'trajectory_fan.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved trajectory fan plot: %s", filepath)

    def plot_landing_scatter(self, output_dir: str) -> None:
        """
        Generate a scatter plot of landing latitude/longitude with a
        2-sigma confidence ellipse.

        Parameters
        ----------
        output_dir : str
            Directory to save the plot image.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Ellipse
        except ImportError:
            logger.warning("matplotlib not available; skipping landing scatter plot.")
            return

        os.makedirs(output_dir, exist_ok=True)

        if self.results is None or self.results.empty:
            logger.warning("No results for landing scatter plot.")
            return

        # Filter to successful runs with landing coordinates
        mask = (
            self.results['success'] &
            self.results['landing_lat'].notna() &
            self.results['landing_lon'].notna()
        )
        landing_data = self.results[mask]

        if landing_data.empty:
            logger.warning("No successful landing data for scatter plot.")
            return

        lat = landing_data['landing_lat']
        lon = landing_data['landing_lon']

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(lon, lat, alpha=0.4, s=15, color='steelblue',
                   label='Landing Points')

        # Compute and plot 2-sigma ellipse
        mean_lon = lon.mean()
        mean_lat = lat.mean()
        std_lon = lon.std()
        std_lat = lat.std()

        if std_lon > 0 and std_lat > 0:
            # Correlation for ellipse rotation
            cov = np.cov(lon, lat)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))

            for n_sigma, alpha in [(1, 0.3), (2, 0.15), (3, 0.05)]:
                width = 2 * n_sigma * np.sqrt(eigenvalues[1])
                height = 2 * n_sigma * np.sqrt(eigenvalues[0])
                ellipse = Ellipse(
                    xy=(mean_lon, mean_lat),
                    width=width, height=height,
                    angle=angle,
                    facecolor='orange', alpha=alpha,
                    edgecolor='darkorange', linewidth=1.5,
                    label=f'{n_sigma}-sigma ellipse',
                )
                ax.add_patch(ellipse)

        ax.plot(mean_lon, mean_lat, 'r+', markersize=15, markeredgewidth=2,
                label=f'Mean ({mean_lat:.4f}, {mean_lon:.4f})')

        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        ax.set_title(f'Monte Carlo Landing Scatter  (N={len(landing_data)})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        filepath = os.path.join(output_dir, 'landing_scatter.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved landing scatter plot: %s", filepath)

    # =========================================================================
    # REPRESENTATION
    # =========================================================================

    def __repr__(self) -> str:
        status = "not run" if self.results is None else f"{len(self.results)} results"
        return (
            f"MonteCarloSim(runs={self.num_runs}, seed={self.seed}, "
            f"status={status})"
        )
