"""
Attitude Trade Study
Studies how pointing errors and orientation affect trajectory accuracy and fuel.
Compares PID, LQR, and Sliding Mode controller performance.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from core.constants import (
    G0, EARTH_MU, MOON_MU, JUPITER_MU,
    EARTH_RADIUS, EARTH_MOON_DIST, EARTH_JUPITER_DIST,
    MISSION_DELTA_V,
)
from trade_studies.plot_utils import PlotStyle


# ---------------------------------------------------------------------------
# Controller gain presets
# ---------------------------------------------------------------------------

CONTROLLER_GAINS = {
    'PID': {
        'kp': 2.0,
        'ki': 0.05,
        'kd': 8.0,
    },
    'LQR': {
        'K': np.array([4.5, 12.0]),   # state-feedback gains [position, rate]
    },
    'SlidingMode': {
        'lambda_': 3.0,   # sliding surface slope
        'eta': 0.8,       # switching gain
    },
}

# Default simulation parameters
DEFAULT_INERTIA = 450.0          # kg*m^2  (single-axis moment of inertia)
DEFAULT_MAX_TORQUE = 0.5         # N*m
DEFAULT_THRUST = 22_000.0        # N
DEFAULT_DELTA_V = 6300.0         # m/s  (representative leg)
DEFAULT_BURN_TIME = 300.0        # s


# ---------------------------------------------------------------------------
# AttitudeTradeStudy
# ---------------------------------------------------------------------------

class AttitudeTradeStudy:
    """Analyse how pointing errors affect mission performance and compare
    attitude-control strategies."""

    def __init__(self, config=None):
        """
        Parameters
        ----------
        config : dict or None
            Full mission configuration.  Currently used keys:

            - ``inertia``     : scalar moment of inertia [kg m^2]
            - ``max_torque``  : max RW/thruster torque   [N m]
            - ``thrust``      : main engine thrust        [N]
            - ``delta_v``     : reference delta-V         [m/s]
            - ``burn_time``   : reference burn duration   [s]
        """
        cfg = config or {}
        self.inertia = cfg.get('inertia', DEFAULT_INERTIA)
        self.max_torque = cfg.get('max_torque', DEFAULT_MAX_TORQUE)
        self.thrust = cfg.get('thrust', DEFAULT_THRUST)
        self.delta_v = cfg.get('delta_v', DEFAULT_DELTA_V)
        self.burn_time = cfg.get('burn_time', DEFAULT_BURN_TIME)

    # ----- internal simulations -------------------------------------------

    def _simulate_pointing_effect(self, error_deg, duration=1000.0, dt=1.0):
        """Propagate orbit with thrust misaligned by *error_deg*.

        Returns
        -------
        delta_v_penalty : float
            Extra delta-V required [m/s] compared to ideal.
        miss_distance : float
            Final position miss [km].
        """
        error_rad = np.radians(error_deg)
        n_steps = int(duration / dt)

        # Ideal (zero error) trajectory -- simple 1-D constant thrust
        mass = 5000.0                       # kg
        accel = self.thrust / mass          # m/s^2
        v_ideal = accel * duration          # m/s  (final speed)
        x_ideal = 0.5 * accel * duration ** 2  # m

        # Misaligned trajectory: effective axial acceleration reduced
        accel_eff = accel * np.cos(error_rad)
        accel_lat = accel * np.sin(error_rad)

        v_actual = accel_eff * duration
        x_actual = 0.5 * accel_eff * duration ** 2
        y_lateral = 0.5 * accel_lat * duration ** 2

        miss_distance = np.sqrt((x_ideal - x_actual) ** 2 + y_lateral ** 2)
        miss_distance_km = miss_distance / 1000.0

        # Delta-V penalty: extra DV to correct the miss
        delta_v_penalty = v_ideal - v_actual + abs(accel_lat * duration)

        return delta_v_penalty, miss_distance_km

    def _simulate_controller(self, controller_type, disturbance_level,
                             duration=500.0, dt=0.1):
        """Single-axis attitude simulation.

        Parameters
        ----------
        controller_type : str
            One of 'PID', 'LQR', 'SlidingMode'.
        disturbance_level : float
            RMS external torque disturbance [N m].
        duration : float
            Simulation time [s].
        dt : float
            Time step [s].

        Returns
        -------
        times : ndarray
        pointing_errors_deg : ndarray
        """
        n = int(duration / dt)
        times = np.linspace(0, duration, n)

        theta = np.radians(5.0)     # initial pointing offset [rad]
        omega = 0.0                 # initial angular rate
        integral = 0.0              # for PID

        errors = np.zeros(n)
        gains = CONTROLLER_GAINS[controller_type]

        for k in range(n):
            errors[k] = np.degrees(abs(theta))

            # Disturbance torque (white noise)
            T_dist = np.random.normal(0, disturbance_level)

            # Control torque
            if controller_type == 'PID':
                kp, ki, kd = gains['kp'], gains['ki'], gains['kd']
                integral += theta * dt
                T_ctrl = -(kp * theta + ki * integral + kd * omega)

            elif controller_type == 'LQR':
                K = gains['K']
                T_ctrl = -(K[0] * theta + K[1] * omega)

            elif controller_type == 'SlidingMode':
                lam = gains['lambda_']
                eta = gains['eta']
                s = omega + lam * theta
                T_ctrl = -(lam * omega + eta * np.sign(s)) * self.inertia

            else:
                T_ctrl = 0.0

            # Saturate torque
            T_ctrl = np.clip(T_ctrl, -self.max_torque, self.max_torque)

            # Euler integration of rotational dynamics
            alpha = (T_ctrl + T_dist) / self.inertia
            omega += alpha * dt
            theta += omega * dt

        return times, errors

    # ----- plotting --------------------------------------------------------

    def run_and_plot(self, output_dir='output/attitude_trade'):
        """Generate all 8 attitude trade-study plots.

        Parameters
        ----------
        output_dir : str
            Directory where plot images are saved.
        """
        PlotStyle.setup_style()

        # ---- Plot 1: Pointing error vs delta-V penalty --------------------
        error_levels = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        dv_penalties = []
        miss_dists_moon = []
        miss_dists_jupiter = []

        for e in error_levels:
            dvp, miss = self._simulate_pointing_effect(e, duration=300.0)
            dv_penalties.append(dvp)

            # Moon-distance burn (shorter, smaller effect)
            _, miss_m = self._simulate_pointing_effect(
                e, duration=100.0)
            miss_dists_moon.append(miss_m)

            # Jupiter-distance burn (longer, larger effect)
            _, miss_j = self._simulate_pointing_effect(
                e, duration=2000.0)
            miss_dists_jupiter.append(miss_j)

        fig, ax = PlotStyle.create_figure()
        ax.semilogy(error_levels, dv_penalties, 'o-', color='steelblue',
                    linewidth=2, markersize=6)
        ax.set_xlabel('Pointing Error [deg]')
        ax.set_ylabel('Delta-V Penalty [m/s]')
        ax.set_title('Pointing Error vs Delta-V Penalty')
        PlotStyle.save_figure(fig, f'{output_dir}/01_error_vs_dv_penalty.png')

        # ---- Plot 2: Pointing error vs miss distance ----------------------
        fig, ax = PlotStyle.create_figure()
        ax.semilogy(error_levels, miss_dists_moon, 's-', color='grey',
                    linewidth=2, markersize=6, label='Moon Arrival')
        ax.semilogy(error_levels, miss_dists_jupiter, 'D-', color='sandybrown',
                    linewidth=2, markersize=6, label='Jupiter Arrival')
        ax.set_xlabel('Pointing Error [deg]')
        ax.set_ylabel('Miss Distance [km]')
        ax.set_title('Pointing Error vs Arrival Miss Distance')
        ax.legend()
        PlotStyle.save_figure(fig, f'{output_dir}/02_error_vs_miss.png')

        # ---- Plot 3: Thrust misalignment vs efficiency --------------------
        misalignments = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        efficiencies = np.cos(np.radians(misalignments)) * 100.0

        fig, ax = PlotStyle.create_figure()
        ax.scatter(misalignments, efficiencies, s=80, c='crimson',
                   zorder=5, label='Computed')
        # Trend line (quadratic fit)
        coeffs = np.polyfit(misalignments, efficiencies, 2)
        x_fit = np.linspace(0, 6, 100)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, '--', color='grey', label='Quadratic fit')
        ax.set_xlabel('Thrust Misalignment [deg]')
        ax.set_ylabel('Thrust Efficiency [%]')
        ax.set_title('Thrust Misalignment vs Effective Efficiency')
        ax.legend()
        ax.set_ylim(95, 100.5)
        PlotStyle.save_figure(fig,
                              f'{output_dir}/03_misalignment_efficiency.png')

        # ---- Plot 4: Controller comparison time history -------------------
        ctrl_types = ['PID', 'LQR', 'SlidingMode']
        ctrl_colours = {'PID': '#1f77b4', 'LQR': '#ff7f0e',
                        'SlidingMode': '#2ca02c'}
        dist_level = 0.005  # N*m  RMS disturbance

        fig, ax = PlotStyle.create_figure(figsize=(11, 5))
        np.random.seed(42)
        for ct in ctrl_types:
            t, err = self._simulate_controller(ct, dist_level, duration=300.0)
            ax.plot(t, err, color=ctrl_colours[ct], label=ct, linewidth=1.4)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Pointing Error [deg]')
        ax.set_title('Controller Comparison -- Pointing Error vs Time')
        ax.legend()
        PlotStyle.save_figure(fig,
                              f'{output_dir}/04_controller_comparison.png')

        # ---- Plot 5: Monte Carlo landing accuracy -------------------------
        error_budgets = [0.1, 1.0, 5.0]
        mc_samples = 100

        fig, axes = PlotStyle.create_figure(nrows=1, ncols=3,
                                            figsize=(15, 5))
        budget_colours = ['#2ca02c', '#ff7f0e', '#d62728']

        for idx, budget in enumerate(error_budgets):
            np.random.seed(idx)
            landing_errors = []
            for _ in range(mc_samples):
                # Random pointing error drawn from budget-sized distribution
                err = abs(np.random.normal(0, budget))
                _, miss = self._simulate_pointing_effect(err, duration=500.0)
                landing_errors.append(miss)
            axes[idx].hist(landing_errors, bins=20,
                           color=budget_colours[idx], edgecolor='black',
                           alpha=0.75)
            axes[idx].set_xlabel('Landing Error [km]')
            axes[idx].set_ylabel('Count')
            axes[idx].set_title(
                f'Error Budget = {budget} deg\n'
                f'Mean = {np.mean(landing_errors):.1f} km')

        fig.suptitle('Monte Carlo Landing Accuracy', fontsize=15, y=1.02)
        PlotStyle.save_figure(fig,
                              f'{output_dir}/05_monte_carlo_landing.png')

        # ---- Plot 6: Bandwidth vs pointing error --------------------------
        #   Approximate closed-loop bandwidth from step response rise time.
        fig, ax = PlotStyle.create_figure()
        np.random.seed(99)
        for ct in ctrl_types:
            t, err = self._simulate_controller(ct, 0.001, duration=200.0,
                                               dt=0.05)
            # Rise time: time from 10% to 90% of initial error
            initial_err = err[0] if err[0] > 0 else 1.0
            thresh_90 = 0.10 * initial_err
            thresh_10 = 0.90 * initial_err
            idx_10 = np.argmax(err < thresh_10) if np.any(err < thresh_10) \
                else len(err) - 1
            idx_90 = np.argmax(err < thresh_90) if np.any(err < thresh_90) \
                else len(err) - 1
            rise_time = max(t[idx_90] - t[idx_10], 0.01)
            bandwidth_hz = 0.35 / rise_time   # classic approximation

            # Steady-state pointing error (mean of last 20 %)
            ss_error = np.mean(err[int(0.8 * len(err)):])

            ax.plot(ss_error, bandwidth_hz, 'o', color=ctrl_colours[ct],
                    markersize=12, label=ct)
            ax.annotate(ct, (ss_error, bandwidth_hz),
                        textcoords='offset points', xytext=(8, 5),
                        fontsize=10)

        ax.set_xlabel('Steady-State Pointing Error [deg]')
        ax.set_ylabel('Closed-Loop Bandwidth [Hz]')
        ax.set_title('Controller Bandwidth vs Pointing Accuracy')
        ax.legend()
        PlotStyle.save_figure(fig,
                              f'{output_dir}/06_bandwidth_vs_error.png')

        # ---- Plot 7: Heat map -- DV penalty(pointing, misalignment) -------
        pt_errors = np.linspace(0.01, 10.0, 40)
        misalign = np.linspace(0.01, 5.0, 40)
        PE, MA = np.meshgrid(pt_errors, misalign)
        DV_penalty = np.zeros_like(PE)

        for i in range(PE.shape[0]):
            for j in range(PE.shape[1]):
                total_err = PE[i, j] + MA[i, j]
                dvp, _ = self._simulate_pointing_effect(total_err,
                                                        duration=300.0)
                DV_penalty[i, j] = dvp

        fig, ax = PlotStyle.create_figure(figsize=(10, 7))
        pcm = ax.pcolormesh(PE, MA, DV_penalty, cmap='hot_r', shading='auto')
        fig.colorbar(pcm, ax=ax, label='Delta-V Penalty [m/s]')
        ax.set_xlabel('Pointing Error [deg]')
        ax.set_ylabel('Thrust Misalignment [deg]')
        ax.set_title('Delta-V Penalty Heat Map')
        PlotStyle.save_figure(fig, f'{output_dir}/07_dv_penalty_heatmap.png')

        # ---- Plot 8: Box plot -- achieved accuracy per controller ---------
        n_trials = 50
        box_data = {}
        np.random.seed(77)

        for ct in ctrl_types:
            accs = []
            for _ in range(n_trials):
                # Random initial condition (1-10 deg offset)
                init_err_deg = np.random.uniform(1.0, 10.0)
                dist = np.random.uniform(0.001, 0.01)

                t, err = self._simulate_controller(ct, dist, duration=200.0)
                # Steady-state accuracy = mean of last 25 % of time
                ss = np.mean(err[int(0.75 * len(err)):])
                accs.append(ss)
            box_data[ct] = accs

        fig, ax = PlotStyle.create_figure(figsize=(8, 6))
        bp = ax.boxplot([box_data[ct] for ct in ctrl_types],
                        labels=ctrl_types, patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 1.5})
        for patch, ct in zip(bp['boxes'], ctrl_types):
            patch.set_facecolor(ctrl_colours[ct])
            patch.set_alpha(0.6)
        ax.set_ylabel('Steady-State Pointing Error [deg]')
        ax.set_title('Controller Accuracy -- 50 Random Initial Conditions')
        PlotStyle.save_figure(fig,
                              f'{output_dir}/08_controller_boxplot.png')

        print(f'[AttitudeTradeStudy] All plots saved to {output_dir}/')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    study = AttitudeTradeStudy()
    study.run_and_plot()
