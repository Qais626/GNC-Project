"""
Propulsion Trade Study
Compares Chemical Bipropellant, Solid Rocket, Ion Thruster, Nuclear Thermal
across the full KSC-Moon-Jupiter-KSC mission.
Generates 8 comparison plots.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from core.constants import (
    G0, EARTH_MU, MOON_MU, JUPITER_MU, SUN_MU,
    EARTH_RADIUS, MOON_RADIUS, JUPITER_RADIUS,
    EARTH_MOON_DIST, EARTH_JUPITER_DIST,
    MISSION_DELTA_V,
)
from trade_studies.plot_utils import PlotStyle


# ---------------------------------------------------------------------------
# Default propulsion catalogue
# ---------------------------------------------------------------------------

DEFAULT_PROPULSION_OPTIONS = [
    {
        'name': 'Chemical Bipropellant',
        'isp': 320.0,            # s
        'thrust': 22_000.0,      # N
        'engine_mass': 85.0,     # kg
        'type': 'chemical',
    },
    {
        'name': 'Solid Rocket',
        'isp': 270.0,
        'thrust': 68_000.0,
        'engine_mass': 120.0,
        'type': 'solid',
    },
    {
        'name': 'Ion Thruster',
        'isp': 3100.0,
        'thrust': 0.5,
        'engine_mass': 25.0,
        'type': 'ion',
    },
    {
        'name': 'Nuclear Thermal',
        'isp': 900.0,
        'thrust': 67_000.0,
        'engine_mass': 2200.0,
        'type': 'nuclear',
    },
]

# Mission legs and their approximate delta-V requirements [m/s]
MISSION_LEGS = [
    ('TLI',            3150.0),
    ('LOI',            800.0),
    ('Lunar Escape',   900.0),
    ('Earth-Jupiter',  6300.0),
    ('JOI',            2000.0),
    ('Jupiter Escape', 2200.0),
    ('Return',         6300.0),
]

# Representative distances for each leg [km]
LEG_DISTANCES_KM = {
    'TLI':            384_400.0,
    'LOI':            50_000.0,
    'Lunar Escape':   384_400.0,
    'Earth-Jupiter':  6.3e8,
    'JOI':            1.0e6,
    'Jupiter Escape': 1.0e6,
    'Return':         6.3e8,
}

# Initial total spacecraft wet mass [kg]
INITIAL_MASS_KG = 12_000.0


# ---------------------------------------------------------------------------
# PropulsionTradeStudy
# ---------------------------------------------------------------------------

class PropulsionTradeStudy:
    """Compare propulsion options across the full mission profile."""

    def __init__(self, config=None):
        """
        Parameters
        ----------
        config : list of dict or None
            Each dict must contain keys: name, isp, thrust, engine_mass, type.
            If *None*, the built-in catalogue is used.
        """
        self.options = config if config is not None else DEFAULT_PROPULSION_OPTIONS
        self.results = None   # populated by run_trade()

    # ----- core equations --------------------------------------------------

    @staticmethod
    def _rocket_equation(isp, m0, mf):
        """Tsiolkovsky rocket equation.  Returns delta-V in m/s."""
        if mf <= 0 or m0 <= 0:
            return 0.0
        return isp * G0 * np.log(m0 / mf)

    @staticmethod
    def _propellant_needed(delta_v, isp, dry_mass):
        """Return propellant mass [kg] required for *delta_v* [m/s].

        Uses the inverse Tsiolkovsky equation:
            m0 = mf * exp(dv / (isp * g0))
            propellant = m0 - mf
        """
        if isp <= 0:
            return np.inf
        mass_ratio = np.exp(delta_v / (isp * G0))
        m0 = dry_mass * mass_ratio
        return m0 - dry_mass

    @staticmethod
    def _transfer_time(delta_v, thrust, mass, distance_km, prop_type):
        """Estimate transfer time [days] for a given leg.

        Chemical / solid / nuclear -- impulsive burns.  Time of flight
        approximated from a Hohmann-like half-period or from the distance
        and mean velocity.

        Ion -- continuous low-thrust spiral.  t = m * dv / thrust.
        """
        if prop_type in ('chemical', 'solid', 'nuclear'):
            # Mean transfer velocity ~ delta_v for short burns, else use
            # vis-viva-ish approximation.  Rough Hohmann TOF:
            mean_v = max(delta_v, 1.0)  # m/s
            tof_s = (distance_km * 1e3) / mean_v
            # For nuclear propulsion transfers are about 60% of chemical time
            if prop_type == 'nuclear':
                tof_s *= 0.6
            return tof_s / 86400.0  # convert to days
        else:
            # Low-thrust spiral
            if thrust <= 0:
                return np.inf
            burn_time_s = mass * delta_v / thrust
            # Coast adds ~20 % overhead
            return burn_time_s * 1.2 / 86400.0

    # ----- main analysis ---------------------------------------------------

    def run_trade(self):
        """Evaluate every propulsion option on every mission leg.

        Populates ``self.results`` -- a list of dicts, one per propulsion
        option, each containing per-leg metrics and cumulative totals.

        Returns
        -------
        summary : pd.DataFrame
        """
        all_results = []

        for opt in self.options:
            name = opt['name']
            isp = opt['isp']
            thrust = opt['thrust']
            eng_mass = opt['engine_mass']
            ptype = opt['type']

            current_mass = INITIAL_MASS_KG
            leg_records = []
            mass_timeline = [current_mass]
            time_cumulative = 0.0  # days

            for leg_name, dv_req in MISSION_LEGS:
                prop_mass = self._propellant_needed(dv_req, isp, current_mass)
                if prop_mass >= current_mass:
                    prop_mass = current_mass * 0.95  # cap to avoid negative
                remaining_mass = current_mass - prop_mass

                dist_km = LEG_DISTANCES_KM.get(leg_name, 1e6)
                tof_days = self._transfer_time(dv_req, thrust,
                                               current_mass, dist_km, ptype)
                time_cumulative += tof_days

                achievable_dv = self._rocket_equation(isp, current_mass,
                                                      remaining_mass)

                leg_records.append({
                    'leg': leg_name,
                    'delta_v_req_m_s': dv_req,
                    'delta_v_achieved_m_s': achievable_dv,
                    'propellant_kg': prop_mass,
                    'mass_after_kg': remaining_mass,
                    'tof_days': tof_days,
                    'cumulative_days': time_cumulative,
                })

                mass_timeline.append(remaining_mass)
                current_mass = remaining_mass

            total_prop = sum(r['propellant_kg'] for r in leg_records)
            total_dv = sum(r['delta_v_achieved_m_s'] for r in leg_records)
            total_time = time_cumulative

            all_results.append({
                'name': name,
                'isp': isp,
                'thrust': thrust,
                'engine_mass': eng_mass,
                'type': ptype,
                'legs': leg_records,
                'total_propellant_kg': total_prop,
                'total_delta_v_m_s': total_dv,
                'total_time_days': total_time,
                'mass_timeline': mass_timeline,
                'final_mass_kg': current_mass,
            })

        self.results = all_results
        return self.get_summary_table()

    # ----- simplified trajectory -------------------------------------------

    def simulate_trajectory(self, prop_config):
        """Return a simplified 2-D trajectory array (N, 2) in km.

        Chemical / solid  -> Hohmann-like elliptic arcs.
        Ion               -> expanding spiral.
        Nuclear           -> faster Hohmann arcs.
        """
        ptype = prop_config['type']
        theta = np.linspace(0, 2 * np.pi, 1000)

        if ptype == 'ion':
            # Spiral from Earth to Jupiter
            r_start = EARTH_RADIUS + 400.0  # LEO
            r_end = EARTH_JUPITER_DIST
            r = r_start + (r_end - r_start) * (theta / (2 * np.pi)) ** 0.7
            x = r * np.cos(theta * 5)
            y = r * np.sin(theta * 5)
        elif ptype == 'nuclear':
            # Faster Hohmann-like
            r_start = EARTH_RADIUS + 400.0
            r_end = EARTH_JUPITER_DIST
            a = (r_start + r_end) / 2.0
            e = (r_end - r_start) / (r_end + r_start)
            r = a * (1 - e ** 2) / (1 + e * np.cos(theta))
            x = r * np.cos(theta)
            y = r * np.sin(theta)
        else:
            # Standard Hohmann
            r_start = EARTH_RADIUS + 400.0
            r_end = EARTH_JUPITER_DIST
            a = (r_start + r_end) / 2.0
            e = (r_end - r_start) / (r_end + r_start)
            r = a * (1 - e ** 2) / (1 + e * np.cos(theta))
            x = r * np.cos(theta)
            y = r * np.sin(theta)

        return np.column_stack([x, y])

    # ----- summary table ---------------------------------------------------

    def get_summary_table(self):
        """Return a pandas DataFrame summarising each propulsion option."""
        if self.results is None:
            return pd.DataFrame()
        rows = []
        for r in self.results:
            rows.append({
                'Propulsion': r['name'],
                'Isp [s]': r['isp'],
                'Thrust [N]': r['thrust'],
                'Engine Mass [kg]': r['engine_mass'],
                'Total Propellant [kg]': round(r['total_propellant_kg'], 1),
                'Total delta-V [m/s]': round(r['total_delta_v_m_s'], 1),
                'Mission Duration [days]': round(r['total_time_days'], 1),
                'Final Mass [kg]': round(r['final_mass_kg'], 1),
            })
        return pd.DataFrame(rows)

    # ----- plotting --------------------------------------------------------

    def run_and_plot(self, output_dir='output/propulsion_trade'):
        """Run the trade study and generate all 8 comparison plots.

        Parameters
        ----------
        output_dir : str
            Directory where plot images are saved.
        """
        if self.results is None:
            self.run_trade()

        PlotStyle.setup_style()
        names = [r['name'] for r in self.results]
        colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        # ---- Plot 1: Total delta-V capability per type --------------------
        fig, ax = PlotStyle.create_figure()
        dvs = [r['total_delta_v_m_s'] / 1000.0 for r in self.results]
        bars = ax.bar(names, dvs, color=colours, edgecolor='black')
        for bar, v in zip(bars, dvs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f'{v:.1f}', ha='center', fontsize=10)
        ax.set_ylabel('Total Delta-V [km/s]')
        ax.set_title('Total Delta-V Capability by Propulsion Type')
        PlotStyle.save_figure(fig, f'{output_dir}/01_total_delta_v.png')

        # ---- Plot 2: Total mission duration per type ----------------------
        fig, ax = PlotStyle.create_figure()
        durations = [r['total_time_days'] for r in self.results]
        bars = ax.bar(names, durations, color=colours, edgecolor='black')
        for bar, d in zip(bars, durations):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f'{d:.0f} d', ha='center', fontsize=10)
        ax.set_ylabel('Mission Duration [days]')
        ax.set_title('Total Mission Duration by Propulsion Type')
        PlotStyle.save_figure(fig, f'{output_dir}/02_mission_duration.png')

        # ---- Plot 3: Stacked bar -- propellant mass by leg ----------------
        fig, ax = PlotStyle.create_figure(figsize=(12, 7))
        leg_names = [l[0] for l in MISSION_LEGS]
        n_legs = len(leg_names)
        x = np.arange(len(names))
        width = 0.18
        leg_colours = cm.get_cmap('Set2')(np.linspace(0, 1, n_legs))

        bottom = np.zeros(len(names))
        for j, leg in enumerate(leg_names):
            vals = []
            for r in self.results:
                vals.append(r['legs'][j]['propellant_kg'])
            vals = np.array(vals)
            ax.bar(x, vals, width=0.6, bottom=bottom, color=leg_colours[j],
                   edgecolor='white', linewidth=0.4, label=leg)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel('Propellant Mass [kg]')
        ax.set_title('Propellant Mass Breakdown by Mission Leg')
        ax.legend(loc='upper right', fontsize=9, ncol=2)
        PlotStyle.save_figure(fig, f'{output_dir}/03_propellant_stacked.png')

        # ---- Plot 4: 2-D trajectory comparison ---------------------------
        fig, ax = PlotStyle.create_figure(figsize=(11, 9))
        for idx, opt in enumerate(self.options):
            traj = self.simulate_trajectory(opt)
            ax.plot(traj[:, 0] / 1e6, traj[:, 1] / 1e6, color=colours[idx],
                    label=opt['name'], linewidth=1.5)

        # Draw bodies
        ax.plot(0, 0, 'o', color='royalblue', markersize=10, label='Earth')
        ax.plot(EARTH_MOON_DIST / 1e6, 0, 'o', color='grey', markersize=6,
                label='Moon')
        ax.plot(EARTH_JUPITER_DIST / 1e6, 0, 'o', color='sandybrown',
                markersize=12, label='Jupiter')
        ax.set_xlabel('X [10$^6$ km]')
        ax.set_ylabel('Y [10$^6$ km]')
        ax.set_title('2-D Trajectory Comparison')
        ax.legend(fontsize=9)
        ax.set_aspect('equal', adjustable='datalim')
        PlotStyle.save_figure(fig, f'{output_dir}/04_trajectory_2d.png')

        # ---- Plot 5: 3-D trajectory comparison ---------------------------
        fig = plt.figure(figsize=(12, 9))
        ax3 = fig.add_subplot(111, projection='3d')
        for idx, opt in enumerate(self.options):
            traj = self.simulate_trajectory(opt)
            z = np.sin(np.linspace(0, np.pi, len(traj))) * 5e7
            ax3.plot(traj[:, 0] / 1e6, traj[:, 1] / 1e6, z / 1e6,
                     color=colours[idx], label=opt['name'], linewidth=1.4)
        ax3.scatter(0, 0, 0, color='royalblue', s=60, label='Earth')
        ax3.scatter(EARTH_JUPITER_DIST / 1e6, 0, 0, color='sandybrown',
                    s=100, label='Jupiter')
        ax3.set_xlabel('X [10$^6$ km]')
        ax3.set_ylabel('Y [10$^6$ km]')
        ax3.set_zlabel('Z [10$^6$ km]')
        ax3.set_title('3-D Trajectory Comparison')
        ax3.legend(fontsize=8, loc='upper left')
        PlotStyle.save_figure(fig, f'{output_dir}/05_trajectory_3d.png')

        # ---- Plot 6: Radar / spider chart ---------------------------------
        categories = ['Isp', 'Thrust', 'Engine Mass', 'Duration', 'Propellant']
        raw = np.zeros((len(self.results), len(categories)))
        for i, r in enumerate(self.results):
            raw[i] = [r['isp'], r['thrust'],
                      r['engine_mass'], r['total_time_days'],
                      r['total_propellant_kg']]

        # Normalise 0-1 (higher is better for Isp and Thrust; invert others)
        normed = np.zeros_like(raw)
        for c in range(len(categories)):
            col = raw[:, c]
            mn, mx = col.min(), col.max()
            if mx - mn == 0:
                normed[:, c] = 1.0
            else:
                normed[:, c] = (col - mn) / (mx - mn)
        # Invert columns where lower is better
        for c_idx in [2, 3, 4]:  # engine mass, duration, propellant
            normed[:, c_idx] = 1.0 - normed[:, c_idx]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
        for i, r in enumerate(self.results):
            vals = np.concatenate([normed[i], [normed[i][0]]])
            ax.plot(angles, vals, color=colours[i], linewidth=2,
                    label=r['name'])
            ax.fill(angles, vals, color=colours[i], alpha=0.10)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_title('Multi-Attribute Propulsion Comparison', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
        PlotStyle.save_figure(fig, f'{output_dir}/06_radar_chart.png')

        # ---- Plot 7: Mass vs time ----------------------------------------
        fig, ax = PlotStyle.create_figure()
        for idx, r in enumerate(self.results):
            t_points = [0.0]
            for leg in r['legs']:
                t_points.append(leg['cumulative_days'])
            ax.plot(t_points, r['mass_timeline'], marker='o', markersize=4,
                    color=colours[idx], label=r['name'])
        ax.set_xlabel('Mission Time [days]')
        ax.set_ylabel('Spacecraft Mass [kg]')
        ax.set_title('Spacecraft Mass History by Propulsion Type')
        ax.legend()
        PlotStyle.save_figure(fig, f'{output_dir}/07_mass_history.png')

        # ---- Plot 8: Velocity vs time ------------------------------------
        fig, ax = PlotStyle.create_figure()
        for idx, r in enumerate(self.results):
            t_points = [0.0]
            v_cumulative = [0.0]
            running_v = 0.0
            for leg in r['legs']:
                running_v += leg['delta_v_achieved_m_s']
                t_points.append(leg['cumulative_days'])
                v_cumulative.append(running_v / 1000.0)  # km/s
            ax.plot(t_points, v_cumulative, marker='s', markersize=4,
                    color=colours[idx], label=r['name'])
        ax.set_xlabel('Mission Time [days]')
        ax.set_ylabel('Cumulative Delta-V [km/s]')
        ax.set_title('Cumulative Delta-V by Propulsion Type')
        ax.legend()
        PlotStyle.save_figure(fig, f'{output_dir}/08_velocity_history.png')

        print(f'[PropulsionTradeStudy] All plots saved to {output_dir}/')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    study = PropulsionTradeStudy()
    summary = study.run_trade()
    print(summary.to_string(index=False))
    study.run_and_plot()
