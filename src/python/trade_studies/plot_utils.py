"""
Plotting Utilities for GNC Mission
Professional publication-quality plots using matplotlib.
Consistent styling, 3D trajectory plots, time histories, control responses.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os


# ---------------------------------------------------------------------------
# PlotStyle -- shared styling helpers
# ---------------------------------------------------------------------------

class PlotStyle:
    """Centralised styling and figure management for all project plots.

    Professional publication-quality styling with:
    - Clean, modern color palette (colorblind-friendly)
    - Proper typography and spacing
    - High-DPI output for print and presentations
    - Consistent styling across all plots
    """

    # Professional color palette (colorblind-friendly)
    COLORS = {
        'primary': '#2E86AB',      # Steel blue
        'secondary': '#A23B72',    # Magenta
        'accent1': '#F18F01',      # Orange
        'accent2': '#C73E1D',      # Red
        'accent3': '#3B1F2B',      # Dark purple
        'success': '#2E7D32',      # Green
        'warning': '#F57C00',      # Amber
        'info': '#1976D2',         # Blue
        'neutral': '#546E7A',      # Blue grey
        'light': '#ECEFF1',        # Light grey
    }

    # Color sequences for multi-line plots
    PALETTE = ['#2E86AB', '#F18F01', '#2E7D32', '#C73E1D', '#7B1FA2', '#00838F']

    @staticmethod
    def setup_style():
        """Set matplotlib rcParams for publication-quality figures."""
        plt.rcParams.update({
            # Typography
            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,

            # Figure settings
            'figure.figsize': (10, 6),
            'figure.dpi': 100,
            'figure.facecolor': 'white',
            'figure.edgecolor': 'white',
            'savefig.dpi': 300,
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'none',
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,

            # Axes styling
            'axes.facecolor': 'white',
            'axes.edgecolor': '#333333',
            'axes.linewidth': 1.0,
            'axes.grid': True,
            'axes.axisbelow': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.prop_cycle': plt.cycler(color=PlotStyle.PALETTE),

            # Grid styling
            'grid.color': '#E0E0E0',
            'grid.linestyle': '-',
            'grid.linewidth': 0.5,
            'grid.alpha': 0.7,

            # Line styling
            'lines.linewidth': 2.0,
            'lines.markersize': 6,
            'lines.markeredgewidth': 1.5,

            # Legend styling
            'legend.frameon': True,
            'legend.framealpha': 0.95,
            'legend.facecolor': 'white',
            'legend.edgecolor': '#CCCCCC',
            'legend.borderpad': 0.5,
            'legend.labelspacing': 0.4,

            # Tick styling
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.major.size': 5,
            'ytick.major.size': 5,
            'xtick.minor.size': 3,
            'ytick.minor.size': 3,
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
        })

    @staticmethod
    def create_figure(nrows=1, ncols=1, figsize=None):
        """Return (fig, axes) with tight_layout enabled.

        Parameters
        ----------
        nrows, ncols : int
            Subplot grid dimensions.
        figsize : tuple or None
            Figure size in inches.  If *None*, use rcParams default.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : matplotlib.axes.Axes or ndarray of Axes
        """
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        fig.set_tight_layout(True)
        return fig, axes

    @staticmethod
    def save_figure(fig, filepath, dpi=300):
        """Save *fig* to *filepath*, creating directories as needed.

        Uses high DPI (300) for print-quality output.
        """
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)

    @staticmethod
    def add_mission_phase_bands(ax, phase_times, phase_names, alpha=0.1):
        """Shade alternating mission-phase bands on *ax*.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        phase_times : list of (t_start, t_end)
            Start / end time for each phase.
        phase_names : list of str
            Human-readable name for each phase.
        alpha : float
            Transparency for the shaded bands.
        """
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan', 'magenta']
        for idx, ((t0, t1), name) in enumerate(zip(phase_times, phase_names)):
            colour = colors[idx % len(colors)]
            ax.axvspan(t0, t1, alpha=alpha, color=colour, label=name)


# ---------------------------------------------------------------------------
# Standalone plotting functions
# ---------------------------------------------------------------------------

def plot_trajectory_3d(position_arrays, labels, title, filepath):
    """3-D trajectory plot with spheres for major bodies.

    Parameters
    ----------
    position_arrays : list of ndarray, each (N, 3)
        Position histories to plot.
    labels : list of str
        Legend label for each trajectory.
    title : str
    filepath : str
    """
    PlotStyle.setup_style()
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Draw body spheres (Earth, Moon, Jupiter) -- representative radii in km
    bodies = {
        'Earth':   {'pos': np.array([0.0, 0.0, 0.0]),       'r': 6371.0,   'color': 'royalblue'},
        'Moon':    {'pos': np.array([384400.0, 0.0, 0.0]),   'r': 1737.4,   'color': 'grey'},
        'Jupiter': {'pos': np.array([7.785e8, 0.0, 0.0]),    'r': 69911.0,  'color': 'sandybrown'},
    }

    for body_name, info in bodies.items():
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        scale = info['r'] * 50  # visual scaling for readability
        xs = scale * np.outer(np.cos(u), np.sin(v)) + info['pos'][0]
        ys = scale * np.outer(np.sin(u), np.sin(v)) + info['pos'][1]
        zs = scale * np.outer(np.ones_like(u), np.cos(v)) + info['pos'][2]
        ax.plot_surface(xs, ys, zs, color=info['color'], alpha=0.6)
        ax.scatter(*info['pos'], color=info['color'], s=40, label=body_name)

    # Plot each trajectory
    cmap_colours = cm.get_cmap('tab10')
    for idx, (pos, lbl) in enumerate(zip(position_arrays, labels)):
        pos = np.asarray(pos)
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                color=cmap_colours(idx), label=lbl, linewidth=1.8)

    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=9)
    PlotStyle.save_figure(fig, filepath)


def plot_orbit_groundtrack(lats_deg, lons_deg, title, filepath):
    """Scatter plot ground track on a rectangular lat/lon grid.

    Parameters
    ----------
    lats_deg, lons_deg : array-like
    title : str
    filepath : str
    """
    PlotStyle.setup_style()
    fig, ax = PlotStyle.create_figure(figsize=(12, 6))

    # Approximate coastline rectangles for Earth reference
    coastline_rects = [
        (-10, -80, 60, 110),   # Africa-ish
        (5, -170, 70, 60),     # North America-ish
        (-55, -80, 60, 30),    # South America-ish
        (10, 60, 55, 90),      # Asia-ish
        (-45, 110, 35, 50),    # Australia-ish
        (35, -15, 35, 50),     # Europe-ish
    ]
    for (lat0, lon0, dlat, dlon) in coastline_rects:
        rect = plt.Rectangle((lon0, lat0), dlon, dlat,
                              linewidth=0.8, edgecolor='green',
                              facecolor='palegreen', alpha=0.25)
        ax.add_patch(rect)

    ax.scatter(lons_deg, lats_deg, s=2, c='navy', alpha=0.6)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.set_title(title)
    PlotStyle.save_figure(fig, filepath)


def plot_state_history(times, state_arrays, labels, title, filepath):
    """Plot one or more state signals vs time.

    If more than 4 signals are provided, they are split across subplots.

    Parameters
    ----------
    times : array-like
    state_arrays : list of array-like
    labels : list of str
    title : str
    filepath : str
    """
    PlotStyle.setup_style()
    n_signals = len(state_arrays)

    if n_signals <= 4:
        fig, ax = PlotStyle.create_figure(figsize=(10, 6))
        for sig, lbl in zip(state_arrays, labels):
            ax.plot(times, sig, label=lbl)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('State')
        ax.set_title(title)
        ax.legend()
    else:
        nrows = int(np.ceil(n_signals / 2))
        fig, axes = PlotStyle.create_figure(nrows=nrows, ncols=2,
                                            figsize=(14, 3.5 * nrows))
        axes = np.atleast_2d(axes).flatten()
        for idx, (sig, lbl) in enumerate(zip(state_arrays, labels)):
            axes[idx].plot(times, sig)
            axes[idx].set_title(lbl)
            axes[idx].set_xlabel('Time [s]')
        # Hide unused subplots
        for j in range(n_signals, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(title, fontsize=15)

    PlotStyle.save_figure(fig, filepath)


def plot_control_response(times, commanded, actual, labels, title, filepath):
    """Plot commanded vs actual control signals for X, Y, Z axes.

    Parameters
    ----------
    times : array-like
    commanded : ndarray (N, 3)
    actual : ndarray (N, 3)
    labels : list of str  -- e.g. ['Roll', 'Pitch', 'Yaw']
    title : str
    filepath : str
    """
    PlotStyle.setup_style()
    fig, axes = PlotStyle.create_figure(nrows=3, ncols=1, figsize=(10, 10))
    commanded = np.asarray(commanded)
    actual = np.asarray(actual)

    for i in range(3):
        axes[i].plot(times, commanded[:, i], '--', label=f'{labels[i]} cmd',
                     linewidth=1.4)
        axes[i].plot(times, actual[:, i], '-', label=f'{labels[i]} actual',
                     linewidth=1.4)
        axes[i].set_ylabel(labels[i])
        axes[i].legend(loc='upper right')

    axes[2].set_xlabel('Time [s]')
    fig.suptitle(title, fontsize=15)
    PlotStyle.save_figure(fig, filepath)


def plot_pointing_error(times, errors_deg, requirement_deg, title, filepath):
    """Plot pointing error vs time with a requirement threshold line.

    Regions where error exceeds the requirement are shaded red.

    Parameters
    ----------
    times : array-like
    errors_deg : array-like
    requirement_deg : float
    title : str
    filepath : str
    """
    PlotStyle.setup_style()
    fig, ax = PlotStyle.create_figure(figsize=(10, 5))
    times = np.asarray(times)
    errors_deg = np.asarray(errors_deg)

    ax.plot(times, errors_deg, color='steelblue', label='Pointing error')
    ax.axhline(y=requirement_deg, color='red', linestyle='--',
               label=f'Requirement ({requirement_deg:.2f} deg)')

    # Shade violations
    violation = errors_deg > requirement_deg
    ax.fill_between(times, errors_deg, requirement_deg,
                    where=violation, color='red', alpha=0.20,
                    label='Violation')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Pointing Error [deg]')
    ax.set_title(title)
    ax.legend()
    PlotStyle.save_figure(fig, filepath)


def plot_filter_performance(times, errors, covariances_3sigma, labels,
                            title, filepath):
    """Plot estimation errors with +/-3-sigma covariance bounds.

    Parameters
    ----------
    times : array-like
    errors : list of array-like  -- one per state being estimated
    covariances_3sigma : list of array-like  -- 3-sigma bounds
    labels : list of str
    title : str
    filepath : str
    """
    PlotStyle.setup_style()
    n = len(errors)
    fig, axes = PlotStyle.create_figure(nrows=n, ncols=1,
                                        figsize=(10, 3.2 * n))
    if n == 1:
        axes = [axes]

    for i in range(n):
        err = np.asarray(errors[i])
        cov3 = np.asarray(covariances_3sigma[i])
        axes[i].plot(times, err, color='steelblue', label='Error')
        axes[i].plot(times, cov3, 'r--', linewidth=1.0, label='+3$\\sigma$')
        axes[i].plot(times, -cov3, 'r--', linewidth=1.0, label='-3$\\sigma$')
        axes[i].fill_between(times, -cov3, cov3, color='red', alpha=0.08)
        axes[i].set_ylabel(labels[i])
        axes[i].legend(loc='upper right', fontsize=9)

    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(title, fontsize=15)
    PlotStyle.save_figure(fig, filepath)


def plot_quaternion_history(times, quats, title, filepath):
    """Plot quaternion components w, x, y, z in four subplots.

    Parameters
    ----------
    times : array-like
    quats : ndarray (N, 4)  -- [w, x, y, z] per row
    title : str
    filepath : str
    """
    PlotStyle.setup_style()
    fig, axes = PlotStyle.create_figure(nrows=4, ncols=1, figsize=(10, 10))
    quats = np.asarray(quats)
    comp_labels = ['$q_w$', '$q_x$', '$q_y$', '$q_z$']
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i in range(4):
        axes[i].plot(times, quats[:, i], color=colours[i])
        axes[i].set_ylabel(comp_labels[i])

    axes[3].set_xlabel('Time [s]')
    fig.suptitle(title, fontsize=15)
    PlotStyle.save_figure(fig, filepath)


def plot_angular_velocity(times, omegas, title, filepath):
    """Plot angular velocity components in three subplots.

    Parameters
    ----------
    times : array-like
    omegas : ndarray (N, 3)  -- [wx, wy, wz] per row
    title : str
    filepath : str
    """
    PlotStyle.setup_style()
    fig, axes = PlotStyle.create_figure(nrows=3, ncols=1, figsize=(10, 8))
    omegas = np.asarray(omegas)
    comp_labels = ['$\\omega_x$', '$\\omega_y$', '$\\omega_z$']
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i in range(3):
        axes[i].plot(times, omegas[:, i], color=colours[i])
        axes[i].set_ylabel(f'{comp_labels[i]} [rad/s]')

    axes[2].set_xlabel('Time [s]')
    fig.suptitle(title, fontsize=15)
    PlotStyle.save_figure(fig, filepath)


def plot_delta_v_budget(phases, delta_vs, title, filepath):
    """Horizontal bar chart of delta-V per mission phase.

    Parameters
    ----------
    phases : list of str
    delta_vs : list of float  -- km/s per phase
    title : str
    filepath : str
    """
    PlotStyle.setup_style()
    fig, ax = PlotStyle.create_figure(figsize=(10, max(5, 0.6 * len(phases))))

    y_pos = np.arange(len(phases))
    colours = cm.get_cmap('viridis')(np.linspace(0.2, 0.85, len(phases)))
    bars = ax.barh(y_pos, delta_vs, color=colours, edgecolor='black',
                   linewidth=0.5)

    # Annotate bar values
    for bar, dv in zip(bars, delta_vs):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f'{dv:.2f} km/s', va='center', fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(phases)
    ax.set_xlabel('Delta-V [km/s]')
    ax.set_title(title)
    ax.invert_yaxis()
    PlotStyle.save_figure(fig, filepath)


def plot_mass_history(times, masses, title, filepath):
    """Plot spacecraft mass vs time with phase annotations.

    Parameters
    ----------
    times : array-like
    masses : array-like
    title : str
    filepath : str
    """
    PlotStyle.setup_style()
    fig, ax = PlotStyle.create_figure(figsize=(10, 5))
    ax.plot(times, masses, color='darkgreen', linewidth=2)
    ax.fill_between(times, masses, alpha=0.15, color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Mass [kg]')
    ax.set_title(title)
    PlotStyle.save_figure(fig, filepath)
