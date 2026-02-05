"""
===============================================================================
GNC PROJECT - Sophisticated Trajectory Visualization System
===============================================================================

Professional-quality trajectory plots for the Miami-Moon-Jupiter mission.

Features:
  - Multi-scenario comparison (Hohmann, bi-elliptic, gravity assist)
  - 3D interactive trajectory visualization
  - Phase-specific detailed plots
  - Monte Carlo dispersion analysis
  - Trade study comparison charts
  - Publication-quality formatting

This module generates all trajectory plots for:
  1. Launch and ascent phase
  2. Earth-Moon transfer
  3. Lunar orbit operations
  4. Jupiter transfer (different trajectory options)
  5. Jupiter orbit operations
  6. Return trajectory
  7. Earth reentry and landing

===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.gridspec as gridspec
from typing import List, Tuple, Dict, Optional
import os

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Custom color palette (professional, colorblind-friendly)
COLORS = {
    'earth': '#3498db',
    'moon': '#95a5a6',
    'jupiter': '#e67e22',
    'sun': '#f1c40f',
    'trajectory_nominal': '#2c3e50',
    'trajectory_alt1': '#e74c3c',
    'trajectory_alt2': '#27ae60',
    'trajectory_alt3': '#9b59b6',
    'monte_carlo': '#bdc3c7',
    'highlight': '#1abc9c',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'success': '#27ae60',
}

# Physical constants
MU_EARTH = 3.986004418e14  # m^3/s^2
MU_MOON = 4.9048695e12
MU_JUPITER = 1.26686534e17
MU_SUN = 1.32712440018e20
R_EARTH = 6.371e6  # m
R_MOON = 1.737e6
R_JUPITER = 6.991e7
AU = 1.496e11  # m


class TrajectoryScenario:
    """Represents a complete trajectory scenario for comparison."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.phases: List[Dict] = []
        self.delta_v_total = 0.0
        self.time_of_flight = 0.0
        self.propellant_mass = 0.0

    def add_phase(self, phase_name: str, positions: np.ndarray,
                  velocities: np.ndarray = None, times: np.ndarray = None,
                  delta_v: float = 0.0, color: str = None):
        """Add a trajectory phase."""
        self.phases.append({
            'name': phase_name,
            'positions': positions,
            'velocities': velocities,
            'times': times,
            'delta_v': delta_v,
            'color': color or COLORS['trajectory_nominal']
        })
        self.delta_v_total += delta_v


def generate_orbital_trajectory(mu: float, r_start: float, r_end: float,
                                 theta_range: Tuple[float, float] = (0, 2*np.pi),
                                 n_points: int = 1000,
                                 eccentricity: float = 0.0) -> np.ndarray:
    """
    Generate 3D orbital trajectory points.

    Args:
        mu: Gravitational parameter [m^3/s^2]
        r_start: Starting radius [m]
        r_end: Ending radius [m] (for transfer orbits)
        theta_range: True anomaly range [rad]
        n_points: Number of points
        eccentricity: Orbital eccentricity (0 = circular)

    Returns:
        positions: (n_points, 3) array of XYZ positions
    """
    theta = np.linspace(theta_range[0], theta_range[1], n_points)

    if eccentricity < 0.001:
        # Circular orbit
        r = r_start
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros_like(theta)
    else:
        # Elliptical orbit
        a = (r_start + r_end) / 2  # Semi-major axis
        e = abs(r_end - r_start) / (r_end + r_start)  # Eccentricity
        r = a * (1 - e**2) / (1 + e * np.cos(theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros_like(theta)

    return np.column_stack([x, y, z])


def generate_hohmann_transfer(r1: float, r2: float, mu: float,
                              n_points: int = 500) -> Tuple[np.ndarray, float, float]:
    """
    Generate Hohmann transfer trajectory.

    Returns:
        positions: Transfer trajectory points
        delta_v1: First burn delta-V [m/s]
        delta_v2: Second burn delta-V [m/s]
    """
    # Transfer orbit parameters
    a_transfer = (r1 + r2) / 2
    e_transfer = abs(r2 - r1) / (r2 + r1)

    # Delta-V calculations
    v_c1 = np.sqrt(mu / r1)  # Circular velocity at r1
    v_c2 = np.sqrt(mu / r2)  # Circular velocity at r2
    v_p = np.sqrt(mu * (2/r1 - 1/a_transfer))  # Velocity at periapsis
    v_a = np.sqrt(mu * (2/r2 - 1/a_transfer))  # Velocity at apoapsis

    delta_v1 = v_p - v_c1
    delta_v2 = v_c2 - v_a

    # Generate transfer arc (half ellipse)
    theta = np.linspace(0, np.pi, n_points)
    r = a_transfer * (1 - e_transfer**2) / (1 + e_transfer * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros_like(theta)

    positions = np.column_stack([x, y, z])
    return positions, delta_v1, delta_v2


def generate_bielliptic_transfer(r1: float, r2: float, r_intermediate: float,
                                  mu: float, n_points: int = 500) -> Tuple[np.ndarray, float]:
    """
    Generate bi-elliptic transfer trajectory.

    Bi-elliptic is more efficient than Hohmann when r2/r1 > 11.94

    Returns:
        positions: Transfer trajectory points
        delta_v_total: Total delta-V [m/s]
    """
    # First ellipse: r1 to r_intermediate
    a1 = (r1 + r_intermediate) / 2
    e1 = (r_intermediate - r1) / (r_intermediate + r1)

    # Second ellipse: r_intermediate to r2
    a2 = (r_intermediate + r2) / 2
    e2 = abs(r_intermediate - r2) / (r_intermediate + r2)

    # Delta-V calculations
    v_c1 = np.sqrt(mu / r1)
    v_c2 = np.sqrt(mu / r2)
    v_p1 = np.sqrt(mu * (2/r1 - 1/a1))
    v_a1 = np.sqrt(mu * (2/r_intermediate - 1/a1))
    v_p2 = np.sqrt(mu * (2/r_intermediate - 1/a2))
    v_a2 = np.sqrt(mu * (2/r2 - 1/a2))

    dv1 = v_p1 - v_c1
    dv2 = v_p2 - v_a1
    dv3 = v_c2 - v_a2

    delta_v_total = abs(dv1) + abs(dv2) + abs(dv3)

    # Generate trajectory
    theta1 = np.linspace(0, np.pi, n_points // 2)
    r_arc1 = a1 * (1 - e1**2) / (1 + e1 * np.cos(theta1))

    theta2 = np.linspace(np.pi, 2*np.pi, n_points // 2)
    r_arc2 = a2 * (1 - e2**2) / (1 + e2 * np.cos(theta2 - np.pi))

    x1 = r_arc1 * np.cos(theta1)
    y1 = r_arc1 * np.sin(theta1)
    x2 = r_arc2 * np.cos(theta2)
    y2 = r_arc2 * np.sin(theta2)

    positions = np.column_stack([
        np.concatenate([x1, x2]),
        np.concatenate([y1, y2]),
        np.zeros(n_points)
    ])

    return positions, delta_v_total


def plot_full_mission_3d(scenarios: List[TrajectoryScenario],
                         output_path: str,
                         show_bodies: bool = True):
    """
    Create a comprehensive 3D plot of the full mission trajectory
    comparing multiple scenarios.
    """
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Set up axes
    ax.set_xlabel('X (km)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y (km)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z (km)', fontsize=12, labelpad=10)

    # Color-coded scenarios
    scenario_colors = [COLORS['trajectory_nominal'],
                       COLORS['trajectory_alt1'],
                       COLORS['trajectory_alt2'],
                       COLORS['trajectory_alt3']]

    # Plot each scenario
    for i, scenario in enumerate(scenarios):
        color = scenario_colors[i % len(scenario_colors)]

        for phase in scenario.phases:
            pos = phase['positions'] / 1000  # Convert to km
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                    color=phase.get('color', color),
                    label=f"{scenario.name}: {phase['name']}",
                    alpha=0.8)

    # Draw Earth at origin
    if show_bodies:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        r_e = R_EARTH / 1000
        x = r_e * np.cos(u) * np.sin(v)
        y = r_e * np.sin(u) * np.sin(v)
        z = r_e * np.cos(v)
        ax.plot_surface(x, y, z, color=COLORS['earth'], alpha=0.7)

    ax.set_title('Mission Trajectory Comparison\nMiami → Moon → Jupiter → Miami',
                 fontsize=16, fontweight='bold', pad=20)

    # Add legend with Delta-V summary
    legend_text = []
    for scenario in scenarios:
        legend_text.append(f"{scenario.name}: ΔV = {scenario.delta_v_total/1000:.2f} km/s")
    ax.legend(title='Trajectory Options', loc='upper left')

    # Add text box with summary
    textstr = '\n'.join(legend_text)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    fig.text(0.02, 0.98, textstr, transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def plot_earth_departure(output_path: str):
    """
    Create detailed plot of Earth departure showing launch, parking orbit,
    and Trans-Lunar Injection.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # === Subplot 1: Launch trajectory profile ===
    ax1 = axes[0, 0]

    # Simulated launch data
    t = np.linspace(0, 600, 1000)  # 10 minutes
    altitude = np.where(t < 170,
                        0.5 * 30 * (t**2) / 1000,  # Stage 1
                        25 + 0.5 * 15 * ((t-170)**2) / 1000)  # Stage 2
    altitude = np.clip(altitude, 0, 200)

    velocity = np.where(t < 170,
                        30 * t / 1000,
                        5 + 15 * (t-170) / 1000)
    velocity = np.clip(velocity, 0, 8)

    ax1.plot(t, altitude, color=COLORS['trajectory_nominal'], linewidth=2,
             label='Altitude (km)')
    ax1.axvline(x=170, color=COLORS['warning'], linestyle='--', alpha=0.7,
                label='Stage 1 Sep')
    ax1.axhline(y=110, color=COLORS['highlight'], linestyle=':', alpha=0.7,
                label='Fairing Jettison')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Altitude (km)')
    ax1.set_title('Launch Trajectory Profile', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.set_xlim(0, 600)
    ax1.set_ylim(0, 250)

    # Add secondary axis for velocity
    ax1b = ax1.twinx()
    ax1b.plot(t, velocity, color=COLORS['trajectory_alt1'], linewidth=2,
              linestyle='--', label='Velocity (km/s)')
    ax1b.set_ylabel('Velocity (km/s)', color=COLORS['trajectory_alt1'])
    ax1b.legend(loc='upper right')

    # === Subplot 2: Parking orbit and TLI ===
    ax2 = axes[0, 1]

    # Earth
    theta_earth = np.linspace(0, 2*np.pi, 100)
    ax2.fill(R_EARTH/1000 * np.cos(theta_earth),
             R_EARTH/1000 * np.sin(theta_earth),
             color=COLORS['earth'], alpha=0.7, label='Earth')

    # Parking orbit (200 km)
    r_park = (R_EARTH + 200e3) / 1000
    ax2.plot(r_park * np.cos(theta_earth), r_park * np.sin(theta_earth),
             color=COLORS['trajectory_nominal'], linewidth=2, label='Parking Orbit')

    # TLI burn location
    ax2.scatter([r_park], [0], c=COLORS['highlight'], s=100, zorder=5,
                marker='*', label='TLI Burn Point')

    # Outbound trajectory (start of Hohmann)
    theta_tli = np.linspace(0, 0.3, 50)
    r_tli = r_park * (1 + 0.5 * theta_tli)
    ax2.plot(r_tli * np.cos(theta_tli), r_tli * np.sin(theta_tli),
             color=COLORS['trajectory_alt2'], linewidth=2, linestyle='--',
             label='TLI Trajectory')

    ax2.set_xlabel('X (km)')
    ax2.set_ylabel('Y (km)')
    ax2.set_title('Parking Orbit & Trans-Lunar Injection', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_aspect('equal')
    ax2.set_xlim(-15000, 15000)
    ax2.set_ylim(-15000, 15000)

    # === Subplot 3: Dynamic pressure (Max-Q) ===
    ax3 = axes[1, 0]

    # Simulated dynamic pressure
    q = np.where(altitude < 80,
                 0.5 * 1.225 * np.exp(-altitude/8.5) * (velocity*1000)**2,
                 0)
    q = q / 1000  # Convert to kPa

    ax3.fill_between(t, 0, q, color=COLORS['trajectory_nominal'], alpha=0.3)
    ax3.plot(t, q, color=COLORS['trajectory_nominal'], linewidth=2)

    # Mark Max-Q
    max_q_idx = np.argmax(q)
    ax3.scatter([t[max_q_idx]], [q[max_q_idx]], color=COLORS['danger'], s=100,
                zorder=5, label=f'Max-Q: {q[max_q_idx]:.1f} kPa')
    ax3.axhline(y=q[max_q_idx], color=COLORS['danger'], linestyle='--', alpha=0.5)

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Dynamic Pressure (kPa)')
    ax3.set_title('Dynamic Pressure Profile (Max-Q Analysis)', fontweight='bold')
    ax3.legend()
    ax3.set_xlim(0, 200)

    # === Subplot 4: Acceleration profile ===
    ax4 = axes[1, 1]

    # Simulated g-loading
    g_load = np.where(t < 170,
                      1 + 0.02 * t,  # Stage 1 increasing
                      1 + 0.015 * (t - 170))  # Stage 2
    g_load = np.clip(g_load, 1, 4.5)

    ax4.plot(t, g_load, color=COLORS['trajectory_nominal'], linewidth=2)
    ax4.axhline(y=4, color=COLORS['warning'], linestyle='--', alpha=0.7,
                label='Human Limit (4g)')
    ax4.fill_between(t, 0, g_load, alpha=0.2, color=COLORS['trajectory_nominal'])

    # Mark staging
    ax4.axvline(x=170, color=COLORS['danger'], linestyle='-', alpha=0.7)
    ax4.annotate('Stage Sep', xy=(170, 3), fontsize=9, color=COLORS['danger'])

    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel("Acceleration (g's)")
    ax4.set_title('Acceleration Profile', fontweight='bold')
    ax4.legend(loc='upper left')
    ax4.set_xlim(0, 600)
    ax4.set_ylim(0, 5)

    plt.suptitle('Earth Departure Phase Analysis\nMiami Launch Site (25.76°N, 80.19°W)',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def plot_jupiter_transfer_comparison(output_path: str):
    """
    Compare different Jupiter transfer strategies:
    1. Direct Hohmann
    2. Bi-elliptic
    3. Venus gravity assist (VGA)
    4. Earth gravity assist (EGA)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Orbital radii (AU -> m)
    r_earth = 1.0 * AU
    r_jupiter = 5.2 * AU
    r_venus = 0.72 * AU

    # === Subplot 1: Hohmann Transfer ===
    ax1 = axes[0, 0]

    # Sun
    ax1.scatter([0], [0], c=COLORS['sun'], s=500, zorder=10, label='Sun')

    # Earth orbit
    theta = np.linspace(0, 2*np.pi, 200)
    ax1.plot(r_earth/AU * np.cos(theta), r_earth/AU * np.sin(theta),
             color=COLORS['earth'], linewidth=2, label='Earth Orbit')

    # Jupiter orbit
    ax1.plot(r_jupiter/AU * np.cos(theta), r_jupiter/AU * np.sin(theta),
             color=COLORS['jupiter'], linewidth=2, label='Jupiter Orbit')

    # Hohmann transfer
    pos_h, dv1_h, dv2_h = generate_hohmann_transfer(r_earth, r_jupiter, MU_SUN)
    ax1.plot(pos_h[:, 0]/AU, pos_h[:, 1]/AU, color=COLORS['trajectory_nominal'],
             linewidth=3, linestyle='--', label='Hohmann Transfer')

    # Add markers
    ax1.scatter([1], [0], c=COLORS['earth'], s=150, zorder=5, marker='o')
    ax1.scatter([-5.2], [0], c=COLORS['jupiter'], s=200, zorder=5, marker='o')

    dv_total_h = abs(dv1_h) + abs(dv2_h)
    tof_h = np.pi * np.sqrt(((r_earth + r_jupiter)/2)**3 / MU_SUN) / (365.25*24*3600)

    ax1.set_title(f'1. Direct Hohmann Transfer\nΔV = {dv_total_h/1000:.2f} km/s, '
                  f'ToF = {tof_h:.1f} years', fontweight='bold')
    ax1.set_xlabel('X (AU)')
    ax1.set_ylabel('Y (AU)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_aspect('equal')
    ax1.set_xlim(-7, 3)
    ax1.set_ylim(-5, 5)

    # === Subplot 2: Bi-elliptic ===
    ax2 = axes[0, 1]

    # Orbits
    ax2.scatter([0], [0], c=COLORS['sun'], s=500, zorder=10)
    ax2.plot(r_earth/AU * np.cos(theta), r_earth/AU * np.sin(theta),
             color=COLORS['earth'], linewidth=2)
    ax2.plot(r_jupiter/AU * np.cos(theta), r_jupiter/AU * np.sin(theta),
             color=COLORS['jupiter'], linewidth=2)

    # Bi-elliptic with intermediate at 10 AU
    r_int = 10 * AU
    pos_be, dv_be = generate_bielliptic_transfer(r_earth, r_jupiter, r_int, MU_SUN)
    ax2.plot(pos_be[:, 0]/AU, pos_be[:, 1]/AU, color=COLORS['trajectory_alt1'],
             linewidth=3, linestyle='--', label='Bi-elliptic')

    # Outer apoapsis
    ax2.scatter([0], [10], c=COLORS['highlight'], s=100, marker='x',
                label=f'Apoapsis at 10 AU')

    ax2.set_title(f'2. Bi-elliptic Transfer (10 AU apoapsis)\nΔV = {dv_be/1000:.2f} km/s, '
                  f'ToF > Hohmann', fontweight='bold')
    ax2.set_xlabel('X (AU)')
    ax2.set_ylabel('Y (AU)')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_aspect('equal')
    ax2.set_xlim(-12, 4)
    ax2.set_ylim(-6, 12)

    # === Subplot 3: Venus Gravity Assist ===
    ax3 = axes[1, 0]

    ax3.scatter([0], [0], c=COLORS['sun'], s=500, zorder=10)
    ax3.plot(r_earth/AU * np.cos(theta), r_earth/AU * np.sin(theta),
             color=COLORS['earth'], linewidth=2, label='Earth')
    ax3.plot(r_venus/AU * np.cos(theta), r_venus/AU * np.sin(theta),
             color='#e91e63', linewidth=2, label='Venus')
    ax3.plot(r_jupiter/AU * np.cos(theta), r_jupiter/AU * np.sin(theta),
             color=COLORS['jupiter'], linewidth=2, label='Jupiter')

    # VGA trajectory (simplified)
    t_vga = np.linspace(0, 2*np.pi, 300)
    r_vga1 = r_earth/AU - 0.28 * t_vga/np.pi  # To Venus
    r_vga2 = r_venus/AU + 4.48 * (t_vga - np.pi)/np.pi  # Venus to Jupiter

    x_vga = np.where(t_vga < np.pi,
                     r_vga1 * np.cos(t_vga * 0.5),
                     r_vga2 * np.cos(np.pi + (t_vga - np.pi) * 0.7))
    y_vga = np.where(t_vga < np.pi,
                     r_vga1 * np.sin(t_vga * 0.5),
                     r_vga2 * np.sin(np.pi + (t_vga - np.pi) * 0.7))

    ax3.plot(x_vga[::3], y_vga[::3], color=COLORS['trajectory_alt2'],
             linewidth=3, linestyle='--', label='VGA Trajectory')

    # Venus encounter
    ax3.scatter([0.72 * np.cos(np.pi/2)], [0.72 * np.sin(np.pi/2)],
                c='#e91e63', s=100, marker='*', zorder=10)

    dv_vga = 8500  # Approximate
    ax3.set_title(f'3. Venus Gravity Assist (VGA)\nΔV ≈ {dv_vga/1000:.2f} km/s, '
                  f'ToF ≈ 3-4 years', fontweight='bold')
    ax3.set_xlabel('X (AU)')
    ax3.set_ylabel('Y (AU)')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.set_aspect('equal')
    ax3.set_xlim(-7, 3)
    ax3.set_ylim(-3, 7)

    # === Subplot 4: Comparison Summary ===
    ax4 = axes[1, 1]

    strategies = ['Hohmann', 'Bi-elliptic\n(10 AU)', 'Venus GA', 'Earth GA\n(VEEGA)']
    delta_vs = [dv_total_h/1000, dv_be/1000, 8.5, 6.5]  # km/s
    tofs = [tof_h, tof_h * 1.8, 3.5, 6.0]  # years

    x_pos = np.arange(len(strategies))
    width = 0.35

    bars1 = ax4.bar(x_pos - width/2, delta_vs, width,
                    label='Total ΔV (km/s)', color=COLORS['trajectory_nominal'])
    bars2 = ax4.bar(x_pos + width/2, tofs, width,
                    label='Time of Flight (years)', color=COLORS['trajectory_alt1'])

    ax4.set_ylabel('Value')
    ax4.set_title('Transfer Strategy Comparison', fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(strategies)
    ax4.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f}',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f}',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)

    ax4.axhline(y=12.45, color=COLORS['danger'], linestyle='--', alpha=0.7,
                label='Mission Budget ΔV')
    ax4.axhline(y=4.5, color=COLORS['warning'], linestyle=':', alpha=0.7,
                label='Mission Target ToF')

    plt.suptitle('Jupiter Transfer Trajectory Trade Study',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def plot_mission_timeline(output_path: str):
    """
    Create a comprehensive mission timeline visualization.
    """
    fig, ax = plt.subplots(figsize=(16, 10))

    # Mission phases with timing (days from launch)
    phases = [
        ('Launch', 0, 0.01, COLORS['danger']),
        ('Stage 1', 0, 0.002, COLORS['trajectory_nominal']),
        ('Stage 2', 0.002, 0.007, COLORS['trajectory_nominal']),
        ('Parking Orbit', 0.007, 0.04, COLORS['earth']),
        ('TLI Burn', 0.04, 0.041, COLORS['highlight']),
        ('Lunar Coast', 0.041, 3, COLORS['trajectory_nominal']),
        ('LOI Burn', 3, 3.01, COLORS['highlight']),
        ('Lunar Orbit 1', 3.01, 3.5, COLORS['moon']),
        ('Inclination Change', 3.5, 3.51, COLORS['highlight']),
        ('Lunar Orbit 2', 3.51, 4, COLORS['moon']),
        ('Lunar Escape', 4, 4.01, COLORS['highlight']),
        ('Earth-Jupiter Transfer', 4.01, 730, COLORS['trajectory_alt2']),
        ('JOI Burn', 730, 730.01, COLORS['highlight']),
        ('Jupiter Orbit (3 rev)', 730.01, 760, COLORS['jupiter']),
        ('Jupiter Escape', 760, 760.01, COLORS['highlight']),
        ('Jupiter-Earth Return', 760.01, 1490, COLORS['trajectory_alt3']),
        ('Earth Reentry', 1490, 1490.1, COLORS['danger']),
        ('Landing', 1490.1, 1490.15, COLORS['success']),
    ]

    # Plot phases as horizontal bars
    y_pos = 0.5
    for i, (name, start, end, color) in enumerate(phases):
        duration = end - start
        if duration < 0.1:
            duration = 0.5  # Minimum visible width

        ax.barh(y_pos, duration, left=start, height=0.3, color=color, alpha=0.8,
                edgecolor='white', linewidth=1)

        # Add labels
        if duration > 30:
            ax.text(start + duration/2, y_pos, name, ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')
        else:
            ax.text(end + 5, y_pos, name, ha='left', va='center',
                    fontsize=8)

    # Format x-axis
    ax.set_xlabel('Mission Day', fontsize=12)
    ax.set_xlim(-10, 1550)

    # Create custom x-ticks showing years
    year_days = [0, 365, 730, 1095, 1460]
    year_labels = ['Launch', 'Year 1', 'Year 2', 'Year 3', 'Year 4']
    ax.set_xticks(year_days)
    ax.set_xticklabels(year_labels)

    # Secondary axis for actual dates (assuming launch 2026)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(year_days)
    ax2.set_xticklabels(['2026', '2027', '2028', '2029', '2030'])
    ax2.set_xlabel('Calendar Year')

    ax.set_yticks([])
    ax.set_title('Mission Timeline: Miami → Moon → Jupiter → Miami\n'
                 'Total Duration: ~4.1 Years',
                 fontsize=16, fontweight='bold', pad=20)

    # Add key milestones
    milestones = [
        (0, 'Launch from Miami'),
        (3, 'Moon Arrival'),
        (730, 'Jupiter Arrival'),
        (1490, 'Earth Return'),
    ]

    for day, label in milestones:
        ax.axvline(x=day, color='gray', linestyle='--', alpha=0.5)
        ax.text(day, 0.9, label, rotation=90, ha='right', va='top', fontsize=9)

    # Add delta-V budget annotation
    dv_text = """ΔV Budget Summary:
    TLI:       3,150 m/s
    LOI:         850 m/s
    Lunar Esc:   900 m/s
    JOI:       2,000 m/s
    Jupiter Esc: 2,200 m/s
    Entry:       ~300 m/s
    ─────────────────
    Total:    ~9,400 m/s
    """
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.85, 0.5, dv_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=props, family='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def plot_monte_carlo_dispersions(output_path: str, n_runs: int = 100):
    """
    Generate Monte Carlo dispersion analysis plots showing
    trajectory uncertainty across multiple runs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    np.random.seed(42)

    # Nominal trajectory (simplified launch profile)
    t = np.linspace(0, 600, 100)
    alt_nom = 0.5 * (t/10)**2
    vel_nom = 0.01 * t + 0.00005 * t**2

    # Generate dispersed trajectories
    alt_runs = []
    vel_runs = []
    landing_errors = []

    for _ in range(n_runs):
        # Apply dispersions
        thrust_disp = 1 + 0.03 * np.random.randn()
        mass_disp = 1 + 0.02 * np.random.randn()
        atm_disp = 1 + 0.1 * np.random.randn()

        alt = alt_nom * thrust_disp / mass_disp
        vel = vel_nom * thrust_disp / mass_disp

        alt_runs.append(alt)
        vel_runs.append(vel)
        landing_errors.append(np.random.randn() * 5 + np.random.randn() * 5j)

    alt_runs = np.array(alt_runs)
    vel_runs = np.array(vel_runs)

    # === Subplot 1: Altitude dispersions ===
    ax1 = axes[0, 0]
    for alt in alt_runs:
        ax1.plot(t, alt, color=COLORS['monte_carlo'], alpha=0.1, linewidth=0.5)
    ax1.plot(t, alt_nom, color=COLORS['trajectory_nominal'], linewidth=2,
             label='Nominal')
    ax1.plot(t, np.percentile(alt_runs, 95, axis=0), 'r--', linewidth=1.5,
             label='95th percentile')
    ax1.plot(t, np.percentile(alt_runs, 5, axis=0), 'b--', linewidth=1.5,
             label='5th percentile')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Altitude (km)')
    ax1.set_title(f'Altitude Dispersions (N={n_runs} runs)', fontweight='bold')
    ax1.legend(loc='upper left')

    # === Subplot 2: Velocity dispersions ===
    ax2 = axes[0, 1]
    for vel in vel_runs:
        ax2.plot(t, vel, color=COLORS['monte_carlo'], alpha=0.1, linewidth=0.5)
    ax2.plot(t, vel_nom, color=COLORS['trajectory_nominal'], linewidth=2,
             label='Nominal')

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (km/s)')
    ax2.set_title(f'Velocity Dispersions (N={n_runs} runs)', fontweight='bold')
    ax2.legend(loc='upper left')

    # === Subplot 3: Landing dispersion scatter ===
    ax3 = axes[1, 0]
    landing_x = [e.real for e in landing_errors]
    landing_y = [e.imag for e in landing_errors]

    ax3.scatter(landing_x, landing_y, c=COLORS['monte_carlo'], alpha=0.5, s=30)
    ax3.scatter([0], [0], c=COLORS['danger'], s=200, marker='X', zorder=10,
                label='Target (Miami)')

    # Draw 1-sigma and 3-sigma ellipses
    from matplotlib.patches import Ellipse
    sigma_x = np.std(landing_x)
    sigma_y = np.std(landing_y)
    for n_sigma, color in [(1, COLORS['success']), (2, COLORS['warning']),
                            (3, COLORS['danger'])]:
        ellipse = Ellipse((0, 0), 2*n_sigma*sigma_x, 2*n_sigma*sigma_y,
                         fill=False, color=color, linewidth=2,
                         label=f'{n_sigma}σ ({n_sigma*100/3:.0f}% CEP)')
        ax3.add_patch(ellipse)

    ax3.set_xlabel('Downrange Error (km)')
    ax3.set_ylabel('Crossrange Error (km)')
    ax3.set_title('Landing Dispersion Analysis', fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.set_aspect('equal')
    ax3.set_xlim(-20, 20)
    ax3.set_ylim(-20, 20)

    # === Subplot 4: Statistical summary ===
    ax4 = axes[1, 1]

    # Create histogram of final altitudes
    final_alts = alt_runs[:, -1]
    ax4.hist(final_alts, bins=20, color=COLORS['trajectory_nominal'],
             alpha=0.7, edgecolor='white')

    ax4.axvline(x=alt_nom[-1], color=COLORS['danger'], linewidth=2,
                label=f'Nominal: {alt_nom[-1]:.1f} km')
    ax4.axvline(x=np.mean(final_alts), color=COLORS['success'], linewidth=2,
                linestyle='--', label=f'Mean: {np.mean(final_alts):.1f} km')

    ax4.set_xlabel('Final Altitude (km)')
    ax4.set_ylabel('Count')
    ax4.set_title('Final Altitude Distribution', fontweight='bold')
    ax4.legend()

    # Add statistics text box
    stats_text = f"""Statistics:
    Mean: {np.mean(final_alts):.2f} km
    Std Dev: {np.std(final_alts):.2f} km
    Min: {np.min(final_alts):.2f} km
    Max: {np.max(final_alts):.2f} km
    Success Rate: {np.sum(final_alts > 190) / n_runs * 100:.1f}%
    """
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax4.text(0.95, 0.95, stats_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right', bbox=props,
             family='monospace')

    plt.suptitle('Monte Carlo Trajectory Dispersion Analysis',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_plots(output_dir: str):
    """Generate all trajectory plots."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Generating Sophisticated Trajectory Plots")
    print("=" * 60)

    # 1. Mission Timeline
    plot_mission_timeline(os.path.join(output_dir, 'mission_timeline.png'))

    # 2. Earth Departure
    plot_earth_departure(os.path.join(output_dir, 'earth_departure.png'))

    # 3. Jupiter Transfer Comparison
    plot_jupiter_transfer_comparison(os.path.join(output_dir, 'jupiter_transfer_comparison.png'))

    # 4. Monte Carlo Dispersions
    plot_monte_carlo_dispersions(os.path.join(output_dir, 'monte_carlo_dispersions.png'))

    # 5. Full Mission 3D (create scenarios)
    scenarios = []

    # Scenario 1: Nominal Hohmann
    s1 = TrajectoryScenario("Hohmann", "Direct minimum-energy transfer")
    pos_h, dv1, dv2 = generate_hohmann_transfer(R_EARTH + 200e3, 384400e3, MU_EARTH)
    s1.add_phase("Earth-Moon", pos_h, delta_v=abs(dv1)+abs(dv2))
    s1.delta_v_total = 9400  # Total mission
    scenarios.append(s1)

    # Scenario 2: Fast transfer
    s2 = TrajectoryScenario("Fast Transfer", "Higher energy, shorter time")
    s2.add_phase("Earth-Moon", pos_h * 0.8, delta_v=4500)
    s2.delta_v_total = 12000
    scenarios.append(s2)

    plot_full_mission_3d(scenarios,
                         os.path.join(output_dir, 'full_mission_3d.png'))

    print("=" * 60)
    print(f"  All plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'output', 'plots')
    generate_all_plots(output_dir)
