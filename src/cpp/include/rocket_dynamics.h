// =============================================================================
// rocket_dynamics.h -- Multi-Stage Rocket Dynamics Engine
// =============================================================================
//
// OVERVIEW:
//   High-fidelity rocket dynamics for launch vehicle simulation including:
//     - Multi-stage mass depletion and staging
//     - Atmospheric flight (drag, pressure effects on thrust)
//     - Gravity turn trajectory optimization
//     - Structural loads and flex dynamics
//     - Thrust vector control (TVC)
//     - Monte Carlo dispersion support
//
// PHYSICS MODELED:
//
//   TRANSLATIONAL DYNAMICS:
//     m * dv/dt = T - D - m*g + F_external
//
//     where:
//       T = thrust (varies with altitude due to nozzle expansion ratio)
//       D = 0.5 * rho * v^2 * Cd * A  (aerodynamic drag)
//       g = mu / r^2  (gravitational acceleration)
//
//   ROTATIONAL DYNAMICS:
//     I * d(omega)/dt = M_control + M_aero + M_gravity_gradient - omega x (I*omega)
//
//   MASS DEPLETION:
//     dm/dt = -T / (g0 * Isp)
//
//   THRUST VARIATION WITH ALTITUDE:
//     T(h) = T_vac - (P_atm(h) - P_exit) * A_exit
//
//   STAGING:
//     At staging event:
//       - Jettison stage mass
//       - Update inertia tensor
//       - Apply separation impulse
//       - Handle any staging errors (Monte Carlo)
//
// COORDINATE FRAMES:
//   - ECI: Earth-Centered Inertial (J2000)
//   - ECEF: Earth-Centered Earth-Fixed
//   - Body: Rocket body frame (X forward, Z up for launch)
//   - Launch: Local launch site frame
//
// =============================================================================

#ifndef GNC_ROCKET_DYNAMICS_H
#define GNC_ROCKET_DYNAMICS_H

#include <cmath>
#include <cstddef>
#include <array>
#include <vector>
#include <functional>
#include <memory>

namespace gnc {

// ============================================================================
// Physical Constants
// ============================================================================
namespace constants {
    constexpr double MU_EARTH = 3.986004418e14;     // m^3/s^2
    constexpr double R_EARTH = 6.378137e6;          // m (equatorial)
    constexpr double OMEGA_EARTH = 7.2921159e-5;    // rad/s
    constexpr double G0 = 9.80665;                  // m/s^2
    constexpr double P0_SEA_LEVEL = 101325.0;       // Pa
    constexpr double RHO0_SEA_LEVEL = 1.225;        // kg/m^3
    constexpr double SCALE_HEIGHT = 8500.0;         // m
    constexpr double GAMMA_AIR = 1.4;               // Ratio of specific heats
    constexpr double R_GAS = 287.05;                // J/(kg*K)
    constexpr double T0_SEA_LEVEL = 288.15;         // K
    constexpr double PI = 3.14159265358979323846;
    constexpr double DEG2RAD = PI / 180.0;
    constexpr double RAD2DEG = 180.0 / PI;
}

// ============================================================================
// Atmospheric Model (Exponential + US Standard Atmosphere 1976)
// ============================================================================
struct AtmosphericState {
    double density;         // kg/m^3
    double pressure;        // Pa
    double temperature;     // K
    double speed_of_sound;  // m/s
    double mach_number;     // dimensionless (requires velocity input)
    double dynamic_pressure; // Pa (requires velocity input)
};

class AtmosphereModel {
public:
    // Get atmospheric properties at altitude (meters above sea level)
    static AtmosphericState get_state(double altitude_m, double velocity_ms = 0.0);

    // Simple exponential model
    static double density_exponential(double altitude_m);

    // US Standard Atmosphere 1976 (piecewise model)
    static double density_us76(double altitude_m);
    static double pressure_us76(double altitude_m);
    static double temperature_us76(double altitude_m);
};

// ============================================================================
// Engine Model
// ============================================================================
struct EngineSpec {
    double thrust_sl;           // Sea-level thrust [N]
    double thrust_vac;          // Vacuum thrust [N]
    double isp_sl;              // Sea-level Isp [s]
    double isp_vac;             // Vacuum Isp [s]
    double exit_area;           // Nozzle exit area [m^2]
    double exit_pressure;       // Nozzle exit pressure [Pa]
    double mass_flow_rate;      // kg/s (computed or specified)
    double throttle_min;        // Minimum throttle (e.g., 0.6 = 60%)
    double throttle_max;        // Maximum throttle (e.g., 1.0 = 100%)
    double gimbal_max_deg;      // Maximum TVC gimbal angle [deg]
    double gimbal_rate_dps;     // Maximum gimbal rate [deg/s]
    int num_engines;            // Number of engines

    // Compute thrust at altitude with throttle
    double get_thrust(double altitude_m, double throttle = 1.0) const;

    // Compute Isp at altitude
    double get_isp(double altitude_m) const;

    // Compute mass flow rate
    double get_mass_flow(double altitude_m, double throttle = 1.0) const;
};

// ============================================================================
// Stage Definition
// ============================================================================
struct StageConfig {
    std::string name;

    // Mass properties
    double dry_mass;            // kg (structure, engines, avionics)
    double propellant_mass;     // kg
    double fairing_mass;        // kg (if applicable, jettisoned separately)

    // Geometry
    double diameter;            // m
    double length;              // m
    double nose_cone_length;    // m (for drag calculation)

    // Aerodynamics
    double cd_subsonic;         // Drag coefficient (M < 0.8)
    double cd_transonic;        // Drag coefficient (0.8 < M < 1.2)
    double cd_supersonic;       // Drag coefficient (M > 1.2)
    double reference_area;      // m^2 (computed from diameter)

    // Inertia (3x3 tensor, symmetric)
    std::array<double, 9> inertia_full;     // Full propellant
    std::array<double, 9> inertia_empty;    // Empty (dry)

    // Engine
    EngineSpec engine;

    // Staging parameters
    double separation_impulse;  // N*s (small kick at separation)
    double separation_delay;    // s (time between engine cutoff and sep)

    // Get current inertia based on propellant fraction
    void get_inertia(double prop_fraction, double* inertia_out) const;

    // Get drag coefficient based on Mach number
    double get_cd(double mach) const;

    // Get cross-sectional area
    double get_area() const { return constants::PI * 0.25 * diameter * diameter; }
};

// ============================================================================
// Rocket State Vector
// ============================================================================
struct RocketState {
    // Position (ECI) [m]
    double pos[3];

    // Velocity (ECI) [m/s]
    double vel[3];

    // Attitude quaternion (body-to-ECI) [qx, qy, qz, qw]
    double quat[4];

    // Angular velocity (body frame) [rad/s]
    double omega[3];

    // Mass properties
    double total_mass;          // Current total mass [kg]
    double propellant_mass;     // Remaining propellant in current stage [kg]

    // Time
    double time;                // Mission elapsed time [s]

    // Control inputs (for logging)
    double throttle;            // Current throttle [0-1]
    double gimbal[2];           // TVC gimbal angles [pitch, yaw] [rad]

    // Derived quantities (computed by propagator)
    double altitude;            // Altitude above sea level [m]
    double velocity_mag;        // Velocity magnitude [m/s]
    double dynamic_pressure;    // Dynamic pressure [Pa] (max-Q tracking)
    double acceleration_g;      // Acceleration in g's
    double downrange;           // Downrange distance [m]
    double flight_path_angle;   // Flight path angle [rad]
    int current_stage;          // Current stage index (1-based)

    // Initialize to default launch state
    void set_launch_state(double latitude_rad, double longitude_rad,
                          double azimuth_rad);

    // Compute derived quantities from primary state
    void update_derived();
};

// ============================================================================
// Control Command
// ============================================================================
struct RocketCommand {
    double throttle;            // Throttle command [0-1]
    double gimbal_pitch;        // TVC pitch [rad]
    double gimbal_yaw;          // TVC yaw [rad]
    bool stage_arm;             // Staging armed flag
    bool fairing_jettison;      // Fairing jettison command
};

// ============================================================================
// Staging Event
// ============================================================================
struct StagingEvent {
    double time;                // Time of event [s]
    int from_stage;             // Stage being jettisoned
    int to_stage;               // New active stage
    double delta_mass;          // Mass removed [kg]
    bool success;               // Separation success flag
    std::string notes;          // Event notes
};

// ============================================================================
// Trajectory Point (for output/plotting)
// ============================================================================
struct TrajectoryPoint {
    double time;
    double altitude;
    double downrange;
    double velocity;
    double acceleration;
    double dynamic_pressure;
    double mass;
    double flight_path_angle;
    double throttle;
    int stage;

    // Position in various frames
    double pos_eci[3];
    double pos_ecef[3];
    double lat, lon, alt;
};

// ============================================================================
// Rocket Dynamics Engine
// ============================================================================
class RocketDynamics {
public:
    // Constructor with stage configurations
    explicit RocketDynamics(const std::vector<StageConfig>& stages);

    ~RocketDynamics() = default;

    // -------------------------------------------------------------------------
    // Initialization
    // -------------------------------------------------------------------------

    // Set launch site (geodetic coordinates)
    void set_launch_site(double latitude_deg, double longitude_deg,
                         double altitude_m = 0.0);

    // Set launch azimuth
    void set_launch_azimuth(double azimuth_deg);

    // Initialize state to launch configuration
    void initialize();

    // -------------------------------------------------------------------------
    // Propagation
    // -------------------------------------------------------------------------

    // Propagate by dt with given command
    void propagate(double dt, const RocketCommand& cmd);

    // Propagate with RK4 (higher accuracy)
    void propagate_rk4(double dt, const RocketCommand& cmd);

    // -------------------------------------------------------------------------
    // Staging
    // -------------------------------------------------------------------------

    // Execute staging event
    void execute_staging();

    // Check if staging conditions are met
    bool check_staging_conditions() const;

    // -------------------------------------------------------------------------
    // Force/Moment Calculations
    // -------------------------------------------------------------------------

    // Compute total forces (inertial frame)
    void compute_forces(const RocketState& state, const RocketCommand& cmd,
                       double forces[3]) const;

    // Compute total moments (body frame)
    void compute_moments(const RocketState& state, const RocketCommand& cmd,
                        double moments[3]) const;

    // Individual force contributions
    void compute_thrust(const RocketState& state, const RocketCommand& cmd,
                       double thrust[3]) const;
    void compute_drag(const RocketState& state, double drag[3]) const;
    void compute_gravity(const RocketState& state, double gravity[3]) const;

    // -------------------------------------------------------------------------
    // Guidance
    // -------------------------------------------------------------------------

    // Gravity turn guidance (pitch program)
    double gravity_turn_pitch(double altitude, double velocity) const;

    // Terminal guidance for orbit insertion
    RocketCommand compute_terminal_guidance(double target_altitude,
                                            double target_velocity) const;

    // -------------------------------------------------------------------------
    // State Access
    // -------------------------------------------------------------------------

    const RocketState& get_state() const { return state_; }
    RocketState& get_state() { return state_; }

    const StageConfig& get_current_stage_config() const;
    int get_current_stage_index() const { return current_stage_; }
    int get_num_stages() const { return static_cast<int>(stages_.size()); }

    // -------------------------------------------------------------------------
    // Trajectory Recording
    // -------------------------------------------------------------------------

    // Record current state to trajectory
    void record_trajectory_point();

    // Get full trajectory
    const std::vector<TrajectoryPoint>& get_trajectory() const {
        return trajectory_;
    }

    // Clear trajectory
    void clear_trajectory() { trajectory_.clear(); }

    // -------------------------------------------------------------------------
    // Events
    // -------------------------------------------------------------------------

    const std::vector<StagingEvent>& get_staging_events() const {
        return staging_events_;
    }

    // -------------------------------------------------------------------------
    // Statistics
    // -------------------------------------------------------------------------

    struct SimStats {
        double max_q;               // Maximum dynamic pressure [Pa]
        double max_q_time;          // Time of max-Q [s]
        double max_g;               // Maximum acceleration [g's]
        double max_g_time;          // Time of max-g [s]
        double total_delta_v;       // Total delta-V expended [m/s]
        double propellant_used;     // Total propellant consumed [kg]
        std::size_t total_steps;    // Number of integration steps
        double wall_time_s;         // Wall clock time [s]
    };

    SimStats get_stats() const { return stats_; }

private:
    // Stage configurations
    std::vector<StageConfig> stages_;
    int current_stage_;             // 0-based index

    // Current state
    RocketState state_;

    // Launch site parameters
    double launch_lat_rad_;
    double launch_lon_rad_;
    double launch_alt_m_;
    double launch_azimuth_rad_;

    // Trajectory history
    std::vector<TrajectoryPoint> trajectory_;

    // Staging events
    std::vector<StagingEvent> staging_events_;

    // Statistics
    SimStats stats_;

    // Internal helpers
    void compute_state_derivative(const RocketState& state,
                                  const RocketCommand& cmd,
                                  double deriv[14]) const;

    void quaternion_derivative(const double q[4], const double omega[3],
                              double dq[4]) const;

    void normalize_quaternion(double q[4]) const;

    void eci_to_geodetic(const double pos_eci[3],
                         double& lat, double& lon, double& alt) const;

    void body_to_eci(const double vec_body[3], const double q[4],
                    double vec_eci[3]) const;

    void update_stats(const RocketState& state);
};

// ============================================================================
// Trajectory Scenarios (for trade studies and Monte Carlo)
// ============================================================================

struct LaunchScenario {
    std::string name;
    std::string description;

    // Launch parameters
    double latitude_deg;
    double longitude_deg;
    double azimuth_deg;

    // Target orbit
    double target_altitude_km;
    double target_velocity_ms;
    double target_inclination_deg;

    // Vehicle configuration
    std::vector<StageConfig> stages;

    // Dispersion parameters (for Monte Carlo)
    double mass_dispersion_percent;
    double thrust_dispersion_percent;
    double isp_dispersion_percent;
    double wind_dispersion_ms;
};

// Run a single scenario
std::vector<TrajectoryPoint> run_scenario(const LaunchScenario& scenario);

// Run Monte Carlo analysis
struct MonteCarloResult {
    std::vector<std::vector<TrajectoryPoint>> trajectories;
    double success_rate;
    double mean_final_altitude;
    double std_final_altitude;
    double mean_final_velocity;
    double std_final_velocity;
    double mean_propellant_used;
};

MonteCarloResult run_monte_carlo(const LaunchScenario& scenario,
                                 int num_runs,
                                 unsigned int seed = 42);

} // namespace gnc

#endif // GNC_ROCKET_DYNAMICS_H
