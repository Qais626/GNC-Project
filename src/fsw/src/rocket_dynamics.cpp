// =============================================================================
// rocket_dynamics.cpp -- Multi-Stage Rocket Dynamics Implementation
// =============================================================================

#include "rocket_dynamics.h"
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <random>
#include <chrono>

namespace gnc {

// ============================================================================
// Atmosphere Model Implementation
// ============================================================================

AtmosphericState AtmosphereModel::get_state(double altitude_m, double velocity_ms) {
    AtmosphericState state;

    // Use US Standard Atmosphere 1976 for accuracy
    state.density = density_us76(altitude_m);
    state.pressure = pressure_us76(altitude_m);
    state.temperature = temperature_us76(altitude_m);

    // Speed of sound: a = sqrt(gamma * R * T)
    state.speed_of_sound = std::sqrt(constants::GAMMA_AIR *
                                     constants::R_GAS *
                                     state.temperature);

    // Mach number
    state.mach_number = (state.speed_of_sound > 0.0) ?
                        velocity_ms / state.speed_of_sound : 0.0;

    // Dynamic pressure: q = 0.5 * rho * v^2
    state.dynamic_pressure = 0.5 * state.density * velocity_ms * velocity_ms;

    return state;
}

double AtmosphereModel::density_exponential(double altitude_m) {
    if (altitude_m < 0.0) altitude_m = 0.0;
    return constants::RHO0_SEA_LEVEL *
           std::exp(-altitude_m / constants::SCALE_HEIGHT);
}

double AtmosphereModel::density_us76(double altitude_m) {
    // US Standard Atmosphere 1976 - piecewise model
    // Valid from 0 to 86 km, exponential decay above

    if (altitude_m < 0.0) altitude_m = 0.0;

    // Above 86 km - exponential decay
    if (altitude_m > 86000.0) {
        double rho_86 = 6.958e-6;  // kg/m^3 at 86 km
        double scale_high = 6000.0; // Scale height above 86 km
        return rho_86 * std::exp(-(altitude_m - 86000.0) / scale_high);
    }

    // Troposphere (0-11 km)
    if (altitude_m <= 11000.0) {
        double T = 288.15 - 0.0065 * altitude_m;
        double p = 101325.0 * std::pow(T / 288.15, 5.2559);
        return p / (constants::R_GAS * T);
    }

    // Tropopause (11-20 km)
    if (altitude_m <= 20000.0) {
        double T = 216.65;
        double p = 22632.0 * std::exp(-constants::G0 * (altitude_m - 11000.0) /
                                      (constants::R_GAS * T));
        return p / (constants::R_GAS * T);
    }

    // Stratosphere lower (20-32 km)
    if (altitude_m <= 32000.0) {
        double T = 216.65 + 0.001 * (altitude_m - 20000.0);
        double p = 5474.9 * std::pow(T / 216.65, -34.163);
        return p / (constants::R_GAS * T);
    }

    // Stratosphere upper (32-47 km)
    if (altitude_m <= 47000.0) {
        double T = 228.65 + 0.0028 * (altitude_m - 32000.0);
        double p = 868.02 * std::pow(T / 228.65, -12.201);
        return p / (constants::R_GAS * T);
    }

    // Stratopause (47-51 km)
    if (altitude_m <= 51000.0) {
        double T = 270.65;
        double p = 110.91 * std::exp(-constants::G0 * (altitude_m - 47000.0) /
                                     (constants::R_GAS * T));
        return p / (constants::R_GAS * T);
    }

    // Mesosphere lower (51-71 km)
    if (altitude_m <= 71000.0) {
        double T = 270.65 - 0.0028 * (altitude_m - 51000.0);
        double p = 66.939 * std::pow(T / 270.65, 12.201);
        return p / (constants::R_GAS * T);
    }

    // Mesosphere upper (71-86 km)
    double T = 214.65 - 0.002 * (altitude_m - 71000.0);
    double p = 3.9564 * std::pow(T / 214.65, 17.082);
    return p / (constants::R_GAS * T);
}

double AtmosphereModel::pressure_us76(double altitude_m) {
    double rho = density_us76(altitude_m);
    double T = temperature_us76(altitude_m);
    return rho * constants::R_GAS * T;
}

double AtmosphereModel::temperature_us76(double altitude_m) {
    if (altitude_m < 0.0) altitude_m = 0.0;

    if (altitude_m <= 11000.0) return 288.15 - 0.0065 * altitude_m;
    if (altitude_m <= 20000.0) return 216.65;
    if (altitude_m <= 32000.0) return 216.65 + 0.001 * (altitude_m - 20000.0);
    if (altitude_m <= 47000.0) return 228.65 + 0.0028 * (altitude_m - 32000.0);
    if (altitude_m <= 51000.0) return 270.65;
    if (altitude_m <= 71000.0) return 270.65 - 0.0028 * (altitude_m - 51000.0);
    if (altitude_m <= 86000.0) return 214.65 - 0.002 * (altitude_m - 71000.0);

    return 186.65;  // Above 86 km
}

// ============================================================================
// Engine Model Implementation
// ============================================================================

double EngineSpec::get_thrust(double altitude_m, double throttle) const {
    // Thrust varies with altitude due to nozzle pressure difference
    // T = T_vac - (P_atm - P_exit) * A_exit
    // At sea level: T = T_sl
    // In vacuum: T = T_vac

    double p_atm = AtmosphereModel::pressure_us76(altitude_m);

    // Linear interpolation based on pressure
    double p_sl = constants::P0_SEA_LEVEL;
    double thrust_interp;

    if (p_atm >= p_sl) {
        thrust_interp = thrust_sl;
    } else if (p_atm <= 0.0) {
        thrust_interp = thrust_vac;
    } else {
        // Linear interpolation
        double frac = p_atm / p_sl;
        thrust_interp = thrust_vac - frac * (thrust_vac - thrust_sl);
    }

    return thrust_interp * throttle * num_engines;
}

double EngineSpec::get_isp(double altitude_m) const {
    double p_atm = AtmosphereModel::pressure_us76(altitude_m);
    double p_sl = constants::P0_SEA_LEVEL;

    if (p_atm >= p_sl) return isp_sl;
    if (p_atm <= 0.0) return isp_vac;

    // Linear interpolation
    double frac = p_atm / p_sl;
    return isp_vac - frac * (isp_vac - isp_sl);
}

double EngineSpec::get_mass_flow(double altitude_m, double throttle) const {
    double thrust = get_thrust(altitude_m, throttle);
    double isp = get_isp(altitude_m);
    return thrust / (constants::G0 * isp);
}

// ============================================================================
// Stage Config Implementation
// ============================================================================

void StageConfig::get_inertia(double prop_fraction, double* inertia_out) const {
    // Linear interpolation between full and empty inertia
    prop_fraction = std::clamp(prop_fraction, 0.0, 1.0);

    for (int i = 0; i < 9; ++i) {
        inertia_out[i] = inertia_empty[i] +
                         prop_fraction * (inertia_full[i] - inertia_empty[i]);
    }
}

double StageConfig::get_cd(double mach) const {
    // Piecewise drag coefficient based on Mach number
    if (mach < 0.8) {
        return cd_subsonic;
    } else if (mach < 1.2) {
        // Transonic interpolation (drag rise)
        double frac = (mach - 0.8) / 0.4;
        return cd_subsonic + frac * (cd_transonic - cd_subsonic);
    } else if (mach < 2.0) {
        // Supersonic interpolation (drag decrease)
        double frac = (mach - 1.2) / 0.8;
        return cd_transonic + frac * (cd_supersonic - cd_transonic);
    } else {
        return cd_supersonic;
    }
}

// ============================================================================
// Rocket State Implementation
// ============================================================================

void RocketState::set_launch_state(double latitude_rad, double longitude_rad,
                                   double azimuth_rad) {
    // Position at launch site (ECI at t=0)
    double r = constants::R_EARTH;
    pos[0] = r * std::cos(latitude_rad) * std::cos(longitude_rad);
    pos[1] = r * std::cos(latitude_rad) * std::sin(longitude_rad);
    pos[2] = r * std::sin(latitude_rad);

    // Initial velocity (Earth surface rotation)
    // v = omega_earth x r
    vel[0] = -constants::OMEGA_EARTH * pos[1];
    vel[1] =  constants::OMEGA_EARTH * pos[0];
    vel[2] = 0.0;

    // Initial attitude: rocket pointing up along launch site vertical
    // Body +X axis = up (along position vector initially)
    // Then we'll rotate about the launch azimuth
    double r_mag = std::sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);
    double up[3] = {pos[0]/r_mag, pos[1]/r_mag, pos[2]/r_mag};

    // For now, simple identity quaternion (will be refined)
    quat[0] = 0.0;  // qx
    quat[1] = 0.0;  // qy
    quat[2] = 0.0;  // qz
    quat[3] = 1.0;  // qw

    // Zero initial angular velocity
    omega[0] = 0.0;
    omega[1] = 0.0;
    omega[2] = 0.0;

    // Initialize other fields
    time = 0.0;
    throttle = 0.0;
    gimbal[0] = 0.0;
    gimbal[1] = 0.0;
    current_stage = 1;

    update_derived();
}

void RocketState::update_derived() {
    // Altitude (simplified - assumes spherical Earth)
    double r = std::sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);
    altitude = r - constants::R_EARTH;
    if (altitude < 0.0) altitude = 0.0;

    // Velocity magnitude
    velocity_mag = std::sqrt(vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);

    // Dynamic pressure
    double rho = AtmosphereModel::density_us76(altitude);
    dynamic_pressure = 0.5 * rho * velocity_mag * velocity_mag;

    // Downrange (great circle distance from launch site - simplified)
    // This would need the launch site position to compute properly
    // For now, approximate using horizontal distance
    downrange = std::sqrt(pos[0]*pos[0] + pos[1]*pos[1]);

    // Flight path angle (angle between velocity and local horizontal)
    double r_vec[3] = {pos[0]/r, pos[1]/r, pos[2]/r};
    double v_dot_r = vel[0]*r_vec[0] + vel[1]*r_vec[1] + vel[2]*r_vec[2];
    double v_hor = std::sqrt(velocity_mag*velocity_mag - v_dot_r*v_dot_r);
    flight_path_angle = std::atan2(v_dot_r, v_hor);
}

// ============================================================================
// Rocket Dynamics Engine Implementation
// ============================================================================

RocketDynamics::RocketDynamics(const std::vector<StageConfig>& stages)
    : stages_(stages)
    , current_stage_(0)
    , launch_lat_rad_(0.0)
    , launch_lon_rad_(0.0)
    , launch_alt_m_(0.0)
    , launch_azimuth_rad_(0.0)
{
    std::memset(&state_, 0, sizeof(state_));
    std::memset(&stats_, 0, sizeof(stats_));
}

void RocketDynamics::set_launch_site(double latitude_deg, double longitude_deg,
                                     double altitude_m) {
    launch_lat_rad_ = latitude_deg * constants::DEG2RAD;
    launch_lon_rad_ = longitude_deg * constants::DEG2RAD;
    launch_alt_m_ = altitude_m;
}

void RocketDynamics::set_launch_azimuth(double azimuth_deg) {
    launch_azimuth_rad_ = azimuth_deg * constants::DEG2RAD;
}

void RocketDynamics::initialize() {
    // Set initial state at launch site
    state_.set_launch_state(launch_lat_rad_, launch_lon_rad_, launch_azimuth_rad_);

    // Compute initial mass
    double total_mass = 0.0;
    for (const auto& stage : stages_) {
        total_mass += stage.dry_mass + stage.propellant_mass;
    }
    if (!stages_.empty() && stages_[0].fairing_mass > 0.0) {
        total_mass += stages_[0].fairing_mass;
    }
    state_.total_mass = total_mass;
    state_.propellant_mass = stages_[0].propellant_mass;

    current_stage_ = 0;
    state_.current_stage = 1;

    // Clear history
    trajectory_.clear();
    staging_events_.clear();
    std::memset(&stats_, 0, sizeof(stats_));

    // Record initial point
    record_trajectory_point();
}

const StageConfig& RocketDynamics::get_current_stage_config() const {
    return stages_[current_stage_];
}

void RocketDynamics::propagate(double dt, const RocketCommand& cmd) {
    // Simple Euler integration (for quick tests)
    // Use propagate_rk4 for better accuracy

    double deriv[14];
    compute_state_derivative(state_, cmd, deriv);

    // Update position
    state_.pos[0] += deriv[0] * dt;
    state_.pos[1] += deriv[1] * dt;
    state_.pos[2] += deriv[2] * dt;

    // Update velocity
    state_.vel[0] += deriv[3] * dt;
    state_.vel[1] += deriv[4] * dt;
    state_.vel[2] += deriv[5] * dt;

    // Update quaternion
    state_.quat[0] += deriv[6] * dt;
    state_.quat[1] += deriv[7] * dt;
    state_.quat[2] += deriv[8] * dt;
    state_.quat[3] += deriv[9] * dt;
    normalize_quaternion(state_.quat);

    // Update angular velocity
    state_.omega[0] += deriv[10] * dt;
    state_.omega[1] += deriv[11] * dt;
    state_.omega[2] += deriv[12] * dt;

    // Update mass
    state_.total_mass += deriv[13] * dt;
    state_.propellant_mass += deriv[13] * dt;

    if (state_.propellant_mass < 0.0) state_.propellant_mass = 0.0;

    // Update time
    state_.time += dt;

    // Store control inputs
    state_.throttle = cmd.throttle;
    state_.gimbal[0] = cmd.gimbal_pitch;
    state_.gimbal[1] = cmd.gimbal_yaw;

    // Update derived quantities
    state_.update_derived();

    // Update statistics
    update_stats(state_);
    stats_.total_steps++;
}

void RocketDynamics::propagate_rk4(double dt, const RocketCommand& cmd) {
    // 4th-order Runge-Kutta integration
    RocketState s0 = state_;
    double k1[14], k2[14], k3[14], k4[14];
    RocketState s_temp;

    // k1 = f(t, y)
    compute_state_derivative(s0, cmd, k1);

    // k2 = f(t + dt/2, y + dt/2 * k1)
    s_temp = s0;
    s_temp.pos[0] += 0.5 * dt * k1[0];
    s_temp.pos[1] += 0.5 * dt * k1[1];
    s_temp.pos[2] += 0.5 * dt * k1[2];
    s_temp.vel[0] += 0.5 * dt * k1[3];
    s_temp.vel[1] += 0.5 * dt * k1[4];
    s_temp.vel[2] += 0.5 * dt * k1[5];
    s_temp.quat[0] += 0.5 * dt * k1[6];
    s_temp.quat[1] += 0.5 * dt * k1[7];
    s_temp.quat[2] += 0.5 * dt * k1[8];
    s_temp.quat[3] += 0.5 * dt * k1[9];
    s_temp.omega[0] += 0.5 * dt * k1[10];
    s_temp.omega[1] += 0.5 * dt * k1[11];
    s_temp.omega[2] += 0.5 * dt * k1[12];
    s_temp.total_mass += 0.5 * dt * k1[13];
    normalize_quaternion(s_temp.quat);
    compute_state_derivative(s_temp, cmd, k2);

    // k3 = f(t + dt/2, y + dt/2 * k2)
    s_temp = s0;
    s_temp.pos[0] += 0.5 * dt * k2[0];
    s_temp.pos[1] += 0.5 * dt * k2[1];
    s_temp.pos[2] += 0.5 * dt * k2[2];
    s_temp.vel[0] += 0.5 * dt * k2[3];
    s_temp.vel[1] += 0.5 * dt * k2[4];
    s_temp.vel[2] += 0.5 * dt * k2[5];
    s_temp.quat[0] += 0.5 * dt * k2[6];
    s_temp.quat[1] += 0.5 * dt * k2[7];
    s_temp.quat[2] += 0.5 * dt * k2[8];
    s_temp.quat[3] += 0.5 * dt * k2[9];
    s_temp.omega[0] += 0.5 * dt * k2[10];
    s_temp.omega[1] += 0.5 * dt * k2[11];
    s_temp.omega[2] += 0.5 * dt * k2[12];
    s_temp.total_mass += 0.5 * dt * k2[13];
    normalize_quaternion(s_temp.quat);
    compute_state_derivative(s_temp, cmd, k3);

    // k4 = f(t + dt, y + dt * k3)
    s_temp = s0;
    s_temp.pos[0] += dt * k3[0];
    s_temp.pos[1] += dt * k3[1];
    s_temp.pos[2] += dt * k3[2];
    s_temp.vel[0] += dt * k3[3];
    s_temp.vel[1] += dt * k3[4];
    s_temp.vel[2] += dt * k3[5];
    s_temp.quat[0] += dt * k3[6];
    s_temp.quat[1] += dt * k3[7];
    s_temp.quat[2] += dt * k3[8];
    s_temp.quat[3] += dt * k3[9];
    s_temp.omega[0] += dt * k3[10];
    s_temp.omega[1] += dt * k3[11];
    s_temp.omega[2] += dt * k3[12];
    s_temp.total_mass += dt * k3[13];
    normalize_quaternion(s_temp.quat);
    compute_state_derivative(s_temp, cmd, k4);

    // y_new = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    for (int i = 0; i < 3; ++i) {
        state_.pos[i] = s0.pos[i] + (dt/6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
        state_.vel[i] = s0.vel[i] + (dt/6.0) * (k1[i+3] + 2*k2[i+3] + 2*k3[i+3] + k4[i+3]);
        state_.omega[i] = s0.omega[i] + (dt/6.0) * (k1[i+10] + 2*k2[i+10] + 2*k3[i+10] + k4[i+10]);
    }
    for (int i = 0; i < 4; ++i) {
        state_.quat[i] = s0.quat[i] + (dt/6.0) * (k1[i+6] + 2*k2[i+6] + 2*k3[i+6] + k4[i+6]);
    }
    normalize_quaternion(state_.quat);

    state_.total_mass = s0.total_mass + (dt/6.0) * (k1[13] + 2*k2[13] + 2*k3[13] + k4[13]);
    state_.propellant_mass += (dt/6.0) * (k1[13] + 2*k2[13] + 2*k3[13] + k4[13]);
    if (state_.propellant_mass < 0.0) state_.propellant_mass = 0.0;

    state_.time += dt;
    state_.throttle = cmd.throttle;
    state_.gimbal[0] = cmd.gimbal_pitch;
    state_.gimbal[1] = cmd.gimbal_yaw;

    state_.update_derived();
    update_stats(state_);
    stats_.total_steps++;
}

void RocketDynamics::compute_state_derivative(const RocketState& state,
                                              const RocketCommand& cmd,
                                              double deriv[14]) const {
    // deriv[0-2] = velocity (d(pos)/dt = vel)
    deriv[0] = state.vel[0];
    deriv[1] = state.vel[1];
    deriv[2] = state.vel[2];

    // Compute forces
    double forces[3];
    compute_forces(state, cmd, forces);

    // deriv[3-5] = acceleration (d(vel)/dt = F/m)
    double m = state.total_mass;
    if (m < 1.0) m = 1.0;  // Prevent division by zero
    deriv[3] = forces[0] / m;
    deriv[4] = forces[1] / m;
    deriv[5] = forces[2] / m;

    // Compute acceleration in g's
    double a_mag = std::sqrt(deriv[3]*deriv[3] + deriv[4]*deriv[4] + deriv[5]*deriv[5]);

    // deriv[6-9] = quaternion derivative
    quaternion_derivative(state.quat, state.omega, &deriv[6]);

    // Compute moments
    double moments[3];
    compute_moments(state, cmd, moments);

    // Get current inertia
    const StageConfig& stage = stages_[current_stage_];
    double prop_frac = (stage.propellant_mass > 0.0) ?
                       state.propellant_mass / stage.propellant_mass : 0.0;
    double I[9];
    stage.get_inertia(prop_frac, I);

    // Simple diagonal inertia for now
    // deriv[10-12] = angular acceleration (d(omega)/dt = I^-1 * M)
    deriv[10] = moments[0] / I[0];
    deriv[11] = moments[1] / I[4];
    deriv[12] = moments[2] / I[8];

    // deriv[13] = mass rate (negative for propellant consumption)
    if (cmd.throttle > 0.0 && state.propellant_mass > 0.0) {
        double mdot = stage.engine.get_mass_flow(state.altitude, cmd.throttle);
        deriv[13] = -mdot;
    } else {
        deriv[13] = 0.0;
    }
}

void RocketDynamics::compute_forces(const RocketState& state,
                                    const RocketCommand& cmd,
                                    double forces[3]) const {
    double thrust[3], drag[3], gravity[3];

    compute_thrust(state, cmd, thrust);
    compute_drag(state, drag);
    compute_gravity(state, gravity);

    for (int i = 0; i < 3; ++i) {
        forces[i] = thrust[i] + drag[i] + gravity[i];
    }
}

void RocketDynamics::compute_thrust(const RocketState& state,
                                    const RocketCommand& cmd,
                                    double thrust[3]) const {
    if (cmd.throttle <= 0.0 || state.propellant_mass <= 0.0) {
        thrust[0] = thrust[1] = thrust[2] = 0.0;
        return;
    }

    const StageConfig& stage = stages_[current_stage_];
    double T_mag = stage.engine.get_thrust(state.altitude, cmd.throttle);

    // Thrust direction in body frame (along +X with gimbal)
    double T_body[3];
    double cp = std::cos(cmd.gimbal_pitch);
    double sp = std::sin(cmd.gimbal_pitch);
    double cy = std::cos(cmd.gimbal_yaw);
    double sy = std::sin(cmd.gimbal_yaw);

    T_body[0] = T_mag * cp * cy;
    T_body[1] = T_mag * sy;
    T_body[2] = T_mag * sp * cy;

    // Transform to ECI
    body_to_eci(T_body, state.quat, thrust);
}

void RocketDynamics::compute_drag(const RocketState& state,
                                  double drag[3]) const {
    double v_mag = state.velocity_mag;
    if (v_mag < 1.0 || state.altitude > 100000.0) {
        drag[0] = drag[1] = drag[2] = 0.0;
        return;
    }

    const StageConfig& stage = stages_[current_stage_];
    AtmosphericState atm = AtmosphereModel::get_state(state.altitude, v_mag);

    // Drag magnitude: D = 0.5 * rho * v^2 * Cd * A
    double Cd = stage.get_cd(atm.mach_number);
    double A = stage.get_area();
    double D_mag = 0.5 * atm.density * v_mag * v_mag * Cd * A;

    // Drag opposes velocity
    drag[0] = -D_mag * state.vel[0] / v_mag;
    drag[1] = -D_mag * state.vel[1] / v_mag;
    drag[2] = -D_mag * state.vel[2] / v_mag;
}

void RocketDynamics::compute_gravity(const RocketState& state,
                                     double gravity[3]) const {
    double r = std::sqrt(state.pos[0]*state.pos[0] +
                        state.pos[1]*state.pos[1] +
                        state.pos[2]*state.pos[2]);

    if (r < 1.0) r = constants::R_EARTH;

    double g_mag = constants::MU_EARTH / (r * r);
    double m = state.total_mass;

    // Gravity force = -m * g * r_hat
    gravity[0] = -m * g_mag * state.pos[0] / r;
    gravity[1] = -m * g_mag * state.pos[1] / r;
    gravity[2] = -m * g_mag * state.pos[2] / r;
}

void RocketDynamics::compute_moments(const RocketState& state,
                                     const RocketCommand& cmd,
                                     double moments[3]) const {
    // For now, simple TVC-generated moments
    // In a full model, would include:
    // - Aerodynamic moments
    // - Gravity gradient
    // - Reaction control system
    // - CG offset effects

    moments[0] = 0.0;
    moments[1] = 0.0;
    moments[2] = 0.0;

    // TVC moment arm (distance from CG to gimbal point)
    const double L_arm = 2.0;  // meters

    if (cmd.throttle > 0.0 && state.propellant_mass > 0.0) {
        const StageConfig& stage = stages_[current_stage_];
        double T_mag = stage.engine.get_thrust(state.altitude, cmd.throttle);

        // Moment from TVC (simplified)
        moments[1] = T_mag * std::sin(cmd.gimbal_pitch) * L_arm;  // Pitch moment
        moments[2] = T_mag * std::sin(cmd.gimbal_yaw) * L_arm;    // Yaw moment
    }
}

void RocketDynamics::quaternion_derivative(const double q[4],
                                           const double omega[3],
                                           double dq[4]) const {
    // dq/dt = 0.5 * q * omega_quat
    // where omega_quat = [0, wx, wy, wz]
    double qx = q[0], qy = q[1], qz = q[2], qw = q[3];
    double wx = omega[0], wy = omega[1], wz = omega[2];

    dq[0] = 0.5 * ( qw*wx - qz*wy + qy*wz);
    dq[1] = 0.5 * ( qz*wx + qw*wy - qx*wz);
    dq[2] = 0.5 * (-qy*wx + qx*wy + qw*wz);
    dq[3] = 0.5 * (-qx*wx - qy*wy - qz*wz);
}

void RocketDynamics::normalize_quaternion(double q[4]) const {
    double n = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (n > 1e-10) {
        q[0] /= n;
        q[1] /= n;
        q[2] /= n;
        q[3] /= n;
    }
}

void RocketDynamics::body_to_eci(const double vec_body[3],
                                 const double q[4],
                                 double vec_eci[3]) const {
    // Rotate vector from body to ECI using quaternion
    double qx = q[0], qy = q[1], qz = q[2], qw = q[3];

    // Rodrigues rotation formula (optimized)
    double t[3];
    t[0] = 2.0 * (qy*vec_body[2] - qz*vec_body[1]);
    t[1] = 2.0 * (qz*vec_body[0] - qx*vec_body[2]);
    t[2] = 2.0 * (qx*vec_body[1] - qy*vec_body[0]);

    vec_eci[0] = vec_body[0] + qw*t[0] + (qy*t[2] - qz*t[1]);
    vec_eci[1] = vec_body[1] + qw*t[1] + (qz*t[0] - qx*t[2]);
    vec_eci[2] = vec_body[2] + qw*t[2] + (qx*t[1] - qy*t[0]);
}

void RocketDynamics::execute_staging() {
    if (current_stage_ >= static_cast<int>(stages_.size()) - 1) {
        return;  // No more stages
    }

    StagingEvent event;
    event.time = state_.time;
    event.from_stage = current_stage_ + 1;
    event.to_stage = current_stage_ + 2;

    // Remove old stage mass
    const StageConfig& old_stage = stages_[current_stage_];
    double jettison_mass = old_stage.dry_mass;
    event.delta_mass = jettison_mass;

    state_.total_mass -= jettison_mass;
    current_stage_++;
    state_.current_stage = current_stage_ + 1;

    // Reset propellant for new stage
    state_.propellant_mass = stages_[current_stage_].propellant_mass;

    event.success = true;
    event.notes = "Nominal separation";
    staging_events_.push_back(event);

    printf("STAGING: Stage %d separated at t=%.1f s, mass=%.0f kg\n",
           event.from_stage, event.time, state_.total_mass);
}

bool RocketDynamics::check_staging_conditions() const {
    if (current_stage_ >= static_cast<int>(stages_.size()) - 1) {
        return false;
    }
    return state_.propellant_mass <= 0.0;
}

void RocketDynamics::record_trajectory_point() {
    TrajectoryPoint pt;
    pt.time = state_.time;
    pt.altitude = state_.altitude;
    pt.downrange = state_.downrange;
    pt.velocity = state_.velocity_mag;
    pt.dynamic_pressure = state_.dynamic_pressure;
    pt.mass = state_.total_mass;
    pt.flight_path_angle = state_.flight_path_angle;
    pt.throttle = state_.throttle;
    pt.stage = state_.current_stage;

    // Acceleration (would need to recompute)
    pt.acceleration = 0.0;

    for (int i = 0; i < 3; ++i) {
        pt.pos_eci[i] = state_.pos[i];
    }

    trajectory_.push_back(pt);
}

void RocketDynamics::update_stats(const RocketState& state) {
    // Track max-Q
    if (state.dynamic_pressure > stats_.max_q) {
        stats_.max_q = state.dynamic_pressure;
        stats_.max_q_time = state.time;
    }

    // Track max-g (would need to compute acceleration)
}

double RocketDynamics::gravity_turn_pitch(double altitude, double velocity) const {
    // Simple gravity turn pitch program
    // Start vertical, gradually pitch over as altitude increases

    const double h_start = 200.0;      // Start pitch at 200m
    const double h_end = 50000.0;      // Reach target pitch by 50km
    const double pitch_target = 15.0 * constants::DEG2RAD;  // 15 deg from vertical

    if (altitude < h_start) {
        return constants::PI / 2.0;  // Vertical (90 deg)
    }

    if (altitude > h_end) {
        return pitch_target;
    }

    // Linear interpolation
    double frac = (altitude - h_start) / (h_end - h_start);
    return constants::PI / 2.0 - frac * (constants::PI / 2.0 - pitch_target);
}

} // namespace gnc
