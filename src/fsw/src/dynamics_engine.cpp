// =============================================================================
// dynamics_engine.cpp -- Real-Time Orbital Dynamics Propagator Implementation
// =============================================================================
//
// This file implements the DynamicsEngine class declared in dynamics_engine.h.
//
// KEY COMPUTATIONAL CONCEPTS:
//
//   1. RK4 INTEGRATION:
//      The 4th-order Runge-Kutta method is the workhorse of aerospace
//      simulation. It evaluates the derivative at 4 points within each time
//      step and combines them with optimal weights:
//
//        k1 = f(t,       y)
//        k2 = f(t + h/2, y + h/2 * k1)
//        k3 = f(t + h/2, y + h/2 * k2)
//        k4 = f(t + h,   y + h * k3)
//        y_{n+1} = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)
//
//      The error is O(h^5) per step, O(h^4) globally. For a 1-second time
//      step in LEO, this gives roughly millimeter-level position accuracy
//      per orbit. For higher fidelity, adaptive step-size (RKF45) is used.
//
//   2. J2 GRAVITATIONAL PERTURBATION:
//      The Earth is not a perfect sphere -- it has an equatorial bulge.
//      The J2 harmonic (the dominant oblateness term) causes:
//        - Nodal regression: the orbital plane precesses westward
//        - Apsidal advance: the argument of perigee rotates
//      These effects are critical for mission planning (Sun-synchronous
//      orbits exploit nodal regression to maintain sun angle).
//
//   3. PERFORMANCE OPTIMIZATIONS:
//      - Manual loop unrolling for the 15-element state vector
//      - Avoidance of virtual function calls in the hot path
//      - Precomputed intermediate values (r^2, r^3, r^5) to reduce
//        redundant sqrt and division operations
//      - Cache-friendly sequential memory access (StateVector is 120 bytes,
//        fits in two 64-byte cache lines)
//
//   4. QUATERNION KINEMATICS:
//      Quaternions avoid the gimbal lock problem of Euler angles and the
//      singularity issues of rotation matrices at certain orientations.
//      The kinematic equation: dq/dt = 0.5 * q (x) omega_quat
//      where omega_quat = [omega_x, omega_y, omega_z, 0] (pure quaternion)
//      and (x) denotes quaternion multiplication.
//
//      Quaternions must be normalized after each step because numerical
//      integration causes the norm to drift from 1.0 over time (a
//      consequence of finite-precision floating-point arithmetic).
//
// =============================================================================

#include "dynamics_engine.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// Constructor
//
// The constructor initializes the propagation statistics. All physical
// constants (mu, J2, R_body) are passed as function parameters rather than
// stored as members, allowing the same engine to propagate orbits around
// different bodies (Earth, Moon, Mars) without reconstructing the object.
//
// The statistics tracking has near-zero overhead: we increment counters and
// perform a single chrono call per propagation step. The chrono overhead is
// ~20 ns on modern systems, negligible compared to the ~1 us RK4 cost.
// ---------------------------------------------------------------------------
DynamicsEngine::DynamicsEngine() {
    stats_.total_steps = 0;
    stats_.total_time_propagated = 0.0;
    stats_.avg_step_compute_us = 0.0;
    stats_.peak_step_compute_us = 0.0;
    stats_.energy_drift_ppm = 0.0;
}

// ---------------------------------------------------------------------------
// gravitational_acceleration()
//
// Computes the acceleration due to central-body gravity with J2 perturbation.
//
// The two-body gravitational acceleration is:
//   a = -mu * r_vec / |r|^3
//
// The J2 perturbation adds correction terms that depend on the z-component
// (out-of-equatorial-plane position). The full expressions are:
//
//   Factor = -mu / r^3
//   J2_factor = 1.5 * J2 * (R_body / r)^2
//   z_ratio_sq = (z / r)^2
//
//   a_x = Factor * x * (1 - J2_factor * (5 * z_ratio_sq - 1))
//   a_y = Factor * y * (1 - J2_factor * (5 * z_ratio_sq - 1))
//   a_z = Factor * z * (1 - J2_factor * (5 * z_ratio_sq - 3))
//
// Note: a_x and a_y have identical J2 correction terms (5*z^2/r^2 - 1),
// while a_z has a different term (5*z^2/r^2 - 3). This asymmetry reflects
// the fact that J2 creates an additional downward pull near the equator
// and a reduced pull near the poles.
//
// PERFORMANCE: We precompute r^2, r^3, r^5 and the common J2 factor to
// avoid redundant multiplications. The total cost is approximately:
//   - 3 multiplications for r^2
//   - 1 sqrt for r
//   - ~15 multiplications and additions for J2 terms
//   - Total: ~25 FLOPs per call
//
// At 100 Hz with 4 RK4 stages, that is 4 * 25 = 100 FLOPs/step, or
// 10,000 FLOPs/second -- trivial for any modern processor.
// ---------------------------------------------------------------------------
void DynamicsEngine::gravitational_acceleration(
    const double pos[3],
    double accel[3],
    double mu,
    double J2,
    double R_body)
{
    // Extract position components for readability
    const double x = pos[0];
    const double y = pos[1];
    const double z = pos[2];

    // Compute radial distance squared: r^2 = x^2 + y^2 + z^2
    // We compute r^2 first because we need it for several intermediate values,
    // and squaring is cheaper than taking a square root.
    const double r_sq = x * x + y * y + z * z;

    // Radial distance: r = sqrt(r^2)
    // std::sqrt is implemented in hardware on modern CPUs (FSQRT on x86,
    // VSQRT on AVX). Latency: ~12-20 cycles, throughput: ~4-8 cycles.
    const double r = std::sqrt(r_sq);

    // Precompute 1/r^3 to avoid repeated division.
    // Division is expensive (~20-40 cycles) compared to multiplication (~3-5).
    // By computing 1/r^3 once and multiplying, we save 2 divisions.
    const double r_cubed = r * r_sq;            // r^3 = r * r^2
    const double inv_r_cubed = 1.0 / r_cubed;   // 1/r^3

    // Base two-body gravitational acceleration factor: -mu / r^3
    // This is the Keplerian (unperturbed) gravity.
    const double base_factor = -mu * inv_r_cubed;

    // J2 perturbation terms:
    //   J2_coeff = 1.5 * J2 * (R_body / r)^2
    //
    // This factor captures the strength of the J2 perturbation relative to
    // the two-body term. For a 400 km LEO orbit:
    //   R_body/r ~ 6378/6778 ~ 0.941
    //   (R_body/r)^2 ~ 0.886
    //   1.5 * J2 * 0.886 ~ 1.5 * 1.083e-3 * 0.886 ~ 1.44e-3
    //
    // So J2 accelerations are about 0.14% of the two-body term -- small but
    // critical for orbit determination over multiple revolutions.
    const double R_over_r_sq = (R_body * R_body) / r_sq;   // (R/r)^2
    const double J2_coeff = 1.5 * J2 * R_over_r_sq;

    // z/r ratio squared: appears in all three J2 correction terms
    // This represents how far the spacecraft is from the equatorial plane
    // relative to its distance from Earth's center.
    const double z_over_r_sq = (z * z) / r_sq;  // (z/r)^2

    // -----------------------------------------------------------------------
    // X-component of acceleration:
    //   a_x = -mu * x / r^3 * (1 - J2_coeff * (5 * (z/r)^2 - 1))
    //
    // The term (5*(z/r)^2 - 1) modifies the radial gravity in the x-direction.
    // When the spacecraft is in the equatorial plane (z=0), this becomes
    // (-1)*J2_coeff, strengthening the gravity slightly (equatorial bulge
    // pulls harder). Near the poles (z/r ~ 1), the factor becomes
    // (5-1)*J2_coeff = 4*J2_coeff, weakening gravity in x,y directions.
    // -----------------------------------------------------------------------
    accel[0] = base_factor * x * (1.0 - J2_coeff * (5.0 * z_over_r_sq - 1.0));

    // -----------------------------------------------------------------------
    // Y-component of acceleration:
    //   a_y = -mu * y / r^3 * (1 - J2_coeff * (5 * (z/r)^2 - 1))
    //
    // Identical J2 correction as the x-component due to the axial symmetry
    // of J2 (the oblateness is symmetric about the z-axis / rotation axis).
    // -----------------------------------------------------------------------
    accel[1] = base_factor * y * (1.0 - J2_coeff * (5.0 * z_over_r_sq - 1.0));

    // -----------------------------------------------------------------------
    // Z-component of acceleration:
    //   a_z = -mu * z / r^3 * (1 - J2_coeff * (5 * (z/r)^2 - 3))
    //
    // The z-component has a different correction factor: (5*(z/r)^2 - 3)
    // instead of (5*(z/r)^2 - 1). The -3 term (vs -1) reflects the
    // additional radial pull toward the equatorial plane. In the equatorial
    // plane (z=0), the z-correction is -(-3)*J2_coeff = +3*J2_coeff,
    // providing a restoring force toward z=0. Near the poles, the
    // correction is (5-3)*J2_coeff = 2*J2_coeff.
    // -----------------------------------------------------------------------
    accel[2] = base_factor * z * (1.0 - J2_coeff * (5.0 * z_over_r_sq - 3.0));
}

// ---------------------------------------------------------------------------
// quaternion_derivative()
//
// Computes the time derivative of the attitude quaternion.
//
// The kinematic equation relates angular velocity to quaternion rate:
//   dq/dt = 0.5 * q (x) omega_quat
//
// where omega_quat = [omega_x, omega_y, omega_z, 0] is the angular velocity
// expressed as a pure quaternion (zero scalar part).
//
// The quaternion multiplication q (x) omega_quat expands to:
//   dqx = 0.5 * ( qw*wx + qy*wz - qz*wy)
//   dqy = 0.5 * ( qw*wy - qx*wz + qz*wx)
//   dqz = 0.5 * ( qw*wz + qx*wy - qy*wx)
//   dqw = 0.5 * (-qx*wx - qy*wy - qz*wz)
//
// Convention: quaternion is [qx, qy, qz, qw] (scalar-last / JPL convention).
//
// NUMERICAL NOTE: The quaternion norm drifts from 1.0 during integration.
// We add a small correction term proportional to (1 - |q|^2) to stabilize
// the norm. This is the Baumgarte stabilization technique.
// ---------------------------------------------------------------------------
void DynamicsEngine::quaternion_derivative(
    const double quat[4],
    const double omega[3],
    double dquat[4])
{
    // Extract quaternion components (scalar-last: [qx, qy, qz, qw])
    const double qx = quat[0];
    const double qy = quat[1];
    const double qz = quat[2];
    const double qw = quat[3];

    // Extract angular velocity components (body frame, rad/s)
    const double wx = omega[0];
    const double wy = omega[1];
    const double wz = omega[2];

    // Quaternion kinematic equations: dq/dt = 0.5 * q (x) [wx, wy, wz, 0]
    //
    // Written out using the Hamilton product rule:
    //   (a1 + b1*i + c1*j + d1*k) * (a2 + b2*i + c2*j + d2*k)
    // where i*j = k, j*k = i, k*i = j, i*i = j*j = k*k = -1
    dquat[0] = 0.5 * ( qw * wx + qy * wz - qz * wy);  // dqx/dt
    dquat[1] = 0.5 * ( qw * wy - qx * wz + qz * wx);  // dqy/dt
    dquat[2] = 0.5 * ( qw * wz + qx * wy - qy * wx);  // dqz/dt
    dquat[3] = 0.5 * (-qx * wx - qy * wy - qz * wz);  // dqw/dt

    // Baumgarte stabilization: add a constraint-restoring term.
    // The quaternion constraint is |q|^2 = 1. If the norm drifts,
    // we add a correction that pushes it back toward 1.
    //
    // lambda controls the stabilization strength. Too high causes
    // high-frequency oscillations; too low allows drift to accumulate.
    // A value of 0.5/dt would be optimal, but we use a fixed value
    // that works well for typical GNC time steps (0.001 to 1.0 s).
    constexpr double lambda = 1.0;  // Stabilization gain
    const double norm_sq = qx * qx + qy * qy + qz * qz + qw * qw;
    const double correction = lambda * (1.0 - norm_sq);

    dquat[0] += correction * qx;
    dquat[1] += correction * qy;
    dquat[2] += correction * qz;
    dquat[3] += correction * qw;
}

// ---------------------------------------------------------------------------
// normalize_quaternion()
//
// Normalizes the quaternion to unit length. Must be called periodically
// (typically every step, or every N steps with monitoring).
//
// The norm drift per RK4 step is approximately O(h^5 * |omega|^2), which
// for typical spacecraft angular rates (~0.01 rad/s) and time steps (~0.1 s)
// is on the order of 1e-12 per step. After 10,000 steps, the drift
// accumulates to ~1e-8, still small but worth correcting.
//
// We use the fast inverse square root followed by multiplication rather
// than dividing by sqrt(norm_sq), because multiplication is faster than
// division on all modern CPUs.
// ---------------------------------------------------------------------------
void DynamicsEngine::normalize_quaternion(double quat[4]) {
    const double norm_sq = quat[0] * quat[0] + quat[1] * quat[1] +
                           quat[2] * quat[2] + quat[3] * quat[3];

    // Guard against degenerate quaternion (all zeros).
    // This should never happen in normal operation, but if it does,
    // reset to identity quaternion rather than dividing by zero.
    if (norm_sq < 1e-30) {
        quat[0] = 0.0;
        quat[1] = 0.0;
        quat[2] = 0.0;
        quat[3] = 1.0;
        return;
    }

    // Compute 1/sqrt(norm_sq) using standard library.
    // On x86, this compiles to RSQRTSD (approximate reciprocal sqrt,
    // ~5 cycles) followed by a Newton-Raphson refinement step for
    // full double precision.
    const double inv_norm = 1.0 / std::sqrt(norm_sq);

    quat[0] *= inv_norm;
    quat[1] *= inv_norm;
    quat[2] *= inv_norm;
    quat[3] *= inv_norm;
}

// ---------------------------------------------------------------------------
// pack_state() / unpack_state()
//
// Serialize/deserialize the StateVector to/from a flat double array for
// the generic RK4 integrator. The flat array has 15 elements:
//   [0..2]  = position (x, y, z)
//   [3..5]  = velocity (vx, vy, vz)
//   [6..9]  = quaternion (qx, qy, qz, qw)
//   [10..12] = angular velocity (wx, wy, wz)
//   [13]    = mass
//   [14]    = time
//
// We use std::memcpy for the contiguous arrays. The compiler will optimize
// this to SIMD moves (MOVAPS/MOVUPD) when the data is aligned. For small
// fixed sizes, memcpy is typically inlined entirely.
// ---------------------------------------------------------------------------
void DynamicsEngine::pack_state(const StateVector& state, double y[15]) {
    y[0]  = state.pos[0];
    y[1]  = state.pos[1];
    y[2]  = state.pos[2];
    y[3]  = state.vel[0];
    y[4]  = state.vel[1];
    y[5]  = state.vel[2];
    y[6]  = state.quat[0];
    y[7]  = state.quat[1];
    y[8]  = state.quat[2];
    y[9]  = state.quat[3];
    y[10] = state.omega[0];
    y[11] = state.omega[1];
    y[12] = state.omega[2];
    y[13] = state.mass;
    y[14] = state.time;
}

void DynamicsEngine::unpack_state(const double y[15], StateVector& state) {
    state.pos[0]   = y[0];
    state.pos[1]   = y[1];
    state.pos[2]   = y[2];
    state.vel[0]   = y[3];
    state.vel[1]   = y[4];
    state.vel[2]   = y[5];
    state.quat[0]  = y[6];
    state.quat[1]  = y[7];
    state.quat[2]  = y[8];
    state.quat[3]  = y[9];
    state.omega[0] = y[10];
    state.omega[1] = y[11];
    state.omega[2] = y[12];
    state.mass     = y[13];
    state.time     = y[14];
}

// ---------------------------------------------------------------------------
// state_derivative()
//
// Computes the right-hand side of the ODE system: dy/dt = f(t, y).
//
// For orbital mechanics, the state derivative is:
//   d(pos)/dt   = vel                    (kinematics)
//   d(vel)/dt   = grav_accel + F/m       (Newton's second law)
//   d(quat)/dt  = quaternion_kinematics  (attitude kinematics)
//   d(omega)/dt = torques / I            (Euler's equation, simplified)
//   d(mass)/dt  = 0                      (no fuel consumption modeled here)
//   d(time)/dt  = 1                      (clock advances)
//
// The output is a 15-element array matching the pack_state() layout.
// ---------------------------------------------------------------------------
void DynamicsEngine::state_derivative(
    const StateVector& state,
    const double* ext_forces,
    const double* ext_torques,
    double deriv[15])
{
    // -- Position derivative = velocity (trivial kinematics) --
    deriv[0] = state.vel[0];   // dx/dt = vx
    deriv[1] = state.vel[1];   // dy/dt = vy
    deriv[2] = state.vel[2];   // dz/dt = vz

    // -- Velocity derivative = gravitational + external accelerations --
    double grav_accel[3];
    gravitational_acceleration(state.pos, grav_accel);

    // Add external forces (thrust, drag, SRP, etc.)
    // F = ma => a = F/m. Guard against zero mass.
    double inv_mass = (state.mass > 0.0) ? (1.0 / state.mass) : 0.0;

    deriv[3] = grav_accel[0];  // dvx/dt
    deriv[4] = grav_accel[1];  // dvy/dt
    deriv[5] = grav_accel[2];  // dvz/dt

    if (ext_forces) {
        deriv[3] += ext_forces[0] * inv_mass;
        deriv[4] += ext_forces[1] * inv_mass;
        deriv[5] += ext_forces[2] * inv_mass;
    }

    // -- Quaternion derivative = kinematic equation --
    double dquat[4];
    quaternion_derivative(state.quat, state.omega, dquat);
    deriv[6]  = dquat[0];  // dqx/dt
    deriv[7]  = dquat[1];  // dqy/dt
    deriv[8]  = dquat[2];  // dqz/dt
    deriv[9]  = dquat[3];  // dqw/dt

    // -- Angular velocity derivative (simplified Euler's equation) --
    // For a rigid body: I * d(omega)/dt = torques - omega x (I * omega)
    // We simplify by assuming a spherical inertia tensor (I = diag(I0, I0, I0)),
    // which eliminates the gyroscopic term: d(omega)/dt = torques / I0
    //
    // A more complete implementation would take the full inertia tensor
    // and compute the cross product term, but for demonstration purposes
    // this captures the essential dynamics.
    constexpr double I0 = 50.0;  // Moment of inertia (kg*m^2), typical small sat
    double inv_I = 1.0 / I0;

    deriv[10] = 0.0;  // d(omega_x)/dt
    deriv[11] = 0.0;  // d(omega_y)/dt
    deriv[12] = 0.0;  // d(omega_z)/dt

    if (ext_torques) {
        deriv[10] = ext_torques[0] * inv_I;
        deriv[11] = ext_torques[1] * inv_I;
        deriv[12] = ext_torques[2] * inv_I;
    }

    // -- Mass derivative = 0 (no propulsion consumption modeled) --
    deriv[13] = 0.0;

    // -- Time derivative = 1 (clock) --
    deriv[14] = 1.0;
}

// ---------------------------------------------------------------------------
// rk4_step()
//
// Classic 4-stage Runge-Kutta integration for the full state vector.
//
// PERFORMANCE CONSIDERATIONS:
//
//   1. MANUAL UNROLLING: We manually unroll the k1/k2/k3/k4 computations
//      rather than using a generic loop over state dimensions. This allows
//      the compiler to:
//      - Keep intermediate values in registers (no memory spills)
//      - Eliminate loop overhead (compare, branch, increment)
//      - Schedule instructions optimally for the CPU pipeline
//
//   2. STACK ALLOCATION: All temporary arrays (k1, k2, k3, k4, y_temp)
//      are allocated on the stack. Stack allocation is O(1) -- just a
//      pointer adjustment. No heap allocation, no system calls, fully
//      deterministic.
//
//   3. FUNCTION CALL OVERHEAD: We call state_derivative() 4 times per
//      step. Each call involves constructing a temporary StateVector.
//      In a production system, we might inline the derivative computation
//      to eliminate call overhead. Here, readability is prioritized.
//
//   4. NUMERICAL STABILITY: RK4 is unconditionally stable for the
//      gravitational ODE when h * |eigenvalue| < 2.78 (stability limit).
//      For LEO orbits, the eigenvalues of the Jacobian are approximately
//      sqrt(mu/r^3) ~ 0.0011 rad/s. At dt = 1.0 s, h*lambda = 0.0011,
//      far within the stability region.
// ---------------------------------------------------------------------------
void DynamicsEngine::rk4_step(
    StateVector& state,
    double dt,
    const double* ext_forces,
    const double* ext_torques)
{
    // Dimension of the full state vector
    constexpr int N = 15;

    // Pack state into flat array for integration
    double y[N];
    pack_state(state, y);

    // -----------------------------------------------------------------------
    // Stage 1: k1 = f(t, y)
    // Evaluate derivative at the current state.
    // -----------------------------------------------------------------------
    double k1[N];
    state_derivative(state, ext_forces, ext_torques, k1);

    // -----------------------------------------------------------------------
    // Stage 2: k2 = f(t + h/2, y + h/2 * k1)
    // Evaluate derivative at the midpoint using Euler half-step from k1.
    //
    // We manually unroll the loop y_temp[i] = y[i] + 0.5*dt*k1[i]
    // for all 15 elements. The compiler will keep these in registers
    // (modern x86 has 16 general-purpose registers and 16 YMM registers).
    // -----------------------------------------------------------------------
    double y_temp[N];
    const double half_dt = 0.5 * dt;

    // Unrolled: compute y + h/2 * k1 for all state elements
    y_temp[0]  = y[0]  + half_dt * k1[0];   // pos x
    y_temp[1]  = y[1]  + half_dt * k1[1];   // pos y
    y_temp[2]  = y[2]  + half_dt * k1[2];   // pos z
    y_temp[3]  = y[3]  + half_dt * k1[3];   // vel x
    y_temp[4]  = y[4]  + half_dt * k1[4];   // vel y
    y_temp[5]  = y[5]  + half_dt * k1[5];   // vel z
    y_temp[6]  = y[6]  + half_dt * k1[6];   // quat x
    y_temp[7]  = y[7]  + half_dt * k1[7];   // quat y
    y_temp[8]  = y[8]  + half_dt * k1[8];   // quat z
    y_temp[9]  = y[9]  + half_dt * k1[9];   // quat w
    y_temp[10] = y[10] + half_dt * k1[10];  // omega x
    y_temp[11] = y[11] + half_dt * k1[11];  // omega y
    y_temp[12] = y[12] + half_dt * k1[12];  // omega z
    y_temp[13] = y[13] + half_dt * k1[13];  // mass
    y_temp[14] = y[14] + half_dt * k1[14];  // time

    StateVector state_temp;
    unpack_state(y_temp, state_temp);

    double k2[N];
    state_derivative(state_temp, ext_forces, ext_torques, k2);

    // -----------------------------------------------------------------------
    // Stage 3: k3 = f(t + h/2, y + h/2 * k2)
    // Another midpoint evaluation, this time using k2.
    // k3 is generally the most accurate of the midpoint estimates.
    // -----------------------------------------------------------------------
    y_temp[0]  = y[0]  + half_dt * k2[0];
    y_temp[1]  = y[1]  + half_dt * k2[1];
    y_temp[2]  = y[2]  + half_dt * k2[2];
    y_temp[3]  = y[3]  + half_dt * k2[3];
    y_temp[4]  = y[4]  + half_dt * k2[4];
    y_temp[5]  = y[5]  + half_dt * k2[5];
    y_temp[6]  = y[6]  + half_dt * k2[6];
    y_temp[7]  = y[7]  + half_dt * k2[7];
    y_temp[8]  = y[8]  + half_dt * k2[8];
    y_temp[9]  = y[9]  + half_dt * k2[9];
    y_temp[10] = y[10] + half_dt * k2[10];
    y_temp[11] = y[11] + half_dt * k2[11];
    y_temp[12] = y[12] + half_dt * k2[12];
    y_temp[13] = y[13] + half_dt * k2[13];
    y_temp[14] = y[14] + half_dt * k2[14];

    unpack_state(y_temp, state_temp);

    double k3[N];
    state_derivative(state_temp, ext_forces, ext_torques, k3);

    // -----------------------------------------------------------------------
    // Stage 4: k4 = f(t + h, y + h * k3)
    // Endpoint evaluation using a full Euler step from k3.
    // -----------------------------------------------------------------------
    y_temp[0]  = y[0]  + dt * k3[0];
    y_temp[1]  = y[1]  + dt * k3[1];
    y_temp[2]  = y[2]  + dt * k3[2];
    y_temp[3]  = y[3]  + dt * k3[3];
    y_temp[4]  = y[4]  + dt * k3[4];
    y_temp[5]  = y[5]  + dt * k3[5];
    y_temp[6]  = y[6]  + dt * k3[6];
    y_temp[7]  = y[7]  + dt * k3[7];
    y_temp[8]  = y[8]  + dt * k3[8];
    y_temp[9]  = y[9]  + dt * k3[9];
    y_temp[10] = y[10] + dt * k3[10];
    y_temp[11] = y[11] + dt * k3[11];
    y_temp[12] = y[12] + dt * k3[12];
    y_temp[13] = y[13] + dt * k3[13];
    y_temp[14] = y[14] + dt * k3[14];

    unpack_state(y_temp, state_temp);

    double k4[N];
    state_derivative(state_temp, ext_forces, ext_torques, k4);

    // -----------------------------------------------------------------------
    // Combine: y_{n+1} = y_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    //
    // The weights (1, 2, 2, 1)/6 are optimal for 4th-order accuracy.
    // They come from matching the Taylor expansion of y(t+h) to 4th order.
    //
    // OPTIMIZATION: We compute dt/6 once and multiply each coefficient.
    // The compiler will fuse the multiply-add into FMA instructions (Fused
    // Multiply-Add) on CPUs that support it (all modern x86/ARM).
    // FMA computes a*b+c in a single instruction with only one rounding
    // error, giving both speed and accuracy benefits.
    // -----------------------------------------------------------------------
    const double dt6 = dt / 6.0;

    y[0]  += dt6 * (k1[0]  + 2.0 * k2[0]  + 2.0 * k3[0]  + k4[0]);
    y[1]  += dt6 * (k1[1]  + 2.0 * k2[1]  + 2.0 * k3[1]  + k4[1]);
    y[2]  += dt6 * (k1[2]  + 2.0 * k2[2]  + 2.0 * k3[2]  + k4[2]);
    y[3]  += dt6 * (k1[3]  + 2.0 * k2[3]  + 2.0 * k3[3]  + k4[3]);
    y[4]  += dt6 * (k1[4]  + 2.0 * k2[4]  + 2.0 * k3[4]  + k4[4]);
    y[5]  += dt6 * (k1[5]  + 2.0 * k2[5]  + 2.0 * k3[5]  + k4[5]);
    y[6]  += dt6 * (k1[6]  + 2.0 * k2[6]  + 2.0 * k3[6]  + k4[6]);
    y[7]  += dt6 * (k1[7]  + 2.0 * k2[7]  + 2.0 * k3[7]  + k4[7]);
    y[8]  += dt6 * (k1[8]  + 2.0 * k2[8]  + 2.0 * k3[8]  + k4[8]);
    y[9]  += dt6 * (k1[9]  + 2.0 * k2[9]  + 2.0 * k3[9]  + k4[9]);
    y[10] += dt6 * (k1[10] + 2.0 * k2[10] + 2.0 * k3[10] + k4[10]);
    y[11] += dt6 * (k1[11] + 2.0 * k2[11] + 2.0 * k3[11] + k4[11]);
    y[12] += dt6 * (k1[12] + 2.0 * k2[12] + 2.0 * k3[12] + k4[12]);
    y[13] += dt6 * (k1[13] + 2.0 * k2[13] + 2.0 * k3[13] + k4[13]);
    y[14] += dt6 * (k1[14] + 2.0 * k2[14] + 2.0 * k3[14] + k4[14]);

    // Unpack back into the state structure
    unpack_state(y, state);

    // Normalize the quaternion to prevent drift from unit sphere.
    // After one RK4 step, the norm drift is typically ~1e-15,
    // but it accumulates over thousands of steps.
    normalize_quaternion(state.quat);
}

// ---------------------------------------------------------------------------
// propagate()
//
// Public propagation method. Advances the state by dt seconds using
// RK4 integration with optional external forces and torques.
//
// This is the primary interface for the flight software loop:
//   1. Navigation filter produces a state estimate
//   2. Guidance computes desired forces/torques
//   3. This function propagates the dynamics forward
//   4. The new state feeds back into navigation
//
// Timing is measured for statistics and watchdog purposes.
// ---------------------------------------------------------------------------
void DynamicsEngine::propagate(
    StateVector& state,
    double dt,
    const double* forces,
    const double* torques)
{
    // Measure computation time for statistics and watchdog monitoring.
    // steady_clock is monotonic (never goes backwards), unlike system_clock
    // which can be adjusted by NTP or manual time changes.
    auto start = std::chrono::steady_clock::now();

    // Perform the RK4 integration step
    rk4_step(state, dt, forces, torques);

    // Update propagation statistics
    auto end = std::chrono::steady_clock::now();
    double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();

    stats_.total_steps++;
    stats_.total_time_propagated += dt;

    // Running average of computation time using cumulative moving average.
    // This avoids storing all previous values (memory-efficient).
    // Formula: avg_new = avg_old + (new_value - avg_old) / n
    stats_.avg_step_compute_us += (elapsed_us - stats_.avg_step_compute_us)
                                   / static_cast<double>(stats_.total_steps);

    if (elapsed_us > stats_.peak_step_compute_us) {
        stats_.peak_step_compute_us = elapsed_us;
    }
}

// ---------------------------------------------------------------------------
// propagate_with_model()
//
// Alternative propagation interface using a force model callback.
// The callback receives the current state and fills force/torque arrays.
// This is more flexible for complex force models (atmospheric drag that
// depends on velocity, SRP that depends on attitude, etc.).
// ---------------------------------------------------------------------------
void DynamicsEngine::propagate_with_model(
    StateVector& state,
    double dt,
    const ForceModel& model)
{
    // Evaluate the force model to get external forces and torques
    double forces[3] = {0.0, 0.0, 0.0};
    double torques[3] = {0.0, 0.0, 0.0};
    model(state, forces, torques);

    // Propagate with the computed forces and torques
    propagate(state, dt, forces, torques);
}

// ---------------------------------------------------------------------------
// elements_to_state()
//
// Convert classical orbital elements to Cartesian state vector.
//
// This is the standard Keplerian orbit to Cartesian conversion:
//   1. Compute position and velocity in the perifocal (PQW) frame
//   2. Rotate from PQW to inertial (ECI) frame using the Euler angles
//      (RAAN, inclination, argument of perigee)
//
// The perifocal frame has:
//   P-axis: toward periapsis
//   Q-axis: in the orbital plane, 90 degrees from P in direction of motion
//   W-axis: normal to orbital plane (angular momentum direction)
//
// NUMERICAL NOTE: This conversion has a singularity for circular orbits
// (e = 0) because the argument of perigee is undefined. For e < 1e-10,
// we set w = 0 by convention.
// ---------------------------------------------------------------------------
StateVector DynamicsEngine::elements_to_state(
    const OrbitalElements& oe,
    double mu)
{
    StateVector state;

    // Semi-latus rectum: p = a * (1 - e^2)
    // This is the radius at true anomaly = 90 degrees
    const double p = oe.a * (1.0 - oe.e * oe.e);

    // Radius at current true anomaly: r = p / (1 + e * cos(nu))
    const double cos_nu = std::cos(oe.nu);
    const double sin_nu = std::sin(oe.nu);
    const double r = p / (1.0 + oe.e * cos_nu);

    // Position in perifocal frame (2D orbit plane)
    const double r_pqw_x = r * cos_nu;   // Along P (toward periapsis)
    const double r_pqw_y = r * sin_nu;   // Along Q (in-plane, perpendicular)

    // Velocity in perifocal frame
    // v_P = -sqrt(mu/p) * sin(nu)
    // v_Q =  sqrt(mu/p) * (e + cos(nu))
    const double sqrt_mu_over_p = std::sqrt(mu / p);
    const double v_pqw_x = -sqrt_mu_over_p * sin_nu;
    const double v_pqw_y =  sqrt_mu_over_p * (oe.e + cos_nu);

    // Rotation matrix from PQW to ECI (3-1-3 Euler rotation)
    // R = Rz(-RAAN) * Rx(-i) * Rz(-w)
    const double cos_O = std::cos(oe.omega);  // cos(RAAN)
    const double sin_O = std::sin(oe.omega);  // sin(RAAN)
    const double cos_i = std::cos(oe.i);       // cos(inclination)
    const double sin_i = std::sin(oe.i);       // sin(inclination)
    const double cos_w = std::cos(oe.w);       // cos(argument of perigee)
    const double sin_w = std::sin(oe.w);       // sin(argument of perigee)

    // Rotation matrix elements (only the needed ones for 2D->3D)
    // R11 = cos_O*cos_w - sin_O*sin_w*cos_i
    // R12 = -cos_O*sin_w - sin_O*cos_w*cos_i
    // R21 = sin_O*cos_w + cos_O*sin_w*cos_i
    // R22 = -sin_O*sin_w + cos_O*cos_w*cos_i
    // R31 = sin_w*sin_i
    // R32 = cos_w*sin_i
    const double R11 =  cos_O * cos_w - sin_O * sin_w * cos_i;
    const double R12 = -cos_O * sin_w - sin_O * cos_w * cos_i;
    const double R21 =  sin_O * cos_w + cos_O * sin_w * cos_i;
    const double R22 = -sin_O * sin_w + cos_O * cos_w * cos_i;
    const double R31 =  sin_w * sin_i;
    const double R32 =  cos_w * sin_i;

    // Transform position: r_ECI = R * r_PQW
    state.pos[0] = R11 * r_pqw_x + R12 * r_pqw_y;
    state.pos[1] = R21 * r_pqw_x + R22 * r_pqw_y;
    state.pos[2] = R31 * r_pqw_x + R32 * r_pqw_y;

    // Transform velocity: v_ECI = R * v_PQW
    state.vel[0] = R11 * v_pqw_x + R12 * v_pqw_y;
    state.vel[1] = R21 * v_pqw_x + R22 * v_pqw_y;
    state.vel[2] = R31 * v_pqw_x + R32 * v_pqw_y;

    // Identity quaternion (no attitude information from orbital elements)
    state.quat[0] = 0.0;
    state.quat[1] = 0.0;
    state.quat[2] = 0.0;
    state.quat[3] = 1.0;

    // Zero angular velocity
    state.omega[0] = 0.0;
    state.omega[1] = 0.0;
    state.omega[2] = 0.0;

    state.mass = 500.0;
    state.time = 0.0;

    return state;
}

// ---------------------------------------------------------------------------
// orbital_energy()
//
// Compute the specific orbital energy: E = v^2/2 - mu/r
//
// For a Keplerian orbit (no perturbations), energy is conserved.
// Monitoring energy drift is a key validation metric for the integrator.
// With RK4 at dt = 1.0 s, energy should be conserved to ~1e-10 relative
// error per orbit (5600 seconds for LEO).
//
// Including J2: energy is NOT strictly conserved because J2 is a
// non-central force (depends on z/r, breaking spherical symmetry).
// However, the Jacobi constant (related to energy in a rotating frame)
// should be approximately conserved. Small energy oscillations with the
// orbital period are expected and normal.
// ---------------------------------------------------------------------------
double DynamicsEngine::orbital_energy(const StateVector& state, double mu) {
    // Kinetic energy per unit mass: T = v^2 / 2
    const double v_sq = state.vel[0] * state.vel[0] +
                        state.vel[1] * state.vel[1] +
                        state.vel[2] * state.vel[2];
    const double T = 0.5 * v_sq;

    // Potential energy per unit mass: U = -mu / r
    const double r = std::sqrt(state.pos[0] * state.pos[0] +
                               state.pos[1] * state.pos[1] +
                               state.pos[2] * state.pos[2]);
    const double U = -mu / r;

    return T + U;  // Specific orbital energy (J/kg)
}

// ---------------------------------------------------------------------------
// angular_momentum()
//
// Compute the magnitude of the specific angular momentum vector:
//   h = r x v (cross product)
//   |h| = sqrt(hx^2 + hy^2 + hz^2)
//
// For Keplerian orbits, angular momentum is conserved (central force).
// J2 causes small periodic oscillations in h because it is a non-central
// perturbation. The average angular momentum should remain constant.
// ---------------------------------------------------------------------------
double DynamicsEngine::angular_momentum(const StateVector& state) {
    // h = r x v (cross product)
    const double hx = state.pos[1] * state.vel[2] - state.pos[2] * state.vel[1];
    const double hy = state.pos[2] * state.vel[0] - state.pos[0] * state.vel[2];
    const double hz = state.pos[0] * state.vel[1] - state.pos[1] * state.vel[0];

    return std::sqrt(hx * hx + hy * hy + hz * hz);
}
