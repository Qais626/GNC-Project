// =============================================================================
// dynamics_engine.h -- Real-Time Orbital Dynamics Propagator
// =============================================================================
//
// OVERVIEW:
//   Propagates spacecraft state (position, velocity, attitude) forward in time
//   using a 4th-order Runge-Kutta integrator with J2 gravitational perturbation.
//
// DATA LAYOUT PHILOSOPHY (SoA vs. AoS):
//
//   Consider two ways to store N state vectors:
//
//   Array of Structures (AoS):
//     struct State { double x, y, z, vx, vy, vz; };
//     State states[N];
//     // Memory: [x0,y0,z0,vx0,vy0,vz0, x1,y1,z1,vx1,vy1,vz1, ...]
//
//   Structure of Arrays (SoA):
//     struct States {
//         double x[N], y[N], z[N];
//         double vx[N], vy[N], vz[N];
//     };
//     // Memory: [x0,x1,x2,..., y0,y1,y2,..., z0,z1,z2,..., ...]
//
//   WHY SoA IS BETTER FOR SIMD:
//
//   SIMD (Single Instruction, Multiple Data) instructions like SSE/AVX process
//   4 doubles (AVX-256) or 8 doubles (AVX-512) simultaneously. To use SIMD,
//   the data must be contiguous in memory:
//
//   AoS: To compute x[i]+x[i+1]+x[i+2]+x[i+3], the x values are 48 bytes
//        apart (stride of 6 doubles). The CPU must gather them from scattered
//        locations -- slow and can't use SIMD efficiently.
//
//   SoA: x[0], x[1], x[2], x[3] are contiguous. One SIMD load grabs all four.
//        The computation runs at 4x speed (or 8x with AVX-512).
//
//   Additionally, SoA improves cache utilization when processing one component
//   at a time. Computing all x-accelerations only touches the x-array, which
//   fits nicely in cache without pulling in unused y, z, vx, vy, vz data.
//
//   TRADE-OFF: SoA has worse locality for per-particle operations (computing
//   the norm of one state vector requires loading from 6 different arrays).
//   For orbit propagation where we process one state at a time, AoS is
//   actually often better. We use AoS for the main StateVector but provide
//   SoA batch processing for constellation-scale propagation.
//
// RK4 INTEGRATOR:
//
//   The 4th-order Runge-Kutta method evaluates the derivative at 4 points
//   per step and combines them with specific weights:
//
//     k1 = f(t,       y)
//     k2 = f(t + h/2, y + h*k1/2)
//     k3 = f(t + h/2, y + h*k2/2)
//     k4 = f(t + h,   y + h*k3)
//     y_next = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
//
//   Error: O(h^5) per step, O(h^4) globally.
//   For orbit propagation at dt = 1s, this gives ~1 mm accuracy per orbit.
//
//   RK4 requires 4 derivative evaluations per step. For our state vector
//   of 13 elements (pos[3] + vel[3] + quat[4] + omega[3]), that's
//   4 * 13 = 52 doubles computed per step. At 100 Hz, that's 5200 doubles
//   per second -- well within real-time budget on modern CPUs.
//
// =============================================================================

#ifndef GNC_DYNAMICS_ENGINE_H
#define GNC_DYNAMICS_ENGINE_H

#include <cmath>
#include <cstddef>
#include <cstring>
#include <array>
#include <functional>

// ---------------------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------------------
namespace constants {
    // Earth gravitational parameter (m^3/s^2)
    constexpr double MU_EARTH = 3.986004418e14;

    // Earth equatorial radius (m)
    constexpr double R_EARTH = 6.378137e6;

    // Earth J2 zonal harmonic (dimensionless)
    // J2 causes orbital precession -- nodal regression and apsidal advance.
    // It's the dominant perturbation for LEO spacecraft.
    constexpr double J2_EARTH = 1.08263e-3;

    // Standard gravitational acceleration (m/s^2)
    constexpr double G0 = 9.80665;

    // Speed of light (m/s) -- for relativistic corrections if needed
    constexpr double C_LIGHT = 299792458.0;

    // Pi -- use our own constant to avoid platform-dependent M_PI
    constexpr double PI = 3.14159265358979323846;

    // Degrees to radians
    constexpr double DEG2RAD = PI / 180.0;
    constexpr double RAD2DEG = 180.0 / PI;
}

// ---------------------------------------------------------------------------
// StateVector: The complete state of a rigid body spacecraft.
//
// Memory layout: 13 doubles + 1 double (mass) + 1 double (time) = 15 doubles
// = 120 bytes. This fits in two 64-byte cache lines.
//
// Quaternion convention: Hamilton, scalar-last [qx, qy, qz, qw]
// This is the JPL/NASA convention, matching Eigen and many flight software
// frameworks. Be careful: some systems use scalar-first!
// ---------------------------------------------------------------------------
struct StateVector {
    double pos[3];     // Position in inertial frame (m) [x, y, z]
    double vel[3];     // Velocity in inertial frame (m/s) [vx, vy, vz]
    double quat[4];    // Attitude quaternion (body-to-inertial) [qx, qy, qz, qw]
    double omega[3];   // Angular velocity in body frame (rad/s) [wx, wy, wz]
    double mass;       // Spacecraft mass (kg) -- changes with fuel consumption
    double time;       // Epoch time (seconds since reference epoch)

    // Initialize to a default LEO orbit
    void set_default() {
        // Circular orbit at 400 km altitude (ISS-like)
        pos[0] = constants::R_EARTH + 400000.0;  // m
        pos[1] = 0.0;
        pos[2] = 0.0;

        // Circular orbital velocity: v = sqrt(mu/r)
        double r = pos[0];
        double v_circ = std::sqrt(constants::MU_EARTH / r);
        vel[0] = 0.0;
        vel[1] = v_circ;
        vel[2] = 0.0;

        // Identity quaternion (no rotation)
        quat[0] = 0.0;  // qx
        quat[1] = 0.0;  // qy
        quat[2] = 0.0;  // qz
        quat[3] = 1.0;  // qw (scalar part)

        // No angular velocity
        omega[0] = 0.0;
        omega[1] = 0.0;
        omega[2] = 0.0;

        mass = 500.0;  // kg (small satellite)
        time = 0.0;
    }
};

// ---------------------------------------------------------------------------
// SoA layout for batch propagation of multiple spacecraft.
// Used for constellation-scale simulation where SIMD is beneficial.
// See header comments for why SoA enables vectorization.
// ---------------------------------------------------------------------------
struct StateVectorSoA {
    static constexpr std::size_t MAX_SPACECRAFT = 256;

    // Position components (each array is contiguous for SIMD)
    alignas(64) double px[MAX_SPACECRAFT];
    alignas(64) double py[MAX_SPACECRAFT];
    alignas(64) double pz[MAX_SPACECRAFT];

    // Velocity components
    alignas(64) double vx[MAX_SPACECRAFT];
    alignas(64) double vy[MAX_SPACECRAFT];
    alignas(64) double vz[MAX_SPACECRAFT];

    // Quaternion components
    alignas(64) double q0[MAX_SPACECRAFT];  // qx
    alignas(64) double q1[MAX_SPACECRAFT];  // qy
    alignas(64) double q2[MAX_SPACECRAFT];  // qz
    alignas(64) double q3[MAX_SPACECRAFT];  // qw

    // Angular rate components
    alignas(64) double wx[MAX_SPACECRAFT];
    alignas(64) double wy[MAX_SPACECRAFT];
    alignas(64) double wz[MAX_SPACECRAFT];

    std::size_t count = 0;  // Number of active spacecraft
};

// ---------------------------------------------------------------------------
// Orbital elements (for initialization and output)
// ---------------------------------------------------------------------------
struct OrbitalElements {
    double a;     // Semi-major axis (m)
    double e;     // Eccentricity
    double i;     // Inclination (rad)
    double omega;  // Right ascension of ascending node (rad)
    double w;     // Argument of perigee (rad)
    double nu;    // True anomaly (rad)
};

// ---------------------------------------------------------------------------
// Force/torque model callback.
// External forces and torques (thrust, drag, SRP, magnetic torquers, etc.)
// are supplied via callback to keep the dynamics engine generic.
// ---------------------------------------------------------------------------
using ForceModel = std::function<void(const StateVector& state,
                                       double forces[3],
                                       double torques[3])>;

// ---------------------------------------------------------------------------
// DynamicsEngine class
// ---------------------------------------------------------------------------
class DynamicsEngine {
public:
    DynamicsEngine();
    ~DynamicsEngine() = default;

    // -----------------------------------------------------------------------
    // propagate() -- Advance state by dt seconds.
    //
    // Uses RK4 integration with gravitational acceleration (including J2)
    // and optional external forces/torques.
    //
    // Parameters:
    //   state   - Current state (modified in place)
    //   dt      - Time step (seconds). Typical: 0.01s for control, 1.0s for nav
    //   forces  - External forces in inertial frame (N) [Fx, Fy, Fz], or nullptr
    //   torques - External torques in body frame (N*m) [Tx, Ty, Tz], or nullptr
    // -----------------------------------------------------------------------
    void propagate(StateVector& state, double dt,
                   const double* forces = nullptr,
                   const double* torques = nullptr);

    // -----------------------------------------------------------------------
    // propagate_with_model() -- Same as propagate but uses a force model callback.
    // This is more flexible for complex force models.
    // -----------------------------------------------------------------------
    void propagate_with_model(StateVector& state, double dt,
                              const ForceModel& model);

    // -----------------------------------------------------------------------
    // gravitational_acceleration() -- Central body gravity with J2 perturbation.
    //
    // J2 is the dominant gravitational perturbation for LEO satellites.
    // It arises from Earth's oblate shape (equatorial bulge).
    //
    // The J2 acceleration in Cartesian coordinates:
    //   a_J2 = -(3/2) * J2 * mu * R^2 / r^5 * [
    //     x * (1 - 5*z^2/r^2),
    //     y * (1 - 5*z^2/r^2),
    //     z * (3 - 5*z^2/r^2)
    //   ]
    //
    // Parameters:
    //   pos    - Position vector [x, y, z] in meters
    //   accel  - Output acceleration [ax, ay, az] in m/s^2
    //   mu     - Gravitational parameter (default: Earth)
    //   J2     - J2 coefficient (default: Earth)
    //   R_body - Body equatorial radius (default: Earth)
    // -----------------------------------------------------------------------
    static void gravitational_acceleration(
        const double pos[3],
        double accel[3],
        double mu = constants::MU_EARTH,
        double J2 = constants::J2_EARTH,
        double R_body = constants::R_EARTH
    );

    // -----------------------------------------------------------------------
    // Utility methods
    // -----------------------------------------------------------------------

    // Convert orbital elements to state vector
    static StateVector elements_to_state(const OrbitalElements& oe,
                                          double mu = constants::MU_EARTH);

    // Compute orbital energy (should be constant for Keplerian orbits)
    static double orbital_energy(const StateVector& state,
                                  double mu = constants::MU_EARTH);

    // Compute orbital angular momentum magnitude
    static double angular_momentum(const StateVector& state);

    // Normalize quaternion (must be done periodically to prevent drift)
    static void normalize_quaternion(double quat[4]);

    // Get propagation statistics
    struct PropStats {
        std::size_t total_steps;
        double total_time_propagated;
        double avg_step_compute_us;     // microseconds
        double peak_step_compute_us;
        double energy_drift_ppm;        // parts per million
    };
    PropStats get_stats() const { return stats_; }

private:
    // -----------------------------------------------------------------------
    // RK4 step: the core integration routine.
    //
    // Template parameter N is the state dimension (13 for our full state).
    // Using a template allows the compiler to unroll loops at compile time,
    // eliminating loop overhead for small, fixed-size vectors.
    //
    // PERFORMANCE NOTE: This function is the hot path. It runs 4 times per
    // propagation step. We use manual loop unrolling and avoid virtual
    // function calls inside the loop.
    // -----------------------------------------------------------------------
    void rk4_step(StateVector& state, double dt,
                  const double* ext_forces, const double* ext_torques);

    // Compute the state derivative (right-hand side of the ODE)
    void state_derivative(const StateVector& state,
                          const double* ext_forces,
                          const double* ext_torques,
                          double deriv[15]);

    // Quaternion kinematics: dq/dt = 0.5 * q * omega_quat
    static void quaternion_derivative(const double quat[4],
                                       const double omega[3],
                                       double dquat[4]);

    // Pack/unpack StateVector to/from a flat array for RK4
    static void pack_state(const StateVector& state, double y[15]);
    static void unpack_state(const double y[15], StateVector& state);

    // Statistics tracking
    PropStats stats_ = {};
};

#endif // GNC_DYNAMICS_ENGINE_H
