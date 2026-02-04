// =============================================================================
// main.cpp -- GNC Real-Time Simulation Entry Point
// =============================================================================
//
// OVERVIEW:
//   This is the main entry point for the GNC real-time simulation. It
//   exercises all major subsystems and generates performance data for
//   integration with the Python analysis pipeline.
//
// EXECUTION SEQUENCE:
//
//   1. DYNAMICS BENCHMARK:
//      Propagates a LEO orbit for 100,000 time steps using the RK4
//      integrator with J2 perturbation. Outputs trajectory to CSV for
//      Python visualization. Measures propagation throughput.
//
//   2. FLIGHT COMPUTER DEMO:
//      Runs the deterministic task scheduler with representative GNC
//      tasks (guidance at 1 Hz, navigation at 10 Hz, control at 50 Hz).
//      Demonstrates rate-monotonic scheduling, watchdog monitoring,
//      and triple modular redundancy (TMR).
//
//   3. MEMORY POOL BENCHMARK:
//      Compares custom pool allocation vs. new/delete and malloc/free.
//      Measures allocation throughput and timing determinism.
//
//   4. RING BUFFER BENCHMARK:
//      Tests lock-free SPSC ring buffer throughput in single-threaded
//      and concurrent producer-consumer configurations. Compares
//      against mutex-protected std::queue.
//
//   5. HIL MOCK LOOP:
//      Runs the hardware-in-the-loop interface in mock mode, simulating
//      sensor data reception and actuator command transmission at 50 Hz
//      for 10 seconds.
//
// OUTPUT:
//   - Console: Detailed benchmark results and statistics
//   - CSV file: Trajectory data for Python visualization
//     Format: time,x,y,z,vx,vy,vz,qx,qy,qz,qw,wx,wy,wz,energy,h_mag
//
// =============================================================================

#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <cmath>

#include "dynamics_engine.h"
#include "flight_computer.h"
#include "hil_interface.h"
#include "memory_pool.h"
#include "ring_buffer.h"

// ---------------------------------------------------------------------------
// Forward declarations for benchmark functions defined in other .cpp files.
// These are free functions (not class methods) that exercise each subsystem.
// ---------------------------------------------------------------------------
extern void benchmark_memory_pool();
extern void benchmark_vs_malloc();
extern void print_memory_stats();
// benchmark_ring_buffer() is declared in ring_buffer.h

// ---------------------------------------------------------------------------
// Helper: Print a banner with consistent formatting.
// Uses ANSI escape codes for emphasis (supported by most modern terminals).
// In flight software, we would use plain text for UART consoles.
// ---------------------------------------------------------------------------
static void print_banner() {
    std::printf("\n");
    std::printf("###############################################################\n");
    std::printf("#                                                             #\n");
    std::printf("#   GNC REAL-TIME SIMULATION ENGINE                           #\n");
    std::printf("#   Guidance, Navigation, and Control                         #\n");
    std::printf("#                                                             #\n");
    std::printf("#   Subsystems:                                               #\n");
    std::printf("#     - RK4 Orbital Dynamics with J2 Perturbation             #\n");
    std::printf("#     - Rate-Monotonic Flight Computer Scheduler              #\n");
    std::printf("#     - Lock-Free SPSC Ring Buffer (Telemetry)                #\n");
    std::printf("#     - Custom Memory Pool Allocator                          #\n");
    std::printf("#     - Hardware-in-the-Loop Mock Interface                   #\n");
    std::printf("#                                                             #\n");
    std::printf("###############################################################\n\n");
}

// ---------------------------------------------------------------------------
// SECTION 1: Dynamics Benchmark
//
// Propagates a spacecraft in a ~400 km circular LEO orbit using the RK4
// integrator. The initial conditions represent an ISS-like orbit.
//
// Physics:
//   - Central body: Earth (mu = 3.986e14 m^3/s^2)
//   - Perturbations: J2 zonal harmonic (1.083e-3)
//   - Orbit: ~400 km altitude, circular (e ~ 0)
//   - Period: ~92 minutes (5554 seconds)
//   - Time step: 1.0 second
//   - Total propagation: 100,000 seconds (~18 orbits)
//
// Outputs:
//   - Console: Throughput statistics (steps/sec, us/step)
//   - CSV: Position, velocity, attitude, energy, angular momentum
//     Written every 100 steps to keep file size manageable
//     (~1000 rows for 100,000 steps)
//
// Validation:
//   - Orbital energy should be approximately conserved (drift < 1 ppm)
//   - Angular momentum should be approximately conserved
//   - Position should trace a closed ellipse (with J2 precession)
// ---------------------------------------------------------------------------
static void run_dynamics_benchmark() {
    std::printf("===== SECTION 1: ORBITAL DYNAMICS BENCHMARK =====\n\n");

    DynamicsEngine engine;

    // Set up initial state: circular orbit at 400 km altitude
    StateVector state;
    state.set_default();

    // Record initial orbital energy for drift monitoring
    double initial_energy = DynamicsEngine::orbital_energy(state);
    double initial_h = DynamicsEngine::angular_momentum(state);

    std::printf("  Initial conditions:\n");
    std::printf("    Position:    [%.1f, %.1f, %.1f] m\n",
                state.pos[0], state.pos[1], state.pos[2]);
    std::printf("    Velocity:    [%.2f, %.2f, %.2f] m/s\n",
                state.vel[0], state.vel[1], state.vel[2]);
    std::printf("    Altitude:    %.1f km\n",
                (std::sqrt(state.pos[0]*state.pos[0] +
                           state.pos[1]*state.pos[1] +
                           state.pos[2]*state.pos[2]) - constants::R_EARTH) / 1000.0);
    std::printf("    Orb. energy: %.6e J/kg\n", initial_energy);
    std::printf("    Ang. moment: %.6e m^2/s\n", initial_h);
    std::printf("\n");

    // Open CSV file for trajectory output.
    // The path is relative to the build directory, going up to the project
    // root and into output/data/. This matches the Python analysis scripts.
    std::string csv_path = "../../output/data/cpp_trajectory.csv";
    std::ofstream csv(csv_path);

    if (!csv.is_open()) {
        // Try absolute fallback path
        csv_path = "/tmp/cpp_trajectory.csv";
        csv.open(csv_path);
        if (!csv.is_open()) {
            std::fprintf(stderr, "  WARNING: Could not open CSV file for writing.\n");
            std::fprintf(stderr, "  Trajectory data will not be saved.\n\n");
        }
    }

    if (csv.is_open()) {
        // CSV header
        csv << "time,x,y,z,vx,vy,vz,qx,qy,qz,qw,wx,wy,wz,energy,h_mag\n";
        std::printf("  Writing trajectory to: %s\n\n", csv_path.c_str());
    }

    // Propagation parameters
    constexpr std::size_t NUM_STEPS = 100000;
    constexpr double DT = 1.0;               // 1 second time step
    constexpr std::size_t CSV_INTERVAL = 100; // Write CSV every 100 steps

    // Benchmark the propagation
    std::printf("  Propagating %zu steps (dt = %.1f s, total = %.0f s = %.1f orbits)...\n",
                NUM_STEPS, DT, static_cast<double>(NUM_STEPS) * DT,
                static_cast<double>(NUM_STEPS) * DT / 5554.0);

    auto bench_start = std::chrono::high_resolution_clock::now();

    for (std::size_t step = 0; step < NUM_STEPS; ++step) {
        // Propagate one step (gravity + J2, no external forces)
        engine.propagate(state, DT);

        // Write to CSV at regular intervals
        if (csv.is_open() && (step % CSV_INTERVAL == 0)) {
            double energy = DynamicsEngine::orbital_energy(state);
            double h_mag = DynamicsEngine::angular_momentum(state);

            csv << state.time << ","
                << state.pos[0] << "," << state.pos[1] << "," << state.pos[2] << ","
                << state.vel[0] << "," << state.vel[1] << "," << state.vel[2] << ","
                << state.quat[0] << "," << state.quat[1] << ","
                << state.quat[2] << "," << state.quat[3] << ","
                << state.omega[0] << "," << state.omega[1] << "," << state.omega[2] << ","
                << energy << "," << h_mag << "\n";
        }
    }

    auto bench_end = std::chrono::high_resolution_clock::now();
    double bench_elapsed_s = std::chrono::duration<double>(bench_end - bench_start).count();
    double bench_elapsed_us = bench_elapsed_s * 1.0e6;

    if (csv.is_open()) {
        csv.close();
    }

    // Compute final orbital parameters
    double final_energy = DynamicsEngine::orbital_energy(state);
    double final_h = DynamicsEngine::angular_momentum(state);
    double energy_drift_ppm = std::fabs((final_energy - initial_energy) / initial_energy) * 1e6;
    double h_drift_ppm = std::fabs((final_h - initial_h) / initial_h) * 1e6;
    double final_altitude = (std::sqrt(state.pos[0]*state.pos[0] +
                                        state.pos[1]*state.pos[1] +
                                        state.pos[2]*state.pos[2]) - constants::R_EARTH) / 1000.0;

    // Get propagation statistics from the engine
    DynamicsEngine::PropStats stats = engine.get_stats();

    // Print results
    std::printf("\n  Results:\n");
    std::printf("    Final position:    [%.1f, %.1f, %.1f] m\n",
                state.pos[0], state.pos[1], state.pos[2]);
    std::printf("    Final altitude:    %.1f km\n", final_altitude);
    std::printf("    Final energy:      %.6e J/kg\n", final_energy);
    std::printf("    Energy drift:      %.3f ppm (%.2e relative)\n",
                energy_drift_ppm, (final_energy - initial_energy) / initial_energy);
    std::printf("    Ang. momentum drift: %.3f ppm\n", h_drift_ppm);
    std::printf("\n");
    std::printf("  Performance:\n");
    std::printf("    Total wall time:   %.3f ms\n", bench_elapsed_us / 1000.0);
    std::printf("    Steps/second:      %.0f\n",
                static_cast<double>(NUM_STEPS) / bench_elapsed_s);
    std::printf("    Microseconds/step: %.2f\n", bench_elapsed_us / static_cast<double>(NUM_STEPS));
    std::printf("    Avg step time:     %.2f us (engine measured)\n", stats.avg_step_compute_us);
    std::printf("    Peak step time:    %.2f us\n", stats.peak_step_compute_us);
    std::printf("\n");
}

// ---------------------------------------------------------------------------
// SECTION 2: Flight Computer Demo
//
// Demonstrates the deterministic task scheduler with three representative
// GNC tasks running at different rates.
//
// Task Set (Rate-Monotonic Scheduling):
//
//   Task          Rate    Period   Priority   Description
//   ----          ----    ------   --------   -----------
//   Control       50 Hz   20 ms    1 (high)   Attitude/orbit control loop
//   Navigation    10 Hz   100 ms   2 (med)    Navigation filter update
//   Guidance       1 Hz   1000 ms  3 (low)    Trajectory planning
//
// The control task runs most frequently (50 Hz) and gets highest priority.
// This follows the Rate-Monotonic Assignment: tasks with shorter periods
// get higher priority (lower number).
//
// Schedulability check:
//   Assuming each task takes ~100 us WCET:
//   U = 100e-6/0.02 + 100e-6/0.1 + 100e-6/1.0
//     = 0.005 + 0.001 + 0.0001
//     = 0.0061 (0.61%)
//   This is well within the RMS bound of 78.0% for 3 tasks.
//
// The flight computer runs for 1 second (100 base cycles at 100 Hz).
// During this time:
//   - Control executes 50 times (50 Hz * 1 s)
//   - Navigation executes 10 times (10 Hz * 1 s)
//   - Guidance executes 1 time (1 Hz * 1 s)
// ---------------------------------------------------------------------------
static void run_flight_computer_demo() {
    std::printf("===== SECTION 2: FLIGHT COMPUTER DEMO =====\n\n");

    // Create flight computer with 100 Hz base rate
    // All task rates must divide evenly into 100 Hz.
    FlightComputer fc(100.0);

    // Register GNC tasks with appropriate callbacks.
    // In a real system, these callbacks would call the actual GNC algorithms.
    // Here, we simulate them with small computation loops to generate
    // realistic timing statistics.

    // Task 1: Control loop (50 Hz, highest priority)
    // Computes torque commands based on attitude error.
    // Typical computation: quaternion error -> PD controller -> torque
    fc.add_task("CTRL_50Hz", 1, 50.0, [](double mission_time) -> int {
        // Simulate control computation (~50-100 us)
        // In real flight software, this would be:
        //   1. Read attitude quaternion from nav filter
        //   2. Compute quaternion error vs. target
        //   3. Apply PD control law: torque = Kp*error + Kd*rate_error
        //   4. Saturate torque to actuator limits
        //   5. Write command to actuator interface
        volatile double result = 0.0;
        for (int i = 0; i < 50; ++i) {
            result += std::sin(mission_time + static_cast<double>(i) * 0.01);
        }
        (void)result;
        return 0;  // Success
    }, false);  // Not TMR-critical (runs at high rate, inherently redundant)

    // Task 2: Navigation filter (10 Hz, medium priority)
    // Fuses sensor measurements to estimate spacecraft state.
    // Typical: Extended Kalman Filter with 15-state vector
    fc.add_task("NAV_10Hz", 2, 10.0, [](double mission_time) -> int {
        // Simulate navigation computation (~200-500 us)
        // In real flight software, this would be:
        //   1. Read IMU, GPS, star tracker measurements
        //   2. Propagate state covariance (P = F*P*F' + Q)
        //   3. Compute Kalman gain (K = P*H' * (H*P*H' + R)^-1)
        //   4. Update state estimate (x += K * (z - H*x))
        //   5. Update covariance (P = (I - K*H) * P)
        volatile double result = 0.0;
        for (int i = 0; i < 200; ++i) {
            result += std::cos(mission_time + static_cast<double>(i) * 0.005);
        }
        (void)result;
        return 0;
    }, false);

    // Task 3: Guidance (1 Hz, lowest priority)
    // Plans the trajectory and computes desired attitude/thrust profiles.
    // Typically the most computationally expensive GNC task.
    fc.add_task("GUID_1Hz", 3, 1.0, [](double mission_time) -> int {
        // Simulate guidance computation (~500-1000 us)
        // In real flight software, this would be:
        //   1. Evaluate current orbital elements
        //   2. Compute required delta-V for orbit maintenance
        //   3. Plan maneuver timing and direction
        //   4. Generate attitude guidance profile for control
        //   5. Update telemetry with guidance solution
        volatile double result = 0.0;
        for (int i = 0; i < 500; ++i) {
            result += std::sqrt(static_cast<double>(i) + mission_time);
        }
        (void)result;
        return 0;
    }, true);  // TMR-critical: guidance errors could lead to wrong orbits

    // Initialize the flight computer
    // This sorts tasks by priority, checks schedulability, arms watchdog
    if (!fc.initialize()) {
        std::fprintf(stderr, "  Flight computer initialization FAILED\n\n");
        return;
    }

    // Run for 100 cycles (1 second at 100 Hz) in fast-simulation mode
    // (realtime=false: don't sleep between cycles)
    fc.run_for(1.0, false);

    // Print the comprehensive timing report
    std::printf("%s", fc.timing_report().c_str());

    // Demonstrate TMR voting
    std::printf("  TMR Voting Demonstration:\n");
    TMRResult tmr_result = fc.tmr_vote(
        [](double t) -> double {
            return std::sin(t) * 100.0;  // Simple computation
        },
        1.0,    // mission_time = 1.0 s
        1e-10   // tolerance
    );
    std::printf("    TMR result:      %.6f\n", tmr_result.value);
    std::printf("    All agree:       %s\n", tmr_result.agreement ? "YES" : "NO");
    std::printf("    Max deviation:   %.2e\n", tmr_result.max_deviation);
    std::printf("\n");

    // Shutdown
    fc.shutdown();
}

// ---------------------------------------------------------------------------
// SECTION 5: HIL Mock Loop
//
// Runs the hardware-in-the-loop interface in mock mode, simulating the
// closed-loop interaction between the dynamics simulation and flight
// hardware.
//
// Loop architecture:
//   1. Receive sensor data from "hardware" (mock: generated from orbit model)
//   2. Process sensor data through a simple control law
//   3. Send actuator commands back to "hardware"
//   4. Sleep to maintain the commanded rate (50 Hz)
//
// This demonstrates:
//   - Real-time loop timing with chrono steady_clock
//   - CRC-16 computation and verification
//   - Packet serialization/deserialization
//   - Mock hardware data generation
//
// Duration: 2 seconds at 50 Hz = 100 cycles
// (Reduced from 10 seconds for faster benchmark completion)
// ---------------------------------------------------------------------------
static void run_hil_mock_loop() {
    std::printf("===== SECTION 5: HIL MOCK LOOP =====\n\n");

    HILInterface hil;

    // Configure and connect in mock mode (default)
    HILConfig config;
    config.protocol = HILConfig::Protocol::UDP;
    config.udp_host = "127.0.0.1";
    config.udp_port_tx = 5000;
    config.udp_port_rx = 5001;

    if (!hil.connect(config)) {
        std::fprintf(stderr, "  HIL connection failed\n\n");
        return;
    }

    // Perform initial time synchronization
    hil.perform_time_sync();
    std::printf("  Clock offset: %.1f us, one-way delay: %.1f us\n",
                hil.get_clock_offset_us(),
                hil.get_stats().one_way_delay_us);

    // Run the HIL loop
    constexpr double LOOP_RATE_HZ = 50.0;
    constexpr double LOOP_PERIOD_S = 1.0 / LOOP_RATE_HZ;
    constexpr double DURATION_S = 2.0;
    std::size_t num_cycles = static_cast<std::size_t>(DURATION_S * LOOP_RATE_HZ);

    std::printf("  Running HIL loop: %.0f Hz for %.1f s (%zu cycles)\n\n",
                LOOP_RATE_HZ, DURATION_S, num_cycles);

    auto loop_start = std::chrono::steady_clock::now();
    auto next_cycle = loop_start;

    std::size_t rx_count = 0;
    std::size_t tx_count = 0;
    double min_cycle_us = 1e9;
    double max_cycle_us = 0.0;
    double total_cycle_us = 0.0;

    for (std::size_t i = 0; i < num_cycles; ++i) {
        auto cycle_start = std::chrono::steady_clock::now();

        // Step 1: Receive sensor data from hardware
        auto sensor_data = hil.receive_sensor_data();
        if (sensor_data.has_value()) {
            rx_count++;

            // Step 2: Simple control law (proportional feedback)
            // In a real system, this would be the full GNC pipeline:
            //   navigation filter -> guidance -> control -> actuator allocation
            ActuatorCommandPacket cmd;
            std::memset(&cmd, 0, sizeof(ActuatorCommandPacket));

            // Proportional control on angular rate (rate damping)
            // Torque = -Kd * omega (simple derivative controller)
            constexpr float Kd = 0.5f;
            cmd.rw_torque_cmd[0] = -Kd * sensor_data->gyro[0];
            cmd.rw_torque_cmd[1] = -Kd * sensor_data->gyro[1];
            cmd.rw_torque_cmd[2] = -Kd * sensor_data->gyro[2];
            cmd.rw_torque_cmd[3] = 0.0f;  // 4th wheel for redundancy

            // Set simulation timestamp
            cmd.sim_timestamp_us = static_cast<uint64_t>(
                static_cast<double>(i) * LOOP_PERIOD_S * 1e6);

            // Step 3: Send actuator commands
            if (hil.send_actuator_commands(cmd)) {
                tx_count++;
            }
        }

        // Measure cycle time
        auto cycle_end = std::chrono::steady_clock::now();
        double cycle_us = std::chrono::duration<double, std::micro>(
            cycle_end - cycle_start).count();

        if (cycle_us < min_cycle_us) min_cycle_us = cycle_us;
        if (cycle_us > max_cycle_us) max_cycle_us = cycle_us;
        total_cycle_us += cycle_us;

        // Step 4: Sleep to maintain loop rate
        // sleep_until compensates for execution time, preventing drift.
        next_cycle += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(LOOP_PERIOD_S));
        std::this_thread::sleep_until(next_cycle);
    }

    auto loop_end = std::chrono::steady_clock::now();
    double total_s = std::chrono::duration<double>(loop_end - loop_start).count();

    // Get statistics
    HILStats stats = hil.get_stats();

    std::printf("  HIL Loop Results:\n");
    std::printf("    Duration:        %.3f s (target: %.1f s)\n", total_s, DURATION_S);
    std::printf("    Effective rate:  %.1f Hz (target: %.0f Hz)\n",
                static_cast<double>(num_cycles) / total_s, LOOP_RATE_HZ);
    std::printf("    Packets sent:    %zu\n", tx_count);
    std::printf("    Packets recv:    %zu\n", rx_count);
    std::printf("    CRC errors:      %zu\n", stats.crc_errors);
    std::printf("    Cycle timing:    min=%.1f us, max=%.1f us, avg=%.1f us\n",
                min_cycle_us, max_cycle_us,
                total_cycle_us / static_cast<double>(num_cycles));
    std::printf("    Jitter (p-p):    %.1f us\n", max_cycle_us - min_cycle_us);
    std::printf("\n");

    // Disconnect
    hil.disconnect();
}

// ===========================================================================
// MAIN
// ===========================================================================
int main() {
    // Print the project banner
    print_banner();

    // Record overall benchmark start time
    auto total_start = std::chrono::high_resolution_clock::now();

    // -----------------------------------------------------------------------
    // Section 1: Dynamics Benchmark
    // Propagate a LEO orbit for 100,000 steps, save trajectory to CSV.
    // -----------------------------------------------------------------------
    auto sec1_start = std::chrono::high_resolution_clock::now();
    run_dynamics_benchmark();
    auto sec1_end = std::chrono::high_resolution_clock::now();
    double sec1_ms = std::chrono::duration<double, std::milli>(sec1_end - sec1_start).count();

    // -----------------------------------------------------------------------
    // Section 2: Flight Computer Demo
    // Run the task scheduler for 100 cycles with GNC tasks.
    // -----------------------------------------------------------------------
    auto sec2_start = std::chrono::high_resolution_clock::now();
    run_flight_computer_demo();
    auto sec2_end = std::chrono::high_resolution_clock::now();
    double sec2_ms = std::chrono::duration<double, std::milli>(sec2_end - sec2_start).count();

    // -----------------------------------------------------------------------
    // Section 3: Memory Pool Benchmark
    // Compare custom pool vs. new/delete vs. malloc/free.
    // -----------------------------------------------------------------------
    auto sec3_start = std::chrono::high_resolution_clock::now();
    benchmark_memory_pool();
    benchmark_vs_malloc();
    print_memory_stats();
    auto sec3_end = std::chrono::high_resolution_clock::now();
    double sec3_ms = std::chrono::duration<double, std::milli>(sec3_end - sec3_start).count();

    // -----------------------------------------------------------------------
    // Section 4: Ring Buffer Benchmark
    // Test lock-free throughput vs. mutex-protected queue.
    // -----------------------------------------------------------------------
    auto sec4_start = std::chrono::high_resolution_clock::now();
    RingBufferPerfResult rb_result = benchmark_ring_buffer(500000);
    auto sec4_end = std::chrono::high_resolution_clock::now();
    double sec4_ms = std::chrono::duration<double, std::milli>(sec4_end - sec4_start).count();

    // -----------------------------------------------------------------------
    // Section 5: HIL Mock Loop
    // Run mock hardware-in-the-loop at 50 Hz for 2 seconds.
    // -----------------------------------------------------------------------
    auto sec5_start = std::chrono::high_resolution_clock::now();
    run_hil_mock_loop();
    auto sec5_end = std::chrono::high_resolution_clock::now();
    double sec5_ms = std::chrono::duration<double, std::milli>(sec5_end - sec5_start).count();

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    std::printf("\n");
    std::printf("###############################################################\n");
    std::printf("#                                                             #\n");
    std::printf("#   BENCHMARK SUMMARY                                         #\n");
    std::printf("#                                                             #\n");
    std::printf("###############################################################\n\n");

    std::printf("  %-35s %10.1f ms\n", "1. Dynamics (100k RK4 steps):", sec1_ms);
    std::printf("  %-35s %10.1f ms\n", "2. Flight Computer (100 cycles):", sec2_ms);
    std::printf("  %-35s %10.1f ms\n", "3. Memory Pool benchmarks:", sec3_ms);
    std::printf("  %-35s %10.1f ms\n", "4. Ring Buffer benchmarks:", sec4_ms);
    std::printf("  %-35s %10.1f ms\n", "5. HIL Mock Loop (50 Hz, 2s):", sec5_ms);
    std::printf("  %-35s %10.1f ms\n", "   TOTAL:", total_ms);
    std::printf("\n");

    std::printf("  Ring Buffer Lock-Free Throughput: %.2f M ops/sec\n",
                rb_result.lock_free_throughput_mops);
    std::printf("  Ring Buffer Speedup vs Mutex:     %.2fx\n",
                rb_result.speedup_factor);
    std::printf("\n");
    std::printf("###############################################################\n\n");

    return 0;
}
