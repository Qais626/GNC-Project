// =============================================================================
// flight_computer.h -- Flight Software Architecture for Real-Time GNC
// =============================================================================
//
// OVERVIEW:
//   This module implements a deterministic real-time task scheduler inspired by
//   flight software architectures used on actual spacecraft (e.g., NASA cFS,
//   ESA TASTE, ARINC 653 partitioned scheduling).
//
// REAL-TIME SCHEDULING CONCEPTS:
//
//   RATE MONOTONIC SCHEDULING (RMS):
//     The optimal fixed-priority scheduling algorithm for periodic tasks.
//     Tasks with shorter periods (higher rates) get higher priority.
//     Example:
//       - Control loop: 100 Hz (period = 10 ms) -> highest priority
//       - Navigation:   10 Hz  (period = 100 ms) -> medium priority
//       - Guidance:      1 Hz  (period = 1000 ms) -> lower priority
//       - Telemetry:     5 Hz  (period = 200 ms) -> low priority
//
//     RMS is provably optimal: if any fixed-priority algorithm can schedule
//     a task set, RMS can too. The schedulability bound is:
//       U = sum(Ci/Ti) <= n * (2^(1/n) - 1)
//     where Ci = worst-case execution time, Ti = period, n = number of tasks.
//     For large n, this approaches ln(2) = 0.693 (69.3% CPU utilization).
//
//   PRIORITY INVERSION:
//     When a high-priority task is blocked by a low-priority task that holds
//     a shared resource (mutex/lock). The danger: a medium-priority task can
//     preempt the low-priority task, causing the high-priority task to wait
//     indefinitely. This is what caused the Mars Pathfinder reset anomaly.
//
//     Solutions:
//     1. Priority Inheritance: When a low-priority task blocks a high-priority
//        task, temporarily boost the low task's priority.
//     2. Priority Ceiling: Set the mutex's priority to the highest priority
//        of any task that might use it.
//     3. Lock-free data structures: Eliminate shared locks entirely (our approach).
//
//   WATCHDOG TIMER:
//     A hardware or software timer that must be periodically "kicked" by the
//     main loop. If the kick doesn't happen within the timeout period, the
//     watchdog triggers a recovery action (usually a system reset).
//
//     This catches infinite loops, deadlocks, and other hangs. On spacecraft,
//     the watchdog is typically the last line of defense -- hardware watchdog
//     timers in the processor reset the entire CPU if software is unresponsive.
//
//   TRIPLE MODULAR REDUNDANCY (TMR):
//     Run three independent copies of a critical computation and take the
//     majority vote. If one copy produces a wrong result (due to a radiation-
//     induced Single Event Upset (SEU), for example), the other two still
//     agree and the correct result is used.
//
//     For floating-point results, we can't do exact equality. Instead, we
//     take the median value or use a tolerance-based voting scheme.
//
//     TMR costs 3x computation but provides radiation hardening without
//     expensive rad-hard processors. Many modern small satellites use
//     commercial-off-the-shelf (COTS) processors with software TMR.
//
// DETERMINISTIC TIMING:
//
//   Real-time doesn't mean "fast" -- it means "predictable." A system that
//   always completes in exactly 5 ms is more real-time than one that
//   usually completes in 1 ms but sometimes takes 50 ms.
//
//   We achieve determinism by:
//   1. Fixed iteration counts (no while loops in the hot path)
//   2. No dynamic memory allocation (use memory pools)
//   3. No system calls (no printf, no file I/O in the loop)
//   4. Pre-computed tables instead of runtime math where possible
//   5. Bounded-time algorithms only (no unbounded search/sort)
//
// =============================================================================

#ifndef GNC_FLIGHT_COMPUTER_H
#define GNC_FLIGHT_COMPUTER_H

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <functional>
#include <string>
#include <vector>
#include <array>
#include <atomic>

// ---------------------------------------------------------------------------
// Task callback type.
// The callback receives the current mission time and returns a status code.
// Non-zero return indicates an error.
// ---------------------------------------------------------------------------
using TaskCallback = std::function<int(double mission_time)>;

// ---------------------------------------------------------------------------
// Task definition
// ---------------------------------------------------------------------------
struct Task {
    std::string    name;           // Human-readable task name
    int            priority;       // Lower number = higher priority (RMS convention)
    double         rate_hz;        // Execution rate in Hz
    double         period_s;       // Period = 1/rate (computed)
    double         last_exec_time; // Last execution timestamp
    double         wcet_us;        // Worst-case execution time (microseconds)
    double         avg_exec_us;    // Average execution time
    double         total_exec_us;  // Total execution time
    std::size_t    exec_count;     // Number of executions
    std::size_t    overrun_count;  // Number of timing overruns
    TaskCallback   callback;       // The function to execute
    bool           enabled;        // Task enable/disable flag
    bool           is_critical;    // If true, use TMR for this task
};

// ---------------------------------------------------------------------------
// Cycle timing statistics
// ---------------------------------------------------------------------------
struct CycleStats {
    double min_cycle_us;       // Best-case cycle time
    double max_cycle_us;       // Worst-case cycle time
    double avg_cycle_us;       // Average cycle time
    double jitter_us;          // max - min (measure of timing determinism)
    double target_cycle_us;    // Expected cycle time
    double cpu_utilization;    // fraction of cycle time used (0.0 to 1.0)
    std::size_t total_cycles;  // Number of cycles completed
    std::size_t overruns;      // Number of cycles that exceeded target
};

// ---------------------------------------------------------------------------
// TMR result for a double-precision value
// ---------------------------------------------------------------------------
struct TMRResult {
    double value;              // The voted/selected value
    bool   agreement;          // True if all 3 copies agreed (within tolerance)
    int    disagreeing_copy;   // Which copy disagreed (-1 if all agree)
    double max_deviation;      // Maximum deviation between any two copies
};

// ---------------------------------------------------------------------------
// System health/status
// ---------------------------------------------------------------------------
enum class SystemMode : uint8_t {
    BOOT,            // System starting up
    INITIALIZATION,  // Loading configs, self-test
    NOMINAL,         // Normal operation
    SAFE_MODE,       // Reduced functionality due to fault
    SHUTDOWN         // Controlled shutdown
};

// ---------------------------------------------------------------------------
// FlightComputer class
// ---------------------------------------------------------------------------
class FlightComputer {
public:
    // -----------------------------------------------------------------------
    // Constructor
    //   base_rate_hz: The fundamental cycle rate of the main loop.
    //                 All task rates must be divisors of this rate.
    //                 Typical values: 100 Hz (spacecraft), 1000 Hz (reaction wheels)
    // -----------------------------------------------------------------------
    explicit FlightComputer(double base_rate_hz = 100.0);
    ~FlightComputer();

    // No copy (owns system resources)
    FlightComputer(const FlightComputer&) = delete;
    FlightComputer& operator=(const FlightComputer&) = delete;

    // -----------------------------------------------------------------------
    // initialize() -- Perform system initialization.
    //
    // This must be called before run_cycle(). It:
    //   1. Validates task schedule (checks RMS schedulability)
    //   2. Pre-computes task execution table
    //   3. Resets all statistics
    //   4. Sets up watchdog timer
    //   5. Transitions to NOMINAL mode
    //
    // Returns true if initialization succeeded.
    // -----------------------------------------------------------------------
    bool initialize();

    // -----------------------------------------------------------------------
    // add_task() -- Register a task with the scheduler.
    //
    // Must be called BEFORE initialize().
    //
    // Parameters:
    //   name      - Task name (for logging/telemetry)
    //   priority  - Priority level (lower = higher priority, per RMS)
    //   rate_hz   - Execution rate. Must be a divisor of base_rate_hz.
    //   callback  - Function to execute
    //   critical  - If true, execute with TMR protection
    //
    // Returns task index, or -1 on failure.
    // -----------------------------------------------------------------------
    int add_task(const std::string& name, int priority, double rate_hz,
                 TaskCallback callback, bool critical = false);

    // -----------------------------------------------------------------------
    // run_cycle() -- Execute one iteration of the main loop.
    //
    // This is the heart of the flight software. Each call:
    //   1. Records cycle start time
    //   2. Kicks the watchdog timer
    //   3. Checks which tasks are due to execute this cycle
    //   4. Executes due tasks in priority order
    //   5. For critical tasks, runs TMR and votes
    //   6. Checks for timing overruns
    //   7. Records cycle end time and updates statistics
    //
    // The caller is responsible for calling this at the base rate.
    // Typically done in a loop with sleep_until() for timing.
    // -----------------------------------------------------------------------
    void run_cycle();

    // -----------------------------------------------------------------------
    // run_for() -- Run the main loop for a specified duration.
    //
    // This is a convenience method that handles the timing loop internally.
    // Uses std::chrono::steady_clock for monotonic timing.
    //
    // Parameters:
    //   duration_s  - How long to run (seconds)
    //   realtime    - If true, sleep between cycles for real-time pacing
    //                 If false, run as fast as possible (simulation mode)
    // -----------------------------------------------------------------------
    void run_for(double duration_s, bool realtime = false);

    // -----------------------------------------------------------------------
    // shutdown() -- Graceful shutdown.
    // Disables all tasks, flushes logs, transitions to SHUTDOWN mode.
    // -----------------------------------------------------------------------
    void shutdown();

    // -----------------------------------------------------------------------
    // TMR voting
    // -----------------------------------------------------------------------

    // Run a computation 3 times and return the voted result.
    // The function takes mission time and returns a double.
    TMRResult tmr_vote(std::function<double(double)> computation,
                       double mission_time,
                       double tolerance = 1e-10);

    // Vote on an array of values (e.g., a 3-element force vector)
    void tmr_vote_array(std::function<void(double, double*, std::size_t)> computation,
                        double mission_time,
                        double* result, std::size_t n,
                        double tolerance = 1e-10);

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    CycleStats    get_cycle_stats() const { return cycle_stats_; }
    SystemMode    get_mode() const { return mode_; }
    double        get_mission_time() const { return mission_time_; }
    std::size_t   get_task_count() const { return tasks_.size(); }
    const Task&   get_task(std::size_t idx) const { return tasks_[idx]; }
    bool          is_running() const { return running_; }

    // Check if the current task set is RMS-schedulable
    bool check_schedulability() const;

    // Get CPU utilization estimate based on measured WCETs
    double estimated_cpu_utilization() const;

    // Generate timing report
    std::string timing_report() const;

private:
    // -----------------------------------------------------------------------
    // Watchdog
    // -----------------------------------------------------------------------
    struct Watchdog {
        std::chrono::steady_clock::time_point last_kick;
        double timeout_ms;
        bool armed;
        std::size_t timeout_count;

        void kick() {
            last_kick = std::chrono::steady_clock::now();
        }

        bool check() const {
            if (!armed) return true;
            auto now = std::chrono::steady_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(
                now - last_kick
            ).count();
            return elapsed_ms < timeout_ms;
        }
    };

    // -----------------------------------------------------------------------
    // Internal methods
    // -----------------------------------------------------------------------

    // Sort tasks by priority (lower number = higher priority)
    void sort_tasks_by_priority();

    // Check if a task is due to execute this cycle
    bool is_task_due(const Task& task, double current_time) const;

    // Execute a single task with timing
    int execute_task(Task& task, double mission_time);

    // Execute a critical task with TMR
    int execute_task_tmr(Task& task, double mission_time);

    // Handle watchdog timeout (safe mode transition)
    void handle_watchdog_timeout();

    // Update cycle statistics
    void update_cycle_stats(double cycle_time_us);

    // -----------------------------------------------------------------------
    // Data members
    // -----------------------------------------------------------------------
    std::vector<Task>  tasks_;           // Registered tasks
    double             base_rate_hz_;    // Fundamental cycle rate
    double             base_period_s_;   // 1 / base_rate_hz_
    double             mission_time_;    // Current mission elapsed time
    std::size_t        cycle_count_;     // Current cycle number
    SystemMode         mode_;            // Current system mode
    Watchdog           watchdog_;        // Watchdog timer
    CycleStats         cycle_stats_;     // Timing statistics
    std::atomic<bool>  running_;         // Flag for run loop
    bool               initialized_;     // Whether initialize() was called
};

#endif // GNC_FLIGHT_COMPUTER_H
