// =============================================================================
// flight_computer.cpp -- Flight Software Architecture Implementation
// =============================================================================
//
// This file implements the FlightComputer class: a deterministic real-time
// task scheduler inspired by aerospace flight software architectures.
//
// KEY CONCEPTS IMPLEMENTED HERE:
//
//   1. RATE-MONOTONIC SCHEDULING (RMS):
//      Tasks with higher execution rates receive higher priority (lower
//      priority number). This is provably optimal for fixed-priority
//      preemptive scheduling. The schedulability test ensures that the
//      total CPU utilization does not exceed the Liu & Layland bound:
//        U = sum(Ci/Ti) <= n * (2^(1/n) - 1)
//      For 3 tasks: U <= 0.780; for 4 tasks: U <= 0.757.
//
//   2. TRIPLE MODULAR REDUNDANCY (TMR):
//      Critical computations are executed three times independently.
//      The median value (or majority vote for discrete values) is selected.
//      This protects against Single Event Upsets (SEUs) caused by cosmic
//      rays flipping bits in memory or register files.
//
//      In LEO, a spacecraft experiences ~1-10 SEUs per day depending on
//      orbit inclination, solar cycle, and shielding. Without TMR, each
//      SEU could produce a corrupted control command.
//
//   3. WATCHDOG TIMER:
//      A deadline monitor that detects when a task exceeds its allocated
//      time budget. In real flight software, watchdog timeouts trigger
//      progressively severe recovery actions:
//        - Level 1: Log warning, re-execute task
//        - Level 2: Reset the offending task
//        - Level 3: Transition to safe mode
//        - Level 4: Hardware watchdog resets the entire processor
//
//   4. DETERMINISTIC EXECUTION:
//      The scheduler runs in constant time regardless of task outcomes.
//      All memory is pre-allocated. No heap operations occur in the hot path.
//      Task callbacks must also be deterministic (bounded loops, no I/O).
//
// =============================================================================

#include "flight_computer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <sstream>
#include <thread>

// ---------------------------------------------------------------------------
// Constructor
//
// Parameters:
//   base_rate_hz: The fundamental scheduling rate. The main loop ticks at
//                 this rate, and individual tasks execute at sub-multiples.
//                 Example: base = 100 Hz, tasks at 1/10/50/100 Hz.
//
// Memory allocation strategy:
//   We reserve vector capacity upfront to avoid reallocation during
//   add_task() calls. The vector will not grow beyond MAX_TASKS in a
//   real system (typically 8-32 tasks).
// ---------------------------------------------------------------------------
FlightComputer::FlightComputer(double base_rate_hz)
    : base_rate_hz_(base_rate_hz),
      base_period_s_(1.0 / base_rate_hz),
      mission_time_(0.0),
      cycle_count_(0),
      mode_(SystemMode::BOOT),
      running_(false),
      initialized_(false)
{
    // Pre-allocate task vector to avoid heap allocations during operation.
    // 32 tasks is more than enough for any single flight computer partition.
    // On the ISS, the GNC partition runs about 15 tasks.
    tasks_.reserve(32);

    // Initialize watchdog with a generous timeout for startup.
    // The timeout will be tightened after initialization based on the
    // actual worst-case cycle time.
    watchdog_.timeout_ms = 2.0 * base_period_s_ * 1000.0;  // 2x the cycle period
    watchdog_.armed = false;
    watchdog_.timeout_count = 0;
    watchdog_.last_kick = std::chrono::steady_clock::now();

    // Zero out cycle statistics
    cycle_stats_.min_cycle_us = 1e9;   // Will be updated to actual minimum
    cycle_stats_.max_cycle_us = 0.0;
    cycle_stats_.avg_cycle_us = 0.0;
    cycle_stats_.jitter_us = 0.0;
    cycle_stats_.target_cycle_us = base_period_s_ * 1.0e6;
    cycle_stats_.cpu_utilization = 0.0;
    cycle_stats_.total_cycles = 0;
    cycle_stats_.overruns = 0;
}

// ---------------------------------------------------------------------------
// Destructor
//
// Ensures clean shutdown: disables all tasks, stops the running flag,
// and transitions to SHUTDOWN mode. In a real system, this would also
// send a final telemetry packet indicating controlled shutdown.
// ---------------------------------------------------------------------------
FlightComputer::~FlightComputer() {
    if (running_) {
        shutdown();
    }
}

// ---------------------------------------------------------------------------
// add_task()
//
// Registers a new periodic task with the scheduler.
//
// The task is configured with:
//   - A name for logging and telemetry identification
//   - A priority level (lower number = higher priority, per RMS convention)
//   - An execution rate in Hz
//   - A callback function that performs the actual work
//   - A critical flag that enables TMR protection
//
// RATE VALIDATION:
//   The task rate should ideally be a divisor of base_rate_hz to ensure
//   clean scheduling. For example, with base = 100 Hz, valid task rates
//   are 1, 2, 5, 10, 20, 25, 50, 100 Hz. Non-divisor rates work but
//   may exhibit slight jitter due to rounding.
//
// Returns the task index (0-based), or -1 on failure.
// ---------------------------------------------------------------------------
int FlightComputer::add_task(
    const std::string& name,
    int priority,
    double rate_hz,
    TaskCallback callback,
    bool critical)
{
    if (initialized_) {
        std::fprintf(stderr, "[FLIGHT_COMPUTER] ERROR: Cannot add tasks after initialization\n");
        return -1;
    }

    if (rate_hz <= 0.0 || rate_hz > base_rate_hz_) {
        std::fprintf(stderr, "[FLIGHT_COMPUTER] ERROR: Task '%s' rate %.1f Hz invalid "
                     "(must be 0 < rate <= %.1f Hz)\n",
                     name.c_str(), rate_hz, base_rate_hz_);
        return -1;
    }

    // Create and configure the task structure
    Task task;
    task.name           = name;
    task.priority       = priority;
    task.rate_hz        = rate_hz;
    task.period_s       = 1.0 / rate_hz;
    task.last_exec_time = -task.period_s;  // Ensure first execution at t=0
    task.wcet_us        = 0.0;
    task.avg_exec_us    = 0.0;
    task.total_exec_us  = 0.0;
    task.exec_count     = 0;
    task.overrun_count  = 0;
    task.callback       = std::move(callback);
    task.enabled        = true;
    task.is_critical    = critical;

    tasks_.push_back(std::move(task));

    int idx = static_cast<int>(tasks_.size()) - 1;
    std::printf("[FLIGHT_COMPUTER] Task '%s' registered: priority=%d, rate=%.0f Hz, "
                "period=%.3f ms, critical=%s (index=%d)\n",
                name.c_str(), priority, rate_hz, task.period_s * 1000.0,
                critical ? "YES" : "no", idx);

    return idx;
}

// ---------------------------------------------------------------------------
// initialize()
//
// Prepares the scheduler for real-time operation:
//   1. Sorts tasks by priority (highest priority = lowest number first)
//   2. Validates RMS schedulability
//   3. Resets all statistics
//   4. Arms the watchdog timer
//   5. Transitions to NOMINAL mode
//
// This must be called after all tasks are added but before run_cycle().
// ---------------------------------------------------------------------------
bool FlightComputer::initialize() {
    std::printf("\n[FLIGHT_COMPUTER] ===== Initializing Flight Computer =====\n");
    std::printf("[FLIGHT_COMPUTER] Base rate: %.1f Hz (period: %.3f ms)\n",
                base_rate_hz_, base_period_s_ * 1000.0);
    std::printf("[FLIGHT_COMPUTER] Tasks registered: %zu\n", tasks_.size());

    if (tasks_.empty()) {
        std::fprintf(stderr, "[FLIGHT_COMPUTER] ERROR: No tasks registered\n");
        return false;
    }

    // Step 1: Sort tasks by priority (lower number = higher priority = executes first)
    // This ensures that in each cycle, high-priority tasks run before low-priority ones.
    sort_tasks_by_priority();

    std::printf("[FLIGHT_COMPUTER] Task execution order (by priority):\n");
    for (std::size_t i = 0; i < tasks_.size(); ++i) {
        std::printf("  [%zu] %-20s  priority=%d  rate=%6.1f Hz  period=%8.3f ms  critical=%s\n",
                    i, tasks_[i].name.c_str(), tasks_[i].priority,
                    tasks_[i].rate_hz, tasks_[i].period_s * 1000.0,
                    tasks_[i].is_critical ? "YES" : "no");
    }

    // Step 2: Check RMS schedulability
    bool schedulable = check_schedulability();
    if (!schedulable) {
        std::fprintf(stderr, "[FLIGHT_COMPUTER] WARNING: Task set may not be RMS-schedulable!\n");
        std::fprintf(stderr, "  CPU utilization exceeds the Liu & Layland bound.\n");
        std::fprintf(stderr, "  Tasks may miss deadlines under worst-case conditions.\n");
        // Continue anyway -- the bound is sufficient but not necessary.
        // Many practical task sets are schedulable beyond the bound.
    }

    // Step 3: Reset statistics
    mission_time_ = 0.0;
    cycle_count_ = 0;
    cycle_stats_.min_cycle_us = 1e9;
    cycle_stats_.max_cycle_us = 0.0;
    cycle_stats_.avg_cycle_us = 0.0;
    cycle_stats_.total_cycles = 0;
    cycle_stats_.overruns = 0;

    for (auto& task : tasks_) {
        task.wcet_us = 0.0;
        task.avg_exec_us = 0.0;
        task.total_exec_us = 0.0;
        task.exec_count = 0;
        task.overrun_count = 0;
        task.last_exec_time = -task.period_s;
    }

    // Step 4: Arm the watchdog
    watchdog_.armed = true;
    watchdog_.kick();
    std::printf("[FLIGHT_COMPUTER] Watchdog armed (timeout: %.1f ms)\n",
                watchdog_.timeout_ms);

    // Step 5: Transition to NOMINAL mode
    mode_ = SystemMode::NOMINAL;
    initialized_ = true;
    running_ = true;

    std::printf("[FLIGHT_COMPUTER] Initialization complete. Mode: NOMINAL\n");
    std::printf("[FLIGHT_COMPUTER] =========================================\n\n");

    return true;
}

// ---------------------------------------------------------------------------
// sort_tasks_by_priority()
//
// Sorts the task vector by priority (ascending = highest priority first).
// Uses std::sort which is O(n log n), called once during initialization.
// During the real-time loop, the order is fixed (no sorting needed).
// ---------------------------------------------------------------------------
void FlightComputer::sort_tasks_by_priority() {
    std::sort(tasks_.begin(), tasks_.end(),
              [](const Task& a, const Task& b) {
                  return a.priority < b.priority;
              });
}

// ---------------------------------------------------------------------------
// is_task_due()
//
// Determines whether a task should execute in the current cycle.
//
// A task is due when the elapsed time since its last execution exceeds
// its period: (current_time - last_exec_time) >= period_s
//
// We subtract a small epsilon (1% of the period) to avoid floating-point
// comparison issues. Without this, rounding errors can cause a task to
// skip a cycle, then execute twice in the next cycle.
// ---------------------------------------------------------------------------
bool FlightComputer::is_task_due(const Task& task, double current_time) const {
    if (!task.enabled) return false;

    double elapsed = current_time - task.last_exec_time;
    double threshold = task.period_s * 0.99;  // 1% tolerance for timing jitter
    return elapsed >= threshold;
}

// ---------------------------------------------------------------------------
// execute_task()
//
// Executes a single task with timing measurement.
//
// Steps:
//   1. Record start time using steady_clock (monotonic, ~1 ns resolution)
//   2. Call the task callback
//   3. Record end time
//   4. Compute execution time and update statistics
//   5. Check for timing overrun (> 2x expected execution time or > 50% of period)
//
// WATCHDOG INTEGRATION:
//   If a task exceeds 2x its worst-case execution time (WCET), we log a
//   warning. If it exceeds 90% of the task's period, it's a critical
//   overrun that could cause the next execution to be missed.
//
// Returns the task callback's return code (0 = success).
// ---------------------------------------------------------------------------
int FlightComputer::execute_task(Task& task, double mission_time) {
    auto start = std::chrono::steady_clock::now();

    // Execute the task callback
    int result = task.callback(mission_time);

    auto end = std::chrono::steady_clock::now();
    double exec_us = std::chrono::duration<double, std::micro>(end - start).count();

    // Update task statistics
    task.exec_count++;
    task.total_exec_us += exec_us;
    task.avg_exec_us = task.total_exec_us / static_cast<double>(task.exec_count);

    // Update worst-case execution time (WCET)
    // WCET is the single most important metric in real-time systems.
    // It determines whether the schedule is feasible.
    if (exec_us > task.wcet_us) {
        task.wcet_us = exec_us;
    }

    // Check for timing overrun.
    // We define an overrun as exceeding 50% of the task's period.
    // This is a conservative threshold -- in production, it would be
    // tuned based on the specific task set and margins.
    double period_us = task.period_s * 1.0e6;
    double overrun_threshold = period_us * 0.5;

    if (exec_us > overrun_threshold) {
        task.overrun_count++;
        std::fprintf(stderr, "[WATCHDOG] Task '%s' OVERRUN: %.1f us (threshold: %.1f us, "
                     "period: %.1f us) -- overrun #%zu\n",
                     task.name.c_str(), exec_us, overrun_threshold, period_us,
                     task.overrun_count);
    }

    // Update last execution time
    task.last_exec_time = mission_time;

    return result;
}

// ---------------------------------------------------------------------------
// execute_task_tmr()
//
// Executes a critical task with Triple Modular Redundancy.
//
// The task callback is executed three times. If all three return the same
// result code, we proceed normally. If one disagrees, we log the discrepancy
// and use the majority vote.
//
// For floating-point outputs (e.g., control commands), the TMR voting is
// handled separately via tmr_vote() and tmr_vote_array().
//
// TMR COST: 3x computation time, but this is acceptable for critical tasks
// that run at low rates (e.g., guidance at 1 Hz uses only 3 ms out of
// 1000 ms available per period).
// ---------------------------------------------------------------------------
int FlightComputer::execute_task_tmr(Task& task, double mission_time) {
    auto start = std::chrono::steady_clock::now();

    // Execute three independent copies of the task
    int result1 = task.callback(mission_time);
    int result2 = task.callback(mission_time);
    int result3 = task.callback(mission_time);

    auto end = std::chrono::steady_clock::now();
    double exec_us = std::chrono::duration<double, std::micro>(end - start).count();

    // Majority vote on the return codes.
    // If at least two agree, use that value. Otherwise, flag an error.
    int voted_result;
    if (result1 == result2 || result1 == result3) {
        voted_result = result1;
    } else if (result2 == result3) {
        voted_result = result2;
    } else {
        // All three disagree -- this is a severe anomaly.
        // In flight, this would trigger a safe mode transition.
        std::fprintf(stderr, "[TMR] CRITICAL: All three copies of '%s' disagree: "
                     "%d, %d, %d\n", task.name.c_str(), result1, result2, result3);
        voted_result = result1;  // Default to first copy
    }

    // Check if any copy disagreed (indicates a potential SEU)
    if (result1 != result2 || result2 != result3) {
        std::fprintf(stderr, "[TMR] Task '%s' disagreement detected: [%d, %d, %d] -> voted: %d\n",
                     task.name.c_str(), result1, result2, result3, voted_result);
    }

    // Update statistics (TMR execution time includes all three runs)
    task.exec_count++;
    task.total_exec_us += exec_us;
    task.avg_exec_us = task.total_exec_us / static_cast<double>(task.exec_count);
    if (exec_us > task.wcet_us) {
        task.wcet_us = exec_us;
    }

    task.last_exec_time = mission_time;
    return voted_result;
}

// ---------------------------------------------------------------------------
// run_cycle()
//
// Executes one iteration of the main scheduling loop.
//
// This is the heart of the flight software. The caller is responsible for
// invoking this at the base rate (e.g., 100 Hz). Each cycle:
//
//   1. Records the cycle start time for deterministic timing measurement
//   2. Kicks the watchdog to prevent timeout
//   3. Iterates through tasks in priority order
//   4. For each task, checks if it is due (elapsed >= period)
//   5. Executes due tasks (with TMR for critical tasks)
//   6. Records cycle end time and computes statistics
//
// The cycle must complete within base_period_s_ seconds. If it does not,
// it is counted as an overrun and may trigger recovery actions.
// ---------------------------------------------------------------------------
void FlightComputer::run_cycle() {
    if (!initialized_ || !running_) return;

    auto cycle_start = std::chrono::steady_clock::now();

    // Kick the watchdog at the start of each cycle.
    // If the previous cycle caused a hang, the watchdog would have fired
    // between cycles (handled by the timing loop in run_for()).
    watchdog_.kick();

    // Check watchdog health -- has the timer expired since last kick?
    if (!watchdog_.check()) {
        handle_watchdog_timeout();
    }

    // Execute all due tasks in priority order.
    // High-priority tasks (lower priority number) execute first.
    for (auto& task : tasks_) {
        if (!task.enabled) continue;

        if (is_task_due(task, mission_time_)) {
            if (task.is_critical) {
                execute_task_tmr(task, mission_time_);
            } else {
                execute_task(task, mission_time_);
            }
        }
    }

    // Advance mission time by one base period
    mission_time_ += base_period_s_;
    cycle_count_++;

    // Compute cycle timing statistics
    auto cycle_end = std::chrono::steady_clock::now();
    double cycle_us = std::chrono::duration<double, std::micro>(cycle_end - cycle_start).count();

    update_cycle_stats(cycle_us);
}

// ---------------------------------------------------------------------------
// run_for()
//
// Convenience method that runs the scheduling loop for a specified duration.
//
// In realtime mode, it sleeps between cycles using std::this_thread::sleep_until
// with steady_clock for monotonic timing. This achieves ~0.1 ms jitter on
// Linux with SCHED_FIFO, or ~1 ms jitter on standard scheduling.
//
// In simulation mode (realtime=false), it runs as fast as possible --
// useful for Monte Carlo analysis and testing.
// ---------------------------------------------------------------------------
void FlightComputer::run_for(double duration_s, bool realtime) {
    if (!initialized_) {
        std::fprintf(stderr, "[FLIGHT_COMPUTER] ERROR: Must call initialize() before run_for()\n");
        return;
    }

    std::size_t total_cycles = static_cast<std::size_t>(duration_s * base_rate_hz_);

    std::printf("[FLIGHT_COMPUTER] Running for %.2f s (%zu cycles at %.0f Hz)%s\n",
                duration_s, total_cycles, base_rate_hz_,
                realtime ? " [REAL-TIME]" : " [FAST-SIM]");

    auto loop_start = std::chrono::steady_clock::now();
    auto next_cycle = loop_start;

    for (std::size_t i = 0; i < total_cycles && running_; ++i) {
        // Execute one scheduling cycle
        run_cycle();

        if (realtime) {
            // Sleep until the next cycle boundary.
            // sleep_until is more accurate than sleep_for because it accounts
            // for the time spent in run_cycle(). With sleep_for, the cycle
            // time would be: execution_time + sleep_time > desired_period.
            next_cycle += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(base_period_s_)
            );
            std::this_thread::sleep_until(next_cycle);
        }
    }

    auto loop_end = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(loop_end - loop_start).count();

    std::printf("[FLIGHT_COMPUTER] Completed %zu cycles in %.3f s (effective rate: %.1f Hz)\n",
                cycle_stats_.total_cycles, elapsed_s,
                static_cast<double>(cycle_stats_.total_cycles) / elapsed_s);
}

// ---------------------------------------------------------------------------
// shutdown()
//
// Graceful shutdown sequence:
//   1. Set running flag to false (stops run_for loop)
//   2. Disable all tasks
//   3. Disarm watchdog
//   4. Print final statistics
//   5. Transition to SHUTDOWN mode
// ---------------------------------------------------------------------------
void FlightComputer::shutdown() {
    std::printf("\n[FLIGHT_COMPUTER] ===== Shutdown Sequence =====\n");

    running_ = false;
    mode_ = SystemMode::SHUTDOWN;
    watchdog_.armed = false;

    for (auto& task : tasks_) {
        task.enabled = false;
    }

    // Print final report
    std::printf("%s", timing_report().c_str());
    std::printf("[FLIGHT_COMPUTER] Shutdown complete. Mode: SHUTDOWN\n");
    std::printf("[FLIGHT_COMPUTER] ==============================\n\n");
}

// ---------------------------------------------------------------------------
// tmr_vote()
//
// Execute a scalar computation three times and return the voted result.
//
// Voting strategy: MEDIAN value.
//   - If all three agree (within tolerance), return the average
//   - If two agree and one disagrees, return the average of the two
//   - If all three disagree, return the median (middle value)
//
// The median is the optimal choice for single-fault tolerance: if one
// value is corrupted, the median ignores it regardless of the corruption
// magnitude. The mean, by contrast, is sensitive to outliers.
// ---------------------------------------------------------------------------
TMRResult FlightComputer::tmr_vote(
    std::function<double(double)> computation,
    double mission_time,
    double tolerance)
{
    // Execute the computation three times
    double v1 = computation(mission_time);
    double v2 = computation(mission_time);
    double v3 = computation(mission_time);

    TMRResult result;

    // Check pairwise agreement
    bool agree_12 = std::fabs(v1 - v2) <= tolerance;
    bool agree_23 = std::fabs(v2 - v3) <= tolerance;
    bool agree_13 = std::fabs(v1 - v3) <= tolerance;

    // Compute maximum deviation for diagnostics
    result.max_deviation = std::max({std::fabs(v1 - v2),
                                     std::fabs(v2 - v3),
                                     std::fabs(v1 - v3)});

    if (agree_12 && agree_23 && agree_13) {
        // All three agree -- return the average for best accuracy
        result.value = (v1 + v2 + v3) / 3.0;
        result.agreement = true;
        result.disagreeing_copy = -1;
    } else if (agree_12) {
        // Copies 1 and 2 agree, copy 3 disagrees
        result.value = (v1 + v2) / 2.0;
        result.agreement = false;
        result.disagreeing_copy = 2;  // 0-indexed: copy 3 = index 2
    } else if (agree_23) {
        // Copies 2 and 3 agree, copy 1 disagrees
        result.value = (v2 + v3) / 2.0;
        result.agreement = false;
        result.disagreeing_copy = 0;  // Copy 1 = index 0
    } else if (agree_13) {
        // Copies 1 and 3 agree, copy 2 disagrees
        result.value = (v1 + v3) / 2.0;
        result.agreement = false;
        result.disagreeing_copy = 1;  // Copy 2 = index 1
    } else {
        // All three disagree -- return the median.
        // Sort the three values; the median is the middle one.
        double vals[3] = {v1, v2, v3};
        if (vals[0] > vals[1]) std::swap(vals[0], vals[1]);
        if (vals[1] > vals[2]) std::swap(vals[1], vals[2]);
        if (vals[0] > vals[1]) std::swap(vals[0], vals[1]);

        result.value = vals[1];  // Median
        result.agreement = false;
        result.disagreeing_copy = -1;  // Can't identify the single faulty copy
    }

    return result;
}

// ---------------------------------------------------------------------------
// tmr_vote_array()
//
// TMR voting for an array of values (e.g., a 3D force vector).
// Each element is voted independently.
// ---------------------------------------------------------------------------
void FlightComputer::tmr_vote_array(
    std::function<void(double, double*, std::size_t)> computation,
    double mission_time,
    double* result_out,
    std::size_t n,
    double tolerance)
{
    // Allocate temporary arrays on the stack (bounded size, no heap).
    // In a real system, n would be bounded by a compile-time constant.
    // For safety, we cap at 64 elements (512 bytes per copy = 1.5 KB total).
    constexpr std::size_t MAX_N = 64;
    double v1[MAX_N], v2[MAX_N], v3[MAX_N];

    if (n > MAX_N) n = MAX_N;

    // Execute three times
    computation(mission_time, v1, n);
    computation(mission_time, v2, n);
    computation(mission_time, v3, n);

    // Vote each element independently
    for (std::size_t i = 0; i < n; ++i) {
        bool agree_12 = std::fabs(v1[i] - v2[i]) <= tolerance;
        bool agree_23 = std::fabs(v2[i] - v3[i]) <= tolerance;
        bool agree_13 = std::fabs(v1[i] - v3[i]) <= tolerance;

        if (agree_12 && agree_23) {
            result_out[i] = (v1[i] + v2[i] + v3[i]) / 3.0;
        } else if (agree_12) {
            result_out[i] = (v1[i] + v2[i]) / 2.0;
        } else if (agree_23) {
            result_out[i] = (v2[i] + v3[i]) / 2.0;
        } else if (agree_13) {
            result_out[i] = (v1[i] + v3[i]) / 2.0;
        } else {
            // Median vote
            double vals[3] = {v1[i], v2[i], v3[i]};
            if (vals[0] > vals[1]) std::swap(vals[0], vals[1]);
            if (vals[1] > vals[2]) std::swap(vals[1], vals[2]);
            if (vals[0] > vals[1]) std::swap(vals[0], vals[1]);
            result_out[i] = vals[1];
        }
    }
}

// ---------------------------------------------------------------------------
// handle_watchdog_timeout()
//
// Called when the watchdog timer expires (system is unresponsive).
//
// Recovery strategy:
//   1. Log the event (critical for post-flight analysis)
//   2. Increment the timeout counter
//   3. If timeouts exceed threshold, transition to SAFE_MODE
//
// In SAFE_MODE, only essential tasks execute (attitude control and
// telemetry). Guidance and payload tasks are suspended. This reduces
// CPU load and prevents cascading failures.
// ---------------------------------------------------------------------------
void FlightComputer::handle_watchdog_timeout() {
    watchdog_.timeout_count++;
    std::fprintf(stderr, "[WATCHDOG] TIMEOUT #%zu detected at mission time %.3f s\n",
                 watchdog_.timeout_count, mission_time_);

    // After 3 consecutive timeouts, enter safe mode
    if (watchdog_.timeout_count >= 3 && mode_ == SystemMode::NOMINAL) {
        std::fprintf(stderr, "[WATCHDOG] Transitioning to SAFE_MODE after %zu timeouts\n",
                     watchdog_.timeout_count);
        mode_ = SystemMode::SAFE_MODE;

        // In safe mode, disable non-essential tasks (keep only high-priority ones)
        for (auto& task : tasks_) {
            if (task.priority > 5) {  // Low-priority tasks (higher number)
                task.enabled = false;
                std::fprintf(stderr, "[WATCHDOG] Disabled non-essential task: %s\n",
                             task.name.c_str());
            }
        }
    }

    // Reset the watchdog
    watchdog_.kick();
}

// ---------------------------------------------------------------------------
// update_cycle_stats()
//
// Updates running statistics for cycle timing.
//
// We use Welford's online algorithm for computing the running average,
// which is numerically stable (no catastrophic cancellation).
//
// JITTER MEASUREMENT:
//   Jitter = max_cycle - min_cycle. This measures the variability in
//   cycle timing, which is the primary metric for determinism.
//   A perfectly deterministic system has zero jitter.
//
//   Typical jitter values:
//     - Bare metal RTOS: 1-10 us
//     - Linux with PREEMPT_RT: 10-100 us
//     - Standard Linux: 100 us - 10 ms
//     - This simulator: depends on OS scheduler (non-deterministic)
// ---------------------------------------------------------------------------
void FlightComputer::update_cycle_stats(double cycle_time_us) {
    cycle_stats_.total_cycles++;

    // Update min/max
    if (cycle_time_us < cycle_stats_.min_cycle_us) {
        cycle_stats_.min_cycle_us = cycle_time_us;
    }
    if (cycle_time_us > cycle_stats_.max_cycle_us) {
        cycle_stats_.max_cycle_us = cycle_time_us;
    }

    // Welford's online algorithm for running average.
    // avg_new = avg_old + (new_val - avg_old) / n
    // This avoids accumulating a sum that could overflow or lose precision.
    cycle_stats_.avg_cycle_us += (cycle_time_us - cycle_stats_.avg_cycle_us)
                                  / static_cast<double>(cycle_stats_.total_cycles);

    // Jitter = peak-to-peak variation
    cycle_stats_.jitter_us = cycle_stats_.max_cycle_us - cycle_stats_.min_cycle_us;

    // CPU utilization = actual_time / available_time
    cycle_stats_.cpu_utilization = cycle_stats_.avg_cycle_us / cycle_stats_.target_cycle_us;

    // Check for overrun (cycle exceeded the target period)
    if (cycle_time_us > cycle_stats_.target_cycle_us) {
        cycle_stats_.overruns++;
    }
}

// ---------------------------------------------------------------------------
// check_schedulability()
//
// Verifies that the current task set is RMS-schedulable using the
// Liu & Layland utilization bound.
//
// The bound is a SUFFICIENT condition (not necessary):
//   - If U <= bound, the task set is guaranteed schedulable.
//   - If U > bound, the task set MAY still be schedulable (need exact
//     analysis using response-time analysis or simulation).
//
// The bound for n tasks: U_bound = n * (2^(1/n) - 1)
// As n approaches infinity: U_bound approaches ln(2) = 0.693
//
// For practical task sets:
//   n=1: U <= 1.000 (100%)
//   n=2: U <= 0.828 (82.8%)
//   n=3: U <= 0.780 (78.0%)
//   n=4: U <= 0.757 (75.7%)
//   n=5: U <= 0.743 (74.3%)
// ---------------------------------------------------------------------------
bool FlightComputer::check_schedulability() const {
    if (tasks_.empty()) return true;

    // We use estimated WCET for initial schedulability check.
    // Since we haven't run yet, we estimate WCET as 100 us per task
    // (a conservative estimate for simple GNC computations).
    constexpr double estimated_wcet_s = 100.0e-6;  // 100 microseconds

    double utilization = 0.0;
    for (const auto& task : tasks_) {
        double wcet = (task.wcet_us > 0.0) ? (task.wcet_us * 1.0e-6) : estimated_wcet_s;
        utilization += wcet / task.period_s;
    }

    // Liu & Layland bound
    std::size_t n = tasks_.size();
    double bound = static_cast<double>(n) * (std::pow(2.0, 1.0 / static_cast<double>(n)) - 1.0);

    std::printf("[FLIGHT_COMPUTER] RMS Schedulability Analysis:\n");
    std::printf("  CPU utilization: %.1f%%\n", utilization * 100.0);
    std::printf("  Liu & Layland bound (n=%zu): %.1f%%\n", n, bound * 100.0);
    std::printf("  Schedulable: %s\n", (utilization <= bound) ? "YES" : "UNCERTAIN (exceeds bound)");

    return utilization <= bound;
}

// ---------------------------------------------------------------------------
// estimated_cpu_utilization()
//
// Computes the actual CPU utilization based on measured WCETs.
// This is more accurate than the pre-run estimate in check_schedulability().
// ---------------------------------------------------------------------------
double FlightComputer::estimated_cpu_utilization() const {
    double utilization = 0.0;
    for (const auto& task : tasks_) {
        if (task.wcet_us > 0.0 && task.exec_count > 0) {
            utilization += (task.avg_exec_us * 1.0e-6) / task.period_s;
        }
    }
    return utilization;
}

// ---------------------------------------------------------------------------
// timing_report()
//
// Generates a comprehensive timing report suitable for post-flight analysis.
//
// This report format is inspired by actual flight software telemetry pages
// used by operators in Mission Control. Each task's execution statistics
// are reported, along with overall cycle timing.
// ---------------------------------------------------------------------------
std::string FlightComputer::timing_report() const {
    std::ostringstream oss;

    oss << "\n";
    oss << "=================================================================\n";
    oss << "  FLIGHT COMPUTER TIMING REPORT\n";
    oss << "=================================================================\n";
    oss << "  Base rate:        " << base_rate_hz_ << " Hz\n";
    oss << "  Mission time:     " << mission_time_ << " s\n";
    oss << "  Total cycles:     " << cycle_stats_.total_cycles << "\n";
    oss << "  Cycle overruns:   " << cycle_stats_.overruns << "\n";
    oss << "  Watchdog timeouts:" << watchdog_.timeout_count << "\n";
    oss << "  System mode:      ";
    switch (mode_) {
        case SystemMode::BOOT:           oss << "BOOT\n"; break;
        case SystemMode::INITIALIZATION: oss << "INITIALIZATION\n"; break;
        case SystemMode::NOMINAL:        oss << "NOMINAL\n"; break;
        case SystemMode::SAFE_MODE:      oss << "SAFE_MODE\n"; break;
        case SystemMode::SHUTDOWN:       oss << "SHUTDOWN\n"; break;
    }
    oss << "\n";

    // Cycle timing
    oss << "  CYCLE TIMING:\n";
    char buf[256];
    std::snprintf(buf, sizeof(buf), "    Min cycle time:  %10.2f us\n", cycle_stats_.min_cycle_us);
    oss << buf;
    std::snprintf(buf, sizeof(buf), "    Max cycle time:  %10.2f us\n", cycle_stats_.max_cycle_us);
    oss << buf;
    std::snprintf(buf, sizeof(buf), "    Avg cycle time:  %10.2f us\n", cycle_stats_.avg_cycle_us);
    oss << buf;
    std::snprintf(buf, sizeof(buf), "    Jitter (p-p):    %10.2f us\n", cycle_stats_.jitter_us);
    oss << buf;
    std::snprintf(buf, sizeof(buf), "    Target cycle:    %10.2f us\n", cycle_stats_.target_cycle_us);
    oss << buf;
    std::snprintf(buf, sizeof(buf), "    CPU utilization:  %8.2f%%\n", cycle_stats_.cpu_utilization * 100.0);
    oss << buf;
    oss << "\n";

    // Per-task statistics
    oss << "  TASK STATISTICS:\n";
    oss << "  ---------------------------------------------------------------\n";
    std::snprintf(buf, sizeof(buf), "  %-18s %6s %8s %10s %10s %10s %6s\n",
                  "Task", "Rate", "Count", "WCET(us)", "Avg(us)", "Total(us)", "Ovrn");
    oss << buf;
    oss << "  ---------------------------------------------------------------\n";
    for (const auto& task : tasks_) {
        std::snprintf(buf, sizeof(buf), "  %-18s %5.0fHz %8zu %10.1f %10.1f %10.0f %6zu\n",
                      task.name.c_str(), task.rate_hz, task.exec_count,
                      task.wcet_us, task.avg_exec_us, task.total_exec_us,
                      task.overrun_count);
        oss << buf;
    }
    oss << "  ---------------------------------------------------------------\n";
    std::snprintf(buf, sizeof(buf), "  Measured CPU utilization: %.2f%%\n",
                  estimated_cpu_utilization() * 100.0);
    oss << buf;
    oss << "=================================================================\n\n";

    return oss.str();
}
