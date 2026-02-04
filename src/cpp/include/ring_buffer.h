// =============================================================================
// ring_buffer.h -- Lock-Free Ring Buffer for Real-Time Telemetry
// =============================================================================
//
// OVERVIEW:
//   A single-producer, single-consumer (SPSC) lock-free ring buffer designed
//   for real-time telemetry streaming between the GNC computation thread and
//   the telemetry downlink thread.
//
// WHY LOCK-FREE?
//
//   In a real-time system, mutexes are dangerous:
//
//   1. PRIORITY INVERSION: A low-priority telemetry thread holding the mutex
//      can block the high-priority GNC thread. Even with priority inheritance,
//      the GNC thread must wait for the telemetry thread's critical section
//      to complete, introducing unbounded latency.
//
//   2. NON-DETERMINISTIC TIMING: Mutex acquisition involves OS kernel calls
//      (futex on Linux, pthread_mutex_lock). The kernel may need to put the
//      calling thread to sleep and wake another -- context switch overhead
//      is typically 1-10 microseconds, but can spike to milliseconds under load.
//
//   3. DEADLOCK RISK: If the telemetry thread crashes while holding the mutex,
//      the GNC thread is permanently blocked. This is a mission-ending failure.
//
//   Lock-free data structures use atomic CPU instructions (compare-and-swap,
//   load-acquire, store-release) instead of OS-level synchronization.
//   These complete in nanoseconds with deterministic timing.
//
// FALSE SHARING:
//
//   When two threads access different variables that share the same cache line,
//   the CPU's cache coherency protocol (MESI/MOESI) forces both cores to
//   invalidate and reload the entire cache line on every write. This is
//   "false sharing" -- the variables are logically independent but physically
//   coupled through the cache line.
//
//   Example of the problem:
//     struct BadBuffer {
//         std::atomic<size_t> head;  // Written by producer
//         std::atomic<size_t> tail;  // Written by consumer
//         // These are on the same cache line! Each write by one thread
//         // invalidates the other thread's cached copy.
//     };
//
//   Solution: Pad head and tail to separate cache lines:
//     alignas(64) std::atomic<size_t> head;  // Own cache line
//     alignas(64) std::atomic<size_t> tail;  // Own cache line
//
//   This eliminates false sharing. Each thread's writes stay in its own
//   cache line without disturbing the other thread.
//
// MEMORY ORDERING (ACQUIRE/RELEASE SEMANTICS):
//
//   Modern CPUs and compilers may reorder memory operations for performance.
//   The x86 memory model is relatively strong (Total Store Order), but ARM
//   and PowerPC have weaker models where reordering is more aggressive.
//
//   C++ std::atomic provides memory ordering guarantees:
//
//   - memory_order_relaxed: No ordering guarantees. Cheapest. Only guarantees
//     atomicity of the operation itself.
//
//   - memory_order_acquire: No reads/writes after this load can be reordered
//     before it. Used by the consumer: "I see the new head value, so I must
//     also see all the data the producer wrote before updating head."
//
//   - memory_order_release: No reads/writes before this store can be reordered
//     after it. Used by the producer: "I've finished writing the data, now
//     I publish the new head value. The consumer will see the data."
//
//   - memory_order_seq_cst: Full sequential consistency. Safest but slowest.
//     We avoid this for performance -- acquire/release is sufficient for SPSC.
//
//   The acquire/release pair creates a "happens-before" relationship:
//     Producer: write data -> store-release(head)
//     Consumer: load-acquire(head) -> read data
//   The consumer is guaranteed to see the data the producer wrote.
//
// CACHE COHERENCY PROTOCOL (MESI):
//
//   M (Modified):  This core has the only valid copy, and it's been written.
//   E (Exclusive): This core has the only valid copy, but it hasn't been written.
//   S (Shared):    Multiple cores have a valid copy. Read-only.
//   I (Invalid):   This cache line is stale and must be fetched from memory/other core.
//
//   In our ring buffer:
//   - The producer writes data slots (M state on producer's core)
//   - The producer writes head_ (M state on producer's core)
//   - The consumer reads head_ -- this forces a cache line transfer from
//     producer's core to consumer's core (transitions to S state)
//   - By keeping head_ and tail_ on separate cache lines, we minimize
//     these inter-core transfers to just the index variables, not the data.
//
// =============================================================================

#ifndef GNC_RING_BUFFER_H
#define GNC_RING_BUFFER_H

#include <atomic>
#include <optional>
#include <cstddef>
#include <cstdint>
#include <array>
#include <type_traits>

// ---------------------------------------------------------------------------
// Compile-time check: Size must be a power of 2 so we can use bitwise AND
// instead of modulo for index wrapping. Modulo requires integer division
// (20-90 cycles on most CPUs). Bitwise AND is 1 cycle.
//
//   index % Size  -->  index & (Size - 1)  (only when Size is power of 2)
//
// This is a classic embedded optimization.
// ---------------------------------------------------------------------------
template <std::size_t N>
constexpr bool is_power_of_two() {
    return N > 0 && (N & (N - 1)) == 0;
}

// ---------------------------------------------------------------------------
// RingBuffer<T, Size>
//
// Template parameters:
//   T    - Element type. Must be trivially copyable for lock-free safety.
//          Non-trivial types would require calling constructors/destructors,
//          which are not atomic operations.
//   Size - Number of slots. MUST be a power of 2 (enforced at compile time).
//          Choose based on expected burst size: if the consumer is slow for
//          N milliseconds and the producer generates M items/ms, you need
//          at least N*M slots plus margin.
// ---------------------------------------------------------------------------
template <typename T, std::size_t Size>
class RingBuffer {
    static_assert(is_power_of_two<Size>(),
                  "RingBuffer size must be a power of 2 for efficient modulo");
    static_assert(std::is_trivially_copyable_v<T>,
                  "RingBuffer elements must be trivially copyable for lock-free safety");

public:
    RingBuffer() : head_(0), tail_(0) {
        // Pre-fault the buffer memory by writing zeros.
        // This ensures all pages are resident in physical RAM.
        buffer_.fill(T{});
    }

    // No copy or move -- the buffer has a fixed location in memory
    // and atomic members aren't copyable
    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;
    RingBuffer(RingBuffer&&) = delete;
    RingBuffer& operator=(RingBuffer&&) = delete;

    // -----------------------------------------------------------------------
    // push() -- Producer-side: add an item to the buffer.
    //
    // Returns true if the item was written, false if the buffer is full.
    // Never blocks -- this is critical for the GNC thread.
    //
    // Memory ordering:
    //   1. We read tail_ with acquire to see the latest consumer position.
    //   2. We write the data into the slot (regular write, will be made
    //      visible by the release store on head_).
    //   3. We update head_ with release, which "publishes" both the new
    //      head value AND the data we wrote in step 2.
    //
    // On x86, the release store is free (x86 has Total Store Order, so all
    // stores are implicitly release). On ARM, it compiles to a DMB (Data
    // Memory Barrier) instruction before the store.
    // -----------------------------------------------------------------------
    bool push(const T& item) noexcept {
        // Load the current head (only we modify it, so relaxed is fine)
        const std::size_t current_head = head_.load(std::memory_order_relaxed);

        // Load the tail to check if buffer is full.
        // We need acquire to see the consumer's latest dequeue.
        const std::size_t current_tail = tail_.load(std::memory_order_acquire);

        // Full condition: head is one slot behind tail (circular)
        // We waste one slot to distinguish full from empty.
        // Alternative: maintain a separate count, but that would be another
        // atomic variable with more cache coherency traffic.
        if (((current_head + 1) & (Size - 1)) == current_tail) {
            return false;  // Buffer full -- drop the item
        }

        // Write the data into the slot
        buffer_[current_head] = item;

        // Publish the new head with release ordering.
        // This ensures the data write above is visible to the consumer
        // before the consumer sees the updated head.
        head_.store((current_head + 1) & (Size - 1), std::memory_order_release);

        return true;
    }

    // -----------------------------------------------------------------------
    // pop() -- Consumer-side: remove an item from the buffer.
    //
    // Returns std::optional<T>: the item if available, std::nullopt if empty.
    // Never blocks.
    //
    // Memory ordering:
    //   1. We read head_ with acquire to see the latest producer position
    //      AND the data the producer wrote before updating head.
    //   2. We read the data from the slot (regular read, guaranteed visible
    //      by the acquire load of head_).
    //   3. We update tail_ with release, telling the producer we've consumed
    //      the slot and it can be reused.
    // -----------------------------------------------------------------------
    std::optional<T> pop() noexcept {
        // Load the current tail (only we modify it, so relaxed is fine)
        const std::size_t current_tail = tail_.load(std::memory_order_relaxed);

        // Load the head to check if buffer has data.
        // We need acquire to see the producer's latest data write.
        const std::size_t current_head = head_.load(std::memory_order_acquire);

        // Empty condition: head == tail
        if (current_head == current_tail) {
            return std::nullopt;  // Buffer empty
        }

        // Read the data from the slot
        T item = buffer_[current_tail];

        // Publish the new tail with release ordering.
        // This tells the producer the slot is now free for reuse.
        tail_.store((current_tail + 1) & (Size - 1), std::memory_order_release);

        return item;
    }

    // -----------------------------------------------------------------------
    // is_empty() / is_full() / size() -- Query methods.
    //
    // NOTE: These are approximate! Between reading head and tail, the other
    // thread may have modified them. These are useful for monitoring and
    // diagnostics, but NEVER use them for synchronization decisions.
    // The push/pop methods handle synchronization internally.
    // -----------------------------------------------------------------------
    bool is_empty() const noexcept {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

    bool is_full() const noexcept {
        const std::size_t h = head_.load(std::memory_order_acquire);
        const std::size_t t = tail_.load(std::memory_order_acquire);
        return ((h + 1) & (Size - 1)) == t;
    }

    std::size_t size() const noexcept {
        const std::size_t h = head_.load(std::memory_order_acquire);
        const std::size_t t = tail_.load(std::memory_order_acquire);
        // Handle wraparound with modular arithmetic
        return (h - t + Size) & (Size - 1);
    }

    static constexpr std::size_t capacity() noexcept {
        return Size - 1;  // One slot is always wasted to distinguish full/empty
    }

    // -----------------------------------------------------------------------
    // reset() -- Clear the buffer. NOT thread-safe!
    // Only call when both producer and consumer are stopped.
    // Used during mission phase transitions.
    // -----------------------------------------------------------------------
    void reset() noexcept {
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
    }

private:
    // -----------------------------------------------------------------------
    // MEMORY LAYOUT -- carefully designed to prevent false sharing.
    //
    // Cache line 0: head_ (written by producer, read by consumer)
    // Cache line 1: tail_ (written by consumer, read by producer)
    // Cache lines 2+: buffer_ data
    //
    // The alignas(CACHE_LINE_SIZE) ensures each starts on its own cache line.
    // The padding arrays explicitly fill the remaining bytes.
    //
    // Without this padding:
    //   - head_ and tail_ would be on the SAME cache line
    //   - Every push() would invalidate the consumer's cache line
    //   - Every pop() would invalidate the producer's cache line
    //   - Result: 10-100x slower than the padded version!
    //
    // This is one of the most important optimizations for multi-core
    // embedded systems. We've seen 50x throughput improvement from this
    // single change in real telemetry systems.
    // -----------------------------------------------------------------------

    // Producer-owned: head index
    alignas(CACHE_LINE_SIZE) std::atomic<std::size_t> head_;
    char pad_head_[CACHE_LINE_SIZE - sizeof(std::atomic<std::size_t>)];

    // Consumer-owned: tail index
    alignas(CACHE_LINE_SIZE) std::atomic<std::size_t> tail_;
    char pad_tail_[CACHE_LINE_SIZE - sizeof(std::atomic<std::size_t>)];

    // Data buffer -- aligned for optimal cache utilization
    alignas(CACHE_LINE_SIZE) std::array<T, Size> buffer_;
};

// ============================================================================
// Helper types for telemetry
// ============================================================================

// A timestamped telemetry sample for the ring buffer
struct TelemetrySample {
    double    timestamp;       // Mission elapsed time (seconds)
    double    position[3];     // Position in ECI frame (m)
    double    velocity[3];     // Velocity in ECI frame (m/s)
    double    quaternion[4];   // Attitude quaternion (scalar last)
    double    angular_rate[3]; // Body angular rate (rad/s)
    uint32_t  sequence_num;    // Monotonic sequence counter for drop detection
    uint32_t  status_flags;    // Bit flags for system health
};

// Common ring buffer types
using TelemetryBuffer = RingBuffer<TelemetrySample, 1024>;   // ~1 second at 1kHz

// Performance measurement result
struct RingBufferPerfResult {
    double lock_free_throughput_mops;   // Million operations per second (lock-free)
    double mutex_throughput_mops;       // Million operations per second (mutex-based)
    double speedup_factor;             // lock_free / mutex
    std::size_t items_transferred;     // Total items transferred
    double elapsed_seconds;            // Wall clock time
};

// Benchmark function declaration (implemented in ring_buffer.cpp)
RingBufferPerfResult benchmark_ring_buffer(std::size_t num_items);

#endif // GNC_RING_BUFFER_H
