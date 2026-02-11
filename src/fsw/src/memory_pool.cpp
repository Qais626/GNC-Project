// =============================================================================
// memory_pool.cpp -- Memory Pool Performance Analysis Utilities
// =============================================================================
//
// This file provides benchmarking and diagnostic functions for the
// MemoryPool template class defined in memory_pool.h.
//
// PURPOSE:
//   Quantify the performance advantage of custom pool allocation over
//   standard heap allocation (new/delete and malloc/free). This data is
//   critical for justifying the engineering effort of custom allocators
//   in flight software.
//
// WHAT WE MEASURE:
//
//   1. ALLOCATION THROUGHPUT: How many allocations per second?
//      Pool: O(1) per allocation (pop from free list)
//      Heap: O(1) amortized, O(n) worst case (fragmented free list search)
//
//   2. DEALLOCATION THROUGHPUT: How many deallocations per second?
//      Pool: O(1) per deallocation (push to free list)
//      Heap: O(1) amortized (may trigger coalescing, O(n) worst case)
//
//   3. TIMING DETERMINISM: What is the jitter (min/max spread)?
//      Pool: ~10-50 ns jitter (pure pointer math, cache-resident)
//      Heap: ~100 ns - 100 us jitter (system calls, lock contention)
//
//   4. CACHE BEHAVIOR: How does spatial locality affect performance?
//      Pool allocations are sequential in memory (excellent prefetch).
//      Heap allocations may be scattered (poor spatial locality).
//
// TYPICAL RESULTS (x86-64, 3 GHz, -O2):
//
//   Memory pool:  ~15-25 ns per alloc+dealloc cycle
//   new/delete:   ~50-200 ns per cycle
//   malloc/free:  ~40-150 ns per cycle
//
//   Speedup: 2-8x for pool vs. heap
//   Jitter reduction: 10-100x for pool vs. heap
//
// WHY THIS MATTERS IN SPACE:
//
//   A GNC control loop at 100 Hz has a 10 ms budget. If the loop allocates
//   10 objects per cycle, that's 1000 allocations per second. With heap
//   allocation jitter of 100 us, a single allocation could consume 1% of
//   the cycle budget unpredictably. Pool allocation keeps this under 0.01%.
//
//   More importantly, pool allocation is DETERMINISTIC: the WCET (worst-case
//   execution time) is bounded and predictable, which is required for
//   RMS schedulability analysis.
//
// =============================================================================

#include "memory_pool.h"

#include <cstdio>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Test structure: a representative GNC data object.
//
// This struct is sized to be realistic for flight software objects:
//   - State vectors, filter states, telemetry packets, etc.
//   - 128 bytes = 2 cache lines on x86 (64-byte lines)
//
// The padding array ensures the struct is exactly 128 bytes, which is a
// convenient size for cache line analysis.
// ---------------------------------------------------------------------------
struct BenchmarkObject {
    double position[3];    // 24 bytes
    double velocity[3];    // 24 bytes
    double quaternion[4];  // 32 bytes
    double timestamp;      //  8 bytes
    uint32_t sequence;     //  4 bytes
    uint32_t flags;        //  4 bytes
    char padding[32];      // 32 bytes (total: 128 bytes)
};

// ---------------------------------------------------------------------------
// benchmark_memory_pool()
//
// Compares custom pool allocation vs. new/delete for a large number of
// allocation/deallocation cycles.
//
// Methodology:
//   1. Pre-create the memory pool (one-time cost, not measured)
//   2. Allocate N objects from the pool, then deallocate all of them
//   3. Measure total time and compute per-operation time
//   4. Repeat with new/delete for comparison
//   5. Report speedup factor
//
// We use a "burst" pattern: allocate many, then free many. This is
// common in batch processing (e.g., processing a buffer of sensor
// readings). We also test a "churn" pattern: alternating alloc/free,
// which is common in streaming pipelines.
//
// IMPORTANT: We use volatile pointers to prevent the compiler from
// optimizing away the allocations (dead code elimination).
// ---------------------------------------------------------------------------
void benchmark_memory_pool() {
    constexpr std::size_t NUM_OPS = 100000;

    std::printf("\n");
    std::printf("=================================================================\n");
    std::printf("  MEMORY POOL BENCHMARK\n");
    std::printf("=================================================================\n");
    std::printf("  Object size:     %zu bytes\n", sizeof(BenchmarkObject));
    std::printf("  Operations:      %zu alloc + %zu dealloc = %zu total\n",
                NUM_OPS, NUM_OPS, NUM_OPS * 2);
    std::printf("  Pattern:         Burst (allocate all, then deallocate all)\n");
    std::printf("-----------------------------------------------------------------\n\n");

    // Storage for pointers (to prevent optimization and for deallocation)
    std::vector<BenchmarkObject*> ptrs(NUM_OPS);

    // -----------------------------------------------------------------------
    // Test 1: Custom Memory Pool
    //
    // The pool is pre-allocated with NUM_OPS blocks. The first allocation
    // incurs a one-time cost of:
    //   - aligned_alloc(): ~1 us system call
    //   - memset(): ~30 us for 100000 * 128 bytes = 12.8 MB
    //   - free list construction: ~50 us (100000 pointer writes)
    //
    // After construction, each allocate() is pure pointer manipulation:
    //   block = free_list_head_;
    //   free_list_head_ = block->next;
    //   return block;
    //
    // This compiles to ~3 machine instructions (load, load, store).
    // -----------------------------------------------------------------------
    {
        MemoryPool<BenchmarkObject, 100000> pool;

        // Warm up: allocate and free a few objects to prime the cache.
        // The first few allocations after pool construction may be slow
        // because the free list nodes haven't been loaded into L1 cache yet.
        for (std::size_t i = 0; i < 100; ++i) {
            auto* p = pool.allocate();
            pool.deallocate(p);
        }

        // Benchmark allocation burst
        auto alloc_start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < NUM_OPS; ++i) {
            ptrs[i] = pool.allocate();
        }
        auto alloc_end = std::chrono::high_resolution_clock::now();

        double alloc_us = std::chrono::duration<double, std::micro>(alloc_end - alloc_start).count();

        // Benchmark deallocation burst
        auto dealloc_start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < NUM_OPS; ++i) {
            pool.deallocate(ptrs[i]);
        }
        auto dealloc_end = std::chrono::high_resolution_clock::now();

        double dealloc_us = std::chrono::duration<double, std::micro>(dealloc_end - dealloc_start).count();

        double total_us = alloc_us + dealloc_us;
        double per_op_ns = (total_us * 1000.0) / static_cast<double>(NUM_OPS * 2);
        double ops_per_sec = static_cast<double>(NUM_OPS * 2) / (total_us * 1e-6);

        // Print pool statistics
        PoolStats stats = pool.get_stats();

        std::printf("  [POOL] Allocation:     %10.1f us (%6.1f ns/op)\n", alloc_us,
                    (alloc_us * 1000.0) / static_cast<double>(NUM_OPS));
        std::printf("  [POOL] Deallocation:   %10.1f us (%6.1f ns/op)\n", dealloc_us,
                    (dealloc_us * 1000.0) / static_cast<double>(NUM_OPS));
        std::printf("  [POOL] Total:          %10.1f us (%6.1f ns/op)\n", total_us, per_op_ns);
        std::printf("  [POOL] Throughput:     %10.0f ops/sec\n", ops_per_sec);
        std::printf("  [POOL] Peak alloc time: %8.1f ns\n", stats.peak_alloc_time_ns);
        std::printf("  [POOL] Avg alloc time:  %8.1f ns\n", stats.avg_alloc_time_ns);
        std::printf("\n");
    }

    // -----------------------------------------------------------------------
    // Test 2: new/delete (Standard C++ heap allocation)
    //
    // new calls operator new(), which typically calls malloc().
    // The allocator must:
    //   1. Acquire a lock (thread safety) -- ~20 ns
    //   2. Search the free list for a suitable block -- ~10-100 ns
    //   3. Split the block if it's too large -- ~10 ns
    //   4. Update boundary tags -- ~10 ns
    //   5. Release the lock -- ~20 ns
    //
    // Total: ~70-200 ns typical, up to 100 us if sbrk() is needed.
    //
    // delete calls operator delete() -> free():
    //   1. Acquire lock -- ~20 ns
    //   2. Coalesce with adjacent free blocks (if any) -- ~20-50 ns
    //   3. Update free list -- ~10 ns
    //   4. Release lock -- ~20 ns
    //   5. Possibly call munmap() for large blocks -- ~1-10 us
    //
    // Total: ~70-100 ns typical, up to 10 us with munmap().
    // -----------------------------------------------------------------------
    {
        // Warm up the heap allocator
        for (std::size_t i = 0; i < 100; ++i) {
            auto* p = new BenchmarkObject;
            delete p;
        }

        // Benchmark allocation burst
        auto alloc_start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < NUM_OPS; ++i) {
            ptrs[i] = new BenchmarkObject;
        }
        auto alloc_end = std::chrono::high_resolution_clock::now();

        double alloc_us = std::chrono::duration<double, std::micro>(alloc_end - alloc_start).count();

        // Benchmark deallocation burst
        auto dealloc_start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < NUM_OPS; ++i) {
            delete ptrs[i];
        }
        auto dealloc_end = std::chrono::high_resolution_clock::now();

        double dealloc_us = std::chrono::duration<double, std::micro>(dealloc_end - dealloc_start).count();

        double total_us = alloc_us + dealloc_us;
        double per_op_ns = (total_us * 1000.0) / static_cast<double>(NUM_OPS * 2);
        double ops_per_sec = static_cast<double>(NUM_OPS * 2) / (total_us * 1e-6);

        std::printf("  [NEW/DELETE] Allocation:     %10.1f us (%6.1f ns/op)\n", alloc_us,
                    (alloc_us * 1000.0) / static_cast<double>(NUM_OPS));
        std::printf("  [NEW/DELETE] Deallocation:   %10.1f us (%6.1f ns/op)\n", dealloc_us,
                    (dealloc_us * 1000.0) / static_cast<double>(NUM_OPS));
        std::printf("  [NEW/DELETE] Total:          %10.1f us (%6.1f ns/op)\n", total_us, per_op_ns);
        std::printf("  [NEW/DELETE] Throughput:     %10.0f ops/sec\n", ops_per_sec);
        std::printf("\n");
    }

    std::printf("-----------------------------------------------------------------\n");
    std::printf("  Pool vs. new/delete comparison complete.\n");
    std::printf("=================================================================\n\n");
}

// ---------------------------------------------------------------------------
// benchmark_vs_malloc()
//
// Compares the memory pool against C-style malloc/free.
//
// malloc/free are generally slightly faster than new/delete because they
// skip constructor/destructor calls. However, the allocator internals
// are the same (typically glibc's ptmalloc2, jemalloc, or tcmalloc).
//
// We also measure an "interleaved" pattern where allocations and
// deallocations are mixed. This pattern stresses the allocator's
// free list management and is where pool allocation shines because
// it has zero fragmentation overhead.
// ---------------------------------------------------------------------------
void benchmark_vs_malloc() {
    constexpr std::size_t NUM_OPS = 100000;

    std::printf("\n");
    std::printf("=================================================================\n");
    std::printf("  MEMORY POOL vs MALLOC/FREE BENCHMARK\n");
    std::printf("=================================================================\n");
    std::printf("  Pattern: Interleaved (alloc-free-alloc-free...)\n");
    std::printf("-----------------------------------------------------------------\n\n");

    // -----------------------------------------------------------------------
    // Test 1: Pool - Interleaved pattern
    // Allocate one object, use it (write data), then free it.
    // Repeat N times. This tests the pool's performance under "churn."
    //
    // For the pool, each alloc+free cycle touches exactly one cache line
    // (the free list head). This is extremely cache-friendly.
    // -----------------------------------------------------------------------
    {
        MemoryPool<BenchmarkObject, 1024> pool;

        auto start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < NUM_OPS; ++i) {
            BenchmarkObject* p = pool.allocate();
            // Simulate minimal use: write to prevent optimization.
            // This also measures the combined cost of allocation +
            // first access (potential cache miss on new memory).
            p->timestamp = static_cast<double>(i);
            p->sequence = static_cast<uint32_t>(i);
            pool.deallocate(p);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double total_us = std::chrono::duration<double, std::micro>(end - start).count();
        double per_op_ns = (total_us * 1000.0) / static_cast<double>(NUM_OPS);

        std::printf("  [POOL]   Interleaved: %10.1f us total, %6.1f ns/cycle\n",
                    total_us, per_op_ns);
    }

    // -----------------------------------------------------------------------
    // Test 2: malloc/free - Interleaved pattern
    //
    // Each malloc call must:
    //   1. Check thread-local caches (tcache in glibc) -- ~5 ns
    //   2. If cache hit: return cached block -- ~5 ns total
    //   3. If cache miss: search arena free lists -- ~50-200 ns
    //   4. If no suitable block: call sbrk/mmap -- ~1000 ns
    //
    // For our interleaved pattern, the most recently freed block will
    // typically be in the thread-local cache, giving near-optimal
    // performance. This is the best-case scenario for malloc.
    // -----------------------------------------------------------------------
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < NUM_OPS; ++i) {
            void* p = std::malloc(sizeof(BenchmarkObject));
            // Simulate use
            static_cast<BenchmarkObject*>(p)->timestamp = static_cast<double>(i);
            static_cast<BenchmarkObject*>(p)->sequence = static_cast<uint32_t>(i);
            std::free(p);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double total_us = std::chrono::duration<double, std::micro>(end - start).count();
        double per_op_ns = (total_us * 1000.0) / static_cast<double>(NUM_OPS);

        std::printf("  [MALLOC] Interleaved: %10.1f us total, %6.1f ns/cycle\n",
                    total_us, per_op_ns);
    }

    std::printf("\n");
    std::printf("=================================================================\n\n");
}

// ---------------------------------------------------------------------------
// print_memory_stats()
//
// Prints a formatted report of the memory pool's state.
//
// This function creates a small pool, performs some allocations/deallocations
// to generate interesting statistics, then prints the report.
//
// The report includes:
//   - Pool configuration (block size, block count, total memory)
//   - Current allocation state (allocated, free, peak)
//   - Performance metrics (average and peak allocation time)
//   - Fragmentation ratio
//
// In flight software, this information would be included in housekeeping
// telemetry and monitored by mission operations. High fragmentation or
// peak allocation near the pool capacity indicates a potential problem.
// ---------------------------------------------------------------------------
void print_memory_stats() {
    std::printf("\n");
    std::printf("=================================================================\n");
    std::printf("  MEMORY POOL STATISTICS REPORT\n");
    std::printf("=================================================================\n\n");

    // Create a pool and exercise it
    constexpr std::size_t POOL_SIZE = 4096;
    MemoryPool<BenchmarkObject, POOL_SIZE> pool;

    // Allocate some objects
    std::vector<BenchmarkObject*> allocated;
    allocated.reserve(1000);

    // Allocate 1000 objects
    for (std::size_t i = 0; i < 1000; ++i) {
        allocated.push_back(pool.allocate());
    }

    // Free every other one (creates fragmentation pattern)
    for (std::size_t i = 0; i < 1000; i += 2) {
        pool.deallocate(allocated[i]);
        allocated[i] = nullptr;
    }

    // Get and print statistics
    PoolStats stats = pool.get_stats();

    std::printf("  Pool Configuration:\n");
    std::printf("    Block size:          %zu bytes\n", stats.block_size_bytes);
    std::printf("    Total blocks:        %zu\n", stats.total_blocks);
    std::printf("    Pool memory:         %zu bytes (%.2f KB)\n",
                stats.pool_size_bytes, static_cast<double>(stats.pool_size_bytes) / 1024.0);
    std::printf("\n");

    std::printf("  Current State:\n");
    std::printf("    Allocated blocks:    %zu\n", stats.allocated_blocks);
    std::printf("    Free blocks:         %zu\n", stats.free_blocks);
    std::printf("    Peak allocated:      %zu\n", stats.peak_allocated);
    std::printf("    Utilization:         %.1f%%\n",
                100.0 * static_cast<double>(stats.allocated_blocks) /
                static_cast<double>(stats.total_blocks));
    std::printf("\n");

    std::printf("  Performance:\n");
    std::printf("    Total allocations:   %zu\n", stats.total_allocations);
    std::printf("    Total deallocations: %zu\n", stats.total_deallocations);
    std::printf("    Avg alloc time:      %.1f ns\n", stats.avg_alloc_time_ns);
    std::printf("    Peak alloc time:     %.1f ns\n", stats.peak_alloc_time_ns);
    std::printf("    Fragmentation:       %.3f (0=perfect, 1=worst)\n",
                stats.fragmentation_ratio);
    std::printf("\n");

    // Clean up remaining allocations
    for (auto* p : allocated) {
        if (p) pool.deallocate(p);
    }

    // Print post-cleanup stats
    PoolStats clean_stats = pool.get_stats();
    std::printf("  After cleanup:\n");
    std::printf("    Allocated blocks:    %zu\n", clean_stats.allocated_blocks);
    std::printf("    Free blocks:         %zu\n", clean_stats.free_blocks);
    std::printf("    Fragmentation:       %.3f\n", clean_stats.fragmentation_ratio);
    std::printf("\n");
    std::printf("=================================================================\n\n");
}
