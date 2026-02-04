// =============================================================================
// memory_pool.h -- Custom Memory Pool Allocator for Real-Time GNC Systems
// =============================================================================
//
// WHY CUSTOM ALLOCATORS MATTER FOR REAL-TIME SYSTEMS:
//
//   The default system allocator (malloc/free, new/delete) has several problems
//   for hard real-time applications:
//
//   1. NON-DETERMINISTIC TIMING: malloc may trigger sbrk() or mmap() system
//      calls, which involve kernel context switches. These can take anywhere
//      from microseconds to milliseconds -- unacceptable when your attitude
//      control loop must complete in < 1 ms.
//
//   2. FRAGMENTATION: Over time, repeated alloc/free creates a Swiss cheese
//      pattern in the heap. The allocator must search free lists to find a
//      suitable block, making allocation time O(n) in the worst case.
//
//   3. CACHE POLLUTION: System allocators interleave allocations from different
//      subsystems. Your nav filter state and your telemetry strings might sit
//      on the same cache line, causing evictions during computation.
//
//   4. PAGE FAULTS: A new allocation may touch a page that hasn't been mapped
//      yet, triggering a page fault and OS intervention.
//
//   This pool allocator solves all of these:
//   - Pre-allocates all memory at initialization (all page faults happen then)
//   - O(1) allocation using a free list (just pointer manipulation)
//   - Zero fragmentation for fixed-size blocks
//   - Cache-line aligned allocations for optimal data access patterns
//
// CACHE LINE ALIGNMENT (64 bytes on x86/ARM):
//
//   Modern CPUs don't read individual bytes from RAM. They read entire cache
//   lines (typically 64 bytes) at a time. If your 48-byte struct straddles
//   two cache lines, the CPU must fetch both lines -- doubling memory latency.
//   By aligning allocations to 64-byte boundaries, we guarantee each object
//   starts at the beginning of a cache line.
//
//   Additionally, when multiple cores access different objects that happen to
//   share a cache line, the MESI coherency protocol forces both cores to
//   invalidate and reload that line. This is "false sharing" and destroys
//   performance on multi-core systems. Alignment eliminates this.
//
// MEMORY LAYOUT OF THIS POOL:
//
//   +---------+---------+---------+---------+-----+
//   | Block 0 | Block 1 | Block 2 | Block 3 | ... |
//   +---------+---------+---------+---------+-----+
//   ^         ^         ^         ^
//   |         |         |         |
//   64-byte   64-byte   64-byte   64-byte
//   aligned   aligned   aligned   aligned
//
//   Each block is max(sizeof(T), sizeof(void*)) bytes, rounded up to a
//   multiple of 64 for alignment. Free blocks form a singly-linked list
//   by storing a pointer in the first 8 bytes of the free block:
//
//   Free block layout:
//   +------------------+----------------------------+
//   | next_free (8B)   | unused padding             |
//   +------------------+----------------------------+
//
//   Allocated block layout:
//   +------------------------------------------------+
//   | T object data                                   |
//   +------------------------------------------------+
//
// =============================================================================

#ifndef GNC_MEMORY_POOL_H
#define GNC_MEMORY_POOL_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <type_traits>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <chrono>

// ---------------------------------------------------------------------------
// CACHE_LINE_SIZE: The size of an L1 data cache line on the target platform.
//
// On x86-64 (Intel/AMD) and ARMv8 (most modern ARM): 64 bytes.
// On Apple M1/M2: 128 bytes for the performance cores, but 64 for efficiency.
// On some PowerPC: 128 bytes.
//
// We default to 64 bytes. For Apple Silicon, you might want 128.
// In a real flight system, this would come from a board support package (BSP).
// ---------------------------------------------------------------------------
constexpr std::size_t CACHE_LINE_SIZE = 64;

// ---------------------------------------------------------------------------
// PRAGMA PACK DEMONSTRATION
// ---------------------------------------------------------------------------
//
// #pragma pack controls struct member alignment. By default, the compiler
// inserts padding to align members to their natural alignment:
//
//   struct Default {      // sizeof = 16 (with padding)
//       char a;           // offset 0, size 1
//       // 7 bytes padding to align double
//       double b;         // offset 8, size 8
//   };
//
// With #pragma pack(1), no padding is inserted:
//
//   #pragma pack(push, 1)
//   struct Packed {       // sizeof = 9 (no padding)
//       char a;           // offset 0, size 1
//       double b;         // offset 1, size 8  <-- MISALIGNED!
//   };
//   #pragma pack(pop)
//
// TRADE-OFFS:
//   - Packed structs save memory (critical for telemetry packets over a link)
//   - But misaligned access is SLOW on x86 (two cache line reads) and
//     ILLEGAL on some ARM processors (causes a hardware fault!)
//   - Use packed structs ONLY for wire protocols and serialization.
//   - Use aligned structs for computation.
// ---------------------------------------------------------------------------

// Example: Telemetry packet for wire transmission (packed, no padding)
#pragma pack(push, 1)
struct TelemetryPacketWire {
    uint8_t  sync_byte;       // 0x7E sync marker
    uint16_t packet_id;       // Packet sequence number
    uint32_t timestamp_ms;    // Mission elapsed time
    float    position[3];     // ECEF position (m)
    float    velocity[3];     // ECEF velocity (m/s)
    float    quaternion[4];   // Attitude quaternion
    uint16_t crc;             // CRC-16 checksum
    // Total: 1 + 2 + 4 + 12 + 12 + 16 + 2 = 49 bytes (no padding)
};
#pragma pack(pop)

// Same data for computation (naturally aligned, padded)
struct TelemetryPacketAligned {
    uint8_t  sync_byte;       // offset 0
    // 1 byte padding
    uint16_t packet_id;       // offset 2
    uint32_t timestamp_ms;    // offset 4
    float    position[3];     // offset 8
    float    velocity[3];     // offset 20
    float    quaternion[4];   // offset 32
    uint16_t crc;             // offset 48
    // 2 bytes padding to reach 52, then further padding for alignment
    // Total: 52 bytes (with padding for natural alignment)
};

// ---------------------------------------------------------------------------
// Pool allocation statistics -- useful for flight software telemetry
// and debugging memory usage patterns during integration testing.
// ---------------------------------------------------------------------------
struct PoolStats {
    std::size_t total_blocks;          // Total blocks in the pool
    std::size_t allocated_blocks;      // Currently allocated
    std::size_t free_blocks;           // Currently free
    std::size_t peak_allocated;        // High-water mark
    std::size_t total_allocations;     // Lifetime allocation count
    std::size_t total_deallocations;   // Lifetime deallocation count
    double      avg_alloc_time_ns;     // Average allocation time in nanoseconds
    double      peak_alloc_time_ns;    // Worst-case allocation time
    double      fragmentation_ratio;   // 0.0 = no fragmentation, 1.0 = fully fragmented
    std::size_t block_size_bytes;      // Size of each block
    std::size_t pool_size_bytes;       // Total pool memory
};

// ---------------------------------------------------------------------------
// MemoryPool<T, BlockSize>
//
// Template parameters:
//   T         - The type to allocate. The pool is homogeneous (all blocks
//               are the same size), which eliminates external fragmentation.
//   BlockSize - Number of blocks to pre-allocate. Default 4096.
//               Choose based on worst-case allocation count during a mission
//               phase, plus margin. Typically 2x expected max.
// ---------------------------------------------------------------------------
template <typename T, std::size_t BlockSize = 4096>
class MemoryPool {
public:
    // -----------------------------------------------------------------------
    // Constructor: pre-allocates the entire pool.
    //
    // This is the ONLY place where a system call (aligned_alloc / posix_memalign)
    // occurs. After this, allocate() and deallocate() are pure pointer math.
    //
    // We use aligned_alloc (C11/C++17) to get cache-line-aligned memory.
    // This ensures:
    //   1. No object straddles a cache line boundary
    //   2. No false sharing between adjacent objects on different cores
    //   3. SIMD instructions (SSE/AVX) work without alignment faults
    // -----------------------------------------------------------------------
    MemoryPool() {
        // Calculate block size: must hold T and also a pointer (for free list)
        // Round up to cache line boundary for alignment
        constexpr std::size_t raw_size = sizeof(T) > sizeof(void*) ? sizeof(T) : sizeof(void*);
        aligned_block_size_ = ((raw_size + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE) * CACHE_LINE_SIZE;

        pool_size_bytes_ = aligned_block_size_ * BlockSize;

        // ---------------------------------------------------------------------------
        // aligned_alloc: Allocates memory aligned to a specified boundary.
        //
        // Why not just malloc?
        //   malloc only guarantees alignment to max_align_t (typically 16 bytes).
        //   We need 64-byte alignment for cache line optimization.
        //
        // Why not posix_memalign?
        //   aligned_alloc is the C++17 standard way. posix_memalign is POSIX-only.
        //   We fall back to posix_memalign on older systems.
        //
        // Note: pool_size_bytes_ must be a multiple of CACHE_LINE_SIZE for
        //       aligned_alloc to work (C11 requirement).
        // ---------------------------------------------------------------------------
        pool_memory_ = static_cast<uint8_t*>(
            std::aligned_alloc(CACHE_LINE_SIZE, pool_size_bytes_)
        );

        if (!pool_memory_) {
            throw std::bad_alloc();
        }

        // Zero-initialize to trigger page faults NOW, not during flight.
        // This is called "memory locking" in a broader sense -- we want all
        // pages to be resident in physical RAM before the real-time loop starts.
        // In a real system, we'd also call mlock() to prevent the OS from
        // swapping these pages out.
        std::memset(pool_memory_, 0, pool_size_bytes_);

        // Build the free list by chaining all blocks together.
        // Each free block stores a pointer to the next free block.
        build_free_list();

        // Initialize statistics
        stats_ = {};
        stats_.total_blocks = BlockSize;
        stats_.free_blocks = BlockSize;
        stats_.block_size_bytes = aligned_block_size_;
        stats_.pool_size_bytes = pool_size_bytes_;

        cumulative_alloc_time_ns_ = 0.0;
    }

    // -----------------------------------------------------------------------
    // Destructor: frees the pre-allocated pool.
    // In a real flight system, the pool typically lives for the entire mission
    // and is never freed (the process exits at end of mission).
    // -----------------------------------------------------------------------
    ~MemoryPool() {
        if (pool_memory_) {
            std::free(pool_memory_);
            pool_memory_ = nullptr;
        }
    }

    // No copy (each pool owns its memory)
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    // Move is allowed
    MemoryPool(MemoryPool&& other) noexcept
        : pool_memory_(other.pool_memory_),
          free_list_head_(other.free_list_head_),
          aligned_block_size_(other.aligned_block_size_),
          pool_size_bytes_(other.pool_size_bytes_),
          stats_(other.stats_),
          cumulative_alloc_time_ns_(other.cumulative_alloc_time_ns_) {
        other.pool_memory_ = nullptr;
        other.free_list_head_ = nullptr;
    }

    MemoryPool& operator=(MemoryPool&& other) noexcept {
        if (this != &other) {
            if (pool_memory_) std::free(pool_memory_);
            pool_memory_ = other.pool_memory_;
            free_list_head_ = other.free_list_head_;
            aligned_block_size_ = other.aligned_block_size_;
            pool_size_bytes_ = other.pool_size_bytes_;
            stats_ = other.stats_;
            cumulative_alloc_time_ns_ = other.cumulative_alloc_time_ns_;
            other.pool_memory_ = nullptr;
            other.free_list_head_ = nullptr;
        }
        return *this;
    }

    // -----------------------------------------------------------------------
    // allocate() -- O(1) allocation from the free list.
    //
    // HOW IT WORKS:
    //   The free list is a singly-linked list threaded through the free blocks
    //   themselves. Each free block's first 8 bytes hold a pointer to the next
    //   free block. To allocate:
    //     1. Pop the head of the free list (one pointer read)
    //     2. Update head to point to next free block (one pointer write)
    //     3. Return the popped block
    //
    //   Total: 2 memory accesses, no branches (except the exhaustion check),
    //   no system calls. This is as fast as allocation can get.
    //
    //   Compare to malloc: must search free lists, possibly split blocks,
    //   update boundary tags, possibly call sbrk(). Worst case: thousands
    //   of instructions.
    // -----------------------------------------------------------------------
    T* allocate() {
        auto start = std::chrono::high_resolution_clock::now();

        if (!free_list_head_) {
            // Pool exhausted. In flight software, this would trigger a
            // CRITICAL fault and potentially a safe mode transition.
            throw std::bad_alloc();
        }

        // Pop from free list head -- O(1)
        FreeBlock* block = free_list_head_;
        free_list_head_ = block->next;

        // Update statistics
        stats_.allocated_blocks++;
        stats_.free_blocks--;
        stats_.total_allocations++;
        if (stats_.allocated_blocks > stats_.peak_allocated) {
            stats_.peak_allocated = stats_.allocated_blocks;
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ns = std::chrono::duration<double, std::nano>(end - start).count();
        cumulative_alloc_time_ns_ += elapsed_ns;
        stats_.avg_alloc_time_ns = cumulative_alloc_time_ns_ / static_cast<double>(stats_.total_allocations);
        if (elapsed_ns > stats_.peak_alloc_time_ns) {
            stats_.peak_alloc_time_ns = elapsed_ns;
        }

        // Return as T* -- the caller is responsible for placement new if needed
        return reinterpret_cast<T*>(block);
    }

    // -----------------------------------------------------------------------
    // deallocate() -- O(1) return to the free list.
    //
    // Simply pushes the block back onto the free list head.
    // No coalescing needed because all blocks are the same size (no external
    // fragmentation possible with a homogeneous pool).
    //
    // IMPORTANT: The caller must call the destructor (~T) before calling
    // deallocate(). This is the same contract as std::allocator.
    // -----------------------------------------------------------------------
    void deallocate(T* ptr) {
        if (!ptr) return;

        // Validate that the pointer is within our pool
        uint8_t* raw = reinterpret_cast<uint8_t*>(ptr);
        if (raw < pool_memory_ || raw >= pool_memory_ + pool_size_bytes_) {
            // Pointer not from this pool! In flight software, log this as a
            // SEVERE error. Memory corruption is the most dangerous failure mode.
            throw std::invalid_argument("Pointer does not belong to this memory pool");
        }

        // Validate alignment
        std::size_t offset = static_cast<std::size_t>(raw - pool_memory_);
        if (offset % aligned_block_size_ != 0) {
            throw std::invalid_argument("Pointer is not aligned to a block boundary");
        }

        // Push onto free list head -- O(1)
        FreeBlock* block = reinterpret_cast<FreeBlock*>(ptr);
        block->next = free_list_head_;
        free_list_head_ = block;

        stats_.allocated_blocks--;
        stats_.free_blocks++;
        stats_.total_deallocations++;
    }

    // -----------------------------------------------------------------------
    // construct() / destroy() -- Placement new and explicit destructor call.
    // These complete the allocator interface.
    // -----------------------------------------------------------------------
    template <typename... Args>
    T* construct(Args&&... args) {
        T* ptr = allocate();
        new (ptr) T(std::forward<Args>(args)...);
        return ptr;
    }

    void destroy(T* ptr) {
        if (ptr) {
            ptr->~T();
            deallocate(ptr);
        }
    }

    // -----------------------------------------------------------------------
    // Statistics and diagnostics
    // -----------------------------------------------------------------------
    PoolStats get_stats() const {
        PoolStats s = stats_;
        s.fragmentation_ratio = compute_fragmentation();
        return s;
    }

    std::size_t capacity() const { return BlockSize; }
    std::size_t allocated() const { return stats_.allocated_blocks; }
    std::size_t available() const { return stats_.free_blocks; }
    bool is_full() const { return free_list_head_ == nullptr; }
    bool is_empty() const { return stats_.allocated_blocks == 0; }

    // -----------------------------------------------------------------------
    // defragment() -- Rebuild the free list in address order.
    //
    // Even though there's no external fragmentation (all blocks same size),
    // the free list can become randomly ordered after many alloc/dealloc
    // cycles. This means sequential allocations may return non-contiguous
    // blocks, leading to poor spatial locality and more cache misses.
    //
    // Defragmentation sorts the free list so sequential allocations return
    // sequential addresses, maximizing cache line reuse and prefetch
    // effectiveness.
    //
    // WHEN TO CALL: Only during non-time-critical phases (e.g., between
    // mission phases, during coasting). NEVER during a real-time loop.
    // -----------------------------------------------------------------------
    void defragment();

    // -----------------------------------------------------------------------
    // Generate a human-readable report of pool state.
    // Used for telemetry and ground support.
    // -----------------------------------------------------------------------
    std::string generate_report() const;

private:
    // -----------------------------------------------------------------------
    // FreeBlock: Intrusive free list node.
    // When a block is free, its first sizeof(void*) bytes hold the pointer
    // to the next free block. When allocated, these bytes are part of the
    // T object. This is a classic embedded systems technique -- zero overhead.
    // -----------------------------------------------------------------------
    struct FreeBlock {
        FreeBlock* next;
    };

    // Build initial free list: chain all blocks in forward address order
    void build_free_list() {
        free_list_head_ = nullptr;

        // Build in reverse so the free list is in forward address order
        // (first allocation returns lowest address -- good for spatial locality)
        for (std::size_t i = BlockSize; i > 0; --i) {
            uint8_t* block_addr = pool_memory_ + (i - 1) * aligned_block_size_;
            FreeBlock* block = reinterpret_cast<FreeBlock*>(block_addr);
            block->next = free_list_head_;
            free_list_head_ = block;
        }
    }

    // -----------------------------------------------------------------------
    // Fragmentation metric: measures how "scattered" the free list is.
    //
    // A perfectly defragmented pool has all free blocks contiguous. We measure
    // fragmentation as: 1 - (largest_contiguous_run / total_free_blocks).
    // 0.0 = all free blocks contiguous, 1.0 = maximally scattered.
    // -----------------------------------------------------------------------
    double compute_fragmentation() const {
        if (stats_.free_blocks <= 1) return 0.0;

        // Walk the free list and count contiguous runs
        std::size_t max_run = 0;
        std::size_t current_run = 1;

        FreeBlock* current = free_list_head_;
        while (current && current->next) {
            uint8_t* current_addr = reinterpret_cast<uint8_t*>(current);
            uint8_t* next_addr = reinterpret_cast<uint8_t*>(current->next);

            // Check if next free block is the adjacent block in the pool
            if (next_addr == current_addr + aligned_block_size_) {
                current_run++;
            } else {
                if (current_run > max_run) max_run = current_run;
                current_run = 1;
            }
            current = current->next;
        }
        if (current_run > max_run) max_run = current_run;

        return 1.0 - static_cast<double>(max_run) / static_cast<double>(stats_.free_blocks);
    }

    uint8_t*    pool_memory_         = nullptr;  // Raw pool storage
    FreeBlock*  free_list_head_      = nullptr;  // Head of free list
    std::size_t aligned_block_size_  = 0;        // Size of each block (aligned)
    std::size_t pool_size_bytes_     = 0;        // Total pool size
    PoolStats   stats_               = {};       // Allocation statistics
    double      cumulative_alloc_time_ns_ = 0.0; // For computing average
};

#endif // GNC_MEMORY_POOL_H
