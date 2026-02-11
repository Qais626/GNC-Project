# =============================================================================
# GNC Real-Time Simulation Build Configuration
# =============================================================================
# Project: gnc_realtime
# Description: Real-time GNC simulation demonstrating embedded systems concepts
#              including custom memory allocators, lock-free data structures,
#              deterministic scheduling, and hardware-in-the-loop interfaces.
#
# Build instructions:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j$(nproc)
#
# For debug with sanitizers:
#   cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON
# =============================================================================

cmake_minimum_required(VERSION 3.14)
project(gnc_realtime
    VERSION 1.0.0
    DESCRIPTION "GNC Real-Time Simulation Engine"
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# C++17 standard -- required for:
#   - std::optional (ring buffer pop)
#   - structured bindings
#   - if constexpr (template metaprogramming in allocator)
#   - std::chrono improvements
#   - std::invoke for task callbacks
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # No GNU extensions -- keep it portable

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
set(SOURCES
    src/memory_pool.cpp
    src/ring_buffer.cpp
    src/dynamics_engine.cpp
    src/flight_computer.cpp
    src/hil_interface.cpp
    src/main.cpp
)

set(HEADERS
    include/memory_pool.h
    include/ring_buffer.h
    include/dynamics_engine.h
    include/flight_computer.h
    include/hil_interface.h
)

# ---------------------------------------------------------------------------
# Executable target
# ---------------------------------------------------------------------------
add_executable(gnc_rt_sim ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(gnc_rt_sim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ---------------------------------------------------------------------------
# Compiler warnings -- strict for flight-quality code
# ---------------------------------------------------------------------------
target_compile_options(gnc_rt_sim PRIVATE
    -Wall           # All standard warnings
    -Wextra         # Extra warnings beyond -Wall
    -Wpedantic      # Strict ISO C++ compliance
    -Wshadow        # Warn on variable shadowing (critical in embedded)
    -Wconversion    # Warn on implicit type conversions
    -Wsign-conversion
    -Wnull-dereference
    -Wdouble-promotion   # Float promoted to double silently (perf issue)
    -Wformat=2           # Printf format string checking
    -Wcast-align         # Warn on alignment issues (important for SIMD/cache)
    -Wunused
    -Woverloaded-virtual
)

# Release builds: optimize aggressively, enable architecture-specific opts
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(gnc_rt_sim PRIVATE
        -O3                  # Full optimization
        -march=native        # Use CPU-specific instructions (SSE/AVX if available)
        -funroll-loops       # Unroll small loops for pipeline efficiency
        -ffast-math          # Allow floating-point optimizations (know the trade-offs!)
        -flto                # Link-time optimization
    )
endif()

# Debug builds: sanitizers for catching memory and threading bugs
option(ENABLE_SANITIZERS "Enable ASan and UBSan" OFF)
if(ENABLE_SANITIZERS)
    target_compile_options(gnc_rt_sim PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )
    target_link_options(gnc_rt_sim PRIVATE
        -fsanitize=address,undefined
    )
endif()

# ---------------------------------------------------------------------------
# Link pthreads -- required for:
#   - std::thread (flight computer real-time loop)
#   - std::atomic (lock-free ring buffer)
#   - POSIX real-time scheduling (sched_setscheduler)
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)
target_link_libraries(gnc_rt_sim PRIVATE Threads::Threads)

# ---------------------------------------------------------------------------
# Platform-specific: enable POSIX real-time extensions on Linux
# ---------------------------------------------------------------------------
if(UNIX AND NOT APPLE)
    target_link_libraries(gnc_rt_sim PRIVATE rt)  # POSIX RT library
endif()

# ---------------------------------------------------------------------------
# Install target (optional)
# ---------------------------------------------------------------------------
install(TARGETS gnc_rt_sim DESTINATION bin)
