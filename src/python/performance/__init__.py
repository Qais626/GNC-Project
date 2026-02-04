"""
performance - Computer Architecture Optimization & Performance Engineering for GNC

This package demonstrates how understanding hardware architecture translates into
measurable software performance gains for spacecraft GNC computations. The three
modules address different layers of the optimization stack:

    benchmarks       - Quantitative benchmarking suite that measures and compares
                       execution time and memory usage across naive vs optimized
                       implementations of core GNC algorithms (matrix ops, propagators,
                       search, sorting, memory allocation).

    cache_optimizer  - Cache-friendly data layout demonstrations that exploit CPU
                       cache hierarchy (L1/L2/L3), spatial and temporal locality,
                       loop tiling, memory alignment for SIMD, branch prediction,
                       and hardware prefetching.

    parallel         - Parallelization and vectorization for GNC workloads using
                       numpy broadcasting (SIMD-style), multiprocessing for Monte
                       Carlo and grid search, and GPU acceleration stubs that
                       explain CUDA memory and execution models.

Together these modules show that GNC flight software performance is not just about
algorithmic complexity -- it depends critically on how data is laid out in memory,
how the CPU cache and branch predictor interact with access patterns, and how work
is distributed across cores and accelerators.
"""
