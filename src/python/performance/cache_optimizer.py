"""
cache_optimizer.py - Cache-Friendly Data Layouts & Memory Alignment Demonstrations

This module makes invisible hardware effects *visible* by benchmarking access
patterns that interact differently with the CPU cache hierarchy.

CPU Cache Primer
----------------
Modern processors have a multi-level cache hierarchy between the register file
and main memory (DRAM):

    Level   Typical size    Latency (cycles)    Latency (ns, approx.)
    -----   ------------    ----------------    ---------------------
    L1d     32-48 KB        4                   ~1
    L2      256 KB-1 MB     12                  ~4
    L3      8-32 MB         40                  ~12
    DRAM    many GB         200+                ~60-100

The cache operates in *cache lines* (typically 64 bytes). When the CPU requests
a single byte, the hardware fetches the entire 64-byte line. If the next access
falls within the same line ("spatial locality"), it is essentially free. If the
program revisits the same line soon ("temporal locality"), it will still be in
the fast cache levels.

Every demonstration below exploits -- or deliberately defeats -- these
properties so that the wall-clock difference reveals the hardware cost.

Demonstrations
--------------
1. SoA vs AoS          -- spatial locality
2. Row vs column access -- stride and cache-line prefetching
3. Loop blocking/tiling -- temporal locality for matrix multiply
4. Memory alignment     -- SIMD load efficiency
5. Branch prediction    -- sorted vs unsorted conditional sum
6. Prefetching pattern  -- sequential vs random access
"""

from __future__ import annotations

import os
import sys
import time
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class CacheAnalysis:
    """
    Interactive demonstrations of CPU-cache and micro-architecture effects.

    Every ``demonstrate_*`` method prints a human-readable summary to stdout
    AND returns a dict of timings (in seconds) so callers can aggregate or
    re-plot the results.
    """

    # ------------------------------------------------------------------
    # Helper: simple timing decorator
    # ------------------------------------------------------------------
    @staticmethod
    def _time_it(func, *args, num_runs: int = 5, **kwargs) -> float:
        """Return the *median* wall-clock time over *num_runs* invocations."""
        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            func(*args, **kwargs)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        times.sort()
        return times[len(times) // 2]  # median

    # ==================================================================
    # 1. Struct-of-Arrays vs Array-of-Structs
    # ==================================================================
    def demonstrate_soa_vs_aos(self, n_particles: int = 100_000) -> Dict[str, float]:
        """
        Compare AoS (list of dicts) vs SoA (dict of contiguous numpy arrays).

        Why SoA wins
        -------------
        When we iterate over only the position fields (x, y, z) of N particles:

        AoS layout in memory (conceptual):
            [x0 y0 z0 vx0 vy0 vz0] [x1 y1 z1 vx1 vy1 vz1] ...
            ^-------- dict --------^ ^-------- dict --------^
            Each dict is a separate heap object; accessing x_i pulls the
            whole dict (plus hash table overhead) into cache, wasting
            space on vx, vy, vz that we never read.

        SoA layout in memory:
            x: [x0 x1 x2 x3 ...]     <- contiguous float64 array
            y: [y0 y1 y2 y3 ...]
            z: [z0 z1 z2 z3 ...]
            Every 64-byte cache line holds 8 consecutive x-values; the
            hardware prefetcher detects the linear stride-8 pattern and
            pre-fills the next line before we ask for it.  100 % of each
            cache line is useful data.
        """
        print("\n" + "=" * 60)
        print("  Demo 1: SoA vs AoS  (n = {:,})".format(n_particles))
        print("=" * 60)

        # --- build AoS ---
        aos: List[Dict[str, float]] = [
            {
                "x": random.gauss(0, 1),
                "y": random.gauss(0, 1),
                "z": random.gauss(0, 1),
                "vx": random.gauss(0, 1),
                "vy": random.gauss(0, 1),
                "vz": random.gauss(0, 1),
            }
            for _ in range(n_particles)
        ]

        # --- build SoA ---
        soa: Dict[str, np.ndarray] = {
            "x": np.random.randn(n_particles),
            "y": np.random.randn(n_particles),
            "z": np.random.randn(n_particles),
            "vx": np.random.randn(n_particles),
            "vy": np.random.randn(n_particles),
            "vz": np.random.randn(n_particles),
        }

        # --- compute distance (AoS) ---
        def distance_aos() -> List[float]:
            return [
                (p["x"] ** 2 + p["y"] ** 2 + p["z"] ** 2) ** 0.5 for p in aos
            ]

        # --- compute distance (SoA) ---
        def distance_soa() -> np.ndarray:
            return np.sqrt(soa["x"] ** 2 + soa["y"] ** 2 + soa["z"] ** 2)

        t_aos = self._time_it(distance_aos, num_runs=5)
        t_soa = self._time_it(distance_soa, num_runs=5)

        speedup = t_aos / t_soa if t_soa > 0 else float("inf")
        print(f"  AoS time : {t_aos:.6f} s")
        print(f"  SoA time : {t_soa:.6f} s")
        print(f"  Speedup  : {speedup:.1f}x")
        print("  Reason   : SoA keeps each field contiguous -> full cache-line")
        print("             utilisation + hardware prefetcher engaged.")

        return {"aos_s": t_aos, "soa_s": t_soa, "speedup": speedup}

    # ==================================================================
    # 2. Row-major vs Column-major iteration
    # ==================================================================
    def demonstrate_row_vs_column_access(self, matrix_size: int = 1000) -> Dict[str, float]:
        """
        Iterate over a 2-D C-contiguous numpy array in row-major vs column-major order.

        Why row-major wins
        ------------------
        numpy allocates 2-D arrays in C (row-major) order by default:

            Physical memory layout:
            row0_col0  row0_col1  row0_col2  ...  row0_colN
            row1_col0  row1_col1  row1_col2  ...  row1_colN
            ...

        Row-major iteration:
            Access pattern: row0_col0, row0_col1, row0_col2, ...
            Memory stride = 8 bytes (one float64).
            A 64-byte cache line covers 8 consecutive elements, so 7 out of
            every 8 accesses are cache hits (87.5 % hit rate from spatial
            locality alone).  The hardware prefetcher sees a constant stride
            and starts fetching the next line before we need it.

        Column-major iteration:
            Access pattern: row0_col0, row1_col0, row2_col0, ...
            Memory stride = matrix_size * 8 bytes.
            For a 1000-column matrix, that is 8000 bytes between successive
            accesses -- 125 cache lines apart.  Every access is a *compulsory
            cache miss*, and the prefetcher cannot help because the stride
            exceeds its detection window on most micro-architectures.
        """
        print("\n" + "=" * 60)
        print(f"  Demo 2: Row vs Column Access  ({matrix_size}x{matrix_size})")
        print("=" * 60)

        # C-contiguous (row-major) array
        mat = np.random.randn(matrix_size, matrix_size)

        def row_major_sum() -> float:
            """Iterate rows then columns -- stride = 8 bytes."""
            total = 0.0
            for i in range(matrix_size):
                for j in range(matrix_size):
                    total += mat[i, j]
            return total

        def col_major_sum() -> float:
            """Iterate columns then rows -- stride = matrix_size * 8 bytes."""
            total = 0.0
            for j in range(matrix_size):
                for i in range(matrix_size):
                    total += mat[i, j]
            return total

        t_row = self._time_it(row_major_sum, num_runs=3)
        t_col = self._time_it(col_major_sum, num_runs=3)

        speedup = t_col / t_row if t_row > 0 else float("inf")
        print(f"  Row-major time : {t_row:.4f} s")
        print(f"  Col-major time : {t_col:.4f} s")
        print(f"  Row advantage  : {speedup:.2f}x faster")
        print("  Reason         : Row-major matches C-contiguous layout;")
        print("                   stride = 8 B -> cache line reuse + prefetcher.")

        return {"row_major_s": t_row, "col_major_s": t_col, "speedup": speedup}

    # ==================================================================
    # 3. Loop Blocking / Tiling for matrix multiply
    # ==================================================================
    def demonstrate_blocking(
        self, matrix_size: int = 512, block_size: int = 64
    ) -> Dict[str, float]:
        """
        Matrix multiplication with and without loop tiling (blocking).

        Why blocking helps
        ------------------
        Naive triple-loop matrix multiply (C = A * B) accesses B in column
        order.  For large matrices, each column of B evicts cache lines that
        were just loaded for the previous column.

        Blocking partitions A, B, C into small sub-matrices (tiles) of size
        ``block_size x block_size``.  If a tile fits in L1 cache (~32 KB, which
        holds a 64x64 float64 tile exactly), all three tiles stay hot for the
        entire inner-product computation of that tile.

        Tile memory budget:
            3 tiles x 64 x 64 x 8 bytes = 96 KB  (fits comfortably in L1+L2)

        This reduces cache misses from O(n^3 / B) to O(n^3 / (B * L)) where B
        is the cache-line size and L is the tile dimension.
        """
        print("\n" + "=" * 60)
        print(f"  Demo 3: Loop Blocking  ({matrix_size}x{matrix_size}, block={block_size})")
        print("=" * 60)

        A = np.random.randn(matrix_size, matrix_size).astype(np.float64)
        B = np.random.randn(matrix_size, matrix_size).astype(np.float64)

        def naive_matmul() -> np.ndarray:
            """Standard triple-loop -- terrible cache behaviour on B columns."""
            n = matrix_size
            C = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(n):
                    s = 0.0
                    for k in range(n):
                        s += A[i, k] * B[k, j]
                    C[i, j] = s
            return C

        def blocked_matmul() -> np.ndarray:
            """
            Tiled triple-loop.

            The six loops iterate over tile indices (ii, jj, kk) then
            intra-tile indices (i, j, k).  All three sub-matrices
            A[ii:ii+bs, kk:kk+bs], B[kk:kk+bs, jj:jj+bs], C[ii:ii+bs, jj:jj+bs]
            fit in L1 cache, keeping data hot for the full inner product.
            """
            n = matrix_size
            bs = block_size
            C = np.zeros((n, n), dtype=np.float64)
            for ii in range(0, n, bs):
                for jj in range(0, n, bs):
                    for kk in range(0, n, bs):
                        # Tile boundaries (handle edge tiles that may be smaller)
                        i_end = min(ii + bs, n)
                        j_end = min(jj + bs, n)
                        k_end = min(kk + bs, n)
                        for i in range(ii, i_end):
                            for j in range(jj, j_end):
                                s = 0.0
                                for k in range(kk, k_end):
                                    s += A[i, k] * B[k, j]
                                C[i, j] += s
            return C

        # For large matrices the naive version is extremely slow in pure Python,
        # so we use a smaller size for the actual timing but keep the demo
        # illustrative.  We also compare against numpy's optimised BLAS path.
        small = min(matrix_size, 128)  # cap at 128 for reasonable runtime
        A_small = A[:small, :small]
        B_small = B[:small, :small]

        # Rebind closures to small matrices
        def naive_small():
            n = small
            C = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(n):
                    s = 0.0
                    for k in range(n):
                        s += A_small[i, k] * B_small[k, j]
                    C[i, j] = s
            return C

        bs = min(block_size, small)

        def blocked_small():
            n = small
            C = np.zeros((n, n), dtype=np.float64)
            for ii in range(0, n, bs):
                for jj in range(0, n, bs):
                    for kk in range(0, n, bs):
                        i_end = min(ii + bs, n)
                        j_end = min(jj + bs, n)
                        k_end = min(kk + bs, n)
                        for i in range(ii, i_end):
                            for j in range(jj, j_end):
                                s = 0.0
                                for k in range(kk, k_end):
                                    s += A_small[i, k] * B_small[k, j]
                                C[i, j] += s
            return C

        def numpy_blas():
            return A_small @ B_small

        t_naive = self._time_it(naive_small, num_runs=2)
        t_blocked = self._time_it(blocked_small, num_runs=2)
        t_blas = self._time_it(numpy_blas, num_runs=10)

        speedup_block = t_naive / t_blocked if t_blocked > 0 else float("inf")
        speedup_blas = t_naive / t_blas if t_blas > 0 else float("inf")

        print(f"  Naive   time : {t_naive:.4f} s  (size {small}x{small})")
        print(f"  Blocked time : {t_blocked:.4f} s  (block={bs})")
        print(f"  BLAS    time : {t_blas:.6f} s  (numpy @)")
        print(f"  Blocked speedup over naive : {speedup_block:.2f}x")
        print(f"  BLAS    speedup over naive : {speedup_blas:.1f}x")
        print("  Reason: Blocking keeps 3 tiles in L1 cache, avoiding")
        print("          column-stride misses on B.")

        return {
            "naive_s": t_naive,
            "blocked_s": t_blocked,
            "blas_s": t_blas,
            "speedup_blocked": speedup_block,
            "speedup_blas": speedup_blas,
        }

    # ==================================================================
    # 4. Memory alignment and SIMD
    # ==================================================================
    def demonstrate_memory_alignment(self) -> Dict[str, float]:
        """
        Show the effect of memory alignment on array operation speed.

        SIMD background
        ---------------
        Modern CPUs provide SIMD (Single Instruction, Multiple Data) extensions:
            - SSE  : 128-bit registers -> 2 float64 per instruction
            - AVX2 : 256-bit registers -> 4 float64 per instruction
            - AVX-512: 512-bit registers -> 8 float64 per instruction

        SIMD loads are most efficient when the source address is aligned to the
        register width (16, 32, or 64 bytes respectively). Unaligned loads
        historically required a separate micro-op and could cross cache-line
        boundaries, causing two cache reads instead of one.

        numpy arrays allocated through ``np.empty`` are 64-byte aligned by
        default (matching AVX-512 and cache-line width).  We deliberately
        create a misaligned view to show the difference.
        """
        print("\n" + "=" * 60)
        print("  Demo 4: Memory Alignment & SIMD")
        print("=" * 60)

        n = 1_000_000

        # Aligned array: numpy default allocation is 64-byte aligned
        aligned = np.random.randn(n).astype(np.float64)
        assert aligned.ctypes.data % 16 == 0, "Expected 16-byte alignment"

        # Create a deliberately misaligned view:
        # Allocate n+1 elements, then take a slice starting at byte offset 4
        # (half a float64).  This forces the array data to begin at an address
        # that is NOT 16-byte aligned.
        buf = np.random.randn(n + 1).view(np.uint8)  # raw bytes
        # Shift by 4 bytes to break alignment, then reinterpret
        misaligned_buf = np.empty(n * 8 + 4, dtype=np.uint8)
        misaligned_buf[4: 4 + n * 8] = aligned.view(np.uint8)
        misaligned = np.frombuffer(misaligned_buf[4: 4 + n * 8], dtype=np.float64)

        def op_aligned():
            return np.sum(aligned * aligned + 2.0 * aligned)

        def op_misaligned():
            return np.sum(misaligned * misaligned + 2.0 * misaligned)

        t_aligned = self._time_it(op_aligned, num_runs=20)
        t_misaligned = self._time_it(op_misaligned, num_runs=20)

        ratio = t_misaligned / t_aligned if t_aligned > 0 else float("inf")
        print(f"  Aligned    time : {t_aligned:.6f} s")
        print(f"  Misaligned time : {t_misaligned:.6f} s")
        print(f"  Ratio (mis/al)  : {ratio:.3f}")
        print("  Note: On modern CPUs (Haswell+) the penalty is small because")
        print("        hardware handles unaligned loads transparently. The penalty")
        print("        grows on older CPUs or when loads span cache-line boundaries.")
        print()
        print("  numpy aligned dtype example:")
        dt_aligned = np.dtype(
            [("pos", np.float64, 3), ("vel", np.float64, 3), ("mass", np.float64)],
            align=True,
        )
        dt_packed = np.dtype(
            [("pos", np.float64, 3), ("vel", np.float64, 3), ("mass", np.float64)],
            align=False,
        )
        print(f"    aligned struct size   : {dt_aligned.itemsize} bytes")
        print(f"    packed  struct size   : {dt_packed.itemsize} bytes")
        print(f"    alignment padding     : {dt_aligned.itemsize - dt_packed.itemsize} bytes")
        print("    Padding ensures each struct starts on a natural boundary,")
        print("    which keeps SIMD vector loads from spanning two cache lines.")

        return {
            "aligned_s": t_aligned,
            "misaligned_s": t_misaligned,
            "ratio": ratio,
            "aligned_struct_bytes": dt_aligned.itemsize,
            "packed_struct_bytes": dt_packed.itemsize,
        }

    # ==================================================================
    # 5. Branch prediction
    # ==================================================================
    def demonstrate_branch_prediction(self, n_elements: int = 500_000) -> Dict[str, float]:
        """
        The classic sorted-vs-unsorted conditional-sum benchmark.

        Branch prediction background
        ----------------------------
        CPUs speculatively execute instructions beyond a conditional branch
        *before* the branch condition is evaluated.  If the prediction is
        correct, execution continues at full speed.  If wrong, the pipeline
        must be flushed and restarted, costing 10-20 cycles on modern cores.

        The branch predictor learns patterns:
            - With sorted data, the branch `if val >= threshold` transitions
              from "always not taken" to "always taken" exactly once.
              After the transition point the predictor is essentially perfect.
            - With random data, the branch is unpredictable, causing ~50 %
              misprediction rate and constant pipeline flushes.

        Python note: CPython is an interpreter, so the branch-prediction
        effect is mixed with Python-level overhead. We use numpy boolean
        indexing to bring the demonstration closer to hardware:
            sorted_data[sorted_data >= threshold].sum()
        The internal C loop still exhibits the sorted-vs-unsorted difference,
        though the magnitude is smaller than in compiled C/C++.
        """
        print("\n" + "=" * 60)
        print(f"  Demo 5: Branch Prediction  (n = {n_elements:,})")
        print("=" * 60)

        data = np.random.randint(0, 256, size=n_elements).astype(np.int64)
        threshold = 128

        data_sorted = np.sort(data)
        data_unsorted = data.copy()

        # --- pure Python loop version (shows branch effect more directly) ---
        def cond_sum_python(arr):
            total = 0
            for val in arr:
                if val >= threshold:
                    total += val
            return total

        # --- numpy boolean-mask version (compiled inner loop) ---
        def cond_sum_numpy(arr):
            return arr[arr >= threshold].sum()

        # Python-loop comparison
        # Use small subset for Python loop since it is slow
        n_py = min(n_elements, 100_000)
        sorted_py = data_sorted[:n_py].tolist()
        unsorted_py = data_unsorted[:n_py].tolist()

        t_sorted_py = self._time_it(lambda: cond_sum_python(sorted_py), num_runs=3)
        t_unsorted_py = self._time_it(lambda: cond_sum_python(unsorted_py), num_runs=3)

        # numpy-mask comparison (full size)
        t_sorted_np = self._time_it(lambda: cond_sum_numpy(data_sorted), num_runs=10)
        t_unsorted_np = self._time_it(lambda: cond_sum_numpy(data_unsorted), num_runs=10)

        ratio_py = t_unsorted_py / t_sorted_py if t_sorted_py > 0 else float("inf")
        ratio_np = t_unsorted_np / t_sorted_np if t_sorted_np > 0 else float("inf")

        print(f"  Python loop ({n_py:,} elements):")
        print(f"    Sorted   : {t_sorted_py:.6f} s")
        print(f"    Unsorted : {t_unsorted_py:.6f} s")
        print(f"    Ratio    : {ratio_py:.2f}x")
        print(f"  Numpy mask ({n_elements:,} elements):")
        print(f"    Sorted   : {t_sorted_np:.6f} s")
        print(f"    Unsorted : {t_unsorted_np:.6f} s")
        print(f"    Ratio    : {ratio_np:.2f}x")
        print("  Insight: In pure Python the interpreter overhead dominates,")
        print("           muting the branch-prediction signal. In numpy's C")
        print("           inner loop the effect is visible but smaller than in")
        print("           hand-written C because numpy uses branchless SIMD")
        print("           where possible.")

        return {
            "sorted_py_s": t_sorted_py,
            "unsorted_py_s": t_unsorted_py,
            "ratio_py": ratio_py,
            "sorted_np_s": t_sorted_np,
            "unsorted_np_s": t_unsorted_np,
            "ratio_np": ratio_np,
        }

    # ==================================================================
    # 6. Prefetching pattern: sequential vs random access
    # ==================================================================
    def demonstrate_prefetching_pattern(self, n_elements: int = 2_000_000) -> Dict[str, float]:
        """
        Sequential access vs random access in a large array.

        Hardware prefetching
        --------------------
        Modern CPUs detect *constant-stride* access patterns and issue
        prefetch requests to the memory controller several cache lines ahead.
        Intel's L2 streamer, for instance, can track up to 32 independent
        streams.

        Sequential access:
            Stride = 8 bytes (one float64).  The prefetcher sees a steady
            stream and keeps the pipeline fed.  Effective bandwidth approaches
            the theoretical DRAM bandwidth.

        Random access:
            Each access lands on a different cache line (with high probability
            for arrays larger than L3).  The prefetcher cannot find a pattern.
            Every access is a full DRAM round-trip (~60-100 ns), reducing
            effective bandwidth to a tiny fraction of peak.

        For GNC systems, this matters when accessing ephemeris tables:
        sequential epoch queries (time-marching) are fast; random historical
        lookups require index structures to compensate.
        """
        print("\n" + "=" * 60)
        print(f"  Demo 6: Prefetching Pattern  (n = {n_elements:,})")
        print("=" * 60)

        data = np.random.randn(n_elements)

        # Number of accesses (same for both patterns so work is identical)
        n_accesses = min(n_elements, 500_000)

        sequential_indices = np.arange(n_accesses, dtype=np.intp)
        random_indices = np.random.randint(0, n_elements, size=n_accesses).astype(np.intp)

        def sequential_sum():
            """
            Access data[0], data[1], data[2], ...
            Stride = 8 B -> cache line reuse + prefetcher engaged.
            """
            return data[sequential_indices].sum()

        def random_sum():
            """
            Access data[rand], data[rand], data[rand], ...
            No detectable stride -> prefetcher helpless -> constant cache misses.
            """
            return data[random_indices].sum()

        t_seq = self._time_it(sequential_sum, num_runs=10)
        t_rand = self._time_it(random_sum, num_runs=10)

        ratio = t_rand / t_seq if t_seq > 0 else float("inf")
        print(f"  Sequential time : {t_seq:.6f} s")
        print(f"  Random     time : {t_rand:.6f} s")
        print(f"  Random / Seq    : {ratio:.2f}x slower")
        print("  Reason: Sequential access lets the HW prefetcher run ahead;")
        print("          random access defeats it and every load stalls on DRAM.")

        return {"sequential_s": t_seq, "random_s": t_rand, "ratio": ratio}

    # ==================================================================
    # Run all and produce plots
    # ==================================================================
    def run_all_demonstrations(self, output_dir: str = "cache_results") -> Dict[str, Dict]:
        """
        Execute every demonstration and write timing plots to *output_dir*.

        Returns
        -------
        dict mapping demonstration name -> result dict
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {}
        results["soa_vs_aos"] = self.demonstrate_soa_vs_aos()
        results["row_vs_col"] = self.demonstrate_row_vs_column_access()
        results["blocking"] = self.demonstrate_blocking()
        results["alignment"] = self.demonstrate_memory_alignment()
        results["branch_pred"] = self.demonstrate_branch_prediction()
        results["prefetching"] = self.demonstrate_prefetching_pattern()

        # ---- Summary bar chart of speedups / ratios ----
        labels = []
        speedups = []

        # Extract the most meaningful ratio from each demo
        mapping = {
            "SoA vs AoS": results["soa_vs_aos"].get("speedup", 1),
            "Row vs Col": results["row_vs_col"].get("speedup", 1),
            "Blocked vs Naive": results["blocking"].get("speedup_blocked", 1),
            "BLAS vs Naive": results["blocking"].get("speedup_blas", 1),
            "Seq vs Random": results["prefetching"].get("ratio", 1),
            "Branch (py)": results["branch_pred"].get("ratio_py", 1),
        }
        for label, value in mapping.items():
            labels.append(label)
            speedups.append(value)

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, speedups, color="teal", edgecolor="black")
        ax.set_ylabel("Speedup / Slowdown factor (x)")
        ax.set_title("Cache & Micro-Architecture Effects")
        ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8)
        for bar, val in zip(bars, speedups):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val:.1f}x",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "cache_effects.png"), dpi=150)
        plt.close(fig)

        # ---- Individual paired bar charts ----

        # SoA vs AoS
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["AoS", "SoA"], [results["soa_vs_aos"]["aos_s"],
                                  results["soa_vs_aos"]["soa_s"]],
               color=["salmon", "mediumseagreen"], edgecolor="black")
        ax.set_ylabel("Time (s)")
        ax.set_title("SoA vs AoS Distance Computation")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "soa_vs_aos.png"), dpi=150)
        plt.close(fig)

        # Row vs Column
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["Row-major", "Col-major"],
               [results["row_vs_col"]["row_major_s"],
                results["row_vs_col"]["col_major_s"]],
               color=["mediumseagreen", "salmon"], edgecolor="black")
        ax.set_ylabel("Time (s)")
        ax.set_title("Row vs Column Access Order")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "row_vs_col.png"), dpi=150)
        plt.close(fig)

        # Sequential vs Random
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["Sequential", "Random"],
               [results["prefetching"]["sequential_s"],
                results["prefetching"]["random_s"]],
               color=["mediumseagreen", "salmon"], edgecolor="black")
        ax.set_ylabel("Time (s)")
        ax.set_title("Sequential vs Random Access (Prefetching)")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "prefetching.png"), dpi=150)
        plt.close(fig)

        print("\n" + "=" * 60)
        print(f"  All plots saved to {os.path.abspath(output_dir)}")
        print("=" * 60)

        return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "cache_results"
    analyzer = CacheAnalysis()
    all_results = analyzer.run_all_demonstrations(output_dir=out)
    print("\nFinal results summary:")
    for name, data in all_results.items():
        print(f"  {name}: {data}")
