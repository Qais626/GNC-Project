"""
benchmarks.py - Performance Benchmarking Suite for GNC Algorithms

Provides a rigorous benchmarking framework that quantifies the runtime and memory
impact of optimization choices across six categories of GNC-relevant computation:

    1. Matrix operations     - Naive Python loops vs numpy vectorized
    2. Propagator comparison - Loop-based RK4 vs vectorized RK4
    3. Data layout           - Struct-of-Arrays vs Array-of-Structs
    4. Search algorithms     - Linear search vs KD-tree (O(n) vs O(log n))
    5. Memory allocation     - Pre-allocated arrays vs dynamic list appending
    6. Sorting               - Python sort vs numpy sort vs custom quicksort

Every benchmark returns structured timing data so results are reproducible and
can be aggregated into summary tables and comparison plots.

Hardware context
----------------
Modern CPUs execute billions of operations per second, but only when the data
they need is already in cache. A single L1 cache miss costs ~4 ns; an L2 miss
costs ~10 ns; a main-memory fetch costs ~100 ns.  Numpy keeps data in contiguous
C arrays and dispatches to BLAS/LAPACK, which means the CPU's prefetcher,
SIMD units, and out-of-order engine all work at full efficiency. Pure-Python
loops, by contrast, chase pointers through a heap of boxed PyObjects, defeating
every one of those hardware optimizations.

The benchmarks below make these costs visible.
"""

from __future__ import annotations

import os
import sys
import time
import statistics
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server / CI environments
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Benchmark utility class
# ---------------------------------------------------------------------------

class Benchmark:
    """
    General-purpose benchmarking harness.

    Provides timing (wall-clock), memory profiling (via tracemalloc), and
    side-by-side comparison of two implementations.  All public methods
    return plain dicts or DataFrames so callers can serialise, plot, or
    aggregate results however they wish.
    """

    # ---- Core measurement helpers ----------------------------------------

    @staticmethod
    def time_function(func: Callable, *args, num_runs: int = 100, **kwargs) -> Dict[str, float]:
        """
        Time *func* over *num_runs* invocations and return descriptive statistics.

        Returns
        -------
        dict with keys: min, max, mean, median, std, total, num_runs
            All times are in **seconds**.
        """
        times: List[float] = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            func(*args, **kwargs)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        return {
            "min": min(times),
            "max": max(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0.0,
            "total": sum(times),
            "num_runs": num_runs,
        }

    @staticmethod
    def memory_profile(func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Measure peak memory and number of allocation events for *func*.

        Uses :mod:`tracemalloc` to take two snapshots (before / after) and
        compute the delta.

        Returns
        -------
        dict with keys: peak_bytes, peak_kb, peak_mb, current_bytes,
                        num_allocations (top-level block count in snapshot)
        """
        tracemalloc.start()
        tracemalloc.reset_peak()

        func(*args, **kwargs)

        current, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = snapshot.statistics("lineno")

        return {
            "peak_bytes": peak,
            "peak_kb": peak / 1024,
            "peak_mb": peak / (1024 * 1024),
            "current_bytes": current,
            "num_allocations": len(stats),
        }

    @staticmethod
    def compare(
        func_a: Callable,
        func_b: Callable,
        *args,
        labels: Tuple[str, str] = ("A", "B"),
        num_runs: int = 100,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Run two functions on the same arguments and return a DataFrame that
        puts their timing statistics side by side.

        An extra 'speedup' row shows how many times faster *func_b* is relative
        to *func_a* (mean time ratio).
        """
        stats_a = Benchmark.time_function(func_a, *args, num_runs=num_runs, **kwargs)
        stats_b = Benchmark.time_function(func_b, *args, num_runs=num_runs, **kwargs)

        df = pd.DataFrame({labels[0]: stats_a, labels[1]: stats_b})
        # Compute speedup (how many X faster B is compared to A based on mean)
        if stats_b["mean"] > 0:
            df.loc["speedup"] = [stats_a["mean"] / stats_b["mean"], 1.0]
        return df

    # ---- Benchmark scenarios --------------------------------------------

    @staticmethod
    def benchmark_matrix_operations(num_runs: int = 500) -> pd.DataFrame:
        """
        Scenario 1 -- Matrix operations
        Compare naive Python-loop implementations vs numpy-vectorized versions
        for 3x3 matrix multiply, cross product, and quaternion multiply.

        Typical result: numpy is 10-100x faster because it dispatches to
        compiled C / Fortran BLAS routines that exploit SIMD (AVX-2/512) and
        avoid Python's per-element boxing overhead.
        """
        # -- 3x3 matrix multiply ------------------------------------------
        def matmul_naive(A: list, B: list) -> list:
            """Triple nested loop -- O(n^3) with Python overhead per element."""
            n = len(A)
            C = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    s = 0.0
                    for k in range(n):
                        s += A[i][k] * B[k][j]
                    C[i][j] = s
            return C

        def matmul_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            """Single numpy call -- dispatches to optimised BLAS dgemm."""
            return A @ B

        A_list = [[float(i * 3 + j) for j in range(3)] for i in range(3)]
        B_list = [[float(i * 3 + j + 1) for j in range(3)] for i in range(3)]
        A_np = np.array(A_list, dtype=np.float64)
        B_np = np.array(B_list, dtype=np.float64)

        matmul_cmp = Benchmark.compare(
            lambda: matmul_naive(A_list, B_list),
            lambda: matmul_numpy(A_np, B_np),
            labels=("naive_matmul", "numpy_matmul"),
            num_runs=num_runs,
        )

        # -- Cross product -------------------------------------------------
        def cross_naive(a: list, b: list) -> list:
            return [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]

        def cross_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.cross(a, b)

        a_list = [1.0, 2.0, 3.0]
        b_list = [4.0, 5.0, 6.0]
        a_np = np.array(a_list)
        b_np = np.array(b_list)

        cross_cmp = Benchmark.compare(
            lambda: cross_naive(a_list, b_list),
            lambda: cross_numpy(a_np, b_np),
            labels=("naive_cross", "numpy_cross"),
            num_runs=num_runs,
        )

        # -- Quaternion multiply -------------------------------------------
        def quat_mul_naive(q1: list, q2: list) -> list:
            """Hamilton product using scalar arithmetic."""
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]

        def quat_mul_numpy(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
            """Hamilton product using numpy element-wise ops + slicing."""
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ])

        q1_list = [1.0, 0.0, 0.0, 0.0]
        q2_list = [0.707, 0.707, 0.0, 0.0]
        q1_np = np.array(q1_list)
        q2_np = np.array(q2_list)

        quat_cmp = Benchmark.compare(
            lambda: quat_mul_naive(q1_list, q2_list),
            lambda: quat_mul_numpy(q1_np, q2_np),
            labels=("naive_quat", "numpy_quat"),
            num_runs=num_runs,
        )

        # Combine into a single multi-indexed DataFrame
        combined = pd.concat(
            {"matmul": matmul_cmp, "cross": cross_cmp, "quat_mul": quat_cmp},
            axis=0,
        )
        print("\n=== Scenario 1: Matrix Operations ===")
        print(combined.to_string())
        return combined

    # ------------------------------------------------------------------

    @staticmethod
    def benchmark_propagator(num_steps: int = 1000, num_runs: int = 20) -> pd.DataFrame:
        """
        Scenario 2 -- RK4 propagator: Python loops vs numpy vectorized.

        We propagate a simple 2-body orbit (mu = 398600.4418 km^3/s^2) for
        *num_steps* fixed-size steps.

        The loop-based version calls the derivative function and updates the
        state one element at a time.  The vectorized version keeps the entire
        state in a (6,) numpy array and performs all arithmetic with numpy
        ufuncs, which the CPU can pipeline through its SIMD units.
        """
        mu = 398_600.4418  # Earth gravitational parameter [km^3/s^2]
        dt = 10.0  # time step [s]

        # --- derivative function (pure python) ---
        def deriv_loop(state: list) -> list:
            x, y, z, vx, vy, vz = state
            r = (x * x + y * y + z * z) ** 0.5
            r3 = r * r * r
            ax = -mu * x / r3
            ay = -mu * y / r3
            az = -mu * z / r3
            return [vx, vy, vz, ax, ay, az]

        def rk4_loop(state: list, n: int) -> list:
            for _ in range(n):
                k1 = deriv_loop(state)
                s2 = [state[i] + 0.5 * dt * k1[i] for i in range(6)]
                k2 = deriv_loop(s2)
                s3 = [state[i] + 0.5 * dt * k2[i] for i in range(6)]
                k3 = deriv_loop(s3)
                s4 = [state[i] + dt * k3[i] for i in range(6)]
                k4 = deriv_loop(s4)
                state = [
                    state[i] + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
                    for i in range(6)
                ]
            return state

        # --- derivative function (numpy) ---
        def deriv_numpy(state: np.ndarray) -> np.ndarray:
            pos = state[:3]
            vel = state[3:]
            r = np.linalg.norm(pos)
            acc = -mu * pos / (r ** 3)
            return np.concatenate([vel, acc])

        def rk4_numpy(state: np.ndarray, n: int) -> np.ndarray:
            for _ in range(n):
                k1 = deriv_numpy(state)
                k2 = deriv_numpy(state + 0.5 * dt * k1)
                k3 = deriv_numpy(state + 0.5 * dt * k2)
                k4 = deriv_numpy(state + dt * k3)
                state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            return state

        state_list = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]
        state_np = np.array(state_list, dtype=np.float64)

        cmp = Benchmark.compare(
            lambda: rk4_loop(list(state_list), num_steps),
            lambda: rk4_numpy(state_np.copy(), num_steps),
            labels=("loop_rk4", "numpy_rk4"),
            num_runs=num_runs,
        )
        print("\n=== Scenario 2: Propagator Comparison ===")
        print(cmp.to_string())
        return cmp

    # ------------------------------------------------------------------

    @staticmethod
    def benchmark_data_layout(n_particles: int = 50_000, num_runs: int = 20) -> pd.DataFrame:
        """
        Scenario 3 -- Struct-of-Arrays (SoA) vs Array-of-Structs (AoS).

        AoS: a Python list of dicts -- each dict holds {x, y, z, vx, vy, vz}.
             Iterating over x-coordinates alone must chase N pointers, pulling
             in y, z, vx, vy, vz into the cache line even though they are unused.

        SoA: a dict of numpy arrays -- {x: array, y: array, ...}.
             All x-values sit in contiguous memory, so a sweep over x fills
             every cache line with useful data (spatial locality).

        We compute a simple per-particle metric: distance = sqrt(x^2 + y^2 + z^2).
        """
        # Build AoS
        aos = [
            {
                "x": np.random.randn(),
                "y": np.random.randn(),
                "z": np.random.randn(),
                "vx": np.random.randn(),
                "vy": np.random.randn(),
                "vz": np.random.randn(),
            }
            for _ in range(n_particles)
        ]

        # Build SoA (contiguous float64 arrays)
        soa = {
            "x": np.random.randn(n_particles),
            "y": np.random.randn(n_particles),
            "z": np.random.randn(n_particles),
            "vx": np.random.randn(n_particles),
            "vy": np.random.randn(n_particles),
            "vz": np.random.randn(n_particles),
        }

        def compute_distance_aos(data: list) -> list:
            return [(p["x"] ** 2 + p["y"] ** 2 + p["z"] ** 2) ** 0.5 for p in data]

        def compute_distance_soa(data: dict) -> np.ndarray:
            return np.sqrt(data["x"] ** 2 + data["y"] ** 2 + data["z"] ** 2)

        cmp = Benchmark.compare(
            lambda: compute_distance_aos(aos),
            lambda: compute_distance_soa(soa),
            labels=("AoS", "SoA"),
            num_runs=num_runs,
        )
        print("\n=== Scenario 3: Data Layout (AoS vs SoA) ===")
        print(cmp.to_string())
        return cmp

    # ------------------------------------------------------------------

    @staticmethod
    def benchmark_search(n_bodies: int = 100_000, num_runs: int = 50) -> pd.DataFrame:
        """
        Scenario 4 -- Search algorithms for nearest body lookup.

        Linear search: O(n) -- iterate through every body to find the nearest.
        KD-tree search: O(log n) -- spatial partitioning via scipy.spatial.KDTree.

        For GNC, nearest-body lookup is needed when switching dominant gravity
        sources during interplanetary transfer (patched conics) or when
        selecting which ephemeris to query for perturbation forces.
        """
        from scipy.spatial import KDTree

        positions = np.random.randn(n_bodies, 3) * 1e6  # random 3-D positions
        query = np.random.randn(3) * 1e6

        def linear_search(pos: np.ndarray, q: np.ndarray) -> int:
            best_idx = 0
            best_d2 = float("inf")
            for i in range(len(pos)):
                dx = pos[i, 0] - q[0]
                dy = pos[i, 1] - q[1]
                dz = pos[i, 2] - q[2]
                d2 = dx * dx + dy * dy + dz * dz
                if d2 < best_d2:
                    best_d2 = d2
                    best_idx = i
            return best_idx

        tree = KDTree(positions)

        def kdtree_search(t: KDTree, q: np.ndarray) -> int:
            _, idx = t.query(q)
            return idx

        cmp = Benchmark.compare(
            lambda: linear_search(positions, query),
            lambda: kdtree_search(tree, query),
            labels=("linear_O(n)", "kdtree_O(logn)"),
            num_runs=num_runs,
        )
        print("\n=== Scenario 4: Search Algorithms ===")
        print(cmp.to_string())
        return cmp

    # ------------------------------------------------------------------

    @staticmethod
    def benchmark_memory_allocation(n_steps: int = 100_000, num_runs: int = 20) -> pd.DataFrame:
        """
        Scenario 5 -- Pre-allocated numpy arrays vs dynamic list appending.

        Dynamic lists:
            - Each ``append`` may trigger a realloc + copy of the underlying
              buffer when capacity is exceeded.
            - The garbage collector must periodically scan all live PyObjects,
              and the growing list creates GC pressure.

        Pre-allocated arrays:
            - A single allocation up-front; subsequent writes are simple
              pointer-offset stores with zero GC involvement.
            - The entire array is contiguous, so the hardware prefetcher
              can stream data efficiently.
        """

        def dynamic_append(n: int) -> list:
            result = []
            for i in range(n):
                result.append(float(i) * 0.1)
            return result

        def preallocated(n: int) -> np.ndarray:
            result = np.empty(n, dtype=np.float64)
            for i in range(n):
                result[i] = float(i) * 0.1
            return result

        def fully_vectorized(n: int) -> np.ndarray:
            """Best case: no Python loop at all."""
            return np.arange(n, dtype=np.float64) * 0.1

        cmp_prealloc = Benchmark.compare(
            lambda: dynamic_append(n_steps),
            lambda: preallocated(n_steps),
            labels=("dynamic_list", "prealloc_array"),
            num_runs=num_runs,
        )

        cmp_vectorized = Benchmark.compare(
            lambda: dynamic_append(n_steps),
            lambda: fully_vectorized(n_steps),
            labels=("dynamic_list", "vectorized"),
            num_runs=num_runs,
        )

        combined = pd.concat(
            {"prealloc": cmp_prealloc, "vectorized": cmp_vectorized}, axis=0
        )
        print("\n=== Scenario 5: Memory Allocation ===")
        print(combined.to_string())
        return combined

    # ------------------------------------------------------------------

    @staticmethod
    def benchmark_sorting(n_elements: int = 100_000, num_runs: int = 30) -> pd.DataFrame:
        """
        Scenario 6 -- Sorting telemetry timestamps / values.

        Python built-in sort (Timsort): optimised for real-world data with
        partially-sorted runs.  Operates on boxed PyObjects, so comparisons
        involve pointer indirection.

        numpy.sort: operates on contiguous typed arrays; uses introsort
        (quicksort + heapsort fallback) and avoids Python object overhead.

        Custom quicksort: in-place pure-Python quicksort to show the cost
        of doing the same algorithmic work without C-level optimisation.
        """
        data_list = list(np.random.randn(n_elements))
        data_np = np.array(data_list, dtype=np.float64)

        def python_sort(d: list) -> list:
            return sorted(d)

        def numpy_sort(d: np.ndarray) -> np.ndarray:
            return np.sort(d)

        def quicksort_python(arr: list) -> list:
            """Naive recursive quicksort (creates new lists -- not in-place)."""
            if len(arr) <= 1:
                return arr
            pivot = arr[len(arr) // 2]
            left = [x for x in arr if x < pivot]
            mid = [x for x in arr if x == pivot]
            right = [x for x in arr if x > pivot]
            return quicksort_python(left) + mid + quicksort_python(right)

        cmp_py_np = Benchmark.compare(
            lambda: python_sort(list(data_list)),
            lambda: numpy_sort(data_np.copy()),
            labels=("python_sort", "numpy_sort"),
            num_runs=num_runs,
        )
        cmp_custom = Benchmark.compare(
            lambda: quicksort_python(list(data_list)),
            lambda: numpy_sort(data_np.copy()),
            labels=("custom_qsort", "numpy_sort"),
            num_runs=num_runs,
        )
        combined = pd.concat(
            {"py_vs_np": cmp_py_np, "custom_vs_np": cmp_custom}, axis=0
        )
        print("\n=== Scenario 6: Sorting ===")
        print(combined.to_string())
        return combined

    # ---- Orchestration ---------------------------------------------------

    @staticmethod
    def run_all_benchmarks(output_dir: str = "benchmark_results") -> pd.DataFrame:
        """
        Execute every benchmark scenario and consolidate into a summary table.

        Parameters
        ----------
        output_dir : str
            Directory where result CSVs and plots will be saved.

        Returns
        -------
        pd.DataFrame
            A summary table with one row per scenario showing the naive mean,
            optimised mean, and speedup factor.
        """
        os.makedirs(output_dir, exist_ok=True)

        scenarios = {
            "matrix_ops": Benchmark.benchmark_matrix_operations,
            "propagator": Benchmark.benchmark_propagator,
            "data_layout": Benchmark.benchmark_data_layout,
            "search": Benchmark.benchmark_search,
            "memory_alloc": Benchmark.benchmark_memory_allocation,
            "sorting": Benchmark.benchmark_sorting,
        }

        all_results: Dict[str, pd.DataFrame] = {}
        summary_rows: List[Dict[str, Any]] = []

        for name, func in scenarios.items():
            print(f"\n{'='*60}")
            print(f"  Running: {name}")
            print(f"{'='*60}")
            df = func()
            all_results[name] = df

            # Save per-scenario CSV
            csv_path = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(csv_path)

            # Extract headline speedup (top-level mean row, first two columns)
            # For multi-indexed frames we grab the first sub-frame
            if isinstance(df.index, pd.MultiIndex):
                first_key = df.index.get_level_values(0).unique()[0]
                sub = df.loc[first_key]
            else:
                sub = df

            cols = sub.columns.tolist()
            if "speedup" in sub.index:
                speedup = sub.loc["speedup", cols[0]]
            else:
                speedup = float("nan")

            naive_mean = sub.loc["mean", cols[0]] if "mean" in sub.index else float("nan")
            opt_mean = sub.loc["mean", cols[1]] if "mean" in sub.index else float("nan")

            summary_rows.append(
                {
                    "scenario": name,
                    "naive_mean_s": naive_mean,
                    "optimized_mean_s": opt_mean,
                    "speedup_x": speedup,
                }
            )

        summary = pd.DataFrame(summary_rows).set_index("scenario")
        summary.to_csv(os.path.join(output_dir, "summary.csv"))

        # --- Generate bar chart of speedups --------------------------------
        fig, ax = plt.subplots(figsize=(10, 5))
        summary["speedup_x"].plot.bar(ax=ax, color="steelblue", edgecolor="black")
        ax.set_ylabel("Speedup (x)")
        ax.set_title("Optimization Speedup by Scenario")
        ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="baseline")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "speedup_bar.png"), dpi=150)
        plt.close(fig)

        # --- Generate timing comparison grouped bar chart -------------------
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(summary))
        width = 0.35
        ax.bar(x - width / 2, summary["naive_mean_s"], width, label="Naive", color="salmon")
        ax.bar(x + width / 2, summary["optimized_mean_s"], width, label="Optimized", color="mediumseagreen")
        ax.set_xticks(x)
        ax.set_xticklabels(summary.index, rotation=30, ha="right")
        ax.set_ylabel("Mean time (s)")
        ax.set_title("Naive vs Optimized Mean Execution Time")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "timing_comparison.png"), dpi=150)
        plt.close(fig)

        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        print(summary.to_string())
        return summary

    # ---- Markdown report -------------------------------------------------

    @staticmethod
    def generate_report(output_dir: str = "benchmark_results") -> str:
        """
        Generate a Markdown report referencing the CSVs and plots created by
        :meth:`run_all_benchmarks`.

        Returns
        -------
        str
            The Markdown text (also written to ``output_dir/report.md``).
        """
        summary_path = os.path.join(output_dir, "summary.csv")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(
                f"{summary_path} not found -- run run_all_benchmarks first."
            )

        summary = pd.read_csv(summary_path, index_col="scenario")

        lines = [
            "# GNC Performance Benchmark Report",
            "",
            "## Summary",
            "",
            "| Scenario | Naive Mean (s) | Optimized Mean (s) | Speedup |",
            "|----------|---------------:|-------------------:|--------:|",
        ]
        for scenario, row in summary.iterrows():
            lines.append(
                f"| {scenario} | {row['naive_mean_s']:.6f} | "
                f"{row['optimized_mean_s']:.6f} | {row['speedup_x']:.1f}x |"
            )

        lines += [
            "",
            "## Speedup Chart",
            "",
            "![Speedup](speedup_bar.png)",
            "",
            "## Timing Comparison",
            "",
            "![Timing](timing_comparison.png)",
            "",
            "## Key Takeaways",
            "",
            "1. **Numpy vectorization** eliminates Python-object overhead and enables "
            "SIMD execution, yielding 10-100x speedups for element-wise math.",
            "2. **Struct-of-Arrays** data layout keeps each field in contiguous memory, "
            "maximising cache-line utilisation and prefetcher effectiveness.",
            "3. **KD-trees** reduce nearest-neighbour search from O(n) to O(log n), "
            "which is critical when the body catalogue grows.",
            "4. **Pre-allocated arrays** avoid repeated realloc/copy and reduce GC "
            "pressure, giving predictable latency.",
            "5. **Numpy sort** operates on contiguous typed data and outperforms "
            "both Python's Timsort and a naive quicksort for large numeric arrays.",
            "",
            "---",
            "*Report generated by benchmarks.py*",
        ]

        report = "\n".join(lines)
        report_path = os.path.join(output_dir, "report.md")
        with open(report_path, "w") as fh:
            fh.write(report)
        print(f"Report written to {report_path}")
        return report


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow overriding output directory from the command line
    out = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results"
    summary = Benchmark.run_all_benchmarks(output_dir=out)
    Benchmark.generate_report(output_dir=out)
    print("\nDone.  Results saved to:", os.path.abspath(out))
