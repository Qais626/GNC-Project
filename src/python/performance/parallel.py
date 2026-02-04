"""
parallel.py - Parallelization & Vectorization for GNC Computations

Three classes at different levels of the parallelism hierarchy:

    VectorizedOps  - SIMD-style data-parallel operations via numpy broadcasting.
                     A single CPU core processes multiple data elements per clock
                     cycle using AVX2/AVX-512 vector instructions under the hood.

    ParallelSim    - Multi-core parallelism via Python's multiprocessing module.
                     Embarrassingly parallel workloads (Monte Carlo, grid search,
                     telemetry map-reduce) are distributed across physical cores.

    GPUStub        - Annotated stubs showing *how* GPU (CUDA) acceleration would
                     be structured, using numpy as a stand-in. Extensive comments
                     explain the GPU memory model, warp execution, coalesced
                     access, and when GPU acceleration helps vs hurts.

Why this matters for GNC
------------------------
Spacecraft GNC processing spans a wide range of computational profiles:

    - Real-time attitude control   : microsecond latency, single-core, must be
                                     deterministic -> vectorized numpy is ideal.
    - Monte Carlo dispersion       : thousands of independent trajectories ->
                                     embarrassingly parallel -> multiprocessing.
    - Trajectory optimisation      : large linear-algebra solves (collocation) ->
                                     GPU BLAS can deliver 10-50x speedups.
    - Telemetry post-processing    : gigabytes of time-series data -> map-reduce
                                     pattern over chunked files.

Each class below addresses one of these profiles.
"""

from __future__ import annotations

import os
import sys
import time
import functools
import multiprocessing
from multiprocessing import Pool, Array as SharedArray
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ========================================================================
# VectorizedOps -- SIMD-style batch operations via numpy
# ========================================================================

class VectorizedOps:
    """
    Batch GNC math using numpy broadcasting to exploit SIMD hardware.

    All ``batch_*`` methods operate on arrays whose first axis is the batch
    dimension N.  Internally numpy dispatches to compiled C loops that the
    compiler auto-vectorises with SSE/AVX intrinsics, processing 4-8 float64
    values per clock cycle.

    Each method includes a ``_sequential`` counterpart that processes the same
    data one element at a time in a Python loop, so the timing comparison
    makes the vectorisation benefit concrete.
    """

    # ----------------------------------------------------------------
    # Quaternion multiply (batch)
    # ----------------------------------------------------------------
    @staticmethod
    def batch_quaternion_multiply(
        q1: np.ndarray, q2: np.ndarray
    ) -> np.ndarray:
        """
        Hamilton product of N quaternion pairs.

        Parameters
        ----------
        q1, q2 : ndarray, shape (N, 4)
            Quaternions in [w, x, y, z] convention.

        Returns
        -------
        ndarray, shape (N, 4)

        Implementation note
        -------------------
        Each line below is a single numpy ufunc call that operates on the
        full length-N column.  On a CPU with AVX-512 this means 8 multiplies
        execute per clock cycle per line.
        """
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        return np.column_stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    @staticmethod
    def _sequential_quaternion_multiply(
        q1: np.ndarray, q2: np.ndarray
    ) -> np.ndarray:
        """One-at-a-time Python loop version for comparison."""
        n = q1.shape[0]
        out = np.empty((n, 4), dtype=np.float64)
        for i in range(n):
            w1, x1, y1, z1 = q1[i]
            w2, x2, y2, z2 = q2[i]
            out[i, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            out[i, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            out[i, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            out[i, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return out

    # ----------------------------------------------------------------
    # Rotate vectors by quaternions (batch)
    # ----------------------------------------------------------------
    @staticmethod
    def batch_rotate_vectors(
        q: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """
        Rotate N vectors by N quaternions using the sandwich product v' = q v q*.

        Implemented with the optimised formula that avoids forming the full
        rotation matrix:
            t = 2 * cross(q_vec, v)
            v' = v + q_w * t + cross(q_vec, t)

        Parameters
        ----------
        q : ndarray, shape (N, 4)  -- [w, x, y, z]
        v : ndarray, shape (N, 3)

        Returns
        -------
        ndarray, shape (N, 3)
        """
        q_w = q[:, 0:1]        # (N, 1) for broadcasting
        q_vec = q[:, 1:4]      # (N, 3)

        t = 2.0 * np.cross(q_vec, v)
        return v + q_w * t + np.cross(q_vec, t)

    @staticmethod
    def _sequential_rotate_vectors(
        q: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """One-at-a-time loop version."""
        n = q.shape[0]
        out = np.empty((n, 3), dtype=np.float64)
        for i in range(n):
            q_w = q[i, 0]
            q_vec = q[i, 1:4]
            t = 2.0 * np.cross(q_vec, v[i])
            out[i] = v[i] + q_w * t + np.cross(q_vec, t)
        return out

    # ----------------------------------------------------------------
    # Gravity acceleration (batch)
    # ----------------------------------------------------------------
    @staticmethod
    def batch_gravity_acceleration(
        positions: np.ndarray,
        mu: float = 398_600.4418,
        J2: float = 1.08263e-3,
        R: float = 6378.137,
    ) -> np.ndarray:
        """
        Compute two-body + J2 gravitational acceleration for N positions.

        Parameters
        ----------
        positions : ndarray, shape (N, 3)  -- [x, y, z] in km
        mu : float -- gravitational parameter [km^3/s^2]
        J2 : float -- second zonal harmonic
        R  : float -- reference body radius [km]

        Returns
        -------
        ndarray, shape (N, 3) -- acceleration [km/s^2]

        Vectorisation note
        ------------------
        All operations below broadcast across the N-axis.  The inner loop is
        a tight sequence of multiplies and adds that the CPU pipelines through
        its FMA (fused multiply-add) units.
        """
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        r2 = r ** 2
        r5 = r ** 5

        # Two-body term
        a_two = -mu / (r ** 3)

        # J2 perturbation terms
        factor = 1.5 * J2 * mu * R ** 2 / r5
        z2_over_r2 = z ** 2 / r2

        ax = a_two * x + factor * x * (5.0 * z2_over_r2 - 1.0)
        ay = a_two * y + factor * y * (5.0 * z2_over_r2 - 1.0)
        az = a_two * z + factor * z * (5.0 * z2_over_r2 - 3.0)

        return np.column_stack([ax, ay, az])

    @staticmethod
    def _sequential_gravity_acceleration(
        positions: np.ndarray,
        mu: float = 398_600.4418,
        J2: float = 1.08263e-3,
        R: float = 6378.137,
    ) -> np.ndarray:
        """One-at-a-time loop version."""
        n = positions.shape[0]
        out = np.empty((n, 3), dtype=np.float64)
        for i in range(n):
            x, y, z = positions[i]
            r = (x ** 2 + y ** 2 + z ** 2) ** 0.5
            r2 = r * r
            r3 = r2 * r
            r5 = r2 * r3
            a_two = -mu / r3
            factor = 1.5 * J2 * mu * R ** 2 / r5
            z2r2 = z * z / r2
            out[i, 0] = a_two * x + factor * x * (5.0 * z2r2 - 1.0)
            out[i, 1] = a_two * y + factor * y * (5.0 * z2r2 - 1.0)
            out[i, 2] = a_two * z + factor * z * (5.0 * z2r2 - 3.0)
        return out

    # ----------------------------------------------------------------
    # Vectorised RK4 step (batch)
    # ----------------------------------------------------------------
    @staticmethod
    def batch_rk4_step(
        states: np.ndarray,
        dt: float,
        force_func: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """
        Advance N state vectors by one RK4 step simultaneously.

        Parameters
        ----------
        states : ndarray, shape (N, M)
            Each row is an M-dimensional state vector [pos; vel; ...].
        dt : float
            Time step.
        force_func : callable
            Derivative function  f(states) -> ndarray (N, M).
            Must accept and return batched arrays.

        Returns
        -------
        ndarray, shape (N, M) -- updated states

        Vectorisation note
        ------------------
        The four derivative evaluations each process all N states in a single
        numpy call.  The final weighted sum is a single array expression.
        """
        k1 = force_func(states)
        k2 = force_func(states + 0.5 * dt * k1)
        k3 = force_func(states + 0.5 * dt * k2)
        k4 = force_func(states + dt * k3)
        return states + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    @staticmethod
    def _sequential_rk4_step(
        states: np.ndarray,
        dt: float,
        force_func_single: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """One-at-a-time loop version. force_func_single takes a single (M,) state."""
        n = states.shape[0]
        out = np.empty_like(states)
        for i in range(n):
            s = states[i]
            k1 = force_func_single(s)
            k2 = force_func_single(s + 0.5 * dt * k1)
            k3 = force_func_single(s + 0.5 * dt * k2)
            k4 = force_func_single(s + dt * k3)
            out[i] = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return out

    # ----------------------------------------------------------------
    # Timing comparison driver
    # ----------------------------------------------------------------
    @classmethod
    def compare_all(cls, n: int = 10_000, num_runs: int = 5) -> Dict[str, Dict]:
        """
        Run vectorized vs sequential versions of every operation and print a
        comparison table.

        Returns dict mapping operation name -> {vectorized_s, sequential_s, speedup}.
        """
        results = {}

        def _time(func, *a, **kw):
            times = []
            for _ in range(num_runs):
                t0 = time.perf_counter()
                func(*a, **kw)
                t1 = time.perf_counter()
                times.append(t1 - t0)
            times.sort()
            return times[len(times) // 2]

        # --- Quaternion multiply ---
        q1 = np.random.randn(n, 4)
        q1 /= np.linalg.norm(q1, axis=1, keepdims=True)
        q2 = np.random.randn(n, 4)
        q2 /= np.linalg.norm(q2, axis=1, keepdims=True)

        tv = _time(cls.batch_quaternion_multiply, q1, q2)
        ts = _time(cls._sequential_quaternion_multiply, q1, q2)
        results["quat_multiply"] = {"vectorized_s": tv, "sequential_s": ts, "speedup": ts / tv}

        # --- Rotate vectors ---
        v = np.random.randn(n, 3)
        tv = _time(cls.batch_rotate_vectors, q1, v)
        ts = _time(cls._sequential_rotate_vectors, q1, v)
        results["rotate_vectors"] = {"vectorized_s": tv, "sequential_s": ts, "speedup": ts / tv}

        # --- Gravity ---
        pos = np.random.randn(n, 3) * 7000 + np.array([7000, 0, 0])
        tv = _time(cls.batch_gravity_acceleration, pos)
        ts = _time(cls._sequential_gravity_acceleration, pos)
        results["gravity"] = {"vectorized_s": tv, "sequential_s": ts, "speedup": ts / tv}

        # --- RK4 step ---
        mu = 398_600.4418
        states = np.zeros((n, 6))
        states[:, 0] = 7000 + np.random.randn(n) * 10
        states[:, 4] = 7.5 + np.random.randn(n) * 0.01

        def deriv_batch(s):
            pos = s[:, :3]
            vel = s[:, 3:]
            r = np.linalg.norm(pos, axis=1, keepdims=True)
            acc = -mu * pos / (r ** 3)
            return np.hstack([vel, acc])

        def deriv_single(s):
            pos = s[:3]
            vel = s[3:]
            r = np.linalg.norm(pos)
            acc = -mu * pos / r ** 3
            return np.concatenate([vel, acc])

        tv = _time(cls.batch_rk4_step, states, 10.0, deriv_batch)
        ts = _time(cls._sequential_rk4_step, states, 10.0, deriv_single)
        results["rk4_step"] = {"vectorized_s": tv, "sequential_s": ts, "speedup": ts / tv}

        # Print table
        print("\n" + "=" * 65)
        print("  VectorizedOps: Sequential vs Vectorized  (N = {:,})".format(n))
        print("=" * 65)
        print(f"  {'Operation':<20} {'Sequential (s)':>15} {'Vectorized (s)':>15} {'Speedup':>10}")
        print("  " + "-" * 62)
        for op, d in results.items():
            print(
                f"  {op:<20} {d['sequential_s']:>15.6f} {d['vectorized_s']:>15.6f} "
                f"{d['speedup']:>9.1f}x"
            )

        return results


# ========================================================================
# ParallelSim -- Multi-core parallelism via multiprocessing
# ========================================================================

def _run_sim(args: Tuple[Callable, Any]) -> Any:
    """
    Top-level function for pickling by multiprocessing.Pool.
    Unpacks (sim_func, config) and calls sim_func(config).
    """
    sim_func, config = args
    return sim_func(config)


def _map_chunk(args: Tuple[Callable, Any]) -> Any:
    """Map function wrapper for pool."""
    map_func, chunk = args
    return map_func(chunk)


class ParallelSim:
    """
    Multi-core parallelism for embarrassingly parallel GNC workloads.

    Uses :class:`multiprocessing.Pool` to distribute independent tasks across
    physical CPU cores. The GIL is irrelevant because each worker is a
    separate OS process with its own Python interpreter.

    Key patterns demonstrated:
        - Process pool for Monte Carlo ensembles
        - Grid search over parameter space
        - MapReduce for telemetry post-processing
        - Shared memory (multiprocessing.Array) for read-heavy data
    """

    def __init__(self, num_workers: Optional[int] = None):
        """
        Parameters
        ----------
        num_workers : int or None
            Number of worker processes. Defaults to ``os.cpu_count()``.
        """
        self.num_workers = num_workers or os.cpu_count() or 4

    # ----------------------------------------------------------------
    # Monte Carlo
    # ----------------------------------------------------------------
    def parallel_monte_carlo(
        self,
        sim_func: Callable[[Any], Any],
        configs: Sequence[Any],
    ) -> List[Any]:
        """
        Run *sim_func* for each configuration in *configs* using a process pool.

        Parameters
        ----------
        sim_func : callable
            A function that takes a single config dict/object and returns a
            result (must be picklable).
        configs : sequence
            One entry per Monte Carlo trial.

        Returns
        -------
        list of results, in the same order as *configs*.

        Parallelism model
        -----------------
        Each trial is independent (no shared mutable state), which makes this
        *embarrassingly parallel*.  The Pool.map call distributes trials evenly
        across workers, collects results, and returns them in order.
        """
        tasks = [(sim_func, c) for c in configs]
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(_run_sim, tasks)
        return results

    # ----------------------------------------------------------------
    # Grid search
    # ----------------------------------------------------------------
    def parallel_trajectory_search(
        self,
        search_func: Callable[[Any], Any],
        param_grid: Sequence[Any],
    ) -> List[Any]:
        """
        Evaluate *search_func* at every point in *param_grid* in parallel.

        This is the trajectory-optimisation equivalent of scikit-learn's
        GridSearchCV: test every combination of departure date, arrival date,
        and launch energy to find the optimal transfer.

        Parameters
        ----------
        search_func : callable
            Evaluates one grid point and returns a scalar or dict.
        param_grid : sequence
            Each element parameterises one grid point.

        Returns
        -------
        list of search_func outputs, same order as param_grid.
        """
        tasks = [(search_func, p) for p in param_grid]
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(_run_sim, tasks)
        return results

    # ----------------------------------------------------------------
    # MapReduce for telemetry
    # ----------------------------------------------------------------
    def map_reduce_telemetry(
        self,
        telemetry_chunks: Sequence[Any],
        map_func: Callable[[Any], Any],
        reduce_func: Callable[[List[Any]], Any],
    ) -> Any:
        """
        Process telemetry in a MapReduce pattern.

        Map phase:   Each chunk is processed independently in a separate worker.
        Reduce phase: All mapped results are combined in the main process.

        Parameters
        ----------
        telemetry_chunks : sequence
            Pre-split data chunks (e.g. one per file or time segment).
        map_func : callable
            Processes a single chunk and returns an intermediate result.
        reduce_func : callable
            Combines a list of intermediate results into a final result.

        Returns
        -------
        The output of reduce_func.

        GNC example
        -----------
        Map:   Compute RMS attitude error per orbit.
        Reduce: Concatenate per-orbit RMS values and compute mission-level
                statistics (mean, 3-sigma, worst-case).
        """
        tasks = [(map_func, chunk) for chunk in telemetry_chunks]
        with Pool(processes=self.num_workers) as pool:
            mapped = pool.map(_map_chunk, tasks)
        return reduce_func(mapped)

    # ----------------------------------------------------------------
    # Shared memory demonstration
    # ----------------------------------------------------------------
    @staticmethod
    def demonstrate_shared_memory(n: int = 1_000_000) -> Dict[str, float]:
        """
        Show how multiprocessing.Array provides shared memory between workers
        without copying.

        Approach
        --------
        We allocate a shared C-type array of n doubles, populate it in the
        main process, then let workers read from it to compute partial sums.
        Because the array lives in shared memory (mmap-backed), no
        serialisation / deserialisation (pickling) overhead occurs.

        Contrast with the default Pool.map approach, which pickles the data
        for every worker -- O(n * num_workers) memory overhead.
        """
        print("\n  Shared memory demo (n = {:,})".format(n))

        # --- Shared array ---
        shared = SharedArray("d", n, lock=False)
        # Fill with random data
        arr_np = np.frombuffer(shared, dtype=np.float64)
        arr_np[:] = np.random.randn(n)

        # Worker reads a slice and returns sum
        def _shared_worker(indices: Tuple[int, int]) -> float:
            arr = np.frombuffer(shared, dtype=np.float64)
            return float(arr[indices[0]: indices[1]].sum())

        num_workers = min(os.cpu_count() or 4, 4)
        chunk = n // num_workers
        slices = [(i * chunk, (i + 1) * chunk) for i in range(num_workers)]
        slices[-1] = (slices[-1][0], n)  # last worker takes remainder

        t0 = time.perf_counter()
        with Pool(processes=num_workers) as pool:
            partial_sums = pool.map(_shared_worker, slices)
        total = sum(partial_sums)
        t_shared = time.perf_counter() - t0

        # --- Copy-based baseline ---
        data_copy = np.array(arr_np)  # full copy

        def _copy_worker(data_slice: np.ndarray) -> float:
            return float(data_slice.sum())

        # Prepare copies of slices (simulates pickling overhead)
        sliced_data = [data_copy[s[0]:s[1]].copy() for s in slices]
        t0 = time.perf_counter()
        with Pool(processes=num_workers) as pool:
            partial_sums_copy = pool.map(_copy_worker, sliced_data)
        total_copy = sum(partial_sums_copy)
        t_copy = time.perf_counter() - t0

        print(f"    Shared memory time : {t_shared:.4f} s  (total = {total:.4f})")
        print(f"    Copy-based time    : {t_copy:.4f} s  (total = {total_copy:.4f})")
        ratio = t_copy / t_shared if t_shared > 0 else float("inf")
        print(f"    Shared / Copy      : {ratio:.2f}x")

        return {"shared_s": t_shared, "copy_s": t_copy, "ratio": ratio}

    # ----------------------------------------------------------------
    # Compare sequential vs parallel Monte Carlo
    # ----------------------------------------------------------------
    @classmethod
    def compare_parallel_vs_sequential(
        cls, num_trials: int = 40, num_workers: int = 4
    ) -> Dict[str, float]:
        """
        Compare sequential execution vs parallel pool for a simple orbit sim.
        """
        mu = 398_600.4418

        def mini_sim(config: dict) -> dict:
            """Tiny two-body propagation (RK4, 200 steps)."""
            state = np.array(config["state"], dtype=np.float64)
            dt = config.get("dt", 10.0)
            n_steps = config.get("n_steps", 200)
            for _ in range(n_steps):
                pos = state[:3]
                vel = state[3:]
                r = np.linalg.norm(pos)
                acc = -mu * pos / r ** 3
                deriv = np.concatenate([vel, acc])
                k1 = deriv
                s2 = state + 0.5 * dt * k1
                pos2, vel2 = s2[:3], s2[3:]
                r2 = np.linalg.norm(pos2)
                k2 = np.concatenate([vel2, -mu * pos2 / r2 ** 3])
                s3 = state + 0.5 * dt * k2
                pos3, vel3 = s3[:3], s3[3:]
                r3 = np.linalg.norm(pos3)
                k3 = np.concatenate([vel3, -mu * pos3 / r3 ** 3])
                s4 = state + dt * k3
                pos4, vel4 = s4[:3], s4[3:]
                r4 = np.linalg.norm(pos4)
                k4 = np.concatenate([vel4, -mu * pos4 / r4 ** 3])
                state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            return {"final_state": state.tolist(), "final_r": float(np.linalg.norm(state[:3]))}

        configs = [
            {"state": [7000 + np.random.randn() * 10, 0, 0, 0, 7.5 + np.random.randn() * 0.01, 0]}
            for _ in range(num_trials)
        ]

        # Sequential
        t0 = time.perf_counter()
        seq_results = [mini_sim(c) for c in configs]
        t_seq = time.perf_counter() - t0

        # Parallel
        ps = cls(num_workers=num_workers)
        t0 = time.perf_counter()
        par_results = ps.parallel_monte_carlo(mini_sim, configs)
        t_par = time.perf_counter() - t0

        speedup = t_seq / t_par if t_par > 0 else float("inf")
        print(f"\n  Monte Carlo comparison ({num_trials} trials, {num_workers} workers):")
        print(f"    Sequential : {t_seq:.4f} s")
        print(f"    Parallel   : {t_par:.4f} s")
        print(f"    Speedup    : {speedup:.2f}x")
        print(f"    Ideal      : {num_workers:.1f}x  (Amdahl's law upper bound)")

        return {"sequential_s": t_seq, "parallel_s": t_par, "speedup": speedup}


# ========================================================================
# GPUStub -- Annotated GPU-acceleration stubs
# ========================================================================

class GPUStub:
    """
    Stub demonstrating how GPU (CUDA) acceleration would be structured for
    GNC computations. Uses numpy as a stand-in for CUDA device arrays.

    GPU Memory Model (CUDA)
    -----------------------
    A discrete GPU has its own DRAM ("device memory", e.g. 16-80 GB HBM).
    Data must be explicitly copied between host (CPU) RAM and device RAM via
    the PCIe or NVLink bus.  This transfer cost is significant:

        PCIe 4.0 x16 : ~25 GB/s
        NVLink 3.0   : ~600 GB/s

    Therefore GPU acceleration only pays off when the *compute time saved*
    exceeds the *transfer overhead*.  Rule of thumb: the compute-to-transfer
    ratio (arithmetic intensity) must be high.

    Execution Model
    ---------------
    - GPU cores ("CUDA cores") are grouped into *Streaming Multiprocessors*
      (SMs). Each SM executes threads in groups of 32 called *warps*.
    - All 32 threads in a warp execute the **same instruction** at the same
      time (SIMT). Divergent branches within a warp serialize execution.
    - Memory accesses within a warp should be *coalesced*: consecutive threads
      accessing consecutive addresses merge into a single wide transaction.

    When GPU helps vs hurts
    -----------------------
    Helps:
        - Large matrix multiplies (BLAS level-3) -- arithmetic intensity O(n)
        - Batch element-wise operations on >100K elements
        - FFTs with > 2^16 points
    Hurts:
        - Small matrices (3x3, 6x6) -- transfer overhead dominates
        - Branchy control flow -- warp divergence
        - Sequential algorithms with loop-carried dependencies
    """

    @staticmethod
    def gpu_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        "GPU" matrix multiply -- uses numpy but annotated with how CUDA
        would implement it.

        CUDA implementation sketch
        --------------------------
        1. Allocate device arrays: d_A, d_B, d_C = cudaMalloc(...)
        2. Copy host -> device:    cudaMemcpy(d_A, h_A, H2D)
        3. Launch kernel:          matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, N)
           - grid  = (N/TILE, N/TILE)    -- one thread block per output tile
           - block = (TILE, TILE)         -- e.g. 16x16 = 256 threads
           - Each thread block loads a TILE x TILE sub-matrix of A and B into
             shared memory (48 KB per SM), computes partial products, and
             accumulates into a register tile.
           - This is the same "blocking" idea as CPU loop tiling, but at the
             warp/block level.
        4. Copy device -> host:    cudaMemcpy(h_C, d_C, D2H)

        In practice one would call cuBLAS sgemm/dgemm, which applies the
        above with auto-tuned tile sizes and double-buffering.

        Parameters
        ----------
        A, B : ndarray (M, K) and (K, N)

        Returns
        -------
        ndarray (M, N)
        """
        # --- simulate host->device transfer ---
        # In real CUDA: cudaMemcpy(d_A, A.ctypes.data, A.nbytes, cudaMemcpyHostToDevice)
        d_A = np.array(A, dtype=np.float64)  # "device copy"
        d_B = np.array(B, dtype=np.float64)

        # --- kernel execution ---
        # cuBLAS: cublasDgemm(handle, ..., d_A, d_B, d_C, ...)
        # Internally: tiled matrix multiply using shared memory
        #
        # Coalesced access pattern:
        #   Thread (tx, ty) in block (bx, by) loads A[by*TILE + ty][k] and
        #   B[k][bx*TILE + tx].  The tx-indexed load on B is coalesced
        #   (consecutive threads access consecutive columns).
        d_C = d_A @ d_B  # numpy BLAS as stand-in

        # --- device->host transfer ---
        # cudaMemcpy(h_C, d_C.ctypes.data, d_C.nbytes, cudaMemcpyDeviceToHost)
        h_C = np.array(d_C)  # "host copy"
        return h_C

    @staticmethod
    def gpu_fft_filter(
        signal: np.ndarray,
        cutoff_fraction: float = 0.1,
    ) -> np.ndarray:
        """
        FFT-based low-pass filter -- annotated with GPU parallelism concepts.

        GPU FFT parallelism (cuFFT)
        ---------------------------
        The Cooley-Tukey FFT decomposes an N-point DFT into log2(N) stages of
        butterfly operations. Each stage is embarrassingly parallel across
        N/2 independent butterflies:

            Stage 0: N/2 butterflies on pairs (0,1), (2,3), ...
            Stage 1: N/2 butterflies on pairs with stride 2
            ...
            Stage k: N/2 butterflies with stride 2^k

        On a GPU, each butterfly is assigned to a CUDA thread. For N = 2^20
        (~1M points), each stage launches 512K threads.  With 10K CUDA cores,
        each stage completes in ~50 iterations, and there are only 20 stages
        -> ~1000 thread-launches total.

        Memory access pattern:
        - Early stages: stride is small -> threads in a warp access nearby
          addresses -> coalesced and cache-friendly.
        - Late stages: stride is large -> threads in a warp access distant
          addresses -> bank conflicts in shared memory.
        - cuFFT handles this by decomposing into radix-2/4/8 sub-FFTs and
          transposing data between stages to restore coalescence.

        Parameters
        ----------
        signal : ndarray, shape (N,)
        cutoff_fraction : float
            Fraction of spectrum to keep (0.0 to 1.0).

        Returns
        -------
        ndarray, shape (N,) -- filtered signal
        """
        n = len(signal)

        # --- Forward FFT (cuFFT would use cufftExecZ2Z for complex double) ---
        # On GPU: plan = cufftPlan1d(&plan, N, CUFFT_Z2Z, 1)
        #         cufftExecZ2Z(plan, d_signal, d_spectrum, CUFFT_FORWARD)
        spectrum = np.fft.rfft(signal)

        # --- Apply brick-wall low-pass filter ---
        # On GPU: a simple element-wise kernel: if freq_index > cutoff -> 0
        # This is trivially parallel (one thread per frequency bin)
        n_freq = len(spectrum)
        cutoff_idx = int(n_freq * cutoff_fraction)
        spectrum[cutoff_idx:] = 0.0  # zero out high frequencies

        # --- Inverse FFT ---
        # cufftExecZ2Z(plan, d_spectrum, d_signal, CUFFT_INVERSE)
        filtered = np.fft.irfft(spectrum, n=n)

        return filtered

    @classmethod
    def demonstrate(cls) -> Dict[str, float]:
        """
        Run both GPU-stub demonstrations and print timing/explanation.

        Returns dict of timings.
        """
        print("\n" + "=" * 65)
        print("  GPUStub: Simulated GPU Operations")
        print("=" * 65)

        # --- Matrix multiply ---
        sizes = [64, 256, 1024]
        mm_times = {}
        for sz in sizes:
            A = np.random.randn(sz, sz)
            B = np.random.randn(sz, sz)
            t0 = time.perf_counter()
            for _ in range(10):
                cls.gpu_matrix_multiply(A, B)
            t = (time.perf_counter() - t0) / 10
            mm_times[sz] = t
            print(f"    matmul {sz}x{sz} : {t:.6f} s")

        print()
        print("    Note: For 3x3 matrices (GNC attitude math), GPU overhead")
        print("    dominates -- CPU is faster. For 1024+ (trajectory collocation),")
        print("    GPU BLAS delivers 10-50x speedup over CPU BLAS.")

        # --- FFT filter ---
        for n in [1024, 65536, 1048576]:
            sig = np.sin(np.linspace(0, 100 * np.pi, n)) + 0.5 * np.random.randn(n)
            t0 = time.perf_counter()
            cls.gpu_fft_filter(sig, cutoff_fraction=0.05)
            t = time.perf_counter() - t0
            mm_times[f"fft_{n}"] = t
            print(f"    FFT filter n={n:>8,} : {t:.6f} s")

        print()
        print("    Note: cuFFT achieves peak throughput at N >= 2^16 (65K).")
        print("    Below that, kernel launch overhead and H2D/D2H transfers")
        print("    negate the parallelism benefit.")

        return mm_times


# ========================================================================
# Plotting helper
# ========================================================================

def _plot_vectorized_comparison(results: Dict[str, Dict], output_dir: str):
    """Bar chart comparing sequential vs vectorized across operations."""
    ops = list(results.keys())
    seq_times = [results[op]["sequential_s"] for op in ops]
    vec_times = [results[op]["vectorized_s"] for op in ops]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Absolute times
    x = np.arange(len(ops))
    w = 0.35
    axes[0].bar(x - w / 2, seq_times, w, label="Sequential", color="salmon")
    axes[0].bar(x + w / 2, vec_times, w, label="Vectorized", color="mediumseagreen")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ops, rotation=30, ha="right")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("Sequential vs Vectorized (absolute)")
    axes[0].legend()

    # Speedup
    speedups = [results[op]["speedup"] for op in ops]
    axes[1].bar(ops, speedups, color="steelblue", edgecolor="black")
    axes[1].set_ylabel("Speedup (x)")
    axes[1].set_title("Vectorization Speedup")
    axes[1].axhline(1.0, color="red", linestyle="--", linewidth=0.8)
    for i, sp in enumerate(speedups):
        axes[1].text(i, sp + 0.3, f"{sp:.1f}x", ha="center", fontsize=9)
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "vectorized_comparison.png"), dpi=150)
    plt.close(fig)


def _plot_parallel_scaling(output_dir: str, max_workers: int = 8, num_trials: int = 40):
    """
    Run Monte Carlo at increasing worker counts and plot scaling efficiency.

    Ideal (Amdahl's law with zero serial fraction) is a straight line y = x.
    Real scaling is sub-linear due to process startup, pickling, and OS
    scheduling overhead.
    """
    worker_counts = list(range(1, max_workers + 1))
    speedups = []

    # Baseline: sequential
    mu = 398_600.4418

    def mini_sim(config):
        state = np.array(config["state"], dtype=np.float64)
        dt = 10.0
        for _ in range(200):
            pos, vel = state[:3], state[3:]
            r = np.linalg.norm(pos)
            acc = -mu * pos / r ** 3
            k1 = np.concatenate([vel, acc])
            s2 = state + 0.5 * dt * k1
            r2 = np.linalg.norm(s2[:3])
            k2 = np.concatenate([s2[3:], -mu * s2[:3] / r2 ** 3])
            s3 = state + 0.5 * dt * k2
            r3 = np.linalg.norm(s3[:3])
            k3 = np.concatenate([s3[3:], -mu * s3[:3] / r3 ** 3])
            s4 = state + dt * k3
            r4 = np.linalg.norm(s4[:3])
            k4 = np.concatenate([s4[3:], -mu * s4[:3] / r4 ** 3])
            state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return {"r": float(np.linalg.norm(state[:3]))}

    configs = [
        {"state": [7000 + np.random.randn() * 10, 0, 0, 0, 7.5 + np.random.randn() * 0.01, 0]}
        for _ in range(num_trials)
    ]

    # Sequential baseline
    t0 = time.perf_counter()
    _ = [mini_sim(c) for c in configs]
    t_base = time.perf_counter() - t0

    for nw in worker_counts:
        ps = ParallelSim(num_workers=nw)
        t0 = time.perf_counter()
        ps.parallel_monte_carlo(mini_sim, configs)
        t = time.perf_counter() - t0
        speedups.append(t_base / t if t > 0 else 0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(worker_counts, speedups, "o-", color="steelblue", label="Measured")
    ax.plot(worker_counts, worker_counts, "--", color="gray", label="Ideal (linear)")
    ax.set_xlabel("Number of Workers")
    ax.set_ylabel("Speedup (x)")
    ax.set_title(f"Parallel Scaling ({num_trials} Monte Carlo trials)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "parallel_scaling.png"), dpi=150)
    plt.close(fig)
    print(f"\n  Scaling plot saved to {os.path.join(output_dir, 'parallel_scaling.png')}")


# ========================================================================
# Main
# ========================================================================

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "parallel_results"
    os.makedirs(out, exist_ok=True)

    print("=" * 65)
    print("  parallel.py -- Parallelization & Vectorization Benchmarks")
    print("=" * 65)

    # 1. Vectorized operations comparison
    vec_results = VectorizedOps.compare_all(n=10_000, num_runs=5)
    _plot_vectorized_comparison(vec_results, out)

    # 2. Parallel Monte Carlo comparison
    par_results = ParallelSim.compare_parallel_vs_sequential(
        num_trials=40, num_workers=min(os.cpu_count() or 4, 4)
    )

    # 3. Shared memory demo
    ParallelSim.demonstrate_shared_memory(n=500_000)

    # 4. GPU stubs
    gpu_results = GPUStub.demonstrate()

    # 5. Parallel scaling plot
    max_w = min(os.cpu_count() or 4, 6)
    _plot_parallel_scaling(out, max_workers=max_w, num_trials=30)

    # 6. MapReduce telemetry demo
    print("\n" + "=" * 65)
    print("  MapReduce Telemetry Demo")
    print("=" * 65)

    # Simulate chunked telemetry: each chunk is a numpy array of "attitude errors"
    n_chunks = 8
    chunk_size = 100_000
    chunks = [np.random.randn(chunk_size) * 0.5 for _ in range(n_chunks)]

    def map_rms(chunk: np.ndarray) -> float:
        """Map: compute RMS of one telemetry chunk."""
        return float(np.sqrt(np.mean(chunk ** 2)))

    def reduce_rms(rms_values: List[float]) -> Dict[str, float]:
        """Reduce: aggregate per-chunk RMS into mission statistics."""
        arr = np.array(rms_values)
        return {
            "mean_rms": float(arr.mean()),
            "max_rms": float(arr.max()),
            "std_rms": float(arr.std()),
        }

    ps = ParallelSim(num_workers=min(os.cpu_count() or 4, 4))

    t0 = time.perf_counter()
    result_par = ps.map_reduce_telemetry(chunks, map_rms, reduce_rms)
    t_par = time.perf_counter() - t0

    t0 = time.perf_counter()
    mapped_seq = [map_rms(c) for c in chunks]
    result_seq = reduce_rms(mapped_seq)
    t_seq = time.perf_counter() - t0

    print(f"    Parallel MapReduce : {t_par:.4f} s  -> {result_par}")
    print(f"    Sequential         : {t_seq:.4f} s  -> {result_seq}")
    print(f"    Speedup            : {t_seq / t_par:.2f}x" if t_par > 0 else "    N/A")

    print("\n" + "=" * 65)
    print(f"  All results and plots saved to: {os.path.abspath(out)}")
    print("=" * 65)
