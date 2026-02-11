"""
Custom data structures for the GNC space mission system.

This module provides mission-specific data structures that wrap well-known
algorithms and storage patterns in an interface tailored for spacecraft
guidance, navigation, and control workloads. Every class documents its
time complexity, memory layout, and the GNC motivation behind its design.

Structures
----------
EventPriorityQueue  -- Min-heap priority queue for mission event scheduling.
StateHistory        -- Fixed-size ring buffer backed by contiguous NumPy arrays.
KDTree3D            -- Lightweight 3-dimensional KD-tree for spatial queries.
TelemetryBuffer     -- Double-buffered writer/reader for telemetry streams.
MissionGraph        -- Directed weighted graph of mission phases with Dijkstra.
DPTable             -- Dynamic-programming memoization grid for trajectory
                       cost-to-go computation.

All public methods carry full type annotations and NumPy-style docstrings.
"""

from __future__ import annotations

import heapq
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. EventPriorityQueue
# ---------------------------------------------------------------------------

class EventType(Enum):
    """Enumeration of schedulable mission events."""
    MANEUVER = auto()
    SENSOR_READING = auto()
    PHASE_TRANSITION = auto()
    TELEMETRY_DOWNLINK = auto()
    ATTITUDE_CORRECTION = auto()
    ORBIT_DETERMINATION = auto()
    HEALTH_CHECK = auto()


@dataclass(order=True)
class MissionEvent:
    """A single schedulable mission event.

    Ordering is defined *first* by ``time`` (earliest first), then by
    ``priority`` (lower numeric value = higher urgency).  The ``event_type``
    and ``data`` fields are excluded from the comparison so that the heap
    property is driven entirely by (time, priority).

    Attributes
    ----------
    time : float
        Mission elapsed time (seconds) at which the event should fire.
    priority : int
        Numeric urgency -- 0 is highest priority.
    event_type : EventType
        Category tag used for downstream dispatching.
    data : Dict[str, Any]
        Arbitrary payload (delta-V vector, sensor id, phase name, etc.).
    """
    time: float
    priority: int
    event_type: EventType = field(compare=False)
    data: Dict[str, Any] = field(default_factory=dict, compare=False)


class EventPriorityQueue:
    """Min-heap priority queue for scheduling mission events.

    Why a priority queue for GNC?
    -----------------------------
    A spacecraft runs dozens of concurrent activities -- attitude
    corrections, thruster firings, sensor polls, telemetry windows, phase
    transitions -- all indexed by *mission time*.  A priority queue lets the
    flight-software main loop always retrieve the chronologically next event
    in O(1) and insert new events in O(log n) without keeping the full list
    sorted.  The secondary ``priority`` key breaks ties when two events
    share the same timestamp (e.g., a critical maneuver should preempt a
    routine health-check).

    Time complexity
    ---------------
    +-----------+----------------+
    | Operation | Worst-case     |
    +===========+================+
    | push      | O(log n)       |
    | pop       | O(log n)       |
    | peek      | O(1)           |
    | is_empty  | O(1)           |
    | size      | O(1)           |
    | clear     | O(1) amortized |
    +-----------+----------------+

    Memory layout
    -------------
    Internally backed by a Python ``list`` arranged as a binary min-heap
    via :mod:`heapq`.  Each element is a :class:`MissionEvent` dataclass
    whose ``__lt__`` compares ``(time, priority)`` tuples, so ``heapq``
    can maintain the heap invariant directly.
    """

    def __init__(self) -> None:
        self._heap: List[MissionEvent] = []

    # -- core operations ---------------------------------------------------

    def push(self, event: MissionEvent) -> None:
        """Insert an event into the queue.

        Parameters
        ----------
        event : MissionEvent
            The event to schedule.

        Complexity
        ----------
        O(log n) -- sift-up through at most log2(n) levels.
        """
        heapq.heappush(self._heap, event)

    def pop(self) -> MissionEvent:
        """Remove and return the earliest / highest-priority event.

        Returns
        -------
        MissionEvent

        Raises
        ------
        IndexError
            If the queue is empty.

        Complexity
        ----------
        O(log n) -- swap root with last element, then sift-down.
        """
        if self.is_empty():
            raise IndexError("pop from an empty EventPriorityQueue")
        return heapq.heappop(self._heap)

    def peek(self) -> MissionEvent:
        """Return the next event without removing it.

        Returns
        -------
        MissionEvent

        Raises
        ------
        IndexError
            If the queue is empty.

        Complexity
        ----------
        O(1) -- the root of a min-heap is always at index 0.
        """
        if self.is_empty():
            raise IndexError("peek on an empty EventPriorityQueue")
        return self._heap[0]

    # -- utility -----------------------------------------------------------

    def is_empty(self) -> bool:
        """Return ``True`` when no events are queued.  O(1)."""
        return len(self._heap) == 0

    def size(self) -> int:
        """Return the current number of queued events.  O(1)."""
        return len(self._heap)

    def clear(self) -> None:
        """Discard all events.  O(1) amortized (list deallocation)."""
        self._heap.clear()

    # -- convenience -------------------------------------------------------

    def push_event(
        self,
        time: float,
        priority: int,
        event_type: EventType,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Shorthand that builds a :class:`MissionEvent` and pushes it.

        Parameters
        ----------
        time : float
            Mission elapsed time in seconds.
        priority : int
            Lower is more urgent.
        event_type : EventType
            Category of the event.
        data : dict, optional
            Event payload.
        """
        self.push(MissionEvent(time, priority, event_type, data or {}))

    def pop_all_before(self, t: float) -> List[MissionEvent]:
        """Pop every event whose time is strictly less than *t*.

        Useful for advancing the simulation clock in discrete steps: drain
        all events that should have fired by now.

        Parameters
        ----------
        t : float
            Cutoff time (exclusive upper bound).

        Returns
        -------
        list of MissionEvent
            Events in chronological / priority order.

        Complexity
        ----------
        O(k log n) where *k* is the number of events returned.
        """
        events: List[MissionEvent] = []
        while not self.is_empty() and self._heap[0].time < t:
            events.append(heapq.heappop(self._heap))
        return events

    def __len__(self) -> int:
        return self.size()

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __repr__(self) -> str:
        return f"EventPriorityQueue(size={self.size()})"


# ---------------------------------------------------------------------------
# 2. StateHistory
# ---------------------------------------------------------------------------

class StateHistory:
    """Fixed-size ring buffer for spacecraft state history.

    Why a ring buffer for GNC?
    --------------------------
    Attitude determination filters (EKFs, UKFs) and control loops need
    access to the last *N* state snapshots -- but storing an unbounded
    history would exhaust memory on an embedded flight computer.  A ring
    buffer provides O(1) append and O(1) random access while guaranteeing a
    hard upper bound on memory usage.  Backing the buffer with a contiguous
    NumPy array ensures cache-friendly sequential reads, which matters when
    the navigation filter sweeps through recent states every guidance cycle.

    Memory layout
    -------------
    Two 1-D arrays are stored:

    * ``_timestamps`` -- ``float64[capacity]``
    * ``_states``     -- ``float64[capacity, state_dim]``

    Both are pre-allocated at construction time so no heap allocation occurs
    during flight.  A ``_head`` index tracks the next write position, and
    ``_count`` tracks how many slots have been filled (up to ``capacity``).
    Reads use modular indexing into the flat array, preserving chronological
    order.

    Time complexity
    ---------------
    +-------------------+------+
    | Operation         | Cost |
    +===================+======+
    | append            | O(1) |
    | get_latest(n)     | O(n) |
    | get_range         | O(k) |
    | to_array          | O(n) |
    | memory_usage      | O(1) |
    +-------------------+------+

    Parameters
    ----------
    capacity : int
        Maximum number of state snapshots to retain.
    state_dim : int
        Dimensionality of each state vector.  For a typical 6-DOF
        spacecraft this is 13 (position 3 + velocity 3 + quaternion 4 +
        angular velocity 3).
    """

    def __init__(self, capacity: int, state_dim: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")

        self._capacity: int = capacity
        self._state_dim: int = state_dim

        # Pre-allocate contiguous storage -- no runtime malloc.
        self._timestamps: np.ndarray = np.zeros(capacity, dtype=np.float64)
        self._states: np.ndarray = np.zeros((capacity, state_dim), dtype=np.float64)

        self._head: int = 0      # Next write position.
        self._count: int = 0     # Number of valid entries (<= capacity).

    # -- write -------------------------------------------------------------

    def append(self, timestamp: float, state: np.ndarray) -> None:
        """Record a new state snapshot, overwriting the oldest if full.

        Parameters
        ----------
        timestamp : float
            Mission elapsed time (seconds).
        state : np.ndarray
            State vector of length ``state_dim``.

        Raises
        ------
        ValueError
            If ``state`` has the wrong dimensionality.

        Complexity
        ----------
        O(1) -- single indexed write into pre-allocated storage.
        """
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (self._state_dim,):
            raise ValueError(
                f"Expected state of shape ({self._state_dim},), got {state.shape}"
            )

        self._timestamps[self._head] = timestamp
        self._states[self._head] = state
        self._head = (self._head + 1) % self._capacity
        self._count = min(self._count + 1, self._capacity)

    # -- read --------------------------------------------------------------

    def get_latest(self, n: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Return the *n* most recent snapshots in chronological order.

        Parameters
        ----------
        n : int
            Number of snapshots to retrieve.  Clamped to available count.

        Returns
        -------
        timestamps : np.ndarray
            Shape ``(k,)`` where ``k = min(n, count)``.
        states : np.ndarray
            Shape ``(k, state_dim)``.

        Complexity
        ----------
        O(n) -- index computation plus array copy.
        """
        n = min(n, self._count)
        if n == 0:
            return np.empty(0, dtype=np.float64), np.empty((0, self._state_dim), dtype=np.float64)

        indices = self._chronological_indices()[-n:]
        return self._timestamps[indices].copy(), self._states[indices].copy()

    def get_range(
        self, t_start: float, t_end: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return all snapshots with timestamps in ``[t_start, t_end]``.

        Parameters
        ----------
        t_start, t_end : float
            Inclusive time bounds (mission elapsed seconds).

        Returns
        -------
        timestamps : np.ndarray
        states : np.ndarray

        Complexity
        ----------
        O(count) -- linear scan of valid entries (ring order).
        """
        if self._count == 0:
            return np.empty(0, dtype=np.float64), np.empty((0, self._state_dim), dtype=np.float64)

        indices = self._chronological_indices()
        ts = self._timestamps[indices]
        mask = (ts >= t_start) & (ts <= t_end)
        selected = indices[mask]
        return self._timestamps[selected].copy(), self._states[selected].copy()

    def to_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the full valid history in chronological order.

        Returns
        -------
        timestamps : np.ndarray, shape ``(count,)``
        states : np.ndarray, shape ``(count, state_dim)``
        """
        return self.get_latest(self._count)

    # -- metadata ----------------------------------------------------------

    def memory_usage(self) -> int:
        """Return approximate memory consumption in bytes.  O(1).

        Only counts the NumPy backing arrays, not Python object overhead.
        """
        return int(self._timestamps.nbytes + self._states.nbytes)

    @property
    def count(self) -> int:
        """Number of valid snapshots currently stored."""
        return self._count

    @property
    def capacity(self) -> int:
        """Maximum number of snapshots the buffer can hold."""
        return self._capacity

    @property
    def is_full(self) -> bool:
        """``True`` when the buffer has wrapped at least once."""
        return self._count == self._capacity

    # -- internals ---------------------------------------------------------

    def _chronological_indices(self) -> np.ndarray:
        """Return physical array indices in chronological order.

        If the buffer has not yet wrapped, this is simply ``[0, 1, ..., count-1]``.
        After wrapping, the oldest entry is at ``_head`` and the newest is at
        ``(_head - 1) % capacity``.
        """
        if self._count < self._capacity:
            return np.arange(self._count)
        # Buffer has wrapped: oldest is at _head.
        return np.roll(np.arange(self._capacity), -self._head)

    def __len__(self) -> int:
        return self._count

    def __repr__(self) -> str:
        usage_kb = self.memory_usage() / 1024
        return (
            f"StateHistory(count={self._count}/{self._capacity}, "
            f"dim={self._state_dim}, mem={usage_kb:.1f} KiB)"
        )


# ---------------------------------------------------------------------------
# 3. KDTree3D
# ---------------------------------------------------------------------------

@dataclass
class _KDNode:
    """Internal node of the 3-D KD-tree.

    Attributes
    ----------
    point : np.ndarray
        The 3-D coordinate stored at this node.
    index : int
        Original index into the input point array (for caller look-ups).
    axis : int
        Splitting axis (0=x, 1=y, 2=z).
    left : _KDNode | None
    right : _KDNode | None
    """
    point: np.ndarray
    index: int
    axis: int
    left: Optional["_KDNode"] = None
    right: Optional["_KDNode"] = None


class KDTree3D:
    """A simple 3-dimensional KD-tree for spatial nearest-neighbor queries.

    Why a KD-tree in GNC?
    ---------------------
    During deep-space navigation, the guidance system must rapidly identify
    the nearest celestial body (for gravitational influence) or the closest
    waypoint on a pre-computed trajectory grid.  Brute-force O(n) scans
    become expensive when the catalog is large (thousands of asteroids, a
    dense waypoint mesh, or debris-field particles).  A KD-tree partitions
    3-D space into axis-aligned half-spaces, enabling average-case O(log n)
    nearest-neighbor lookup and O(k + log n) radius queries.

    Time complexity
    ---------------
    +---------------------+----------------------------+
    | Operation           | Average / Worst            |
    +=====================+============================+
    | build (constructor) | O(n log n) / O(n log n)    |
    | query_nearest       | O(log n)   / O(n)          |
    | query_radius        | O(k+log n) / O(n)          |
    +---------------------+----------------------------+
    ``n`` = number of points, ``k`` = number of points inside the radius.
    Worst case degrades when the point distribution is pathological.

    Memory layout
    -------------
    Each tree node is a small Python dataclass holding a 3-element NumPy
    array.  The tree is pointer-based (left/right children).  For extremely
    large catalogs (>10^6 points) a flat array layout would be preferable,
    but for typical GNC workloads (hundreds to low-thousands of bodies) the
    pointer-based layout keeps the code clear with negligible overhead.

    Parameters
    ----------
    points : np.ndarray
        Shape ``(n, 3)`` array of 3-D positions (e.g., in ECI km).
    """

    def __init__(self, points: np.ndarray) -> None:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected (n, 3) array, got shape {points.shape}")

        self._size: int = len(points)
        self._points: np.ndarray = points.copy()

        # Build index list and construct recursively.
        indices = list(range(self._size))
        self._root: Optional[_KDNode] = self._build(indices, depth=0)

    # -- construction ------------------------------------------------------

    def _build(self, indices: List[int], depth: int) -> Optional[_KDNode]:
        """Recursively build the KD-tree by median-split on cycling axes.

        Parameters
        ----------
        indices : list of int
            Subset of point indices to partition.
        depth : int
            Current recursion depth -- determines splitting axis.

        Returns
        -------
        _KDNode or None
        """
        if not indices:
            return None

        axis = depth % 3
        # Sort by the splitting axis coordinate and pick the median.
        indices.sort(key=lambda i: self._points[i, axis])
        mid = len(indices) // 2

        return _KDNode(
            point=self._points[indices[mid]],
            index=indices[mid],
            axis=axis,
            left=self._build(indices[:mid], depth + 1),
            right=self._build(indices[mid + 1:], depth + 1),
        )

    # -- queries -----------------------------------------------------------

    def query_nearest(
        self, target: np.ndarray
    ) -> Tuple[int, float]:
        """Find the nearest point to *target*.

        Parameters
        ----------
        target : array-like, shape (3,)
            Query position in the same frame as the stored points.

        Returns
        -------
        index : int
            Index into the original ``points`` array.
        distance : float
            Euclidean distance to that point.

        Raises
        ------
        RuntimeError
            If the tree is empty.

        Complexity
        ----------
        Average O(log n), worst O(n).
        """
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (3,):
            raise ValueError(f"Expected shape (3,), got {target.shape}")
        if self._root is None:
            raise RuntimeError("Cannot query an empty KDTree3D")

        best_idx: int = self._root.index
        best_dist_sq: float = float("inf")

        def _search(node: Optional[_KDNode]) -> None:
            nonlocal best_idx, best_dist_sq
            if node is None:
                return

            dist_sq = float(np.sum((node.point - target) ** 2))
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_idx = node.index

            axis = node.axis
            diff = target[axis] - node.point[axis]

            # Visit the side of the splitting plane that contains the target
            # first (more likely to prune the other side).
            near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)
            _search(near)

            # Only visit the far side if the splitting plane is closer than
            # the current best -- this is the key pruning step.
            if diff * diff < best_dist_sq:
                _search(far)

        _search(self._root)
        return best_idx, math.sqrt(best_dist_sq)

    def query_radius(
        self, center: np.ndarray, radius: float
    ) -> List[Tuple[int, float]]:
        """Find all points within *radius* of *center*.

        Parameters
        ----------
        center : array-like, shape (3,)
        radius : float
            Search radius (same units as the stored coordinates).

        Returns
        -------
        list of (index, distance)
            Sorted by ascending distance.

        Complexity
        ----------
        Average O(k + log n), worst O(n).
        """
        center = np.asarray(center, dtype=np.float64)
        if center.shape != (3,):
            raise ValueError(f"Expected shape (3,), got {center.shape}")

        radius_sq = radius * radius
        results: List[Tuple[int, float]] = []

        def _search(node: Optional[_KDNode]) -> None:
            if node is None:
                return

            dist_sq = float(np.sum((node.point - center) ** 2))
            if dist_sq <= radius_sq:
                results.append((node.index, math.sqrt(dist_sq)))

            axis = node.axis
            diff = center[axis] - node.point[axis]

            near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)
            _search(near)

            if diff * diff <= radius_sq:
                _search(far)

        _search(self._root)
        results.sort(key=lambda pair: pair[1])
        return results

    def query_k_nearest(
        self, target: np.ndarray, k: int
    ) -> List[Tuple[int, float]]:
        """Return the *k* closest points to *target*.

        Uses a max-heap of size *k* so that the farthest candidate is
        always at the top and can be evicted in O(log k).

        Parameters
        ----------
        target : array-like, shape (3,)
        k : int
            Number of neighbors to return.

        Returns
        -------
        list of (index, distance)
            Sorted by ascending distance.

        Complexity
        ----------
        Average O(k log k + log n), worst O(n log k).
        """
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (3,):
            raise ValueError(f"Expected shape (3,), got {target.shape}")
        k = min(k, self._size)
        if k <= 0 or self._root is None:
            return []

        # Max-heap (negate distance so heapq min-heap behaves as max-heap).
        heap: List[Tuple[float, int]] = []  # (-dist_sq, index)

        def _search(node: Optional[_KDNode]) -> None:
            if node is None:
                return

            dist_sq = float(np.sum((node.point - target) ** 2))

            if len(heap) < k:
                heapq.heappush(heap, (-dist_sq, node.index))
            elif dist_sq < -heap[0][0]:
                heapq.heapreplace(heap, (-dist_sq, node.index))

            axis = node.axis
            diff = target[axis] - node.point[axis]
            near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)

            _search(near)
            # Prune: only explore far side if the plane is closer than
            # the worst candidate in the heap.
            if len(heap) < k or diff * diff < -heap[0][0]:
                _search(far)

        _search(self._root)
        results = [(idx, math.sqrt(-neg_d2)) for neg_d2, idx in heap]
        results.sort(key=lambda pair: pair[1])
        return results

    # -- metadata ----------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of points stored in the tree."""
        return self._size

    def get_point(self, index: int) -> np.ndarray:
        """Retrieve the original coordinates for *index*.  O(1)."""
        return self._points[index].copy()

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"KDTree3D(n={self._size})"


# ---------------------------------------------------------------------------
# 4. TelemetryBuffer
# ---------------------------------------------------------------------------

class TelemetryBuffer:
    """Double-buffered telemetry storage.

    Why double-buffering for telemetry?
    ------------------------------------
    On a real spacecraft, sensor drivers produce telemetry at high frequency
    while the downlink subsystem reads completed frames at its own pace.  A
    naive shared buffer requires locking on every read and write.  Double
    buffering eliminates contention: one buffer (the *write buffer*)
    accumulates new samples while the other (the *read buffer*) is handed
    off to consumers.  When the writer finishes a frame, the two buffers
    are swapped atomically (pointer swap -- O(1)).  This pattern is common
    in graphics (front/back framebuffers) and in real-time telemetry.

    Although CPython's GIL prevents true data races, the design still
    isolates read and write indices, making it trivially safe and a clean
    demonstration of the concurrent-access pattern.

    Time complexity
    ---------------
    +-------------------+------+
    | Operation         | Cost |
    +===================+======+
    | write             | O(1) |
    | swap              | O(1) |
    | read_buffer       | O(k) |
    | write_buffer_view | O(1) |
    | is_write_full     | O(1) |
    +-------------------+------+

    Memory layout
    -------------
    Two NumPy arrays of shape ``(frame_size, channel_count)`` are
    pre-allocated.  ``_write_idx`` and ``_read_idx`` select which array is
    currently being written to and which is available for reading.
    ``_write_pos`` tracks how many rows have been written into the current
    write frame.

    Parameters
    ----------
    frame_size : int
        Number of telemetry samples per frame (rows).
    channel_count : int
        Number of telemetry channels per sample (columns).  For example,
        a 6-axis IMU produces 6 channels (3 accel + 3 gyro).
    """

    def __init__(self, frame_size: int, channel_count: int) -> None:
        if frame_size <= 0 or channel_count <= 0:
            raise ValueError("frame_size and channel_count must be positive")

        self._frame_size: int = frame_size
        self._channel_count: int = channel_count

        self._buffers: List[np.ndarray] = [
            np.zeros((frame_size, channel_count), dtype=np.float64),
            np.zeros((frame_size, channel_count), dtype=np.float64),
        ]

        self._write_idx: int = 0   # Index into _buffers for writing.
        self._read_idx: int = 1    # Index into _buffers for reading.
        self._write_pos: int = 0   # Row cursor in the write buffer.
        self._read_count: int = 0  # Rows valid in the read buffer.
        self._swap_count: int = 0  # Total number of swaps performed.

    # -- write side --------------------------------------------------------

    def write(self, sample: np.ndarray) -> bool:
        """Append a single telemetry sample to the write buffer.

        Parameters
        ----------
        sample : np.ndarray
            Shape ``(channel_count,)`` -- one row of telemetry data.

        Returns
        -------
        bool
            ``True`` if the write succeeded; ``False`` if the write buffer
            is full (caller should :meth:`swap` first).

        Complexity
        ----------
        O(1) -- indexed write into pre-allocated array.
        """
        if self._write_pos >= self._frame_size:
            return False  # Buffer full; caller must swap.

        sample = np.asarray(sample, dtype=np.float64)
        if sample.shape != (self._channel_count,):
            raise ValueError(
                f"Expected sample shape ({self._channel_count},), got {sample.shape}"
            )

        self._buffers[self._write_idx][self._write_pos] = sample
        self._write_pos += 1
        return True

    def write_batch(self, samples: np.ndarray) -> int:
        """Write as many rows from *samples* as the buffer can accept.

        Parameters
        ----------
        samples : np.ndarray
            Shape ``(m, channel_count)``.

        Returns
        -------
        int
            Number of rows actually written.
        """
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != self._channel_count:
            raise ValueError(
                f"Expected shape (m, {self._channel_count}), got {samples.shape}"
            )

        space = self._frame_size - self._write_pos
        n = min(len(samples), space)
        if n > 0:
            self._buffers[self._write_idx][self._write_pos: self._write_pos + n] = samples[:n]
            self._write_pos += n
        return n

    def is_write_full(self) -> bool:
        """Return ``True`` when the write buffer has no remaining space."""
        return self._write_pos >= self._frame_size

    def write_buffer_usage(self) -> Tuple[int, int]:
        """Return ``(used, capacity)`` for the current write buffer."""
        return self._write_pos, self._frame_size

    # -- swap --------------------------------------------------------------

    def swap(self) -> int:
        """Swap the write and read buffers.

        After swapping, the old write buffer becomes the new read buffer
        (with its accumulated data), and the old read buffer is zeroed and
        becomes the new write target.

        Returns
        -------
        int
            The number of valid rows now available in the read buffer.

        Complexity
        ----------
        O(1) -- pointer (index) swap plus a single ``ndarray.fill`` to
        clear the new write buffer.  The fill touches every element but is
        a highly optimised memset internally.
        """
        # Publish the write buffer for readers.
        self._read_count = self._write_pos

        # Swap indices.
        self._write_idx, self._read_idx = self._read_idx, self._write_idx

        # Reset the (new) write buffer.
        self._buffers[self._write_idx][:] = 0.0
        self._write_pos = 0
        self._swap_count += 1

        return self._read_count

    # -- read side ---------------------------------------------------------

    def read_buffer(self) -> np.ndarray:
        """Return a *copy* of the valid portion of the read buffer.

        Returns
        -------
        np.ndarray
            Shape ``(read_count, channel_count)``.  Returns an empty array
            if no swap has occurred yet.

        Complexity
        ----------
        O(k) where *k* is the number of valid rows.
        """
        return self._buffers[self._read_idx][: self._read_count].copy()

    def read_buffer_view(self) -> np.ndarray:
        """Return a *view* (zero-copy) of the valid read data.

        **Caution:** the returned array shares memory with the internal
        buffer.  It remains valid only until the next :meth:`swap`.
        """
        return self._buffers[self._read_idx][: self._read_count]

    @property
    def read_count(self) -> int:
        """Number of valid rows in the current read buffer."""
        return self._read_count

    @property
    def swap_count(self) -> int:
        """Total number of buffer swaps performed so far."""
        return self._swap_count

    # -- metadata ----------------------------------------------------------

    def memory_usage(self) -> int:
        """Approximate memory in bytes (both buffers)."""
        return sum(b.nbytes for b in self._buffers)

    def __repr__(self) -> str:
        used, cap = self.write_buffer_usage()
        return (
            f"TelemetryBuffer(frame={self._frame_size}x{self._channel_count}, "
            f"write={used}/{cap}, reads={self._read_count}, swaps={self._swap_count})"
        )


# ---------------------------------------------------------------------------
# 5. MissionGraph
# ---------------------------------------------------------------------------

@dataclass
class PhaseNode:
    """A mission phase (node in the :class:`MissionGraph`).

    Attributes
    ----------
    name : str
        Unique human-readable identifier (e.g. ``"LEO_Parking"``).
    data : dict
        Arbitrary metadata -- duration estimates, required delta-V, etc.
    """
    name: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Transition:
    """A directed edge between two mission phases.

    Attributes
    ----------
    source : str
        Name of the origin phase.
    target : str
        Name of the destination phase.
    cost : float
        Transition cost (delta-V in m/s, time in seconds, or an abstract
        weight -- interpretation is up to the mission planner).
    condition : callable or None
        An optional predicate ``(state_dict) -> bool`` that must return
        ``True`` for the transition to be enabled.
    data : dict
        Extra metadata (engine mode, required attitude, etc.).
    """
    source: str
    target: str
    cost: float = 1.0
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    data: Dict[str, Any] = field(default_factory=dict)


class MissionGraph:
    """Directed weighted graph of mission phases and transitions.

    Why a graph for mission planning?
    ----------------------------------
    A spacecraft mission is naturally modeled as a finite state machine
    whose states are *mission phases* (launch, LEO parking orbit, trans-
    lunar injection, lunar orbit insertion, powered descent, landing) and
    whose edges are *transitions* gated by physical conditions (delta-V
    budget, attitude alignment, ground-station contact).  Representing
    this as an explicit directed graph enables:

    * **Validation** -- check that every phase is reachable from the
      initial state and that no deadlocks exist.
    * **Optimal sequencing** -- find the lowest-cost (delta-V, time)
      path from the current phase to a goal phase via Dijkstra.
    * **Contingency analysis** -- enumerate alternative paths when a
      primary transition is disabled.

    Time complexity
    ---------------
    +-------------------+----------------------------+
    | Operation         | Cost                       |
    +===================+============================+
    | add_phase         | O(1)                       |
    | add_transition    | O(1)                       |
    | get_next_phases   | O(degree)                  |
    | can_transition    | O(degree)                  |
    | shortest_path     | O((V+E) log V)  (Dijkstra) |
    | all_paths         | O(V!) worst case (DFS)     |
    +-------------------+----------------------------+

    Memory layout
    -------------
    Adjacency list representation: a ``dict`` maps each phase name to a
    list of :class:`Transition` objects.  This is space-efficient for the
    sparse graphs typical of mission planning (tens of nodes, tens of
    edges).

    Parameters
    ----------
    None.  Phases and transitions are added incrementally.
    """

    def __init__(self) -> None:
        self._phases: Dict[str, PhaseNode] = {}
        self._adjacency: Dict[str, List[Transition]] = {}

    # -- construction ------------------------------------------------------

    def add_phase(
        self, name: str, data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a new mission phase.

        Parameters
        ----------
        name : str
            Unique phase identifier.
        data : dict, optional
            Arbitrary metadata for the phase.

        Raises
        ------
        ValueError
            If the phase already exists.
        """
        if name in self._phases:
            raise ValueError(f"Phase '{name}' already exists")
        self._phases[name] = PhaseNode(name, data or {})
        self._adjacency[name] = []

    def add_transition(
        self,
        source: str,
        target: str,
        cost: float = 1.0,
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a directed edge from *source* to *target*.

        Parameters
        ----------
        source, target : str
            Phase names (must already exist).
        cost : float
            Non-negative transition cost.
        condition : callable, optional
            ``(state_dict) -> bool`` predicate gating the transition.
        data : dict, optional

        Raises
        ------
        KeyError
            If either phase has not been added.
        ValueError
            If cost is negative.
        """
        if source not in self._phases:
            raise KeyError(f"Source phase '{source}' not found")
        if target not in self._phases:
            raise KeyError(f"Target phase '{target}' not found")
        if cost < 0:
            raise ValueError(f"Transition cost must be non-negative, got {cost}")

        self._adjacency[source].append(
            Transition(source, target, cost, condition, data or {})
        )

    # -- queries -----------------------------------------------------------

    def get_next_phases(
        self,
        phase: str,
        state: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Return reachable phases from *phase*, optionally filtering by
        transition conditions evaluated against *state*.

        Parameters
        ----------
        phase : str
        state : dict, optional
            If provided, only transitions whose condition returns ``True``
            (or that have no condition) are included.

        Returns
        -------
        list of str
        """
        if phase not in self._adjacency:
            raise KeyError(f"Phase '{phase}' not found")

        results: List[str] = []
        for t in self._adjacency[phase]:
            if state is not None and t.condition is not None:
                if not t.condition(state):
                    continue
            results.append(t.target)
        return results

    def can_transition(
        self, source: str, target: str, state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Return ``True`` if a direct edge from *source* to *target*
        exists and (optionally) its condition is satisfied.

        Parameters
        ----------
        source, target : str
        state : dict, optional

        Complexity
        ----------
        O(degree of source).
        """
        if source not in self._adjacency:
            raise KeyError(f"Phase '{source}' not found")

        for t in self._adjacency[source]:
            if t.target != target:
                continue
            if state is not None and t.condition is not None:
                if not t.condition(state):
                    continue
            return True
        return False

    # -- shortest path (Dijkstra) ------------------------------------------

    def shortest_path(
        self,
        start: str,
        goal: str,
        state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], float]:
        """Compute the minimum-cost path from *start* to *goal* using
        Dijkstra's algorithm.

        Parameters
        ----------
        start, goal : str
            Phase names.
        state : dict, optional
            If given, only transitions whose conditions pass are traversed.

        Returns
        -------
        path : list of str
            Sequence of phase names from *start* to *goal* inclusive.
        cost : float
            Total transition cost along the path.

        Raises
        ------
        KeyError
            If start or goal not in graph.
        ValueError
            If no path exists.

        Complexity
        ----------
        O((V + E) log V) using a binary min-heap.
        """
        if start not in self._phases:
            raise KeyError(f"Phase '{start}' not found")
        if goal not in self._phases:
            raise KeyError(f"Phase '{goal}' not found")

        # dist[phase] = best known cost from start.
        dist: Dict[str, float] = {name: float("inf") for name in self._phases}
        dist[start] = 0.0
        prev: Dict[str, Optional[str]] = {name: None for name in self._phases}

        # Min-heap entries: (cost, phase_name).
        heap: List[Tuple[float, str]] = [(0.0, start)]
        visited: Set[str] = set()

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)

            if u == goal:
                break

            for t in self._adjacency[u]:
                if t.target in visited:
                    continue
                if state is not None and t.condition is not None:
                    if not t.condition(state):
                        continue
                nd = d + t.cost
                if nd < dist[t.target]:
                    dist[t.target] = nd
                    prev[t.target] = u
                    heapq.heappush(heap, (nd, t.target))

        if dist[goal] == float("inf"):
            raise ValueError(
                f"No feasible path from '{start}' to '{goal}'"
            )

        # Reconstruct path.
        path: List[str] = []
        node: Optional[str] = goal
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()
        return path, dist[goal]

    # -- topology helpers --------------------------------------------------

    def phases(self) -> List[str]:
        """Return all phase names (insertion order)."""
        return list(self._phases.keys())

    def transitions_from(self, phase: str) -> List[Transition]:
        """Return all outgoing transitions from *phase*."""
        if phase not in self._adjacency:
            raise KeyError(f"Phase '{phase}' not found")
        return list(self._adjacency[phase])

    @property
    def num_phases(self) -> int:
        return len(self._phases)

    @property
    def num_transitions(self) -> int:
        return sum(len(edges) for edges in self._adjacency.values())

    def get_phase(self, name: str) -> PhaseNode:
        """Retrieve the :class:`PhaseNode` for *name*."""
        if name not in self._phases:
            raise KeyError(f"Phase '{name}' not found")
        return self._phases[name]

    def __contains__(self, phase: str) -> bool:
        return phase in self._phases

    def __repr__(self) -> str:
        return (
            f"MissionGraph(phases={self.num_phases}, "
            f"transitions={self.num_transitions})"
        )


# ---------------------------------------------------------------------------
# 6. DPTable
# ---------------------------------------------------------------------------

class DPTable:
    """Dynamic-programming memoization table for trajectory optimisation.

    Why DP for trajectory planning?
    --------------------------------
    Many GNC problems are naturally formulated as optimal-control problems
    on a discretised state grid.  For example, minimum-fuel orbit transfer
    can be solved backwards in time: for each grid point, store the
    *cost-to-go* (remaining delta-V to reach the target orbit) and the
    *optimal control* (thrust direction and magnitude).  The Bellman
    recursion propagates these values from the terminal state back to the
    initial state, yielding a globally optimal policy.

    This class provides a thin wrapper around a NumPy N-D array that
    represents the discretised state space plus a time axis.  It supports:

    * **Direct lookup** -- O(1) per grid cell.
    * **Trilinear interpolation** -- for querying costs at states that
      fall between grid nodes.
    * **Optimal-path extraction** -- greedy walk from any start cell to
      the terminal state following the gradient of the cost-to-go.

    Memory layout
    -------------
    The cost-to-go values are stored in a single contiguous NumPy array
    of shape ``(n_time, *grid_shape)`` so that the time axis is the
    outermost dimension.  This layout is cache-friendly for the backwards
    sweep (iterate over decreasing time index, visiting the full spatial
    grid at each step).  An optional ``_policy`` array of the same shape
    (plus trailing control dimensions) stores the optimal control at each
    cell.

    Parameters
    ----------
    grid_shape : tuple of int
        Number of nodes along each spatial dimension.  For a 3-D state
        space (e.g., energy, inclination, RAAN) this might be (50, 30, 60).
    n_time : int
        Number of time steps.
    state_bounds : np.ndarray
        Shape ``(ndim, 2)`` giving ``[low, high]`` for each spatial axis.
        Used to map between physical coordinates and grid indices.
    control_dim : int, optional
        If provided, allocate a parallel policy table of shape
        ``(n_time, *grid_shape, control_dim)`` for storing the optimal
        control vector at each cell.
    default_value : float
        Initial fill value for the cost table (typically ``np.inf`` for a
        minimisation problem).
    """

    def __init__(
        self,
        grid_shape: Tuple[int, ...],
        n_time: int,
        state_bounds: np.ndarray,
        control_dim: int = 0,
        default_value: float = np.inf,
    ) -> None:
        if n_time <= 0:
            raise ValueError(f"n_time must be positive, got {n_time}")
        state_bounds = np.asarray(state_bounds, dtype=np.float64)
        if state_bounds.shape != (len(grid_shape), 2):
            raise ValueError(
                f"state_bounds shape {state_bounds.shape} does not match "
                f"grid_shape dimensionality {len(grid_shape)}"
            )

        self._grid_shape: Tuple[int, ...] = tuple(grid_shape)
        self._n_time: int = n_time
        self._ndim: int = len(grid_shape)
        self._state_bounds: np.ndarray = state_bounds.copy()
        self._default_value: float = default_value

        # Pre-compute per-axis step sizes for coordinate <-> index mapping.
        self._steps: np.ndarray = np.array([
            (state_bounds[d, 1] - state_bounds[d, 0]) / max(grid_shape[d] - 1, 1)
            for d in range(self._ndim)
        ], dtype=np.float64)

        # Cost-to-go table.
        full_shape = (n_time, *grid_shape)
        self._table: np.ndarray = np.full(full_shape, default_value, dtype=np.float64)

        # Optional policy table.
        self._control_dim: int = control_dim
        self._policy: Optional[np.ndarray] = None
        if control_dim > 0:
            self._policy = np.zeros((*full_shape, control_dim), dtype=np.float64)

    # -- coordinate helpers ------------------------------------------------

    def _state_to_index(self, state: np.ndarray) -> Tuple[int, ...]:
        """Map a continuous state vector to the nearest grid index.

        Parameters
        ----------
        state : np.ndarray, shape ``(ndim,)``

        Returns
        -------
        tuple of int
            Clamped to valid grid bounds.
        """
        idx: List[int] = []
        for d in range(self._ndim):
            lo, hi = self._state_bounds[d]
            raw = (state[d] - lo) / self._steps[d] if self._steps[d] != 0 else 0.0
            clamped = int(np.clip(round(raw), 0, self._grid_shape[d] - 1))
            idx.append(clamped)
        return tuple(idx)

    def _state_to_continuous_index(self, state: np.ndarray) -> np.ndarray:
        """Map to fractional grid indices (for interpolation).

        Returns
        -------
        np.ndarray, shape ``(ndim,)``
            Clamped to ``[0, grid_shape[d]-1]`` per axis.
        """
        cidx = np.empty(self._ndim, dtype=np.float64)
        for d in range(self._ndim):
            lo = self._state_bounds[d, 0]
            raw = (state[d] - lo) / self._steps[d] if self._steps[d] != 0 else 0.0
            cidx[d] = np.clip(raw, 0.0, self._grid_shape[d] - 1)
        return cidx

    def _index_to_state(self, index: Tuple[int, ...]) -> np.ndarray:
        """Map a grid index back to a physical state coordinate."""
        return np.array([
            self._state_bounds[d, 0] + index[d] * self._steps[d]
            for d in range(self._ndim)
        ], dtype=np.float64)

    # -- read / write ------------------------------------------------------

    def set_value(
        self,
        time_idx: int,
        state: np.ndarray,
        value: float,
        control: Optional[np.ndarray] = None,
    ) -> None:
        """Store a cost-to-go value (and optionally the optimal control)
        at the grid cell closest to *state* at time step *time_idx*.

        Parameters
        ----------
        time_idx : int
        state : np.ndarray, shape ``(ndim,)``
        value : float
        control : np.ndarray, optional
            Shape ``(control_dim,)`` -- stored in the policy table.

        Complexity
        ----------
        O(1) -- single indexed write.
        """
        idx = self._state_to_index(np.asarray(state, dtype=np.float64))
        self._table[(time_idx, *idx)] = value

        if control is not None and self._policy is not None:
            self._policy[(time_idx, *idx)] = np.asarray(control, dtype=np.float64)

    def get_value(self, time_idx: int, state: np.ndarray) -> float:
        """Return the cost-to-go at the nearest grid cell.  O(1)."""
        idx = self._state_to_index(np.asarray(state, dtype=np.float64))
        return float(self._table[(time_idx, *idx)])

    def get_control(self, time_idx: int, state: np.ndarray) -> Optional[np.ndarray]:
        """Return the stored optimal control vector at the nearest cell.

        Returns ``None`` if no policy table was allocated.
        """
        if self._policy is None:
            return None
        idx = self._state_to_index(np.asarray(state, dtype=np.float64))
        return self._policy[(time_idx, *idx)].copy()

    # -- interpolation -----------------------------------------------------

    def interpolate(self, time_idx: int, state: np.ndarray) -> float:
        """Return the cost-to-go interpolated multilinearly at *state*.

        For a 3-D grid this is *trilinear* interpolation; the method
        generalises to any dimensionality via recursive linear interpolation
        along each axis.

        Parameters
        ----------
        time_idx : int
        state : np.ndarray, shape ``(ndim,)``

        Returns
        -------
        float
            Interpolated cost-to-go.

        Complexity
        ----------
        O(2^ndim) -- evaluate all corners of the enclosing hyper-cell.
        """
        state = np.asarray(state, dtype=np.float64)
        cidx = self._state_to_continuous_index(state)

        # For each axis, compute the two bracketing integer indices and
        # the fractional weight.
        lowers: List[int] = []
        uppers: List[int] = []
        fracs: List[float] = []
        for d in range(self._ndim):
            lo = int(math.floor(cidx[d]))
            hi = min(lo + 1, self._grid_shape[d] - 1)
            frac = cidx[d] - lo
            lowers.append(lo)
            uppers.append(hi)
            fracs.append(frac)

        # Enumerate all 2^ndim corners of the bounding hyper-cell.
        n_corners = 1 << self._ndim
        result = 0.0
        for mask in range(n_corners):
            weight = 1.0
            corner: List[int] = []
            for d in range(self._ndim):
                if mask & (1 << d):
                    corner.append(uppers[d])
                    weight *= fracs[d]
                else:
                    corner.append(lowers[d])
                    weight *= (1.0 - fracs[d])
            val = float(self._table[(time_idx, *corner)])
            # Skip infinite cells so they do not corrupt the interpolation
            # when the grid is only partially filled.
            if not math.isfinite(val):
                continue
            result += weight * val

        return result

    # -- optimal-path extraction -------------------------------------------

    def get_optimal_path(
        self,
        start_time_idx: int,
        start_state: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[float], List[Optional[np.ndarray]]]:
        """Extract the optimal state trajectory by greedily following the
        cost-to-go gradient forward in time.

        Starting from ``(start_time_idx, start_state)``, the method looks
        up the stored policy control at the nearest cell and records the
        state and cost at each step until the final time index is reached.

        Parameters
        ----------
        start_time_idx : int
        start_state : np.ndarray, shape ``(ndim,)``

        Returns
        -------
        states : list of np.ndarray
            The state at each time step along the optimal path.
        costs : list of float
            The cost-to-go at each step.
        controls : list of np.ndarray or None
            The optimal control at each step (``None`` entries if no policy
            table exists).

        Notes
        -----
        If no policy table is allocated, the method falls back to a
        *greedy neighbor search*: at each time step it evaluates all
        adjacent grid cells at the next time index and moves to the one
        with the lowest cost-to-go.

        Complexity
        ----------
        O(n_time * 3^ndim) worst case for the greedy neighbor fallback.
        O(n_time) when a policy table is available.
        """
        start_state = np.asarray(start_state, dtype=np.float64)
        states: List[np.ndarray] = [start_state.copy()]
        costs: List[float] = [self.get_value(start_time_idx, start_state)]
        controls: List[Optional[np.ndarray]] = []

        current_idx = self._state_to_index(start_state)

        for t in range(start_time_idx, self._n_time - 1):
            ctrl = self.get_control(t, self._index_to_state(current_idx))
            controls.append(ctrl)

            if self._policy is not None and ctrl is not None:
                # With a policy table we could propagate the dynamics; here
                # we simply step to the nearest cell (placeholder for a
                # full integrator hook).
                next_idx = self._greedy_step(t + 1, current_idx)
            else:
                next_idx = self._greedy_step(t + 1, current_idx)

            current_idx = next_idx
            s = self._index_to_state(current_idx)
            states.append(s)
            costs.append(float(self._table[(t + 1, *current_idx)]))

        # Final step has no outgoing control.
        controls.append(None)
        return states, costs, controls

    def _greedy_step(
        self, next_time: int, current_idx: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """Move to the neighbor (including self) with the lowest cost-to-go
        at *next_time*.

        Explores all cells within +/-1 in each dimension (3^ndim neighbors).
        """
        best_val = float("inf")
        best_idx = current_idx

        # Generate neighbor offsets for each dimension (-1, 0, +1).
        offsets = [(-1, 0, 1)] * self._ndim
        for combo in _cartesian_product(offsets):
            neighbor: List[int] = []
            valid = True
            for d in range(self._ndim):
                ni = current_idx[d] + combo[d]
                if ni < 0 or ni >= self._grid_shape[d]:
                    valid = False
                    break
                neighbor.append(ni)
            if not valid:
                continue
            ntuple = tuple(neighbor)
            val = float(self._table[(next_time, *ntuple)])
            if val < best_val:
                best_val = val
                best_idx = ntuple

        return best_idx

    # -- metadata ----------------------------------------------------------

    def memory_usage(self) -> int:
        """Approximate memory in bytes for the cost and policy tables."""
        total = self._table.nbytes
        if self._policy is not None:
            total += self._policy.nbytes
        return int(total)

    def filled_fraction(self) -> float:
        """Fraction of cost-table cells that hold finite values."""
        finite = np.isfinite(self._table).sum()
        return float(finite) / self._table.size

    @property
    def grid_shape(self) -> Tuple[int, ...]:
        return self._grid_shape

    @property
    def n_time(self) -> int:
        return self._n_time

    def reset(self) -> None:
        """Reset all cells to the default value."""
        self._table[:] = self._default_value
        if self._policy is not None:
            self._policy[:] = 0.0

    def __repr__(self) -> str:
        shape_str = "x".join(str(s) for s in self._grid_shape)
        mem_kb = self.memory_usage() / 1024
        return (
            f"DPTable(time={self._n_time}, grid={shape_str}, "
            f"filled={self.filled_fraction():.1%}, mem={mem_kb:.1f} KiB)"
        )


# ---------------------------------------------------------------------------
# Private utility
# ---------------------------------------------------------------------------

def _cartesian_product(pools: List[Tuple[int, ...]]) -> Iterator[Tuple[int, ...]]:
    """Yield the Cartesian product of the given index pools.

    Equivalent to ``itertools.product(*pools)`` but avoids the import and
    keeps the dependency list minimal.
    """
    result: List[Tuple[int, ...]] = [()]
    for pool in pools:
        result = [x + (y,) for x in result for y in pool]
    yield from result
