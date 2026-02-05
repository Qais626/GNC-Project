# GNC Project Interview Preparation Guide

## Complete Technical Deep-Dive for Your Interview

This guide covers **everything** in your GNC project in interview-ready format.

---

# Table of Contents

1. [Project Overview](#project-overview)
2. [Control Systems](#control-systems)
3. [Navigation & State Estimation](#navigation--state-estimation)
4. [Orbital Mechanics](#orbital-mechanics)
5. [Real-Time Systems (C++)](#real-time-systems-c)
6. [Software Architecture](#software-architecture)
7. [Challenges & Solutions](#challenges--solutions)
8. [Key Interview Questions & Answers](#key-interview-questions--answers)

---

# Project Overview

## Mission Summary
**"Design and simulate a GNC system for a spacecraft mission from Miami to the Moon (2 orbits), to Jupiter (3 orbits), and back to Miami."**

### Key Numbers to Remember:
- **Mission Duration**: ~4.1 years
- **Total ΔV Budget**: ~10.5 km/s
- **Spacecraft Mass**: 6,000 kg (3,200 kg dry + 2,800 kg propellant)
- **18 Mission Phases**: Pre-launch → Ascent → Parking → TLI → Lunar Ops → Jupiter Transfer → Jupiter Ops → Return → Reentry

### Languages & Tools:
- **Python**: Core simulation, algorithms (~15,000 lines)
- **C++**: Real-time flight software components (~3,000 lines)
- **MATLAB/Simulink**: Control system verification
- **YAML**: Mission configuration

---

# Control Systems

## 1. PID Controller (Classical)

**File**: `control/attitude_control.py` - `PIDController` class

**What it does**: Three-axis attitude control with proportional, integral, and derivative terms.

**Key equation**:
```
τ = Kp·e + Ki·∫e·dt + Kd·(de/dt)
```

**Special features**:
- **Anti-windup**: Clamps integrator to prevent saturation
- **Derivative filtering**: Low-pass filter on derivative to reduce noise
- **Per-axis tuning**: Different gains for roll, pitch, yaw

**When to use**: Detumble mode, coarse pointing

**Interview answer**: *"I implemented a PID controller with anti-windup and derivative filtering because raw derivative amplifies high-frequency noise. The anti-windup prevents integrator saturation during large errors."*

---

## 2. LQR Controller (Optimal)

**File**: `control/attitude_control.py` - `LQRController` class

**What it does**: Minimizes a quadratic cost function to find optimal gains.

**Key concept**:
```
Minimize: J = ∫(x'Qx + u'Ru)dt

State: x = [θ_err, ω_err] (6×1)
Control: u = τ (3×1)
Feedback law: u = -Kx
```

**How I implemented it**:
1. Build linearized state-space model (A, B matrices)
2. Solve the Continuous Algebraic Riccati Equation (CARE)
3. Compute optimal gain: K = R⁻¹B'P

**Interview answer**: *"LQR provides optimal full-state feedback. I solve the Riccati equation iteratively since we don't have scipy.linalg in flight software. The Q matrix weights state errors, R weights control effort - I tuned these for 0.1° pointing with minimal fuel use."*

---

## 3. Sliding Mode Control (Robust)

**File**: `control/attitude_control.py` - `SlidingModeController` class

**What it does**: Robust to modeling errors and disturbances.

**Key concepts**:
```
Sliding surface: s = ω_err + Λ·θ_err
Control law: u = u_eq + u_sw
  u_eq = -J(Λ·ω_err) + ω×(Jω)  (equivalent control)
  u_sw = -η·sat(s/φ)           (switching control)
```

**The sat() function**: Replaces sgn() with saturation to reduce chattering.

**Interview answer**: *"Sliding mode is ideal for large-angle slews because it's inherently robust to bounded uncertainties. I use a boundary layer (φ) to prevent chattering - without it, the actuators would oscillate at high frequency."*

---

## 4. LQG Controller (Optimal + Estimation)

**File**: `control/optimal_control.py` - `LQGController` class

**What it does**: Combines LQR with Kalman filter for output feedback.

**Separation Principle**: You can design the estimator and controller independently; their cascade is optimal.

**Interview answer**: *"LQG is essential when you can't measure all states directly. The Kalman filter estimates the full state from noisy sensors, then LQR computes the optimal control. By the separation principle, this is still optimal."*

---

## 5. MPC (Model Predictive Control)

**File**: `control/optimal_control.py` - `MPCController` class

**What it does**: Solves optimization over a prediction horizon, applies first control.

**Key advantages**:
- Handles constraints (torque limits, rate limits)
- Preview of reference trajectory
- Receding horizon adapts to changes

**How it works**:
1. Predict future states: X = Φ·x₀ + Γ·U
2. Solve QP: min (X-X_ref)'Q(X-X_ref) + U'RU
3. Apply only u₀, repeat next step

**Interview answer**: *"MPC is the most flexible controller because it naturally handles constraints. I pre-compute the prediction matrices for efficiency. The receding horizon means we re-plan every step, making it robust to disturbances."*

---

## 6. H-infinity Controller (Robust)

**File**: `control/optimal_control.py` - `HInfinityController` class

**What it does**: Minimizes worst-case disturbance amplification.

**Key concept**: ||T_zw||∞ < γ (bound on transfer function gain)

**Interview answer**: *"H-infinity is useful when you have model uncertainty. It minimizes the worst-case gain from disturbances to outputs. The γ parameter trades off robustness vs. performance."*

---

# Navigation & State Estimation

## Extended Kalman Filter (EKF)

**File**: `navigation/ekf.py`

### State Vector (15 states):
```
x = [pos(3), vel(3), att_err(3), gyro_bias(3), accel_bias(3)]
```

### Key Innovation: Multiplicative Quaternion Error

**Why not put quaternion in state vector?**
- Quaternion has 4 elements but only 3 DOF
- Unit-norm constraint violates Gaussian assumption
- Solution: Estimate small-angle error (3 states), maintain reference quaternion separately

### Reset Step:
After each update:
1. Fold error into reference: q_ref = δq × q_ref
2. Reset error states to zero

### Joseph Form Covariance Update:
```
P = (I-KH)P(I-KH)' + KRK'
```
More numerically stable than P = (I-KH)P

**Interview answer**: *"I use the multiplicative quaternion formulation because it respects the unit-norm constraint. After each update, I reset the attitude error by folding it into the reference quaternion. The Joseph form covariance update prevents numerical issues during long-duration missions."*

---

## Sensors Modeled

| Sensor | Accuracy | Use Case |
|--------|----------|----------|
| IMU (gyro) | 0.1 mrad/s bias | Attitude rates (100 Hz) |
| Star Tracker | 5 arcsec | Absolute attitude (10 Hz) |
| Sun Sensor | 0.5° | Coarse attitude, eclipse detection |
| GPS | 10 m / 0.1 m/s | Near-Earth navigation |
| Deep Space Network | 5 m range | Jupiter cruise navigation |

---

# Orbital Mechanics

## Key Algorithms

### Hohmann Transfer
**File**: `guidance/maneuver_planner.py`

```python
ΔV₁ = √(μ/r₁) × (√(2r₂/(r₁+r₂)) - 1)
ΔV₂ = √(μ/r₂) × (1 - √(2r₁/(r₁+r₂)))
```

### Bi-elliptic Transfer
Better than Hohmann when r₂/r₁ > 11.94

### Lambert Solver
Given two positions and time of flight, find the orbit connecting them.

**Interview answer**: *"I implemented Hohmann for simple orbit raising and Lambert for arbitrary two-point boundary value problems. The Lambert solver is essential for interplanetary trajectory design."*

---

## Propagation

### RK4 Integrator (C++)
**File**: `dynamics_engine.cpp`

4th-order accuracy, 4 function evaluations per step:
```
k₁ = f(t, y)
k₂ = f(t + h/2, y + h·k₁/2)
k₃ = f(t + h/2, y + h·k₂/2)
k₄ = f(t + h, y + h·k₃)
y_next = y + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)
```

### J2 Perturbation
Earth's oblateness causes:
- Nodal regression (RAAN precession)
- Apsidal advance

---

# Real-Time Systems (C++)

## Flight Computer Architecture

**File**: `flight_computer.h`

### Rate Monotonic Scheduling (RMS)
- Tasks with shorter periods get higher priority
- Optimal for fixed-priority periodic tasks
- Schedulability bound: U ≤ n(2^(1/n) - 1)

### Task Priorities:
1. Control loop: 100 Hz (highest)
2. Navigation: 10 Hz
3. Guidance: 1 Hz
4. Telemetry: 5 Hz (lowest)

### Triple Modular Redundancy (TMR)
Run computation 3 times, take majority vote. Protects against:
- Radiation-induced bit flips (SEU)
- Transient hardware faults

**Interview answer**: *"I implemented rate-monotonic scheduling because it's provably optimal for periodic tasks. TMR provides radiation tolerance using COTS processors - we run critical calculations 3 times and vote."*

---

## Memory Pool (Zero-Allocation)

**File**: `memory_pool.h`

**Why no malloc in real-time?**
- Non-deterministic timing
- Fragmentation
- Page faults

**Solution**: Pre-allocate pools of fixed-size blocks.

```cpp
template<typename T, size_t N>
class MemoryPool {
    alignas(64) T blocks_[N];  // Cache-aligned
    T* free_list_;             // Intrusive linked list
};
```

**Interview answer**: *"Dynamic memory allocation is forbidden in flight software because malloc has non-deterministic timing. I use pre-allocated memory pools with O(1) allocate/free. The blocks are cache-line aligned for optimal performance."*

---

## Lock-Free Ring Buffer

**File**: `ring_buffer.h`

**Single-producer, single-consumer (SPSC)**:
- No locks needed
- Use atomic operations for head/tail
- Essential for ISR-to-task communication

---

# Software Architecture

## Python Package Structure

```
src/python/
├── core/
│   ├── quaternion.py     # 1,280 lines - comprehensive quaternion math
│   ├── constants.py      # Physical constants
│   ├── frames.py         # Coordinate transformations
│   └── data_structures.py
├── dynamics/
│   ├── orbital_mechanics.py
│   ├── attitude_dynamics.py
│   └── spacecraft.py
├── guidance/
│   ├── mission_planner.py
│   ├── trajectory_opt.py
│   └── maneuver_planner.py
├── navigation/
│   ├── sensors.py
│   ├── ekf.py           # 15-state Extended Kalman Filter
│   └── ukf.py
├── control/
│   ├── attitude_control.py  # PID, LQR, SMC
│   ├── optimal_control.py   # LQG, H-inf, MPC
│   └── actuators.py
├── simulation/
│   ├── sim_engine.py    # Central orchestrator
│   └── monte_carlo.py
└── autonomy/
    ├── anomaly_detection.py
    └── attitude_predictor.py
```

---

# Challenges & Solutions

## Challenge 1: Quaternion Normalization Drift

**Problem**: Numerical integration causes quaternion norm to drift from 1.

**Solution**: Renormalize after every integration step + use multiplicative error formulation in EKF.

---

## Challenge 2: Controller Sign Errors

**Problem**: Initial PD controller had wrong sign, causing divergence.

**Root Cause**: Confusion between error quaternion conventions.

**Solution**: Standardized on q_err = q_target × q_current⁻¹ with scalar-positive convention.

---

## Challenge 3: Anti-Windup

**Problem**: Large initial errors caused integrator to saturate.

**Solution**: Per-axis clamping on integrator output.

---

## Challenge 4: Real-Time Determinism

**Problem**: malloc() in hot path caused timing jitter.

**Solution**: Pre-allocated memory pools, eliminated all dynamic allocation.

---

## Challenge 5: Propellant Budget

**Problem**: Initial ΔV budget exceeded propellant capacity.

**Solution**: Identified need for staged propulsion or gravity assists (documented in analysis).

---

# Key Interview Questions & Answers

## Q1: "Walk me through your GNC architecture."

**Answer**: *"My GNC follows the classical loop: Guidance determines where we want to go, Navigation estimates where we are, and Control commands actuators to reduce the error. The sim engine runs at 1 Hz for coast and 100 Hz for burns. I use an Extended Kalman Filter with 15 states for navigation, and support multiple control laws - PID for detumble, LQR for fine pointing, sliding mode for slews, and MPC when constraints matter."*

---

## Q2: "Why quaternions instead of Euler angles?"

**Answer**: *"Euler angles have gimbal lock at ±90° pitch. Quaternions have no singularities, require only 4 parameters (vs 9 for DCM), and compose nicely via multiplication. The trade-off is the unit-norm constraint, which I handle via renormalization and the multiplicative EKF formulation."*

---

## Q3: "How do you handle sensor noise in your EKF?"

**Answer**: *"I characterize each sensor's noise as a covariance matrix R. The Kalman gain optimally trades off between trusting the model (small Q) vs trusting measurements (small R). I use the Joseph form for covariance updates to maintain numerical stability over long missions."*

---

## Q4: "What's special about your real-time C++ implementation?"

**Answer**: *"Flight software must be deterministic. I eliminated dynamic allocation with memory pools, used lock-free data structures for inter-task communication, and implemented rate-monotonic scheduling. For radiation tolerance, I added TMR (triple modular redundancy) on critical computations."*

---

## Q5: "How would you validate this system for flight?"

**Answer**: *"Multiple levels: unit tests for each algorithm, integrated simulation with Monte Carlo analysis (100+ runs with dispersions), hardware-in-the-loop testing with actual sensors, and formal verification of safety-critical paths. I've implemented the first two; HIL and formal methods would be next."*

---

## Q6: "What would you do differently?"

**Answer**: *"I'd add more sophisticated gravity assist trajectory optimization - the current direct transfer requires more propellant than available. I'd also implement an Unscented Kalman Filter for highly nonlinear phases like planetary flybys, where EKF linearization error becomes significant."*

---

# Quick Reference Card

## Equations to Know:

**Rocket Equation**: ΔV = Isp·g₀·ln(m₀/m_f)

**Hohmann ΔV**: ΔV₁ = √(μ/r₁)·(√(2r₂/(r₁+r₂)) - 1)

**Vis-Viva**: v² = μ(2/r - 1/a)

**Quaternion Kinematics**: q̇ = ½q⊗ω_q

**Euler's Equations**: I·ω̇ = τ - ω×(Iω)

**Kalman Gain**: K = PH'(HPH' + R)⁻¹

---

# Debugging & Error Resolution Log

This section documents every significant error encountered during development and how each was resolved. This demonstrates debugging skills and systematic problem-solving.

---

## Error 1: PD Controller Sign Error (Simulink)

**Symptom**: Attitude diverged instead of converging - spacecraft spun out of control.

**Root Cause**: The error quaternion convention was inconsistent. Using `q_err = q_current * q_target^(-1)` instead of `q_err = q_target * q_current^(-1)`.

**Fix**: Standardized on the convention:
```
q_err = q_target × q_current⁻¹
```
Then ensured the scalar part is always positive (short-rotation path):
```python
if q_err.w < 0:
    q_err = -q_err  # Flip to positive scalar
```

**Interview Answer**: *"The initial controller had the wrong sign because I was computing the error quaternion backwards. Quaternion multiplication is non-commutative, so `q1 * q2 ≠ q2 * q1`. I standardized on `q_err = q_target × q_current⁻¹` with scalar-positive enforcement."*

---

## Error 2: Quaternion Normalization Drift

**Symptom**: After many integration steps, quaternion norm drifted from 1.0, causing attitude corruption.

**Root Cause**: Numerical integration accumulates floating-point errors. Even small errors compound over thousands of steps.

**Fix**: Renormalize quaternion after every integration step:
```python
def normalize(q):
    norm = sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
    return Quaternion(q.w/norm, q.x/norm, q.y/norm, q.z/norm)
```

Also implemented the multiplicative quaternion formulation in the EKF, which avoids putting the full quaternion in the state vector.

**Interview Answer**: *"Quaternions have a unit-norm constraint that the Kalman filter's Gaussian assumption can't enforce. I use the multiplicative formulation where I estimate a small 3-DOF attitude error and maintain the reference quaternion separately. After each update, I fold the error into the reference and reset."*

---

## Error 3: Integrator Windup in PID Controller

**Symptom**: After large initial errors, the controller output saturated and stayed saturated even after the error reduced.

**Root Cause**: The integrator accumulated a huge value during the large error period. Even when error went to zero, the integrator term dominated.

**Fix**: Implemented anti-windup with per-axis clamping:
```python
self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
```

Also implemented back-calculation anti-windup for better performance:
```python
if abs(u_commanded) > u_max:
    self.integral -= Kb * (u_commanded - u_saturated)
```

**Interview Answer**: *"Integrator windup is a classic control problem. My PID uses clamping anti-windup to bound the integrator state. For more aggressive scenarios, I implemented back-calculation where the integrator is reduced when the actuator saturates."*

---

## Error 4: malloc() Timing Jitter in Flight Software

**Symptom**: Control loop timing varied from 8ms to 50ms unpredictably, violating real-time requirements.

**Root Cause**: Dynamic memory allocation (malloc/new) has non-deterministic timing due to heap fragmentation and potential page faults.

**Fix**: Eliminated all dynamic allocation in the hot path:
1. Pre-allocated memory pools for fixed-size objects
2. Used stack allocation for temporary variables
3. Pre-sized all vectors during initialization

```cpp
template<typename T, size_t N>
class MemoryPool {
    alignas(64) T blocks_[N];  // Cache-aligned
    T* free_list_;
public:
    T* allocate() { /* O(1) from free list */ }
    void deallocate(T* p) { /* O(1) return to free list */ }
};
```

**Interview Answer**: *"Dynamic allocation is forbidden in flight software because malloc has unbounded worst-case timing. I use pre-allocated memory pools with O(1) allocate and free. The pools are cache-line aligned (64 bytes) to prevent false sharing."*

---

## Error 5: EKF Numerical Instability (Long-Duration Mission)

**Symptom**: After several days of simulated time, the EKF covariance matrix became non-positive-definite, causing filter divergence.

**Root Cause**: The standard covariance update `P = (I - KH)P` is numerically unstable. Small errors accumulate and can make P lose symmetry and positive-definiteness.

**Fix**: Switched to the Joseph form covariance update:
```
P = (I - KH) P (I - KH)' + K R K'
```

This form is algebraically equivalent but numerically stable because it's guaranteed to produce a symmetric positive-semidefinite result.

**Interview Answer**: *"The Joseph form covariance update is critical for long-duration missions. It maintains numerical stability by explicitly preserving symmetry and positive-definiteness, which the standard form can violate due to floating-point errors."*

---

## Error 6: Propellant Budget Exceeded

**Symptom**: Mission analysis showed ~10.5 km/s total delta-V required, but rocket equation showed only ~2 km/s available with 2,800 kg propellant.

**Root Cause**: Direct Hohmann transfer to Jupiter requires enormous delta-V. The initial mass budget was unrealistic for a chemical propulsion system.

**Analysis Using Rocket Equation**:
```
ΔV = Isp × g₀ × ln(m₀/m_f)
2,000 m/s = 320s × 9.81 × ln(6000/3200)
```

For 10.5 km/s with Isp=320s:
```
m₀/m_f = exp(10500 / (320 × 9.81)) = exp(3.34) = 28.3
```
This means 96.5% of the spacecraft would need to be propellant!

**Fix**: Documented the problem and proposed solutions:
1. Use gravity assists (Venus-Earth-Earth or VEEGA)
2. Add additional propulsion stages
3. Consider electric propulsion (higher Isp)
4. Use aerocapture at Earth return

**Interview Answer**: *"I identified that direct chemical propulsion to Jupiter isn't mass-feasible. Real Jupiter missions like Galileo and Juno use gravity assists. I documented this constraint and proposed using VEEGA (Venus-Earth-Earth Gravity Assist) which reduces delta-V to ~6 km/s."*

---

## Error 7: Python Module Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'pandas'` when running simulation.

**Root Cause**: System Python on macOS uses PEP 668 restrictions that prevent pip installs without virtual environments.

**Fix**: Used `--break-system-packages` flag for quick install, or better: create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas matplotlib numpy
```

**Interview Answer**: *"Modern Python environments enforce isolation to prevent dependency conflicts. For development, I use virtual environments. For quick testing, the `--break-system-packages` flag works but isn't recommended for production."*

---

## Error 8: Gimbal Lock in Euler Angle Representation

**Symptom**: At 90° pitch, roll and yaw became coupled - couldn't control them independently.

**Root Cause**: Euler angles have a mathematical singularity at ±90° pitch (gimbal lock). At this orientation, one degree of freedom is lost.

**Fix**: Used quaternions instead of Euler angles for all internal attitude representation:
- No singularities
- Smooth interpolation (SLERP)
- Compact (4 numbers vs 9 for DCM)
- Efficient composition (quaternion multiply)

**Interview Answer**: *"I use quaternions exclusively for attitude representation because they have no singularities. Euler angles are only used for human-readable output and plotting. The trade-off is the unit-norm constraint, which I handle through periodic renormalization."*

---

## Error 9: Rate Monotonic Schedulability Violation

**Symptom**: Occasionally, lower-priority tasks (telemetry) missed their deadlines.

**Root Cause**: Total CPU utilization exceeded the RMS schedulability bound.

**Analysis**:
```
U = C₁/T₁ + C₂/T₂ + C₃/T₃ + C₄/T₄
U = 5/10 + 8/100 + 15/1000 + 20/200
U = 0.5 + 0.08 + 0.015 + 0.1 = 0.695
```

RMS bound for n=4 tasks: `4 × (2^(1/4) - 1) = 0.757`

We were within bounds, but WCET measurements were optimistic.

**Fix**:
1. Re-measured worst-case execution times under stress
2. Added margin to WCET estimates (1.2× measured)
3. Optimized the navigation filter to reduce computation

**Interview Answer**: *"RMS provides a sufficient but not necessary schedulability condition. I measure actual WCETs under worst-case conditions and add 20% margin. If utilization approaches the bound, I profile and optimize the longest-running tasks."*

---

## Error 10: Star Tracker Noise Spikes

**Symptom**: Occasional large attitude errors when star tracker returned invalid readings.

**Root Cause**: Star tracker can fail during:
- Sun in field of view
- Earth/Moon limb in field of view
- Insufficient stars visible
- Radiation-induced bit flips

**Fix**: Implemented measurement validation in the EKF:
```python
innovation = z - H @ x_pred
mahalanobis = innovation.T @ S_inv @ innovation
if mahalanobis > chi2_threshold:
    # Reject measurement as outlier
    return
```

Also added sensor fusion with IMU for attitude rate (gyros rarely fail).

**Interview Answer**: *"I use chi-squared gating to reject outlier measurements. The Mahalanobis distance tells us if a measurement is statistically consistent with our prediction. Star tracker dropouts are handled by propagating with the gyro until valid measurements return."*

---

# Summary: Key Debugging Lessons

1. **Sign conventions matter**: Document and enforce consistent conventions
2. **Numerical stability**: Use Joseph form, renormalize quaternions
3. **Real-time requires determinism**: No malloc, bounded algorithms
4. **Validate early**: Catch physics violations before they propagate
5. **Test under stress**: WCET measurements must be worst-case
6. **Sensor fusion**: No single sensor is reliable enough alone

---

# Good Luck Tomorrow!

Remember: You built this. You understand it. You debugged it. Be confident.
