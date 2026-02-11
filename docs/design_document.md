# GNC Mission Design Document

## 1. Mission Overview

**Mission Name:** KSC-Moon-Jupiter Round Trip
**Objective:** Launch from Kennedy Space Center, FL, orbit the Moon (2 revolutions with inclination change), transfer to Jupiter, orbit Jupiter (3 revolutions), return and land at Kennedy Space Center.

### 1.1 Mission Phases

| # | Phase | Description | Key Event |
|---|-------|-------------|-----------|
| 1 | Pre-Launch | Countdown, system checks | T-10s to T-0 |
| 2 | Stage 1 Ascent | First stage burn through atmosphere | Max-Q, gravity turn |
| 3 | Stage Separation | Drop Stage 1 | Mass discontinuity |
| 4 | Stage 2 Ascent | Upper stage burn to orbit | Fairing jettison |
| 5 | Parking Orbit | 200 km circular LEO coast | Systems checkout |
| 6 | TLI Burn | Trans-Lunar Injection | Leave Earth orbit |
| 7 | Lunar Coast | 3-day transfer to Moon | Mid-course corrections |
| 8 | Lunar Orbit Insertion | LOI burn to capture at Moon | Enter 100 km orbit |
| 9 | Lunar Orbit 1 | First revolution (equatorial) | Science/service ops |
| 10 | Inclination Change | Plane change to 45 deg | Delta-V maneuver |
| 11 | Lunar Orbit 2 | Second revolution (inclined) | Science/service ops |
| 12 | Lunar Escape | Escape burn from Moon | Leave Moon SOI |
| 13 | Earth-Jupiter Transfer | ~2 year cruise | Deep space navigation |
| 14 | Jupiter Orbit Insertion | JOI burn | Enter Jupiter orbit |
| 15 | Jupiter Orbits | 3 revolutions | Radiation exposure |
| 16 | Jupiter Escape | Escape burn from Jupiter | Begin return |
| 17 | Jupiter-Earth Return | ~2 year cruise | Return navigation |
| 18 | Earth Reentry | Atmospheric entry at ~12 km/s | Heat shield, blackout |
| 19 | Descent & Landing | Final approach to KSC | Precision landing |

### 1.2 Delta-V Budget (Estimated)

| Maneuver | Delta-V (m/s) | Notes |
|----------|---------------|-------|
| TLI | ~3,150 | Hohmann-like to Moon |
| LOI | ~850 | Lunar orbit insertion |
| Inclination change | ~200 | 0 to 45 deg at Moon |
| Lunar escape | ~900 | Escape Moon SOI |
| Earth-Jupiter transfer | ~6,000 | Includes gravity assist potential |
| JOI | ~2,000 | Jupiter orbit insertion |
| Jupiter escape | ~2,200 | Escape Jupiter |
| Return corrections | ~500 | Mid-course |
| Reentry/landing | ~100 | Final approach |
| **Total** | **~15,900** | Significant propellant mass needed |

## 2. System Architecture

### 2.1 Software Architecture

```
+------------------+     +------------------+     +------------------+
|    GUIDANCE       |     |   NAVIGATION     |     |    CONTROL       |
|                   |     |                  |     |                  |
| Mission Planner   |---->| EKF / UKF        |---->| Attitude Control |
| Trajectory Opt    |     | Sensor Fusion    |     | Orbit Control    |
| Maneuver Planner  |     | Signal Model     |     | Optimal Control  |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
+--------------------------------------------------------------+
|                    SIMULATION ENGINE                          |
|  Time-stepped loop: Guidance -> Navigation -> Control        |
|  -> Dynamics Propagation -> Telemetry Logging                |
+--------------------------------------------------------------+
        |                        |                        |
        v                        v                        v
+------------------+     +------------------+     +------------------+
|    DYNAMICS       |     |   AUTONOMY       |     |   PERFORMANCE    |
|                   |     |                  |     |                  |
| Orbital Mechanics |     | Anomaly Detect   |     | Benchmarks       |
| Attitude Dynamics |     | Auto Trajectory  |     | Cache Optimizer  |
| Environment       |     | Attitude Predict |     | Parallelization  |
| Spacecraft Model  |     |    (AI/ML)       |     |                  |
+------------------+     +------------------+     +------------------+
```

### 2.2 Language Integration

| Language | Role | Interface |
|----------|------|-----------|
| Python | Main simulation, plotting, ML, Monte Carlo | Central hub |
| C++ | Real-time flight software, HIL, performance-critical dynamics | ctypes/subprocess from Python |
| MATLAB | Control design, Bode/root locus plots, Simulink models | CSV/JSON data exchange |
| SQL | Telemetry storage and querying | sqlite3 from Python |
| CSV/Excel | Parameter tables, data exchange | pandas from Python |

### 2.3 Data Flow

```
Config (YAML) --> Python main.py --> Initialize Subsystems
                                         |
                  +----------------------+----------------------+
                  |                      |                      |
            Dynamics Engine      GNC Algorithms           Trade Studies
            (propagate state)    (estimate, control)      (compare options)
                  |                      |                      |
                  +----------+-----------+                      |
                             |                                  |
                      Telemetry (pandas DataFrame)              |
                             |                                  |
                  +----------+-----------+----------+-----------+
                  |          |           |          |
               SQLite      CSV       Plots     MATLAB data
               (.db)      (.csv)     (.png)    (.mat/.csv)
```

## 3. Non-Ideal Effects Modeled

### 3.1 Structural Defects
- **Flex modes**: 3 structural vibration modes (0.8, 2.3, 5.1 Hz) coupled to rigid-body dynamics
- **Mass uncertainty**: +/- 2% mass knowledge error affects control gains
- **CG offset**: 5-8 mm offset creates disturbance torques during thrusting
- **Inertia products**: Non-zero off-diagonal inertia terms from manufacturing asymmetry

### 3.2 Aerodynamic Effects
- **Drag**: Exponential atmosphere model during launch and reentry
- **Transonic drag rise**: Cd increases through Mach 0.8-1.2
- **Dynamic pressure (max-Q)**: Peak structural loading during ascent
- **Heating**: Sutton-Graves stagnation point heating model for reentry

### 3.3 Radiation Effects
- **Solar radiation pressure**: Photon momentum on spacecraft surfaces
- **Jupiter radiation belt**: Intense radiation degrades electronics and solar panels
- **Cumulative dose tracking**: Panel efficiency decreases, electronics reliability decreases
- **Shielding model**: Dose rate attenuated by shielding thickness

### 3.4 Gravitational Perturbations
- **J2 oblateness**: Earth, Moon, Jupiter J2 effects on orbit
- **J3**: Earth J3 for asymmetric gravity
- **Third-body**: Sun, Moon, Jupiter mutual gravitational perturbation
- **Gravity gradient torque**: Differential gravity across spacecraft creates torque

## 4. Attitude Determination and Control

### 4.1 Sensors
| Sensor | Accuracy | Rate | Availability |
|--------|----------|------|-------------|
| IMU (Gyro) | 0.1 mrad/s bias, ARW 0.03 mrad/sqrt(s) | 100 Hz | Always |
| IMU (Accel) | 0.5 mm/s^2 bias | 100 Hz | Always |
| Star Tracker | 5 arcsec | 10 Hz | Sun/Moon exclusion, max rate 2 deg/s |
| Sun Sensor | 0.5 deg | 5 Hz | Sun in FOV |
| GPS | 10 m position | 1 Hz | Below 3000 km only |
| DSN | 5 m range, 1 mm/s range-rate | Variable | Earth contact only |

### 4.2 Actuators
| Actuator | Capability | Use |
|----------|-----------|-----|
| Reaction Wheels (x4) | 0.2 Nm torque, 50 Nms momentum | Fine pointing |
| CMGs (x4) | 250 Nm torque | Large slew maneuvers |
| RCS Thrusters (x16) | 22 N each | Orbit maneuvers, wheel desaturation |

### 4.3 Control Modes
- **Detumble**: PID control, large attitude errors after separation
- **Pointing**: LQR, fine pointing at Moon or Jupiter (< 0.1 deg accuracy)
- **Slew**: Sliding mode, large reorientation maneuvers
- **Optimal**: LQG/H-infinity during critical phases

### 4.4 Pointing Requirements
The spacecraft must point its instrument axis at the target body:
- During lunar orbits: point at Moon center
- During Jupiter orbits: point at Jupiter center
- During transfer: point along velocity vector (for burns) or at Sun (for power)
- Accuracy requirement: 0.1 degrees (3-sigma)

## 5. Navigation Architecture

### 5.1 Extended Kalman Filter (EKF)
- **State vector (15 states)**: position(3), velocity(3), attitude error(3), gyro bias(3), accel bias(3)
- **Multiplicative quaternion formulation**: 3-element attitude error in filter, reference quaternion propagated separately
- **Joseph form** covariance update for numerical stability
- **Measurement updates**: GPS (near Earth), star tracker (attitude), DSN (deep space)

### 5.2 Unscented Kalman Filter (UKF)
- Same 15-state vector
- Sigma point propagation through nonlinear dynamics (no Jacobian needed)
- Better performance for highly nonlinear regimes (e.g., near Jupiter)
- Tuning parameters: alpha=1e-3, beta=2, kappa=0

## 6. Trade Studies

### 6.1 Propulsion Trade
Compare 4 propulsion types across the full mission:
1. **Chemical Bipropellant** (Isp=316s): Standard, proven, moderate mass
2. **Solid Rocket Motor** (Isp=265s): Simple, low Isp, heavy
3. **Ion Thruster** (Isp=3000s): Very efficient, very low thrust, spiral trajectories
4. **Nuclear Thermal** (Isp=900s): High Isp + high thrust, heavy engine, regulatory issues

Metrics: total delta-V, mission duration, propellant mass, trajectory shape.

### 6.2 Attitude Pointing Impact Trade
Study how pointing errors affect trajectory and fuel:
- Pointing errors of 0.01, 0.1, 1.0, 5.0, 10.0 degrees
- Thrust vector misalignment effects
- Controller comparison: PID vs LQR vs Sliding Mode

## 7. AI/ML and Autonomy

### 7.1 Anomaly Detection
- Autoencoder neural network trained on nominal sensor data
- Detects off-nominal readings (sensor failures, environmental anomalies)
- Statistical Mahalanobis distance as backup method

### 7.2 Autonomous Trajectory Correction
- Q-learning agent for deciding trajectory correction maneuvers
- Balances fuel usage vs trajectory accuracy
- Trained on simplified environment, deployed on full sim

### 7.3 Attitude Prediction
- Feedforward neural network predicts future attitude state
- Faster than numerical propagation for planning lookahead
- Comparison with RK4 propagator (speed vs accuracy tradeoff)

## 8. Real-Time Systems (C++)

### 8.1 Flight Computer Architecture
- Deterministic scheduling with rate-monotonic priority
- Watchdog timer for task overrun detection
- Triple Modular Redundancy (TMR) for critical computations

### 8.2 Memory Architecture
- Custom memory pool allocator (cache-line aligned, 64 bytes)
- Lock-free ring buffer for real-time telemetry streaming
- Zero-allocation hot path after initialization

### 8.3 Computer Architecture Optimizations
- Cache-line alignment for data structures
- Struct-of-Arrays layout for SIMD-friendly access
- False sharing prevention with padding
- Branch prediction optimization in hot loops

## 9. Testing Strategy

### 9.1 Unit Tests (pytest)
- Quaternion math: roundtrip conversions, known rotations
- Orbital mechanics: energy conservation, known transfers
- EKF: convergence, covariance properties, consistency
- Control: stability, performance requirements met

### 9.2 Performance Tests
- Prove vectorized > loop (5-100x speedup)
- Prove pre-allocation > dynamic allocation
- Prove SoA > AoS for batch operations
- Prove KD-tree > linear search (O(log n) vs O(n))

### 9.3 Integration Tests
- Full simulation runs without error
- Phase transitions occur correctly
- Telemetry recording and database write/read
- SIL fault injection and recovery

### 9.4 SIL/HIL
- SIL: Full simulation with fault injection capability
- HIL: Interface framework for hardware testing (serial/UDP)
