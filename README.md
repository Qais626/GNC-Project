# GNC Space Mission Simulation

**Guidance, Navigation, and Control (GNC) System for a KSC-Moon-Jupiter Round Trip Mission**

A multi-fidelity spacecraft simulation implementing the full GNC pipeline for an interplanetary
round trip mission: launch from Kennedy Space Center, Florida, transit to and orbit the Moon (2 revolutions),
transit to and orbit Jupiter (3 revolutions), and return to a landing at the Kennedy Space Center launch site.
The project spans Python for high-level simulation and analysis, C++ for real-time embedded
flight software prototyping, MATLAB for validation and visualization, and SQLite for mission
telemetry storage.

---

## Mission Overview

The **Explorer-VII** spacecraft executes the following mission profile aboard the
**Heavy Lifter Mk-IV** launch vehicle:

| Phase                     | Description                                             | Key Parameter            |
|---------------------------|---------------------------------------------------------|--------------------------|
| Pre-Launch                | Countdown and system checks at KSC pad                  | Lat 28.57 N, Lon 80.65 W|
| Stage 1 Ascent            | Boost phase, 7.5 MN thrust, 170 s burn                 | Isp = 275/310 s          |
| Stage Separation          | Stage 1 jettison                                        | 5 s coast                |
| Stage 2 Ascent            | Upper stage burn, 380 s                                 | Isp = 348 s (vac)        |
| LEO Parking Orbit         | 200 km circular orbit, ~45 min coast                    | V = 7.79 km/s            |
| Trans-Lunar Injection     | TLI burn, delta-V = 3150 m/s                            | 3-day coast to Moon      |
| Lunar Orbit Insertion     | LOI burn into 100 km lunar orbit                        | delta-V = 850 m/s        |
| Lunar Orbit (Rev 1)       | Equatorial orbit, 1 revolution                          | i = 0 deg                |
| Lunar Inclination Change  | Plane change maneuver, delta-V = 200 m/s                | i = 0 -> 45 deg          |
| Lunar Orbit (Rev 2)       | Inclined orbit, 1 revolution                            | i = 45 deg               |
| Lunar Escape              | Departure burn, delta-V = 900 m/s                       | Transfer to Jupiter      |
| Earth-Jupiter Transfer    | ~2 year heliocentric cruise                             | Deep space navigation    |
| Jupiter Orbit Insertion   | JOI burn into high Jupiter orbit                        | delta-V = 2000 m/s       |
| Jupiter Orbits            | 3 revolutions in Jupiter orbit                          | Radiation environment    |
| Jupiter Escape            | Departure burn, delta-V = 2200 m/s                      | Return to Earth          |
| Jupiter-Earth Return      | ~2 year heliocentric cruise                             | Deep space navigation    |
| Earth Re-Entry            | Atmospheric entry at 12 km/s, 120 km altitude           | High thermal loads       |
| Descent and Landing       | Guided descent to KSC landing site                      | Target: 28.57N, 80.65W  |

**Total mission delta-V budget: ~12,450 m/s**

---

## Directory Structure

```
GNC Project/
|
|-- config/
|   |-- mission_config.yaml        # Master mission configuration (all parameters)
|
|-- database/
|   |-- schema.sql                 # SQLite database schema (tables, indexes, views)
|
|-- docs/                          # Documentation and design notes
|
|-- excel/
|   |-- mission_parameters.csv     # Parameter reference table (pandas/Excel compatible)
|
|-- output/
|   |-- plots/                     # Generated plots (trajectories, attitude, errors)
|   |-- trade_studies/             # Trade study results and comparison charts
|   |-- matlab/                    # MATLAB-compatible exported data (.mat, .csv)
|   |-- data/                      # Simulation databases, CSVs, logs
|
|-- scripts/
|   |-- setup.sh                   # Environment setup (venv, dependencies, directories)
|   |-- run_full_sim.sh            # Master pipeline: build, simulate, analyze, test
|   |-- run_tests.sh               # Test runner with coverage reporting
|
|-- src/
|   |-- core/                      # Constants, quaternion math, data structures
|   |-- dynamics/                  # Spacecraft and environment models
|   |-- guidance/                  # Trajectory optimization and mission planning
|   |-- navigation/                # Sensor models and state estimation
|   |-- control/                   # Attitude control laws and actuator models
|   |-- simulation/                # Simulation engine and Monte Carlo
|   |-- autonomy/                  # Fault detection and autonomous operations
|   |-- trade_studies/             # Design trade study analysis
|   |-- optimization/              # Trajectory and MDO optimization
|   |-- performance/               # Benchmarking and profiling tools
|   |-- database/                  # SQLite telemetry database interface
|   |-- visualization/             # Trajectory plotting and visualization
|   |-- tests/                     # Unit and integration tests
|   |-- main.py                    # Main simulation entry point
|   |
|   |-- fsw/                       # C++ real-time flight software
|   |   |-- CMakeLists.txt         # Build configuration
|   |   |-- include/               # C++ headers (memory pool, ring buffer, etc.)
|   |   |-- src/                   # C++ source files
|   |
|   |-- tools/                     # MATLAB analysis and validation tools
|       |-- matlab/                # Control design, plotting, trajectory analysis
|       |-- simulink/              # Simulink attitude control model
|
|-- README.md                      # This file
```

---

## Setup Instructions

### Prerequisites

- Python 3.8 or later
- C++17 compatible compiler (g++ or clang++) -- optional, for real-time module
- CMake 3.14+ -- optional, for C++ build
- MATLAB R2020a+ -- optional, for validation plots

### Quick Start

```bash
# 1. Clone or download the project
cd "GNC Project"

# 2. Run the setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# 3. Activate the virtual environment
source venv/bin/activate

# 4. Run the full simulation pipeline
chmod +x scripts/run_full_sim.sh
./scripts/run_full_sim.sh
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy scipy matplotlib pandas pyyaml scikit-learn pytest pytest-cov

# Create output directories
mkdir -p output/{plots,trade_studies,matlab,data}

# Build C++ flight software module (optional)
cd src/fsw
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../../..
```

---

## How to Run

### Full Simulation Pipeline

```bash
./scripts/run_full_sim.sh
```

This executes the complete pipeline: environment check, C++ build, main simulation,
trade studies, Monte Carlo analysis (10 runs), performance benchmarks, and unit tests.

### Individual Modules

```bash
# Activate environment first
source venv/bin/activate
export PYTHONPATH=src:$PYTHONPATH

# Run specific modules
python3 src/main.py                          # Main simulation
python3 src/simulation/monte_carlo.py        # Monte Carlo analysis
python3 src/trade_studies/propulsion_trade.py # Propulsion trade study
python3 src/performance/benchmark.py          # Performance benchmarks
```

### Tests

```bash
./scripts/run_tests.sh

# Or directly with pytest
python3 -m pytest src/tests/ -v --cov=src
```

---

## Technology Stack

| Technology | Purpose                                                    |
|------------|------------------------------------------------------------|
| **Python** | Primary simulation language: dynamics, guidance, navigation, control, analysis |
| **NumPy/SciPy** | Numerical computation, linear algebra, ODE integration, optimization |
| **pandas** | Data management, telemetry I/O, CSV/database bulk operations |
| **matplotlib** | Plotting: trajectories, attitude histories, trade study charts |
| **scikit-learn** | Anomaly detection (Isolation Forest), sensor fusion ML components |
| **PyYAML** | Mission configuration file parsing                          |
| **C++17**  | Real-time flight software prototyping: memory pools, lock-free buffers, deterministic scheduling |
| **CMake**  | C++ build system                                            |
| **SQLite** | Mission telemetry database with structured queries and views |
| **MATLAB** | Independent validation, high-fidelity reference plots (optional) |
| **CSV**    | Data exchange format between Python, MATLAB, and Excel      |

---

## Module Descriptions

### Core (`src/core/`)

Foundation layer providing physical and astronomical constants (Earth, Moon, Jupiter, Sun
gravitational parameters, radii, J2 coefficients), quaternion mathematics for attitude
representation (multiplication, rotation, interpolation, error computation), and custom data
structures optimized for GNC workloads (priority queues for event scheduling, ring buffers for
state history, KD-trees for spatial queries, double-buffered telemetry streams, mission phase
graphs with Dijkstra pathfinding, and dynamic programming tables for trajectory cost-to-go
computation).

### Dynamics (`src/dynamics/`)

Physical models for the spacecraft and its environment. The spacecraft model tracks mass
properties (with fuel depletion), inertia tensor (with structural defects and CG offsets), and
flexible body modes. The environment module implements a J2+J3 gravity model with third-body
perturbations from the Sun, Moon, and Jupiter, exponential atmospheric drag, solar radiation
pressure, and Jupiter radiation belt dosimetry. Attitude dynamics are propagated using quaternion
kinematics with Euler's rotational equations of motion.

### Guidance (`src/guidance/`)

Trajectory optimization and mission planning. Includes a mission phase sequencer that manages
transitions through the 18+ mission phases, a maneuver planner that computes Hohmann transfers,
bi-elliptic transfers, plane changes, and powered descent profiles, and a trajectory optimizer
that uses scipy.optimize to minimize delta-V budgets subject to orbital constraints. The Lambert
solver handles interplanetary transfer orbit design for the Earth-Jupiter legs.

### Navigation (`src/navigation/`)

Sensor models and state estimation. Five sensor types are modeled with realistic error budgets:
IMU (gyro bias drift, angle random walk, accelerometer noise), star tracker (accuracy, FOV,
exclusion angles, angular rate limits), sun sensor, GPS (position/velocity noise, altitude
limits), and Deep Space Network (range, range-rate, angular measurements with light-time delay).
State estimation combines these measurements through an Extended Kalman Filter (EKF) for
orbit determination and attitude determination.

### Control (`src/control/`)

Attitude control laws and actuator models. Implements three control algorithms -- PID, LQR
(Linear Quadratic Regulator), and sliding mode control -- selectable per mission phase. Actuator
models include reaction wheels (4-wheel pyramid configuration with momentum saturation and
desaturation logic), control moment gyroscopes (CMGs), and RCS thrusters with minimum impulse
bit constraints. The controller manages momentum dumping, slew maneuvers, and fine pointing.

### Simulation (`src/simulation/`)

The simulation engine that ties all subsystems together. Runs the main time loop with
configurable step sizes (1 s nominal, 0.01 s for critical maneuvers), manages phase transitions,
and orchestrates the guidance-navigation-control cycle. The Monte Carlo module runs dispersed
simulations with randomized initial conditions, sensor biases, thrust magnitudes, and Isp values
to characterize mission success probability and sensitivity to uncertainties.

### Trade Studies (`src/trade_studies/`)

Parametric analysis comparing design alternatives. The propulsion trade study evaluates four
engine options (chemical bipropellant, solid rocket motor, xenon ion thruster, nuclear thermal)
across metrics of total fuel mass, trip time, thrust-to-weight ratio, and complexity. Sensor
trade studies evaluate navigation accuracy as a function of sensor suite configuration. Results
are exported as comparison tables and plots to `output/trade_studies/`.

### Autonomy (`src/autonomy/`)

Fault detection, isolation, and recovery (FDIR) capabilities. Uses scikit-learn Isolation Forest
for anomaly detection on sensor telemetry streams, identifying off-nominal readings from sensor
degradation, environmental transients, or hardware faults. Includes rule-based logic for
autonomous mode transitions (e.g., safe mode entry on persistent anomalies) and sensor
reconfiguration when primary sensors fail.

### Performance (`src/performance/`)

Benchmarking and profiling tools that measure execution time and memory usage of each subsystem.
Compares Python simulation speed against the C++ real-time module to quantify the performance
gap and validate that the C++ implementation meets real-time constraints. Generates timing
reports and identifies computational bottlenecks for optimization.

### Database (`src/database/`)

SQLite-backed telemetry storage using the schema defined in `database/schema.sql`. The
`TelemetryDatabase` class provides a pandas-integrated interface for inserting individual records
(real-time logging) and bulk DataFrames (post-processing). Supports querying by time range,
mission phase, and sensor type, with views for mission summaries and phase duration analysis.
All tables can be exported to CSV for external analysis in MATLAB or Excel.

### Flight Software (`src/fsw/`)

Embedded flight software prototype written in C++17 demonstrating real-time GNC patterns:
custom memory pool allocators (zero-allocation after init), lock-free ring buffers for
inter-thread telemetry passing, a deterministic dynamics engine with fixed-step integration,
and a flight computer scheduler with priority-based task execution. Built with CMake and
compiled with strict warning flags appropriate for flight-quality code.

---

## Dependencies

### Python (installed via `scripts/setup.sh`)

| Package        | Version  | Purpose                              |
|----------------|----------|--------------------------------------|
| numpy          | >= 1.21  | Array computation, linear algebra    |
| scipy          | >= 1.7   | ODE integration, optimization        |
| matplotlib     | >= 3.4   | Plotting and visualization           |
| pandas         | >= 1.3   | Data manipulation and database I/O   |
| pyyaml         | >= 5.4   | YAML configuration parsing           |
| scikit-learn   | >= 1.0   | Anomaly detection (Isolation Forest) |
| pytest         | >= 7.0   | Unit testing framework               |
| pytest-cov     | >= 3.0   | Test coverage reporting              |

### C++ (system packages)

| Component      | Version  | Purpose                              |
|----------------|----------|--------------------------------------|
| C++ compiler   | C++17    | g++ >= 7 or clang++ >= 5             |
| CMake          | >= 3.14  | Build system                         |
| pthreads       | POSIX    | Threading (included with compiler)   |

### Optional

| Component      | Purpose                                         |
|----------------|-------------------------------------------------|
| MATLAB         | Independent validation and high-fidelity plots   |

---

## Architecture

```
                    +---------------------------+
                    |    Mission Configuration   |
                    |    (mission_config.yaml)   |
                    +-------------+-------------+
                                  |
                                  v
+------------------------------------------------------------------+
|                      SIMULATION ENGINE                            |
|                                                                  |
|  +------------+    +-----------+    +-----------+    +----------+ |
|  |  Guidance   |--->| Navigation|--->|  Control  |--->| Actuators| |
|  | (Trajectory |    | (EKF,     |    | (PID/LQR/ |    | (RW, CMG,| |
|  |  Optimizer, |    |  Sensors) |    |  Sliding) |    |  RCS)    | |
|  |  Mission    |    |           |    |           |    |          | |
|  |  Planner)   |    |           |    |           |    |          | |
|  +------+------+    +-----+-----+    +-----+-----+    +----+-----+ |
|         |                |                |                |      |
|         v                v                v                v      |
|  +-----------------------------------------------------------+   |
|  |                    Dynamics Engine                          |   |
|  |  (Orbital Mechanics, Attitude Dynamics, Environment)       |   |
|  +----------------------------+-------------------------------+   |
|                               |                                   |
+------------------------------------------------------------------+
                                |
              +-----------------+-----------------+
              |                 |                 |
              v                 v                 v
     +--------+------+  +------+-------+  +------+-------+
     |   Telemetry    |  |    Trade     |  |    Monte     |
     |   Database     |  |   Studies    |  |    Carlo     |
     |   (SQLite)     |  |  (Analysis)  |  |  (Dispersion)|
     +--------+------+  +------+-------+  +------+-------+
              |                 |                 |
              v                 v                 v
     +--------+------+  +------+-------+  +------+-------+
     |   CSV Export   |  |    Plots     |  | Statistics   |
     |   & Database   |  | (matplotlib) |  |  & Reports   |
     +---------------+  +--------------+  +--------------+

     +------------------------------------------------------------------+
     |                   C++ REAL-TIME MODULE                            |
     |  +-------------+  +-------------+  +------------+  +----------+  |
     |  | Memory Pool |  | Ring Buffer |  | Dynamics   |  | Flight   |  |
     |  | Allocator   |  | (Lock-Free) |  | Engine     |  | Computer |  |
     |  +-------------+  +-------------+  +------------+  +----------+  |
     +------------------------------------------------------------------+
```

---

## Example Outputs

After running the full simulation pipeline, the following outputs are generated:

- **`output/plots/`** -- Trajectory plots (3D and 2D projections), attitude quaternion histories,
  angular velocity profiles, pointing error time series, fuel consumption curves, and sensor
  measurement residual plots for each mission phase.

- **`output/trade_studies/`** -- Bar charts and tables comparing propulsion options (chemical vs.
  solid vs. ion vs. nuclear) on fuel mass, trip time, and delta-V margin. Sensor suite trade
  study showing navigation accuracy vs. cost/complexity.

- **`output/data/`** -- SQLite database with complete telemetry, CSV exports of all tables,
  Monte Carlo result summaries with success/failure statistics and landing dispersion ellipses,
  C++ benchmark timing data, and simulation run logs.

- **`output/matlab/`** -- MATLAB-compatible data files for independent validation of Python
  simulation results against MATLAB reference implementations.

---

## Credits

This project was developed as a comprehensive GNC simulation demonstrating the full breadth of
spacecraft guidance, navigation, and control engineering for an ambitious interplanetary mission
profile. It integrates concepts from orbital mechanics, attitude dynamics, sensor modeling,
control theory, autonomy, real-time systems, and data management into a single cohesive
simulation framework.

**Mission:** KSC-Moon-Jupiter Round Trip (Explorer-VII / Heavy Lifter Mk-IV)

**Technology:** Python 3, C++17, MATLAB, SQLite, pandas, NumPy, SciPy, scikit-learn
