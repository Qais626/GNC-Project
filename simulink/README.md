# GNC Simulink Model

## Miami-Moon-Jupiter Round Trip Mission

This directory contains a complete Simulink model for the spacecraft Guidance, Navigation, and Control (GNC) system.

## Files

| File | Description |
|------|-------------|
| `build_gnc_model.m` | MATLAB script that programmatically builds the Simulink model |
| `GNC_System_Init.m` | Parameter initialization script (run before simulation) |
| `run_simulation.m` | Quick-start script to run the simulation |
| `GNC_System.slx` | The generated Simulink model (created by build script) |

## Quick Start

### Step 1: Build the Model
```matlab
>> cd('/path/to/GNC Project/simulink')
>> build_gnc_model
```
This creates `GNC_System.slx` with all subsystems.

### Step 2: Initialize Parameters
```matlab
>> GNC_System_Init
```
This loads all spacecraft, sensor, actuator, and controller parameters.

### Step 3: Run Simulation
```matlab
>> sim('GNC_System', 100)  % Run for 100 seconds
```
Or use the helper script:
```matlab
>> run_simulation
```

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GNC_System                                   │
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Guidance    │───>│  Navigation  │───>│   Control    │          │
│  │   System     │    │   Filter     │    │   System     │          │
│  └──────────────┘    │  (EKF/UKF)   │    │(PID/LQR/SMC) │          │
│         │            └──────────────┘    └──────────────┘          │
│         │                   ▲                    │                   │
│         ▼                   │                    ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Mission    │    │   Sensor     │<───│  Actuator    │          │
│  │   Planner    │    │   Models     │    │   Models     │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                             ▲                    │                   │
│                             │                    ▼                   │
│                      ┌──────────────────────────────┐               │
│                      │    Spacecraft Dynamics       │               │
│                      │  ┌────────┐  ┌────────────┐  │               │
│                      │  │  6-DOF │  │  Flexible  │  │               │
│                      │  │ Motion │  │   Modes    │  │               │
│                      │  └────────┘  └────────────┘  │               │
│                      └──────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

## Subsystem Details

### 1. Spacecraft Dynamics
- **Translational motion**: 3-DOF with gravity (two-body + J2)
- **Rotational motion**: Euler's equations with full inertia tensor
- **Quaternion kinematics**: Scalar-last convention with normalization
- **Flexible modes**: 2nd-order oscillators coupled to rigid body

### 2. Attitude Dynamics
- Rigid-flexible coupled dynamics
- Gravity-gradient torque computation
- External disturbance torques (SRP, drag, magnetic)

### 3. Sensor Models
| Sensor | Model Features |
|--------|---------------|
| GPS | Position/velocity with Gaussian noise, outage above 3000 km |
| Star Tracker | Quaternion output, Sun/Moon exclusion zones |
| IMU Gyro | ARW, bias instability, scale factor, misalignment |
| IMU Accel | Noise, bias, scale factor |
| Sun Sensor | Coarse attitude reference |
| DSN | Range/range-rate for deep space |

### 4. Navigation Filter
- **Extended Kalman Filter (EKF)**: 19-state filter
  - Position (3), Velocity (3)
  - Quaternion (4), Angular rate (3)
  - Gyro bias (3), Accel bias (3)
  - Solar radiation pressure coefficient (1)
- **Multiplicative quaternion formulation**: Avoids quaternion normalization issues
- **Unscented Kalman Filter (UKF)**: Alternative for high nonlinearity

### 5. Guidance System
- Mission phase state machine
- Attitude target generation (nadir, Sun-pointing, thrust vector)
- Delta-V command generation

### 6. Control System
| Controller | Description |
|------------|-------------|
| **PID** | Proportional-Integral-Derivative with anti-windup |
| **LQR** | Linear Quadratic Regulator (optimal state feedback) |
| **SMC** | Sliding Mode Control with boundary layer |
| **MPC** | Model Predictive Control (constrained optimization) |

Controller selection via mode switch (1=PID, 2=LQR, 3=SMC).

### 7. Actuator Models
- **Reaction Wheels**: 4-wheel pyramid, torque/momentum limits, motor dynamics
- **Thrusters**: Main engine + 16 RCS, response dynamics, PWM modulation

## Parameters

Key parameters are defined in `GNC_System_Init.m`:

### Spacecraft
- Mass: 4285 kg (wet), 1820 kg (dry)
- Inertia: Ixx=1200, Iyy=1350, Izz=980 kg·m²

### Sensors
- GPS: 1.5 m position, 0.01 m/s velocity noise
- Star Tracker: 3 arcsec attitude noise
- Gyro: 0.003°/√hr ARW, 0.0001°/hr bias instability

### Actuators
- Reaction Wheels: 0.2 N·m max torque, 50 N·m·s max momentum
- Main Engine: 440 N thrust, 320 s Isp

### Controllers
- PID: Kp=[12.5, 14, 11], Ki=[0.08, 0.1, 0.06], Kd=[45, 50, 40]
- LQR: Q=diag([100,100,100,10,10,10]), R=diag([1,1,1])
- SMC: λ=3, η=0.8, φ=0.01 rad

## Simulation Outputs

After running the simulation, data is available in the MATLAB workspace:

```matlab
% Logged signals
attitude_log  % Quaternion history
omega_log     % Angular rate history
torque_log    % Control torque history

% Access via Simulink.SimulationOutput
out.yout      % All logged signals
out.tout      % Time vector
```

## Customization

### Change Controller Mode
```matlab
% In model or before simulation:
set_param('GNC_System/Controller_Mode', 'Value', '2');  % 1=PID, 2=LQR, 3=SMC
```

### Modify Initial Conditions
Edit values in `GNC_System_Init.m`:
```matlab
r0_eci = [7000e3; 0; 0];    % New initial position [m]
q0 = [0; 0; 0.707; 0.707];  % New initial attitude
```

### Add Disturbances
Modify the disturbance source in the model or programmatically:
```matlab
set_param('GNC_System/Disturbance_Torque', 'Cov', '[1e-4, 1e-4, 1e-4]');
```

## Requirements

- MATLAB R2020a or later
- Simulink
- Control System Toolbox (for LQR design)
- Aerospace Blockset (optional, for visualization)

## Troubleshooting

### "Undefined variable" errors
Run `GNC_System_Init` before simulation.

### Model won't open
Run `build_gnc_model` to regenerate the model.

### Simulation runs slowly
- Increase fixed step size: `set_param('GNC_System', 'FixedStep', '0.1')`
- Reduce simulation time

### Numerical instability
- Decrease step size
- Check actuator saturation limits
- Verify initial conditions are reasonable

## References

1. Wertz, J.R., "Space Mission Engineering: The New SMAD", 2011
2. Schaub, H., "Analytical Mechanics of Space Systems", 4th ed., 2018
3. Wie, B., "Space Vehicle Dynamics and Control", 2nd ed., 2008
4. Crassidis, J.L., "Optimal Estimation of Dynamic Systems", 2012
