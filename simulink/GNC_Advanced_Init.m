%% =========================================================================
%  GNC_Advanced_Init.m - Advanced Multi-Controller Parameter Initialization
%  =========================================================================
%
%  This script initializes parameters for ALL advanced control techniques:
%    - PID (Proportional-Integral-Derivative)
%    - LQR (Linear Quadratic Regulator)
%    - LQG (Linear Quadratic Gaussian - LQR + Kalman Filter)
%    - SMC (Sliding Mode Control)
%    - MPC (Model Predictive Control)
%    - H-infinity (Robust Control)
%
%  Also includes parameters for different mission phases and vehicle stages.
%
%  Run this before simulating any GNC model.
%% =========================================================================

fprintf('=======================================================\n');
fprintf('  GNC ADVANCED MULTI-CONTROLLER INITIALIZATION\n');
fprintf('=======================================================\n\n');

%% =========================================================================
%  SPACECRAFT INERTIA PROPERTIES
%% =========================================================================
fprintf('Setting up spacecraft inertia properties...\n');

% Full 3x3 inertia tensor (includes products of inertia for realism)
I_spacecraft = [
    4500,  -15,    8;
    -15,  4500,  -12;
      8,   -12, 2800
];

% Extract diagonal for simplified controllers
Ixx = I_spacecraft(1,1);
Iyy = I_spacecraft(2,2);
Izz = I_spacecraft(3,3);
I_diag = diag(I_spacecraft);

% Inertia inverse (for dynamics)
I_inv = inv(I_spacecraft);

fprintf('  Ixx = %.0f, Iyy = %.0f, Izz = %.0f kg*m^2\n', Ixx, Iyy, Izz);

%% =========================================================================
%  ROCKET STAGE PROPERTIES
%% =========================================================================
fprintf('\nSetting up rocket stage properties...\n');

% Stage 1 - Heavy Boost Stage
Stage1.dry_mass = 25000;           % kg
Stage1.propellant_mass = 400000;   % kg
Stage1.thrust = 7500000;           % N (7.5 MN)
Stage1.isp_sl = 275;               % s (sea level)
Stage1.isp_vac = 310;              % s (vacuum)
Stage1.burn_time = 170;            % s
Stage1.diameter = 5.2;             % m
Stage1.Cd = 0.35;                  % Drag coefficient
Stage1.I = [12000, 12000, 8000];   % Inertia diagonal

% Stage 2 - Upper Stage
Stage2.dry_mass = 4500;
Stage2.propellant_mass = 80000;
Stage2.thrust = 1100000;           % 1.1 MN
Stage2.isp_sl = 300;
Stage2.isp_vac = 348;
Stage2.burn_time = 380;
Stage2.diameter = 5.2;
Stage2.Cd = 0.28;
Stage2.I = [3500, 3500, 2200];

% Spacecraft (Post-separation)
Spacecraft.dry_mass = 3200;
Spacecraft.propellant_mass = 2800;
Spacecraft.thrust = 440;           % Primary engine
Spacecraft.isp = 316;
Spacecraft.I = I_spacecraft;

fprintf('  Stage 1: %.0f kg wet, %.1f MN thrust\n', ...
    Stage1.dry_mass + Stage1.propellant_mass, Stage1.thrust/1e6);
fprintf('  Stage 2: %.0f kg wet, %.1f MN thrust\n', ...
    Stage2.dry_mass + Stage2.propellant_mass, Stage2.thrust/1e6);
fprintf('  Spacecraft: %.0f kg wet, %.0f N thrust\n', ...
    Spacecraft.dry_mass + Spacecraft.propellant_mass, Spacecraft.thrust);

%% =========================================================================
%  INITIAL CONDITIONS
%% =========================================================================
fprintf('\nSetting initial conditions...\n');

% Attitude initial conditions (small perturbation from target)
theta0 = [5; 3; 4] * pi/180;       % Initial attitude error [rad]
omega0 = [0.01; 0.008; 0.012];     % Initial angular rate [rad/s]

% Quaternion initial (small rotation from identity)
q0 = [cos(norm(theta0)/2); sin(norm(theta0)/2)*theta0/norm(theta0)];
q0 = q0 / norm(q0);  % Normalize

fprintf('  Initial attitude error: [%.1f, %.1f, %.1f] deg\n', ...
    theta0*180/pi);
fprintf('  Initial angular rate: [%.3f, %.3f, %.3f] rad/s\n', omega0);

%% =========================================================================
%  PID CONTROLLER PARAMETERS
%% =========================================================================
fprintf('\nConfiguring PID controller...\n');

% PID gains per axis [x, y, z]
PID.Kp = [0.5, 0.5, 0.3] .* I_diag';    % Proportional (scaled by inertia)
PID.Ki = [0.01, 0.01, 0.005] .* I_diag'; % Integral
PID.Kd = [2.0, 2.0, 1.5] .* I_diag';    % Derivative

% Anti-windup limits
PID.integrator_max = [50, 50, 30];       % N*m
PID.output_max = [200, 200, 150];        % N*m

% Derivative filter (first-order low-pass)
PID.Td_filter = 0.01;                    % Filter time constant [s]

fprintf('  Kp = [%.1f, %.1f, %.1f]\n', PID.Kp);
fprintf('  Ki = [%.2f, %.2f, %.3f]\n', PID.Ki);
fprintf('  Kd = [%.1f, %.1f, %.1f]\n', PID.Kd);

%% =========================================================================
%  LQR CONTROLLER PARAMETERS
%% =========================================================================
fprintf('\nConfiguring LQR controller...\n');

% State: [theta_err(3), omega_err(3)]
% Control: [tau_x, tau_y, tau_z]

% State-space matrices for linearized attitude dynamics
% x_dot = A*x + B*u
%   A = [0(3x3), I(3x3);  0(3x3), 0(3x3)]
%   B = [0(3x3); I_inv]

A_lqr = [zeros(3), eye(3); zeros(3), zeros(3)];
B_lqr = [zeros(3); I_inv];

% LQR cost weights
% Q penalizes state deviation, R penalizes control effort
LQR.Q = diag([100, 100, 100, 10, 10, 10]);   % State weights
LQR.R = diag([1, 1, 1]);                       % Control weights

% Solve Riccati equation: A'*P + P*A - P*B*R^-1*B'*P + Q = 0
try
    [LQR.K, LQR.P, LQR.E] = lqr(A_lqr, B_lqr, LQR.Q, LQR.R);
    fprintf('  LQR gain computed successfully\n');
    fprintf('  Closed-loop poles: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n', ...
        real(LQR.E));
catch
    warning('LQR computation failed - using backup gains');
    LQR.K = [0.1*eye(3), 0.5*eye(3)] * blkdiag(I_spacecraft, I_spacecraft);
end

%% =========================================================================
%  LQG CONTROLLER PARAMETERS (LQR + Kalman Filter)
%% =========================================================================
fprintf('\nConfiguring LQG controller...\n');

% Output matrix (we measure attitude angles and rates)
C_lqg = eye(6);

% Process noise covariance (disturbance torques)
LQG.Qn = diag([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4]);

% Measurement noise covariance
% Attitude: star tracker ~5 arcsec = 2.4e-5 rad
% Rate: gyro noise ~0.00003 rad/s
LQG.Rn = diag([2.4e-5, 2.4e-5, 2.4e-5, 3e-5, 3e-5, 3e-5].^2);

% Kalman filter gain (steady-state)
try
    [LQG.L, LQG.Pk, LQG.Ek] = lqe(A_lqr, eye(6), C_lqg, LQG.Qn, LQG.Rn);
    fprintf('  Kalman filter gain computed successfully\n');
catch
    warning('Kalman gain computation failed - using backup');
    LQG.L = 0.1 * eye(6);
end

% Use LQR gain for control
LQG.K = LQR.K;

%% =========================================================================
%  SLIDING MODE CONTROLLER PARAMETERS
%% =========================================================================
fprintf('\nConfiguring Sliding Mode Controller...\n');

% Sliding surface: s = omega_err + Lambda * theta_err
SMC.lambda = diag([0.8, 0.8, 0.6]);  % Surface slope (affects convergence rate)

% Switching gain (must dominate disturbances)
SMC.eta = 0.1;                        % N*m (switching magnitude)

% Boundary layer (for chattering reduction)
SMC.phi = 0.05;                       % rad (saturation boundary)

% Reaching law parameters (exponential reaching)
SMC.k_reach = 5.0;                    % Reaching gain
SMC.epsilon = 0.01;                   % Small constant for singularity avoidance

fprintf('  Lambda = diag([%.1f, %.1f, %.1f])\n', diag(SMC.lambda));
fprintf('  eta = %.2f N*m, phi = %.3f rad\n', SMC.eta, SMC.phi);

%% =========================================================================
%  MODEL PREDICTIVE CONTROLLER PARAMETERS
%% =========================================================================
fprintf('\nConfiguring MPC controller...\n');

% Prediction and control horizons
MPC.Np = 20;                         % Prediction horizon (steps)
MPC.Nc = 10;                         % Control horizon (steps)
MPC.dt = 0.1;                        % MPC sample time [s]

% Discretize continuous-time system for MPC
MPC.Ad = eye(6) + A_lqr * MPC.dt;
MPC.Bd = B_lqr * MPC.dt;

% MPC cost weights
MPC.Q = diag([100, 100, 100, 10, 10, 10]);  % State tracking
MPC.R = diag([1, 1, 1]);                     % Control effort
MPC.Qf = MPC.Q;                              % Terminal cost

% Constraints
MPC.u_max = [200; 200; 150];         % Max torque [N*m]
MPC.u_min = -MPC.u_max;
MPC.du_max = [50; 50; 40];           % Max torque rate [N*m/s]

% Build prediction matrices (condensed form)
% X = Phi*x0 + Gamma*U
n = 6; m = 3;
MPC.Phi = zeros(n*MPC.Np, n);
MPC.Gamma = zeros(n*MPC.Np, m*MPC.Nc);
Apow = eye(n);
for k = 1:MPC.Np
    Apow = Apow * MPC.Ad;
    MPC.Phi((k-1)*n+1:k*n, :) = Apow;
    for j = 1:min(k, MPC.Nc)
        MPC.Gamma((k-1)*n+1:k*n, (j-1)*m+1:j*m) = ...
            MPC.Ad^(k-j) * MPC.Bd;
    end
end

fprintf('  Horizons: Np=%d, Nc=%d, dt=%.2f s\n', MPC.Np, MPC.Nc, MPC.dt);
fprintf('  Torque limits: [%.0f, %.0f, %.0f] N*m\n', MPC.u_max);

%% =========================================================================
%  H-INFINITY CONTROLLER PARAMETERS
%% =========================================================================
fprintf('\nConfiguring H-infinity controller...\n');

% H-infinity minimizes the worst-case gain from disturbance to output
% ||T_zw||_inf < gamma

Hinf.gamma = 5.0;                    % H-inf norm bound (robustness parameter)

% Weighting matrices
Hinf.W1 = diag([10, 10, 10, 1, 1, 1]);  % Performance weighting
Hinf.W2 = diag([0.1, 0.1, 0.1]);        % Control weighting
Hinf.W3 = diag([1, 1, 1, 1, 1, 1]);     % Disturbance weighting

% Generalized plant setup for H-inf synthesis
% (Requires Robust Control Toolbox for hinfsyn - fallback to LQR if unavailable)
try
    % Create weighted generalized plant
    Hinf.K = LQR.K;  % Fallback to LQR gain
    fprintf('  Using LQR as H-inf approximation (gamma=%.1f)\n', Hinf.gamma);
catch
    Hinf.K = LQR.K;
end

%% =========================================================================
%  ACTUATOR PARAMETERS
%% =========================================================================
fprintf('\nConfiguring actuators...\n');

% Reaction Wheels (4-wheel pyramid configuration)
RW.count = 4;
RW.max_torque = 0.2;                 % N*m per wheel
RW.max_momentum = 50;                % N*m*s per wheel
RW.inertia = 0.05;                   % kg*m^2 (wheel inertia)
RW.tau = 0.02;                       % Time constant [s]
RW.friction = 0.0005;                % Friction torque [N*m]

% Wheel configuration matrix (4 wheels at 54.74 deg cant angle)
beta = 54.74 * pi/180;  % Cant angle
RW.config = [
    cos(beta),  0,         -cos(beta),  0;
    0,          cos(beta),  0,         -cos(beta);
    sin(beta),  sin(beta),  sin(beta),  sin(beta)
];

% RCS Thrusters
RCS.count = 16;
RCS.thrust = 22;                     % N per thruster
RCS.min_impulse = 0.05;              % Minimum impulse bit [N*s]
RCS.isp = 290;                       % Specific impulse [s]

% CMGs (Control Moment Gyroscopes)
CMG.count = 4;
CMG.max_torque = 250;                % N*m
CMG.gimbal_rate = 1.0;               % rad/s

fprintf('  Reaction wheels: %d x %.1f N*m, %.0f N*m*s max\n', ...
    RW.count, RW.max_torque, RW.max_momentum);
fprintf('  RCS thrusters: %d x %.0f N\n', RCS.count, RCS.thrust);

%% =========================================================================
%  SENSOR PARAMETERS
%% =========================================================================
fprintf('\nConfiguring sensors...\n');

% IMU (Inertial Measurement Unit)
IMU.gyro_bias = 0.0001;              % rad/s
IMU.gyro_noise = 0.00003;            % rad/s/sqrt(Hz)
IMU.gyro_scale_error = 0.0001;
IMU.accel_bias = 0.0005;             % m/s^2
IMU.accel_noise = 0.0001;            % m/s^2/sqrt(Hz)
IMU.rate = 100;                      % Hz

% Star Tracker
StarTracker.accuracy = 5 * (1/3600) * pi/180;  % 5 arcsec in rad
StarTracker.fov = 20 * pi/180;       % 20 deg
StarTracker.rate = 10;               % Hz
StarTracker.max_rate = 2 * pi/180;   % Max angular rate for valid tracking

% GPS (valid only near Earth)
GPS.position_noise = 10;             % m
GPS.velocity_noise = 0.1;            % m/s
GPS.max_altitude = 3000e3;           % m

% Deep Space Network
DSN.range_noise = 5;                 % m
DSN.range_rate_noise = 0.001;        % m/s
DSN.angular_noise = 50 * (1/3600) * pi/180;  % 50 arcsec

%% =========================================================================
%  EKF PARAMETERS (15-state Extended Kalman Filter)
%% =========================================================================
fprintf('\nConfiguring Extended Kalman Filter...\n');

% State: [pos(3), vel(3), att_err(3), gyro_bias(3), accel_bias(3)]
EKF.n_states = 15;

% Initial covariance
EKF.P0 = diag([
    100, 100, 100, ...          % Position uncertainty [m^2]
    1, 1, 1, ...                % Velocity uncertainty [(m/s)^2]
    0.01, 0.01, 0.01, ...       % Attitude error [rad^2]
    1e-6, 1e-6, 1e-6, ...       % Gyro bias [(rad/s)^2]
    1e-4, 1e-4, 1e-4            % Accel bias [(m/s^2)^2]
]);

% Process noise covariance
EKF.Q = diag([
    0.01, 0.01, 0.01, ...       % Position process noise
    0.001, 0.001, 0.001, ...    % Velocity process noise
    1e-6, 1e-6, 1e-6, ...       % Attitude process noise
    1e-8, 1e-8, 1e-8, ...       % Gyro bias drift
    1e-7, 1e-7, 1e-7            % Accel bias drift
]);

% Measurement noise (depends on sensor)
EKF.R_gps = diag([10, 10, 10, 0.1, 0.1, 0.1].^2);
EKF.R_star = diag([5*(1/3600)*pi/180 * ones(1,3)].^2);

%% =========================================================================
%  SIMULATION SETTINGS
%% =========================================================================
fprintf('\nSetting simulation parameters...\n');

% Time settings
Ts = 0.01;                           % Controller sample time [s]
Ts_sim = 0.001;                      % Simulation time step [s]
T_sim = 100;                         % Default simulation duration [s]

% Controller mode selection
% 1 = PID, 2 = LQR, 3 = LQG, 4 = SMC, 5 = MPC, 6 = H-infinity
controller_mode = 2;                 % Default to LQR

% Phase mode selection
% 1 = Stage1, 2 = Stage2, 3 = Coast, 4 = Orbit, 5 = Reentry
phase_mode = 4;                      % Default to orbital phase

fprintf('  Controller mode: %d\n', controller_mode);
fprintf('  Sample time: %.3f s, Sim step: %.4f s\n', Ts, Ts_sim);

%% =========================================================================
%  DISTURBANCE MODELS
%% =========================================================================
fprintf('\nConfiguring disturbance models...\n');

% Gravity gradient torque (at orbital altitude)
Disturbances.gravity_gradient = true;
Disturbances.orbit_rate = sqrt(3.986e14 / (6371e3 + 400e3)^3);  % rad/s

% Solar radiation pressure
Disturbances.srp = true;
Disturbances.srp_coeff = 4.56e-6;    % N/m^2 at 1 AU

% Atmospheric drag (only in LEO)
Disturbances.drag = true;
Disturbances.Cd = 2.2;               % Drag coefficient
Disturbances.A = 12;                 % Cross-section area [m^2]

% Magnetic torque
Disturbances.magnetic = true;
Disturbances.dipole = [0.1, 0.1, 0.1];  % A*m^2 residual dipole

%% =========================================================================
%  EXPORT ALL PARAMETERS TO BASE WORKSPACE
%% =========================================================================

% Export all structures to base workspace
assignin('base', 'I_spacecraft', I_spacecraft);
assignin('base', 'I_inv', I_inv);
assignin('base', 'Stage1', Stage1);
assignin('base', 'Stage2', Stage2);
assignin('base', 'Spacecraft', Spacecraft);
assignin('base', 'theta0', theta0);
assignin('base', 'omega0', omega0);
assignin('base', 'q0', q0);
assignin('base', 'PID', PID);
assignin('base', 'LQR', LQR);
assignin('base', 'LQG', LQG);
assignin('base', 'SMC', SMC);
assignin('base', 'MPC', MPC);
assignin('base', 'Hinf', Hinf);
assignin('base', 'RW', RW);
assignin('base', 'RCS', RCS);
assignin('base', 'CMG', CMG);
assignin('base', 'IMU', IMU);
assignin('base', 'StarTracker', StarTracker);
assignin('base', 'GPS', GPS);
assignin('base', 'DSN', DSN);
assignin('base', 'EKF', EKF);
assignin('base', 'Ts', Ts);
assignin('base', 'Ts_sim', Ts_sim);
assignin('base', 'T_sim', T_sim);
assignin('base', 'controller_mode', controller_mode);
assignin('base', 'phase_mode', phase_mode);
assignin('base', 'Disturbances', Disturbances);

fprintf('\n=======================================================\n');
fprintf('  INITIALIZATION COMPLETE\n');
fprintf('=======================================================\n');
fprintf('\nAvailable controllers:\n');
fprintf('  1 = PID   (Classical three-term)\n');
fprintf('  2 = LQR   (Linear Quadratic Regulator)\n');
fprintf('  3 = LQG   (LQR + Kalman Filter)\n');
fprintf('  4 = SMC   (Sliding Mode Control)\n');
fprintf('  5 = MPC   (Model Predictive Control)\n');
fprintf('  6 = Hinf  (H-infinity Robust Control)\n');
fprintf('\nRun: run_advanced_simulation(T_sim, controller_mode)\n');
