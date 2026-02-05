%% GNC_System_Init.m - Parameter Initialization
%  Run this before simulating GNC_System.slx

fprintf('Initializing GNC parameters...\n');

%% Spacecraft Properties
Ixx = 1200;  % Moment of inertia about x-axis [kg*m^2]
Iyy = 1350;  % Moment of inertia about y-axis [kg*m^2]
Izz = 980;   % Moment of inertia about z-axis [kg*m^2]

%% Initial Conditions
theta0_x = 5 * pi/180;   % Initial attitude error [rad] (5 degrees)
omega0_x = 0.01;         % Initial angular rate [rad/s]

%% Controller Gains (tuned for Ixx)
% Natural frequency and damping
wn = 0.5;       % Natural frequency [rad/s]
zeta = 0.7;     % Damping ratio

Kp_x = Ixx * wn^2;          % Proportional gain
Kd_x = 2 * zeta * wn * Ixx; % Derivative gain

%% Actuator Parameters
rw_tau = 0.02;       % Reaction wheel time constant [s]
tau_max = 0.5;       % Max torque [N*m]

%% Simulation Settings
dt = 0.01;           % Time step [s]
T_sim = 100;         % Duration [s]

fprintf('Parameters initialized.\n');
fprintf('  Inertia:    Ixx = %.0f kg*m^2\n', Ixx);
fprintf('  Init error: %.1f deg\n', theta0_x * 180/pi);
fprintf('  Kp = %.2f, Kd = %.2f\n', Kp_x, Kd_x);
fprintf('\nRun: sim(''GNC_System'', %d)\n', T_sim);
