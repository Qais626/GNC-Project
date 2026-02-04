%% ========================================================================
%  GNC Mission - Control System Design and Analysis
%  ========================================================================
%  This script designs and analyzes attitude control systems for a
%  spacecraft GNC (Guidance, Navigation & Control) mission using MATLAB
%  control toolbox functions.
%
%  Controllers designed:
%    1. PID  - Classical proportional-integral-derivative
%    2. LQR  - Linear Quadratic Regulator (optimal state feedback)
%    3. H-infinity / Loop-Shaping (robust control)
%
%  Outputs:
%    - Bode diagrams, root locus, Nyquist, Nichols charts
%    - Step response comparisons
%    - Gain/phase margin summary table
%    - All figures saved to ../../output/matlab/
%
%  Author : GNC Project Team
%  Date   : 2026-01-31
%  ========================================================================

clear; clc; close all;

%% ------------------------------------------------------------------------
%  0. Configuration and Output Directory Setup
%  ------------------------------------------------------------------------

% Output directory for saved figures
outputDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'output', 'matlab');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
    fprintf('[INFO] Created output directory: %s\n', outputDir);
end

% Simulation time vector for step responses
tFinal = 300;          % seconds - allow enough time for settling
dt     = 0.01;         % time step (s)
tSim   = 0:dt:tFinal;  % time vector

fprintf('=============================================================\n');
fprintf('  GNC Mission - Control System Design and Analysis\n');
fprintf('=============================================================\n\n');

%% ========================================================================
%  1. LINEARIZED ATTITUDE DYNAMICS (SINGLE-AXIS)
%  ========================================================================
%  The single-axis attitude dynamics are:
%      J * theta_ddot = tau
%  where:
%      J     = moment of inertia about the control axis [kg*m^2]
%      theta = attitude angle [rad]
%      tau   = control torque [N*m]
%
%  State-space form:
%      x = [theta; theta_dot]
%      x_dot = A*x + B*u
%      y     = C*x + D*u

fprintf('[STEP 1] Defining linearized attitude dynamics...\n');

J = 4500;  % Moment of inertia [kg*m^2] - typical for a medium spacecraft

% State-space matrices
A_plant = [0, 1;
           0, 0];

B_plant = [0;
           1/J];

C_plant = [1, 0];  % Output is attitude angle theta

D_plant = 0;       % No direct feedthrough

% Create state-space and transfer function models
sys_ss = ss(A_plant, B_plant, C_plant, D_plant);
sys_tf = tf(sys_ss);  % Transfer function: 1/(J*s^2)

fprintf('  Moment of inertia J = %.0f kg*m^2\n', J);
fprintf('  Plant transfer function: G(s) = 1 / (%.0f * s^2)\n', J);
fprintf('  Open-loop poles: %s\n\n', mat2str(pole(sys_tf), 4));

%% ========================================================================
%  2. PID CONTROLLER DESIGN
%  ========================================================================
%  Classical PID controller:
%      C_pid(s) = Kp + Ki/s + Kd*s
%
%  In proper transfer function form (with derivative filter):
%      C_pid(s) = Kp + Ki/s + Kd*s/(tau_f*s + 1)
%
%  Gains chosen for the double-integrator plant to achieve:
%    - Reasonable bandwidth (~0.01 rad/s)
%    - Adequate phase margin (>30 deg)
%    - Zero steady-state error to step reference

fprintf('[STEP 2] Designing PID controller...\n');

% PID gains
Kp = 0.5;    % Proportional gain
Ki = 0.01;   % Integral gain
Kd = 2.0;    % Derivative gain

% Derivative filter time constant (to make controller proper)
tau_f = 0.1;  % seconds

% Construct PID transfer function
%   C_pid = Kp + Ki/s + Kd*s/(tau_f*s + 1)
s = tf('s');
C_pid = Kp + Ki/s + Kd * s / (tau_f * s + 1);

fprintf('  PID Gains: Kp = %.2f, Ki = %.3f, Kd = %.2f\n', Kp, Ki, Kd);
fprintf('  Derivative filter tau_f = %.2f s\n', tau_f);

% Open-loop transfer function: L_pid = C_pid * G
L_pid = C_pid * sys_tf;

% Closed-loop transfer function: T_pid = L_pid / (1 + L_pid)
T_pid = feedback(L_pid, 1);

% Sensitivity function: S_pid = 1 / (1 + L_pid)
S_pid = feedback(1, L_pid);

% Compute gain and phase margins
[Gm_pid, Pm_pid, Wcg_pid, Wcp_pid] = margin(L_pid);
Gm_pid_dB = 20 * log10(Gm_pid);

fprintf('  Open-loop bandwidth (0dB crossing): %.4f rad/s\n', Wcp_pid);
fprintf('  Gain Margin : %.2f dB (at %.4f rad/s)\n', Gm_pid_dB, Wcg_pid);
fprintf('  Phase Margin: %.2f deg (at %.4f rad/s)\n', Pm_pid, Wcp_pid);

% Step response characteristics
stepInfo_pid = stepinfo(T_pid);
fprintf('  Rise Time    : %.2f s\n', stepInfo_pid.RiseTime);
fprintf('  Settling Time: %.2f s\n', stepInfo_pid.SettlingTime);
fprintf('  Overshoot    : %.2f %%\n', stepInfo_pid.Overshoot);
fprintf('  Steady-State : %.4f\n\n', dcgain(T_pid));

%% ========================================================================
%  3. LQR CONTROLLER DESIGN
%  ========================================================================
%  Linear Quadratic Regulator minimizes:
%      J_cost = integral( x'*Q*x + u'*R*u ) dt
%
%  State weighting Q penalizes attitude error and angular rate.
%  Control weighting R penalizes torque usage.
%
%  The optimal gain K is found by solving the algebraic Riccati equation.

fprintf('[STEP 3] Designing LQR controller...\n');

% Cost function weights
Q_lqr = diag([100, 10]);  % Penalize angle error (100) and rate (10)
R_lqr = 1;                % Penalize control effort

% Compute optimal LQR gain
[K_lqr, S_riccati, eigCL] = lqr(A_plant, B_plant, Q_lqr, R_lqr);

fprintf('  Q = diag([%.0f, %.0f]), R = %.0f\n', Q_lqr(1,1), Q_lqr(2,2), R_lqr);
fprintf('  LQR Gain K = [%.4f, %.4f]\n', K_lqr(1), K_lqr(2));
fprintf('  Closed-loop eigenvalues: %.4f, %.4f\n', eigCL(1), eigCL(2));

% Closed-loop state-space with LQR feedback: u = -K*x + K(1)*r
% For step response tracking, we use a reference input with prefilter
A_cl_lqr = A_plant - B_plant * K_lqr;
B_cl_lqr = B_plant * K_lqr(1);  % Scale by K(1) for unit DC gain on theta
C_cl_lqr = C_plant;
D_cl_lqr = 0;

sys_cl_lqr = ss(A_cl_lqr, B_cl_lqr, C_cl_lqr, D_cl_lqr);

% Adjust DC gain to unity (prefilter / reference scaling)
dc_lqr = dcgain(sys_cl_lqr);
if abs(dc_lqr) > 1e-10
    N_bar = 1 / dc_lqr;  % Prefilter gain for unity steady-state
else
    N_bar = 1;
end
B_cl_lqr_scaled = B_cl_lqr * N_bar;
sys_cl_lqr = ss(A_cl_lqr, B_cl_lqr_scaled, C_cl_lqr, D_cl_lqr);

% Transfer function form for Bode analysis
T_lqr = tf(sys_cl_lqr);

% Compute equivalent open-loop for margin analysis
% L_lqr = K * (sI - A)^{-1} * B
sys_openloop_lqr = ss(A_plant, B_plant, K_lqr, 0);
L_lqr = tf(sys_openloop_lqr);

[Gm_lqr, Pm_lqr, Wcg_lqr, Wcp_lqr] = margin(L_lqr);
Gm_lqr_dB = 20 * log10(Gm_lqr);

% Step response analysis
stepInfo_lqr = stepinfo(sys_cl_lqr);
fprintf('  Rise Time    : %.2f s\n', stepInfo_lqr.RiseTime);
fprintf('  Settling Time: %.2f s\n', stepInfo_lqr.SettlingTime);
fprintf('  Overshoot    : %.2f %%\n', stepInfo_lqr.Overshoot);
fprintf('  DC Gain      : %.4f (with prefilter N_bar = %.4f)\n', ...
    dcgain(sys_cl_lqr), N_bar);
fprintf('  Gain Margin  : %.2f dB\n', Gm_lqr_dB);
fprintf('  Phase Margin : %.2f deg\n\n', Pm_lqr);

%% ========================================================================
%  4. H-INFINITY / LOOP-SHAPING CONTROLLER DESIGN
%  ========================================================================
%  Mixed sensitivity H-infinity design:
%      minimize || [W1*S ; W2*T] ||_inf
%  where:
%      S = 1/(1+L)     sensitivity function
%      T = L/(1+L)     complementary sensitivity
%      W1 = s/(s+0.1)  low-frequency performance weight
%      W2 = 100*s/(s+100)  high-frequency robustness weight
%
%  If the Robust Control Toolbox is not available, we fall back to a
%  manual loop-shaping design using lead-lag compensation.

fprintf('[STEP 4] Designing H-infinity / loop-shaping controller...\n');

% Define weighting functions
W1 = s / (s + 0.1);        % Low-frequency: force good tracking below 0.1 rad/s
W2 = 100 * s / (s + 100);  % High-frequency: force rolloff above 100 rad/s

% Attempt H-infinity synthesis using mixsyn (Robust Control Toolbox)
hasRobustToolbox = false;
try
    % Check if mixsyn is available
    if exist('mixsyn', 'file')
        hasRobustToolbox = true;
    end
catch
    hasRobustToolbox = false;
end

if hasRobustToolbox
    %% --- Robust Control Toolbox available: use mixsyn ---
    fprintf('  Robust Control Toolbox detected. Using mixsyn...\n');

    % Mixed sensitivity synthesis
    % Minimize ||[W1*S; W2*T]||_inf
    [C_hinf, CL_hinf, gamma_hinf] = mixsyn(sys_tf, W1, [], W2);

    fprintf('  Achieved gamma (H-inf norm): %.4f\n', gamma_hinf);

else
    %% --- Manual loop-shaping fallback ---
    fprintf('  Robust Control Toolbox not found. Using manual loop-shaping.\n');

    % Design a lead-lag compensator that shapes the loop to satisfy
    % the performance and robustness weights approximately.
    %
    % Strategy:
    %   - Lag section: boost low-frequency gain for tracking
    %   - Lead section: add phase at crossover for stability
    %   - Gain: set crossover frequency around 0.05 rad/s

    % Lag compensator: boost gain below break frequency
    % C_lag = K_lag * (s + z_lag) / (s + p_lag)
    z_lag = 0.01;    % zero location
    p_lag = 0.001;   % pole location (lower = more gain boost at low freq)
    K_lag = p_lag / z_lag;  % Normalize DC gain
    C_lag = (s + z_lag) / (s + p_lag) * K_lag;

    % Lead compensator: add phase near crossover
    % C_lead = K_lead * (s + z_lead) / (s + p_lead)
    z_lead = 0.02;   % zero location
    p_lead = 0.2;    % pole location
    K_lead = 1;
    C_lead = (s + z_lead) / (s + p_lead) * K_lead;

    % Overall gain to set desired crossover
    K_overall = 250;

    % Combined loop-shaping controller
    C_hinf = K_overall * C_lag * C_lead;

    fprintf('  Loop-shaping controller designed with:\n');
    fprintf('    Lag  : zero=%.3f, pole=%.4f\n', z_lag, p_lag);
    fprintf('    Lead : zero=%.3f, pole=%.3f\n', z_lead, p_lead);
    fprintf('    Gain : %.1f\n', K_overall);
end

% Form open-loop and closed-loop for H-infinity design
L_hinf = C_hinf * sys_tf;
T_hinf = feedback(L_hinf, 1);
S_hinf = feedback(1, L_hinf);

% Margins
[Gm_hinf, Pm_hinf, Wcg_hinf, Wcp_hinf] = margin(L_hinf);
Gm_hinf_dB = 20 * log10(Gm_hinf);

% Step response
stepInfo_hinf = stepinfo(T_hinf);
fprintf('  Rise Time    : %.2f s\n', stepInfo_hinf.RiseTime);
fprintf('  Settling Time: %.2f s\n', stepInfo_hinf.SettlingTime);
fprintf('  Overshoot    : %.2f %%\n', stepInfo_hinf.Overshoot);
fprintf('  Gain Margin  : %.2f dB\n', Gm_hinf_dB);
fprintf('  Phase Margin : %.2f deg\n\n', Pm_hinf);

%% ========================================================================
%  5. GENERATE COMPARISON PLOTS
%  ========================================================================

fprintf('[STEP 5] Generating comparison plots...\n');

% Color scheme for consistent plot styling
color_pid  = [0.0, 0.45, 0.74];   % Blue
color_lqr  = [0.85, 0.33, 0.10];  % Red-orange
color_hinf = [0.47, 0.67, 0.19];  % Green

lineWidth = 1.8;
fontSize  = 12;

%% --- Figure 1: Bode Plots Comparison (All 3 Controllers) ---
figure('Name', 'Bode Comparison', 'Position', [100, 400, 900, 600]);

% Use bode plot options for formatting
opts = bodeoptions;
opts.FreqUnits  = 'rad/s';
opts.Grid       = 'on';
opts.Title.FontSize = 14;
opts.XLabel.FontSize = fontSize;
opts.YLabel.FontSize = fontSize;

% Plot open-loop Bode for all three controllers
bodeplot(L_pid, 'b-', L_lqr, 'r--', L_hinf, 'g-.', opts);
legend('PID', 'LQR', 'H-inf/Loop-Shaping', ...
       'Location', 'southwest', 'FontSize', fontSize-1);
title('Open-Loop Bode Diagram Comparison', 'Interpreter', 'latex', 'FontSize', 14);

% Save figure
saveas(gcf, fullfile(outputDir, 'bode_comparison.png'));
saveas(gcf, fullfile(outputDir, 'bode_comparison.fig'));
fprintf('  Saved: bode_comparison.png/.fig\n');

%% --- Figure 2: Step Response Comparison (All 3 Controllers) ---
figure('Name', 'Step Response Comparison', 'Position', [150, 350, 900, 500]);

hold on; grid on;

% Compute step responses
[y_pid, t_pid]   = step(T_pid, tSim);
[y_lqr, t_lqr]   = step(sys_cl_lqr, tSim);
[y_hinf, t_hinf] = step(T_hinf, tSim);

% Plot with consistent colors
plot(t_pid, y_pid, '-', 'Color', color_pid, 'LineWidth', lineWidth);
plot(t_lqr, y_lqr, '--', 'Color', color_lqr, 'LineWidth', lineWidth);
plot(t_hinf, y_hinf, '-.', 'Color', color_hinf, 'LineWidth', lineWidth);

% Reference line
yline(1, 'k:', 'Reference', 'LineWidth', 1.0, 'LabelHorizontalAlignment', 'left');

% 2% settling band
yline(1.02, 'k--', 'LineWidth', 0.5);
yline(0.98, 'k--', 'LineWidth', 0.5);

xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', fontSize);
ylabel('Attitude Angle $\theta$ [rad]', 'Interpreter', 'latex', 'FontSize', fontSize);
title('Closed-Loop Step Response Comparison', 'Interpreter', 'latex', 'FontSize', 14);
legend('PID', 'LQR', 'H$_\infty$/Loop-Shaping', '$\pm 2\%$ Band', ...
       'Interpreter', 'latex', 'Location', 'southeast', 'FontSize', fontSize-1);
xlim([0 min(tFinal, 200)]);
hold off;

saveas(gcf, fullfile(outputDir, 'step_response_comparison.png'));
saveas(gcf, fullfile(outputDir, 'step_response_comparison.fig'));
fprintf('  Saved: step_response_comparison.png/.fig\n');

%% --- Figure 3: Root Locus of PID Design ---
figure('Name', 'Root Locus - PID', 'Position', [200, 300, 700, 600]);

% Root locus of open-loop L_pid with variable gain
rlocus(L_pid);
title('Root Locus: PID Controller with Double-Integrator Plant', ...
      'Interpreter', 'latex', 'FontSize', 14);
grid on;

% Mark closed-loop poles
hold on;
cl_poles = pole(T_pid);
plot(real(cl_poles), imag(cl_poles), 'rx', 'MarkerSize', 12, 'LineWidth', 2);
legend('Root Locus', 'Closed-Loop Poles', 'Location', 'best');
hold off;

saveas(gcf, fullfile(outputDir, 'root_locus_pid.png'));
saveas(gcf, fullfile(outputDir, 'root_locus_pid.fig'));
fprintf('  Saved: root_locus_pid.png/.fig\n');

%% --- Figure 4: Nyquist Plot ---
figure('Name', 'Nyquist Plot', 'Position', [250, 250, 700, 600]);

hold on;
nyquist(L_pid);
title('Nyquist Plot: PID Open-Loop', 'Interpreter', 'latex', 'FontSize', 14);

% Mark critical point (-1, 0)
plot(-1, 0, 'r+', 'MarkerSize', 15, 'LineWidth', 2);
hold off;

saveas(gcf, fullfile(outputDir, 'nyquist_pid.png'));
saveas(gcf, fullfile(outputDir, 'nyquist_pid.fig'));
fprintf('  Saved: nyquist_pid.png/.fig\n');

%% --- Figure 5: Nichols Chart ---
figure('Name', 'Nichols Chart', 'Position', [300, 200, 800, 600]);

% Nichols chart for all three controllers
nichols(L_pid, 'b-', L_lqr, 'r--', L_hinf, 'g-.');
legend('PID', 'LQR', 'H-inf/Loop-Shaping', 'Location', 'best');
title('Nichols Chart Comparison', 'Interpreter', 'latex', 'FontSize', 14);
grid on;

saveas(gcf, fullfile(outputDir, 'nichols_chart.png'));
saveas(gcf, fullfile(outputDir, 'nichols_chart.fig'));
fprintf('  Saved: nichols_chart.png/.fig\n');

%% --- Additional Figure: Sensitivity Functions ---
figure('Name', 'Sensitivity Functions', 'Position', [350, 150, 900, 500]);

% Compute sensitivity and complementary sensitivity for PID
S_pid_tf = feedback(1, L_pid);
T_pid_tf = feedback(L_pid, 1);

subplot(1,2,1);
bodemag(S_pid_tf, 'b-', S_hinf, 'g-.', {1e-4, 1e2});
hold on;
bodemag(1/W1, 'k:', {1e-4, 1e2});
hold off;
title('Sensitivity $S(s)$', 'Interpreter', 'latex', 'FontSize', 13);
legend('PID', 'H-inf', '$1/W_1$ bound', ...
       'Interpreter', 'latex', 'Location', 'best');
grid on;

subplot(1,2,2);
bodemag(T_pid_tf, 'b-', T_hinf, 'g-.', {1e-4, 1e2});
hold on;
bodemag(1/W2, 'k:', {1e-4, 1e2});
hold off;
title('Complementary Sensitivity $T(s)$', 'Interpreter', 'latex', 'FontSize', 13);
legend('PID', 'H-inf', '$1/W_2$ bound', ...
       'Interpreter', 'latex', 'Location', 'best');
grid on;

saveas(gcf, fullfile(outputDir, 'sensitivity_functions.png'));
saveas(gcf, fullfile(outputDir, 'sensitivity_functions.fig'));
fprintf('  Saved: sensitivity_functions.png/.fig\n');

%% ========================================================================
%  6. PERFORMANCE SUMMARY TABLE
%  ========================================================================

fprintf('\n');
fprintf('=============================================================\n');
fprintf('  CONTROL SYSTEM PERFORMANCE SUMMARY\n');
fprintf('=============================================================\n');
fprintf('%-20s | %-12s | %-12s | %-15s\n', ...
    'Metric', 'PID', 'LQR', 'H-inf/LoopShp');
fprintf('-------------------------------------------------------------\n');
fprintf('%-20s | %10.2f s | %10.2f s | %13.2f s\n', ...
    'Rise Time', stepInfo_pid.RiseTime, stepInfo_lqr.RiseTime, stepInfo_hinf.RiseTime);
fprintf('%-20s | %10.2f s | %10.2f s | %13.2f s\n', ...
    'Settling Time', stepInfo_pid.SettlingTime, stepInfo_lqr.SettlingTime, stepInfo_hinf.SettlingTime);
fprintf('%-20s | %10.2f %% | %10.2f %% | %13.2f %%\n', ...
    'Overshoot', stepInfo_pid.Overshoot, stepInfo_lqr.Overshoot, stepInfo_hinf.Overshoot);
fprintf('%-20s | %10.2f dB | %10.2f dB | %13.2f dB\n', ...
    'Gain Margin', Gm_pid_dB, Gm_lqr_dB, Gm_hinf_dB);
fprintf('%-20s | %10.2f   | %10.2f   | %13.2f   \n', ...
    'Phase Margin [deg]', Pm_pid, Pm_lqr, Pm_hinf);
fprintf('%-20s | %10.4f   | %10.4f   | %13.4f   \n', ...
    'DC Gain', dcgain(T_pid), dcgain(sys_cl_lqr), dcgain(T_hinf));
fprintf('-------------------------------------------------------------\n');
fprintf('  Plant: G(s) = 1/(%.0f s^2)  |  Single-axis attitude dynamics\n', J);
fprintf('=============================================================\n');

%% --- Settling time comparison (PID vs LQR) ---
fprintf('\n  Settling Time Comparison (PID vs LQR):\n');
if stepInfo_pid.SettlingTime > stepInfo_lqr.SettlingTime
    improvement = (stepInfo_pid.SettlingTime - stepInfo_lqr.SettlingTime) / ...
                   stepInfo_pid.SettlingTime * 100;
    fprintf('    LQR is %.1f%% faster in settling than PID.\n', improvement);
else
    improvement = (stepInfo_lqr.SettlingTime - stepInfo_pid.SettlingTime) / ...
                   stepInfo_lqr.SettlingTime * 100;
    fprintf('    PID is %.1f%% faster in settling than LQR.\n', improvement);
end

%% ========================================================================
%  7. DISCRETIZATION FOR FLIGHT SOFTWARE IMPLEMENTATION
%  ========================================================================
%  Convert continuous controllers to discrete-time for onboard computer.

fprintf('\n[STEP 7] Discretizing controllers for flight software...\n');

Ts = 0.1;  % Sample time for onboard computer [s] (10 Hz control loop)

% Discretize PID controller using Tustin (bilinear) method
C_pid_d = c2d(C_pid, Ts, 'tustin');
fprintf('  PID controller discretized at Ts = %.2f s (Tustin method)\n', Ts);

% Discretize plant for LQR analysis
sys_d = c2d(sys_ss, Ts, 'zoh');  % Zero-order hold for plant
fprintf('  Plant discretized at Ts = %.2f s (ZOH method)\n', Ts);

% Discrete LQR
[K_lqr_d, ~, ~] = dlqr(sys_d.A, sys_d.B, Q_lqr, R_lqr);
fprintf('  Discrete LQR Gain K_d = [%.4f, %.4f]\n', K_lqr_d(1), K_lqr_d(2));

%% ========================================================================
%  COMPLETE
%  ========================================================================
fprintf('\n=============================================================\n');
fprintf('  Control design analysis complete.\n');
fprintf('  All figures saved to: %s\n', outputDir);
fprintf('=============================================================\n');
