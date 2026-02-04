%% =========================================================================
%  Attitude Control Simulink Model (Programmatic Generation / Pure MATLAB Sim)
%  Since Simulink .slx files are binary, this implements the equivalent
%  simulation in pure MATLAB using state-space models and lsim.
%  =========================================================================
clear; clc; close all;

%% ------------------------------------------------------------------------
%  Output Directory
%  ------------------------------------------------------------------------

output_dir = fullfile(fileparts(mfilename('fullpath')), ...
    '..', '..', '..', 'output', 'matlab');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('=============================================================\n');
fprintf('  Attitude Control Model - PID vs LQR Comparison\n');
fprintf('=============================================================\n\n');

%% ========================================================================
%  Section 1: Define Plant - Single-Axis Attitude Dynamics
%  ========================================================================
%  The rotational dynamics for a single axis are:
%    J * theta_ddot = tau
%  In state-space form with x = [theta; theta_dot]:
%    x_dot = A*x + B*u
%    y     = C*x + D*u

fprintf('[STEP 1] Defining single-axis attitude dynamics plant...\n');

J = 4500;                          % Moment of inertia [kg*m^2]

A_plant = [0  1;                   % State matrix: [d(theta)/dt = omega]
           0  0];                  %                [d(omega)/dt = tau/J]

B_plant = [0;                      % Input matrix
           1/J];

C_plant = [1  0];                  % Output: theta (angle)

D_plant = 0;                       % No direct feedthrough

plant = ss(A_plant, B_plant, C_plant, D_plant);

% Verify plant properties
fprintf('  Moment of inertia J     : %.0f kg*m^2\n', J);
fprintf('  Plant poles             : %.4f, %.4f\n', eig(A_plant));
fprintf('  Plant is open-loop unstable (double integrator at origin)\n\n');

%% ========================================================================
%  Section 2: PID Controller Design
%  ========================================================================
%  PID controller: C(s) = Kp + Ki/s + Kd*s
%  Tuned for the double-integrator attitude plant.

fprintf('[STEP 2] Designing PID controller...\n');

Kp = 0.5;                         % Proportional gain
Ki = 0.01;                        % Integral gain
Kd = 2.0;                         % Derivative gain

% Create PID controller transfer function
C_pid = pid(Kp, Ki, Kd);

% Form the closed-loop system: G_cl = C*P / (1 + C*P)
sys_pid_cl = feedback(C_pid * plant, 1);

fprintf('  Kp = %.2f, Ki = %.2f, Kd = %.2f\n', Kp, Ki, Kd);
fprintf('  Closed-loop PID poles   :');
pid_poles = pole(sys_pid_cl);
for k = 1:length(pid_poles)
    if imag(pid_poles(k)) >= 0
        fprintf(' %.4f + %.4fj', real(pid_poles(k)), imag(pid_poles(k)));
    end
end
fprintf('\n\n');

%% ========================================================================
%  Section 3: LQR Controller Design
%  ========================================================================
%  Full-state feedback: u = -K*x + K*x_ref
%  Minimize J = integral( x'*Q*x + u'*R*u ) dt

fprintf('[STEP 3] Designing LQR controller...\n');

Q_lqr = diag([100, 10]);          % State weighting: penalize angle > rate
R_lqr = 1;                        % Control effort weighting

% Compute optimal LQR gain
K_lqr = lqr(A_plant, B_plant, Q_lqr, R_lqr);

% Closed-loop state matrix with LQR feedback
A_cl_lqr = A_plant - B_plant * K_lqr;

% Build closed-loop state-space system for LQR
% Input: reference angle (mapped through B*K to track reference)
sys_lqr_cl = ss(A_cl_lqr, B_plant, C_plant, D_plant);

fprintf('  Q = diag([%.0f, %.0f]), R = %.0f\n', Q_lqr(1,1), Q_lqr(2,2), R_lqr);
fprintf('  LQR gain K             : [%.4f, %.4f]\n', K_lqr(1), K_lqr(2));
fprintf('  Closed-loop LQR poles  : %.4f, %.4f\n', eig(A_cl_lqr));
fprintf('\n');

%% ========================================================================
%  Section 4: Simulate Step Response (Linear, No Noise)
%  ========================================================================

fprintf('[STEP 4] Computing clean step responses...\n');

t = (0:0.01:60)';                 % 60 seconds at 100 Hz

% Unit step response for both controllers
[y_pid, t_pid] = step(sys_pid_cl, t);
[y_lqr, t_lqr] = step(sys_lqr_cl, t);

% Normalize LQR response (scale input so steady-state = 1)
% For LQR tracking, the DC gain needs correction
dc_gain_lqr = dcgain(sys_lqr_cl);
if abs(dc_gain_lqr) > 1e-10
    y_lqr_norm = y_lqr / dc_gain_lqr;
else
    % DC gain is zero or very small; use manual scaling
    y_lqr_norm = y_lqr * (K_lqr(1) * J);
end

fprintf('  PID final value         : %.6f\n', y_pid(end));
fprintf('  LQR DC gain             : %.6f\n', dc_gain_lqr);
fprintf('\n');

%% ========================================================================
%  Section 5: Time-Domain Simulation with Sensor Noise and Disturbances
%  ========================================================================

fprintf('[STEP 5] Running time-domain sim with noise and disturbances...\n');

rng(42);  % Reproducible random number seed

% Simulation time vector
dt  = 0.01;                        % Time step [s]
t_sim = (0:dt:60)';
N_sim = length(t_sim);

% Noise and disturbance signals
sensor_noise = 0.001 * randn(N_sim, 1);           % Measurement noise [rad]
disturbance  = 0.0001 * sin(0.8 * 2*pi * t_sim);  % Flex mode excitation [N*m]

% Reference command: 0.1 rad step (approximately 5.7 deg)
theta_ref = 0.1;

% --- PID Controller Simulation ---
x_pid = [0; 0];                    % Initial state [theta; omega]
y_noisy_pid   = zeros(N_sim, 1);
u_pid_history = zeros(N_sim, 1);
integral_err_pid = 0;
prev_err_pid = 0;

for i = 1:N_sim
    % Measured output with sensor noise
    theta_meas = x_pid(1) + sensor_noise(i);

    % PID error computation
    err = theta_ref - theta_meas;
    integral_err_pid = integral_err_pid + err * dt;

    % Derivative term: use rate feedback (negative of omega) to avoid
    % derivative kick on reference step
    u_pid = Kp * err + Ki * integral_err_pid + Kd * (-x_pid(2));

    % Clamp control torque to actuator limits
    u_max = 5.0;  % Maximum torque [N*m]
    u_pid = max(min(u_pid, u_max), -u_max);

    % Store outputs
    y_noisy_pid(i)   = x_pid(1);
    u_pid_history(i) = u_pid;

    % Propagate state: x_dot = A*x + B*(u + disturbance)
    if i < N_sim
        x_dot = A_plant * x_pid + B_plant * (u_pid + disturbance(i));
        x_pid = x_pid + x_dot * dt;  % Forward Euler integration
    end
end

% --- LQR Controller Simulation ---
x_lqr = [0; 0];                    % Initial state [theta; omega]
y_noisy_lqr   = zeros(N_sim, 1);
u_lqr_history = zeros(N_sim, 1);

for i = 1:N_sim
    % Measured state with sensor noise
    theta_meas_lqr = x_lqr(1) + sensor_noise(i);
    omega_meas_lqr = x_lqr(2) + 0.0005 * randn;  % Rate gyro noise

    % State error (reference tracking)
    x_err = [theta_ref - theta_meas_lqr; ...
             0          - omega_meas_lqr];

    % LQR control law: u = K * x_error
    u_lqr = K_lqr * x_err;

    % Clamp control torque
    u_lqr = max(min(u_lqr, u_max), -u_max);

    % Store outputs
    y_noisy_lqr(i)   = x_lqr(1);
    u_lqr_history(i) = u_lqr;

    % Propagate state with disturbance
    if i < N_sim
        x_dot = A_plant * x_lqr + B_plant * (u_lqr + disturbance(i));
        x_lqr = x_lqr + x_dot * dt;
    end
end

fprintf('  PID final angle (noisy) : %.6f rad (ref = %.4f rad)\n', ...
    y_noisy_pid(end), theta_ref);
fprintf('  LQR final angle (noisy) : %.6f rad (ref = %.4f rad)\n', ...
    y_noisy_lqr(end), theta_ref);
fprintf('\n');

%% ========================================================================
%  Section 6: Generate Comparison Plots
%  ========================================================================

fprintf('[STEP 6] Generating comparison plots...\n\n');

% Color palette
c_pid = [0.0 0.45 0.74];          % Blue for PID
c_lqr = [0.85 0.33 0.10];         % Orange for LQR
c_ref = [0.5 0.5 0.5];            % Gray for reference

% ---- Figure 1: Clean Step Response Comparison ----
fig1 = figure('Name', 'Step Response Comparison', ...
    'Position', [50 400 900 500], 'Color', 'w');

plot(t_pid, y_pid, '-', 'Color', c_pid, 'LineWidth', 2); hold on;
plot(t_lqr, y_lqr_norm, '-', 'Color', c_lqr, 'LineWidth', 2);
yline(1.0, '--', 'Color', c_ref, 'LineWidth', 1);
yline(1.02, ':', 'Color', [0.8 0.2 0.2], 'LineWidth', 0.8);
yline(0.98, ':', 'Color', [0.8 0.2 0.2], 'LineWidth', 0.8);
hold off;

xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Angle / Reference', 'Interpreter', 'latex', 'FontSize', 12);
title('Step Response: PID vs LQR (No Noise)', ...
    'Interpreter', 'latex', 'FontSize', 14);
legend({'PID', 'LQR', 'Reference', '$\pm 2\%$ Band'}, ...
    'Interpreter', 'latex', 'Location', 'best', 'FontSize', 10);
grid on;
xlim([0 60]);
set(gca, 'FontSize', 11, 'TickLabelInterpreter', 'latex');

print(fig1, fullfile(output_dir, 'step_response_comparison'), '-dpng', '-r150');
fprintf('  Saved: step_response_comparison.png\n');

% ---- Figure 2: Control Effort Comparison ----
fig2 = figure('Name', 'Control Effort Comparison', ...
    'Position', [100 350 900 500], 'Color', 'w');

subplot(2, 1, 1);
plot(t_sim, u_pid_history, '-', 'Color', c_pid, 'LineWidth', 1.2);
ylabel('$\tau_{PID}$ [N$\cdot$m]', 'Interpreter', 'latex', 'FontSize', 11);
title('Control Effort: PID vs LQR (With Noise \& Disturbance)', ...
    'Interpreter', 'latex', 'FontSize', 14);
grid on;
xlim([0 60]);
yline(0, 'k-', 'LineWidth', 0.5);
set(gca, 'FontSize', 10, 'TickLabelInterpreter', 'latex');

subplot(2, 1, 2);
plot(t_sim, u_lqr_history, '-', 'Color', c_lqr, 'LineWidth', 1.2);
ylabel('$\tau_{LQR}$ [N$\cdot$m]', 'Interpreter', 'latex', 'FontSize', 11);
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 12);
grid on;
xlim([0 60]);
yline(0, 'k-', 'LineWidth', 0.5);
set(gca, 'FontSize', 10, 'TickLabelInterpreter', 'latex');

% Link axes
ax2 = findall(fig2, 'Type', 'axes');
linkaxes(ax2, 'x');

print(fig2, fullfile(output_dir, 'control_effort_comparison'), '-dpng', '-r150');
fprintf('  Saved: control_effort_comparison.png\n');

% ---- Figure 3: Noisy Response with Disturbance ----
fig3 = figure('Name', 'Noisy Response Comparison', ...
    'Position', [150 300 900 550], 'Color', 'w');

subplot(2, 1, 1);
plot(t_sim, y_noisy_pid * 180/pi, '-', 'Color', c_pid, 'LineWidth', 1.0);
hold on;
yline(theta_ref * 180/pi, '--', 'Color', c_ref, 'LineWidth', 1.5);
hold off;
ylabel('$\theta_{PID}$ [$^\circ$]', 'Interpreter', 'latex', 'FontSize', 11);
title('Response with Sensor Noise and Flex-Mode Disturbance', ...
    'Interpreter', 'latex', 'FontSize', 14);
legend({'PID Output', 'Reference'}, ...
    'Interpreter', 'latex', 'Location', 'best', 'FontSize', 9);
grid on;
xlim([0 60]);
set(gca, 'FontSize', 10, 'TickLabelInterpreter', 'latex');

subplot(2, 1, 2);
plot(t_sim, y_noisy_lqr * 180/pi, '-', 'Color', c_lqr, 'LineWidth', 1.0);
hold on;
yline(theta_ref * 180/pi, '--', 'Color', c_ref, 'LineWidth', 1.5);
hold off;
ylabel('$\theta_{LQR}$ [$^\circ$]', 'Interpreter', 'latex', 'FontSize', 11);
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 12);
legend({'LQR Output', 'Reference'}, ...
    'Interpreter', 'latex', 'Location', 'best', 'FontSize', 9);
grid on;
xlim([0 60]);
set(gca, 'FontSize', 10, 'TickLabelInterpreter', 'latex');

ax3 = findall(fig3, 'Type', 'axes');
linkaxes(ax3, 'x');

print(fig3, fullfile(output_dir, 'noisy_response_comparison'), '-dpng', '-r150');
fprintf('  Saved: noisy_response_comparison.png\n');

% ---- Figure 4: Bode Diagram of Open-Loop Plant ----
fig4 = figure('Name', 'Open-Loop Bode', ...
    'Position', [200 250 900 600], 'Color', 'w');

% Open-loop transfer function (plant alone)
[mag_p, phase_p, w_p] = bode(plant);
mag_p   = squeeze(mag_p);
phase_p = squeeze(phase_p);

subplot(2, 1, 1);
semilogx(w_p, 20*log10(mag_p), 'b-', 'LineWidth', 2);
ylabel('Magnitude [dB]', 'Interpreter', 'latex', 'FontSize', 11);
title('Open-Loop Plant Bode Diagram ($1/Js^2$)', ...
    'Interpreter', 'latex', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 10, 'TickLabelInterpreter', 'latex');

subplot(2, 1, 2);
semilogx(w_p, phase_p, 'b-', 'LineWidth', 2);
ylabel('Phase [$^\circ$]', 'Interpreter', 'latex', 'FontSize', 11);
xlabel('Frequency [rad/s]', 'Interpreter', 'latex', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 10, 'TickLabelInterpreter', 'latex');

print(fig4, fullfile(output_dir, 'bode_openloop_plant'), '-dpng', '-r150');
fprintf('  Saved: bode_openloop_plant.png\n');

% ---- Figure 5: Closed-Loop Bode (PID and LQR) ----
fig5 = figure('Name', 'Closed-Loop Bode', ...
    'Position', [250 200 900 600], 'Color', 'w');

% PID closed-loop frequency response
[mag_pid_cl, phase_pid_cl, w_pid_cl] = bode(sys_pid_cl);
mag_pid_cl   = squeeze(mag_pid_cl);
phase_pid_cl = squeeze(phase_pid_cl);

% LQR closed-loop frequency response
% Build the transfer function from reference to output for LQR
% u = K * (x_ref - x), so the closed-loop TF with reference scaling:
sys_lqr_tf = ss(A_cl_lqr, B_plant * K_lqr(1), C_plant, 0);
[mag_lqr_cl, phase_lqr_cl, w_lqr_cl] = bode(sys_lqr_tf);
mag_lqr_cl   = squeeze(mag_lqr_cl);
phase_lqr_cl = squeeze(phase_lqr_cl);

subplot(2, 1, 1);
semilogx(w_pid_cl, 20*log10(mag_pid_cl), '-', ...
    'Color', c_pid, 'LineWidth', 2);
hold on;
semilogx(w_lqr_cl, 20*log10(mag_lqr_cl), '-', ...
    'Color', c_lqr, 'LineWidth', 2);
yline(-3, 'k--', 'LineWidth', 0.8);
hold off;
ylabel('Magnitude [dB]', 'Interpreter', 'latex', 'FontSize', 11);
title('Closed-Loop Bode Diagram: PID vs LQR', ...
    'Interpreter', 'latex', 'FontSize', 14);
legend({'PID', 'LQR', '$-3$ dB line'}, ...
    'Interpreter', 'latex', 'Location', 'best', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10, 'TickLabelInterpreter', 'latex');

subplot(2, 1, 2);
semilogx(w_pid_cl, phase_pid_cl, '-', 'Color', c_pid, 'LineWidth', 2);
hold on;
semilogx(w_lqr_cl, phase_lqr_cl, '-', 'Color', c_lqr, 'LineWidth', 2);
hold off;
ylabel('Phase [$^\circ$]', 'Interpreter', 'latex', 'FontSize', 11);
xlabel('Frequency [rad/s]', 'Interpreter', 'latex', 'FontSize', 12);
legend({'PID', 'LQR'}, 'Interpreter', 'latex', ...
    'Location', 'best', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10, 'TickLabelInterpreter', 'latex');

print(fig5, fullfile(output_dir, 'bode_closedloop_comparison'), '-dpng', '-r150');
fprintf('  Saved: bode_closedloop_comparison.png\n');

%% ========================================================================
%  Section 7: Performance Metrics
%  ========================================================================

fprintf('\n[STEP 7] Computing performance metrics...\n\n');

% --- Helper function: compute step response metrics ---
%     Uses the noisy simulation data (more realistic).

% PID metrics (from noisy sim)
[ts_pid, os_pid, sse_pid] = compute_metrics(t_sim, y_noisy_pid, theta_ref);

% LQR metrics (from noisy sim)
[ts_lqr, os_lqr, sse_lqr] = compute_metrics(t_sim, y_noisy_lqr, theta_ref);

% Control effort: RMS torque
rms_pid = sqrt(mean(u_pid_history.^2));
rms_lqr = sqrt(mean(u_lqr_history.^2));

% Peak torque
peak_pid = max(abs(u_pid_history));
peak_lqr = max(abs(u_lqr_history));

% Print performance comparison table
fprintf('=============================================================\n');
fprintf('  PERFORMANCE METRICS COMPARISON\n');
fprintf('=============================================================\n\n');
fprintf('  %-30s  %12s  %12s\n', 'Metric', 'PID', 'LQR');
fprintf('  %s\n', repmat('-', 1, 56));
fprintf('  %-30s  %12.3f  %12.3f\n', 'Settling Time (2%) [s]', ts_pid, ts_lqr);
fprintf('  %-30s  %12.2f  %12.2f\n', 'Overshoot [%]', os_pid, os_lqr);
fprintf('  %-30s  %12.6f  %12.6f\n', 'Steady-State Error [rad]', sse_pid, sse_lqr);
fprintf('  %-30s  %12.6f  %12.6f\n', 'Steady-State Error [deg]', ...
    sse_pid*180/pi, sse_lqr*180/pi);
fprintf('  %-30s  %12.4f  %12.4f\n', 'RMS Control Torque [N*m]', rms_pid, rms_lqr);
fprintf('  %-30s  %12.4f  %12.4f\n', 'Peak Control Torque [N*m]', peak_pid, peak_lqr);
fprintf('  %s\n', repmat('-', 1, 56));

% Bandwidth comparison
bw_pid = bandwidth(sys_pid_cl);
fprintf('\n  PID closed-loop bandwidth : %.4f rad/s\n', bw_pid);

% LQR bandwidth (from transfer function)
try
    bw_lqr = bandwidth(sys_lqr_tf);
    fprintf('  LQR closed-loop bandwidth : %.4f rad/s\n', bw_lqr);
catch
    fprintf('  LQR closed-loop bandwidth : (could not compute)\n');
end

fprintf('\n  Controller gains:\n');
fprintf('    PID: Kp = %.2f, Ki = %.2f, Kd = %.2f\n', Kp, Ki, Kd);
fprintf('    LQR: K  = [%.4f, %.4f]\n', K_lqr(1), K_lqr(2));
fprintf('    LQR: Q  = diag([%.0f, %.0f]), R = %.0f\n', ...
    Q_lqr(1,1), Q_lqr(2,2), R_lqr);
fprintf('    Plant inertia J = %.0f kg*m^2\n', J);

fprintf('\n=============================================================\n');
fprintf('  Attitude control analysis complete.\n');
fprintf('  All plots saved to: %s\n', output_dir);
fprintf('=============================================================\n');

%% ========================================================================
%  Local Functions
%  ========================================================================

function [t_settle, overshoot_pct, ss_error] = compute_metrics(t, y, y_ref)
%COMPUTE_METRICS  Compute step response performance metrics.
%
%  Inputs:
%    t     - time vector [s]
%    y     - output signal (angle in rad)
%    y_ref - reference (steady-state target) value [rad]
%
%  Outputs:
%    t_settle      - 2% settling time [s]
%    overshoot_pct - percent overshoot relative to reference [%]
%    ss_error      - absolute steady-state error [rad]

    % Steady-state error: use last 10% of data
    n_ss   = round(0.1 * length(y));
    ss_val = mean(y(end-n_ss+1:end));
    ss_error = abs(y_ref - ss_val);

    % Percent overshoot: peak value relative to reference
    y_peak = max(y);
    if y_ref > 0
        overshoot_pct = max(0, (y_peak - y_ref) / y_ref * 100);
    else
        overshoot_pct = 0;
    end

    % Settling time (2%): last time the response exits the +/- 2% band
    tol = 0.02 * abs(y_ref);
    settled = abs(y - y_ref) <= tol;

    % Find the last index where the signal was outside the band
    outside = find(~settled);
    if isempty(outside)
        t_settle = 0;   % Already within tolerance from the start
    else
        last_outside = outside(end);
        if last_outside < length(t)
            t_settle = t(last_outside + 1);
        else
            t_settle = t(end);  % Never fully settled
        end
    end
end
