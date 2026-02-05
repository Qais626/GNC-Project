%% generate_report_figures.m - Generate figures for the GNC report
% Creates model screenshot and detailed simulation plots

fprintf('=== Generating Report Figures ===\n\n');

% Initialize parameters
GNC_System_Init;

% Build model if needed
if ~exist('GNC_System.slx', 'file')
    fprintf('Building model...\n');
    build_gnc_model;
end

% Load model
fprintf('Loading model...\n');
load_system('GNC_System');

%% Figure 1: Simulink Model Screenshot
fprintf('Capturing model screenshot...\n');
open_system('GNC_System');
set_param('GNC_System', 'ZoomFactor', '100');

% Print model to PNG
print('-sGNC_System', '-dpng', '-r150', 'Simulink_Model_Block_Diagram.png');
fprintf('Saved: Simulink_Model_Block_Diagram.png\n');

%% Run Simulation
fprintf('\nRunning simulation...\n');
simOut = sim('GNC_System', 'StopTime', '100');

% Extract data
t = simOut.tout;
theta = simOut.theta_log.Data;
omega = simOut.omega_log.Data;
torque = simOut.torque_log.Data;

% Convert to degrees
theta_deg = theta * 180/pi;
omega_deg = omega * 180/pi;

%% Figure 2: Attitude Response (detailed)
fprintf('Generating attitude response plot...\n');
fig2 = figure('Position', [100 100 900 600], 'Color', 'w');

subplot(2,1,1);
plot(t, theta_deg, 'b-', 'LineWidth', 1.5);
hold on;
yline(0, 'k--', 'LineWidth', 1);
yline(0.05, 'r--', 'Requirement', 'LineWidth', 1);
yline(-0.05, 'r--', 'LineWidth', 1);
xlabel('Time [s]', 'FontSize', 11);
ylabel('Attitude Angle [deg]', 'FontSize', 11);
title('Spacecraft Attitude Response', 'FontSize', 12, 'FontWeight', 'bold');
legend('Attitude', 'Reference', '\pm0.05 deg Req.', 'Location', 'best');
grid on;
xlim([0 100]);

subplot(2,1,2);
plot(t, omega_deg, 'r-', 'LineWidth', 1.5);
hold on;
yline(0, 'k--', 'LineWidth', 1);
xlabel('Time [s]', 'FontSize', 11);
ylabel('Angular Rate [deg/s]', 'FontSize', 11);
title('Angular Rate Response', 'FontSize', 12, 'FontWeight', 'bold');
legend('Angular Rate', 'Target', 'Location', 'best');
grid on;
xlim([0 100]);

sgtitle('Simulink Attitude Control Simulation Results', 'FontSize', 14, 'FontWeight', 'bold');
saveas(fig2, 'Simulink_Attitude_Response.png');
fprintf('Saved: Simulink_Attitude_Response.png\n');

%% Figure 3: Control Torque and Actuator Behavior
fprintf('Generating control torque plot...\n');
fig3 = figure('Position', [100 100 900 600], 'Color', 'w');

subplot(2,1,1);
plot(t, torque, 'g-', 'LineWidth', 1.5);
hold on;
yline(tau_max, 'r--', 'Saturation Limit', 'LineWidth', 1.5);
yline(-tau_max, 'r--', 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 11);
ylabel('Control Torque [N{\cdot}m]', 'FontSize', 11);
title('Reaction Wheel Torque Command', 'FontSize', 12, 'FontWeight', 'bold');
legend('Commanded Torque', 'Saturation Limits', 'Location', 'best');
grid on;
xlim([0 100]);

subplot(2,1,2);
% Compute cumulative torque effort
torque_effort = cumtrapz(t, abs(torque));
plot(t, torque_effort, 'm-', 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 11);
ylabel('Cumulative Effort [N{\cdot}m{\cdot}s]', 'FontSize', 11);
title('Cumulative Control Effort', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
xlim([0 100]);

sgtitle('Actuator Performance Analysis', 'FontSize', 14, 'FontWeight', 'bold');
saveas(fig3, 'Simulink_Control_Torque.png');
fprintf('Saved: Simulink_Control_Torque.png\n');

%% Figure 4: Phase Portrait and Performance Metrics
fprintf('Generating phase portrait...\n');
fig4 = figure('Position', [100 100 900 500], 'Color', 'w');

subplot(1,2,1);
plot(theta_deg, omega_deg, 'b-', 'LineWidth', 1.2);
hold on;
plot(theta_deg(1), omega_deg(1), 'go', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Start');
plot(theta_deg(end), omega_deg(end), 'rs', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'End');
xlabel('Attitude [deg]', 'FontSize', 11);
ylabel('Angular Rate [deg/s]', 'FontSize', 11);
title('Phase Portrait', 'FontSize', 12, 'FontWeight', 'bold');
legend('Trajectory', 'Start', 'End', 'Location', 'best');
grid on;
axis equal;

subplot(1,2,2);
% Error convergence on log scale
error_abs = abs(theta_deg);
error_abs(error_abs < 1e-6) = 1e-6; % Avoid log(0)
semilogy(t, error_abs, 'b-', 'LineWidth', 1.5);
hold on;
yline(0.05, 'r--', '0.05 deg Requirement', 'LineWidth', 1.5);
xlabel('Time [s]', 'FontSize', 11);
ylabel('Pointing Error [deg]', 'FontSize', 11);
title('Error Convergence (Log Scale)', 'FontSize', 12, 'FontWeight', 'bold');
legend('|Error|', 'Requirement', 'Location', 'best');
grid on;
xlim([0 100]);
ylim([1e-3 10]);

sgtitle('Control System Performance Analysis', 'FontSize', 14, 'FontWeight', 'bold');
saveas(fig4, 'Simulink_Phase_Portrait.png');
fprintf('Saved: Simulink_Phase_Portrait.png\n');

%% Compute and display metrics
fprintf('\n========================================\n');
fprintf('  Performance Metrics\n');
fprintf('========================================\n');
fprintf('  Initial error:     %.2f deg\n', theta_deg(1));
fprintf('  Final error:       %.6f deg\n', abs(theta_deg(end)));
fprintf('  Max overshoot:     %.2f deg\n', max(abs(theta_deg)) - abs(theta_deg(1)));
fprintf('  Max torque:        %.4f N*m\n', max(abs(torque)));
fprintf('  Total effort:      %.2f N*m*s\n', torque_effort(end));

% Settling time to 0.1 deg
settle_idx = find(abs(theta_deg) < 0.1, 1);
if ~isempty(settle_idx)
    fprintf('  Settling (0.1 deg): %.1f s\n', t(settle_idx));
end

% Settling time to 0.05 deg
settle_idx2 = find(abs(theta_deg) < 0.05, 1);
if ~isempty(settle_idx2)
    fprintf('  Settling (0.05 deg): %.1f s\n', t(settle_idx2));
end

fprintf('========================================\n');
fprintf('\nAll figures generated successfully.\n');

% Close model (catch any errors)
try
    close_system('GNC_System', 0);
catch
    % Model may already be closed
end
