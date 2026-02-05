%% run_and_plot.m - Run GNC simulation and generate plots

fprintf('=== GNC Simulation Runner ===\n\n');

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

% Run simulation
fprintf('Running simulation...\n');
tic;
simOut = sim('GNC_System', 'StopTime', '100');
elapsed = toc;
fprintf('Complete in %.1f seconds.\n\n', elapsed);

% Extract data
t = simOut.tout;
theta = simOut.theta_log.Data;
omega = simOut.omega_log.Data;
torque = simOut.torque_log.Data;

% Convert to degrees
theta_deg = theta * 180/pi;
omega_deg = omega * 180/pi;

% Save
save('simulation_results.mat', 't', 'theta', 'omega', 'torque');
fprintf('Saved: simulation_results.mat\n');

% Plot
fprintf('Generating plots...\n');
figure('Position', [100 100 1000 700]);

subplot(2,2,1);
plot(t, theta_deg, 'b-', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Attitude [deg]');
title('Attitude Angle');
grid on;

subplot(2,2,2);
plot(t, omega_deg, 'r-', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Rate [deg/s]');
title('Angular Rate');
grid on;

subplot(2,2,3);
plot(t, torque, 'g-', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Torque [N*m]');
title('Control Torque');
grid on;

subplot(2,2,4);
plot(t, abs(theta_deg), 'b-', 'LineWidth', 1.5);
hold on;
yline(0.05, 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Error [deg]');
title('Pointing Error');
legend('Error', 'Requirement', 'Location', 'best');
grid on;

sgtitle('GNC Attitude Control Simulation', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, 'GNC_Simulation_Results.png');
fprintf('Saved: GNC_Simulation_Results.png\n');

% Summary
fprintf('\n========================================\n');
fprintf('  Simulation Summary\n');
fprintf('========================================\n');
fprintf('  Initial error:  %.2f deg\n', theta_deg(1));
fprintf('  Final error:    %.4f deg\n', abs(theta_deg(end)));
fprintf('  Max torque:     %.4f N*m\n', max(abs(torque)));

settle_idx = find(abs(theta_deg) < 0.1, 1);
if ~isempty(settle_idx)
    fprintf('  Settling time:  %.1f s\n', t(settle_idx));
end
fprintf('========================================\n');
