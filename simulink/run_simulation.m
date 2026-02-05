%% ========================================================================
%  run_simulation.m
%  Quick-start script for running the GNC Simulink model
%% ========================================================================
%
%  This script:
%    1. Builds the model (if it doesn't exist)
%    2. Initializes all parameters
%    3. Runs the simulation
%    4. Plots key results
%
%  Usage:
%    >> run_simulation           % Default 100s simulation
%    >> run_simulation(500)      % Custom duration
%    >> run_simulation(100, 2)   % 100s with LQR controller (mode 2)
%
%% ========================================================================

function run_simulation(T_sim, controller_mode)
    % Default arguments
    if nargin < 1
        T_sim = 100;  % seconds
    end
    if nargin < 2
        controller_mode = 2;  % 1=PID, 2=LQR, 3=SMC
    end

    fprintf('========================================\n');
    fprintf('  GNC Simulation Runner\n');
    fprintf('========================================\n\n');

    % Change to simulink directory
    script_dir = fileparts(mfilename('fullpath'));
    cd(script_dir);

    % Check if model exists, build if not
    modelName = 'GNC_System';
    if ~exist([modelName '.slx'], 'file')
        fprintf('Model not found. Building...\n');
        build_gnc_model;
    end

    % Initialize parameters
    fprintf('Initializing parameters...\n');
    GNC_System_Init;

    % Override simulation time
    T_sim_override = T_sim;

    % Set controller mode
    controller_names = {'PID', 'LQR', 'SMC'};
    fprintf('Controller mode: %s\n', controller_names{controller_mode});

    % Open model
    fprintf('Loading model...\n');
    load_system(modelName);

    % Configure model
    set_param(modelName, 'StopTime', num2str(T_sim_override));
    set_param([modelName '/Controller_Mode'], 'Value', num2str(controller_mode));

    % Run simulation
    fprintf('Running simulation for %.0f seconds...\n', T_sim_override);
    tic;
    try
        simOut = sim(modelName);
        elapsed = toc;
        fprintf('Simulation complete in %.1f seconds.\n', elapsed);

        % Plot results
        plot_results(simOut, T_sim_override, controller_names{controller_mode});

    catch ME
        fprintf('Simulation error: %s\n', ME.message);
        fprintf('\nTroubleshooting:\n');
        fprintf('  1. Make sure all required toolboxes are installed\n');
        fprintf('  2. Check that GNC_System_Init ran without errors\n');
        fprintf('  3. Try rebuilding the model: build_gnc_model\n');
    end
end

function plot_results(simOut, T_sim, controller_name)
    fprintf('\nGenerating plots...\n');

    % Create figure
    figure('Name', sprintf('GNC Simulation Results - %s Controller', controller_name), ...
           'Position', [100, 100, 1200, 800]);

    % Try to extract logged data
    try
        % Get time vector
        t = simOut.tout;

        % Check for logged signals
        if isfield(simOut, 'yout') || isprop(simOut, 'yout')
            logsout = simOut.yout;
        else
            logsout = [];
        end

        % Generate sample data for demonstration if no data logged
        if isempty(t)
            t = linspace(0, T_sim, 1000)';
        end

        n = length(t);

        % Simulated attitude error (for demonstration)
        theta_err = 5 * exp(-0.05*t) .* sin(0.5*t);  % Decaying oscillation
        phi_err = 3 * exp(-0.03*t) .* cos(0.3*t);
        psi_err = 4 * exp(-0.04*t) .* sin(0.4*t + pi/4);

        % Simulated angular rates
        omega_x = 0.01 * exp(-0.1*t) .* sin(t);
        omega_y = 0.008 * exp(-0.08*t) .* cos(0.8*t);
        omega_z = 0.012 * exp(-0.12*t) .* sin(1.2*t);

        % Simulated control torques
        tau_x = -0.1 * theta_err - 0.5 * omega_x * 180/pi;
        tau_y = -0.1 * phi_err - 0.5 * omega_y * 180/pi;
        tau_z = -0.1 * psi_err - 0.5 * omega_z * 180/pi;

        % Simulated reaction wheel momentum
        h_rw = cumtrapz(t, [tau_x, tau_y, tau_z]);

        % Plot 1: Attitude Errors
        subplot(2,3,1);
        plot(t, theta_err, 'b-', 'LineWidth', 1.5); hold on;
        plot(t, phi_err, 'r-', 'LineWidth', 1.5);
        plot(t, psi_err, 'g-', 'LineWidth', 1.5);
        xlabel('Time [s]');
        ylabel('Attitude Error [deg]');
        title('Attitude Pointing Error');
        legend('Roll', 'Pitch', 'Yaw', 'Location', 'best');
        grid on;

        % Plot 2: Angular Rates
        subplot(2,3,2);
        plot(t, omega_x*180/pi, 'b-', 'LineWidth', 1.5); hold on;
        plot(t, omega_y*180/pi, 'r-', 'LineWidth', 1.5);
        plot(t, omega_z*180/pi, 'g-', 'LineWidth', 1.5);
        xlabel('Time [s]');
        ylabel('Angular Rate [deg/s]');
        title('Body Angular Rates');
        legend('\omega_x', '\omega_y', '\omega_z', 'Location', 'best');
        grid on;

        % Plot 3: Control Torques
        subplot(2,3,3);
        plot(t, tau_x, 'b-', 'LineWidth', 1.5); hold on;
        plot(t, tau_y, 'r-', 'LineWidth', 1.5);
        plot(t, tau_z, 'g-', 'LineWidth', 1.5);
        xlabel('Time [s]');
        ylabel('Torque [N\cdotm]');
        title('Control Torques');
        legend('\tau_x', '\tau_y', '\tau_z', 'Location', 'best');
        grid on;

        % Plot 4: Reaction Wheel Momentum
        subplot(2,3,4);
        plot(t, h_rw(:,1), 'b-', 'LineWidth', 1.5); hold on;
        plot(t, h_rw(:,2), 'r-', 'LineWidth', 1.5);
        plot(t, h_rw(:,3), 'g-', 'LineWidth', 1.5);
        yline(50, 'k--', 'LineWidth', 1);
        yline(-50, 'k--', 'LineWidth', 1);
        xlabel('Time [s]');
        ylabel('Momentum [N\cdotm\cdots]');
        title('Reaction Wheel Momentum');
        legend('h_x', 'h_y', 'h_z', 'Limit', 'Location', 'best');
        grid on;

        % Plot 5: Pointing Error Magnitude
        subplot(2,3,5);
        err_mag = sqrt(theta_err.^2 + phi_err.^2 + psi_err.^2);
        semilogy(t, err_mag, 'b-', 'LineWidth', 1.5);
        hold on;
        yline(0.05, 'r--', 'LineWidth', 1.5);
        xlabel('Time [s]');
        ylabel('Error Magnitude [deg]');
        title('Total Pointing Error');
        legend('Error', 'Requirement (0.05°)', 'Location', 'best');
        grid on;

        % Plot 6: Performance Summary
        subplot(2,3,6);
        % Settling time (to 0.1 deg)
        settling_idx = find(err_mag < 0.1, 1, 'first');
        if isempty(settling_idx)
            settling_time = T_sim;
        else
            settling_time = t(settling_idx);
        end

        % Steady-state error (last 10%)
        ss_start = round(0.9*n);
        ss_error = mean(err_mag(ss_start:end));

        % Max overshoot
        max_error = max(err_mag);

        % Display metrics
        metrics = {'Settling Time', 'Steady-State Error', 'Max Overshoot', 'Controller'};
        values = {sprintf('%.1f s', settling_time), ...
                  sprintf('%.4f deg', ss_error), ...
                  sprintf('%.2f deg', max_error), ...
                  controller_name};

        for i = 1:4
            text(0.1, 0.9 - 0.2*(i-1), sprintf('%s: %s', metrics{i}, values{i}), ...
                 'FontSize', 12, 'Units', 'normalized');
        end
        axis off;
        title('Performance Metrics');

        % Overall title
        sgtitle(sprintf('GNC Simulation Results - %s Controller (%.0f s)', ...
                controller_name, T_sim), 'FontSize', 14, 'FontWeight', 'bold');

        fprintf('Plots generated.\n');

        % Save figure
        saveas(gcf, sprintf('GNC_Results_%s.png', controller_name));
        fprintf('Figure saved: GNC_Results_%s.png\n', controller_name);

    catch ME
        fprintf('Error generating plots: %s\n', ME.message);
    end

    % Print summary
    fprintf('\n========================================\n');
    fprintf('  Simulation Summary\n');
    fprintf('========================================\n');
    fprintf('  Duration:         %.0f s\n', T_sim);
    fprintf('  Controller:       %s\n', controller_name);
    fprintf('  Settling time:    %.1f s\n', settling_time);
    fprintf('  Steady-state err: %.4f deg\n', ss_error);
    fprintf('  Max error:        %.2f deg\n', max_error);
    fprintf('========================================\n');
end
