%% =========================================================================
%  Publication Quality Plot Generation from Simulation Data
%  Reads CSV output from Python simulation and generates MATLAB-quality plots
%  =========================================================================
clear; clc; close all;

%% ------------------------------------------------------------------------
%  Configuration
%  ------------------------------------------------------------------------

% Output directory for saved figures
output_dir = fullfile(fileparts(mfilename('fullpath')), ...
    '..', '..', '..', 'output', 'matlab');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Input data directory (where Python simulation writes CSV files)
data_dir = fullfile(fileparts(mfilename('fullpath')), ...
    '..', '..', '..', 'output', 'data');

% Physical constants
mu_earth = 3.986e14;   % Earth gravitational parameter [m^3/s^2]
R_earth  = 6371e3;     % Earth mean radius [m]
r_orbit  = 6571e3;     % Orbital radius: 200 km altitude [m]

% Global plot defaults for publication quality
set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultTextFontSize', 11);
set(0, 'DefaultLineLineWidth', 1.5);
set(0, 'DefaultAxesTickLabelInterpreter', 'latex');
set(0, 'DefaultTextInterpreter', 'latex');
set(0, 'DefaultLegendInterpreter', 'latex');

fprintf('=============================================================\n');
fprintf('  Publication Quality Plot Generation\n');
fprintf('=============================================================\n\n');

%% ------------------------------------------------------------------------
%  Load Telemetry Data (or generate sample if CSV not found)
%  ------------------------------------------------------------------------

fprintf('[STEP 1] Loading telemetry data...\n');

try
    % Attempt to read CSV data produced by the Python simulation
    T = readtable(fullfile(data_dir, 'telemetry.csv'));
    fprintf('  Loaded telemetry CSV: %d rows, %d columns.\n', ...
        height(T), width(T));

    % Extract columns by expected names
    t     = T.time;
    pos_x = T.pos_x;
    pos_y = T.pos_y;
    pos_z = T.pos_z;
    vel_x = T.vel_x;
    vel_y = T.vel_y;
    vel_z = T.vel_z;
    q1    = T.q1;
    q2    = T.q2;
    q3    = T.q3;
    q4    = T.q4;
    omega_x = T.omega_x;
    omega_y = T.omega_y;
    omega_z = T.omega_z;
    tau_x   = T.tau_x;
    tau_y   = T.tau_y;
    tau_z   = T.tau_z;
    mass    = T.mass;

    data_source = 'CSV';

catch
    % ------------------------------------------------------------------
    %  Generate representative sample data for demonstration
    % ------------------------------------------------------------------
    fprintf('  CSV not found. Generating sample telemetry data...\n');
    data_source = 'GENERATED';

    N  = 10000;
    t  = linspace(0, 86400, N)';   % 1 day of data [s]
    dt = t(2) - t(1);

    rng(42);  % Reproducible random seed

    % --- Orbital position: circular orbit at 200 km altitude ---
    omega_orb = sqrt(mu_earth / r_orbit^3);   % Orbital angular rate [rad/s]
    pos_x = r_orbit * cos(omega_orb * t);
    pos_y = r_orbit * sin(omega_orb * t);
    pos_z = 30e3 * sin(2 * omega_orb * t);    % Small out-of-plane oscillation

    % --- Orbital velocity ---
    vel_x = -r_orbit * omega_orb * sin(omega_orb * t);
    vel_y =  r_orbit * omega_orb * cos(omega_orb * t);
    vel_z =  30e3 * 2 * omega_orb * cos(2 * omega_orb * t);

    % --- Attitude quaternion: smooth slew then fine pointing ---
    slew_duration = 600;                   % 10 min slew maneuver [s]
    target_angle  = 30 * pi / 180;        % 30 deg slew about z-axis

    q1 = zeros(N, 1);
    q2 = zeros(N, 1);
    q3 = zeros(N, 1);
    q4 = ones(N, 1);

    for i = 1:N
        if t(i) < slew_duration
            % Sigmoid slew profile
            angle_i = target_angle / (1 + exp(-0.02 * (t(i) - slew_duration/2)));
        else
            angle_i = target_angle;
        end
        q3(i) = sin(angle_i / 2);
        q4(i) = cos(angle_i / 2);
    end

    % Add small noise and renormalize
    qn = 1e-4;
    q1 = q1 + qn * randn(N, 1);
    q2 = q2 + qn * randn(N, 1);
    q3 = q3 + qn * randn(N, 1);
    q4 = q4 + qn * randn(N, 1);
    q_norm = sqrt(q1.^2 + q2.^2 + q3.^2 + q4.^2);
    q1 = q1 ./ q_norm;
    q2 = q2 ./ q_norm;
    q3 = q3 ./ q_norm;
    q4 = q4 ./ q_norm;

    % --- Angular velocity [rad/s] ---
    omega_x = 0.0005 * sin(0.005 * t) + 3e-4 * randn(N, 1);
    omega_y = 0.0005 * cos(0.005 * t) + 3e-4 * randn(N, 1);
    omega_z = zeros(N, 1);
    for i = 1:N
        if t(i) < slew_duration
            frac = t(i) / slew_duration;
            omega_z(i) = (target_angle / slew_duration) * ...
                4 * frac * (1 - frac);  % Parabolic rate profile
        end
    end
    omega_z = omega_z + 1e-4 * randn(N, 1);

    % --- Control torques [N*m] ---
    tau_x = -0.3 * omega_x + 0.005 * randn(N, 1);
    tau_y = -0.3 * omega_y + 0.005 * randn(N, 1);
    tau_z = zeros(N, 1);
    for i = 1:N
        if t(i) < slew_duration
            tau_z(i) = 1.5 * ((target_angle/slew_duration) - omega_z(i)) ...
                       + 0.01 * randn;
        else
            tau_z(i) = -0.3 * omega_z(i) + 0.005 * randn;
        end
    end

    % --- Spacecraft mass [kg] ---
    % Small linear decrease representing station-keeping propellant usage
    mass = 6000 - 0.005 * t;

    fprintf('  Generated %d time steps over %.0f s (%.1f hours).\n\n', ...
        N, t(end), t(end)/3600);
end

%% ========================================================================
%  Figure 1: 3D Trajectory
%  ========================================================================

fprintf('[PLOT 1] 3D Trajectory...\n');

fig1 = figure('Name', '3D Trajectory', ...
    'Position', [100 100 800 600], 'Color', 'w');

% Plot spacecraft orbit
plot3(pos_x/1e3, pos_y/1e3, pos_z/1e3, 'b', 'LineWidth', 1.5);
hold on;

% Draw Earth sphere
[xs, ys, zs] = sphere(20);
surf(xs * R_earth/1e3, ys * R_earth/1e3, zs * R_earth/1e3, ...
    'FaceColor', [0.2 0.4 0.8], 'EdgeColor', 'none', ...
    'FaceAlpha', 0.7, 'FaceLighting', 'gouraud');
light('Position', [1 0.5 0.5], 'Style', 'infinite');
material dull;

% Mark start and end points
plot3(pos_x(1)/1e3, pos_y(1)/1e3, pos_z(1)/1e3, ...
    'g^', 'MarkerSize', 12, 'MarkerFaceColor', [0.3 0.8 0.3]);
plot3(pos_x(end)/1e3, pos_y(end)/1e3, pos_z(end)/1e3, ...
    'rs', 'MarkerSize', 12, 'MarkerFaceColor', [0.9 0.2 0.2]);

xlabel('X [km]', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Y [km]', 'Interpreter', 'latex', 'FontSize', 12);
zlabel('Z [km]', 'Interpreter', 'latex', 'FontSize', 12);
title('Mission Trajectory', 'Interpreter', 'latex', 'FontSize', 14);
legend({'Orbit', 'Earth', 'Start', 'End'}, ...
    'Interpreter', 'latex', 'Location', 'best');
grid on; axis equal; view(30, 25);
hold off;

print(fig1, fullfile(output_dir, 'trajectory_3d'), '-dpng', '-r150');
fprintf('  Saved: trajectory_3d.png\n');

%% ========================================================================
%  Figure 2: Attitude Quaternion History (4 subplots)
%  ========================================================================

fprintf('[PLOT 2] Attitude quaternion history...\n');

fig2 = figure('Name', 'Quaternion History', ...
    'Position', [120 120 1000 700], 'Color', 'w');

q_data   = [q1, q2, q3, q4];
q_labels = {'$q_1$', '$q_2$', '$q_3$', '$q_4$ (scalar)'};
q_colors = {[0.0 0.45 0.74], [0.85 0.33 0.10], ...
            [0.47 0.67 0.19], [0.49 0.18 0.56]};

for k = 1:4
    subplot(4, 1, k);
    plot(t/3600, q_data(:, k), '-', 'Color', q_colors{k}, 'LineWidth', 1.2);
    ylabel(q_labels{k}, 'Interpreter', 'latex', 'FontSize', 11);
    grid on;
    if k == 1
        title('Attitude Quaternion History', ...
            'Interpreter', 'latex', 'FontSize', 14);
    end
    if k < 4
        set(gca, 'XTickLabel', []);
    end
    xlim([t(1) t(end)] / 3600);
    set(gca, 'TickLabelInterpreter', 'latex');
end
xlabel('Time [hours]', 'Interpreter', 'latex', 'FontSize', 12);

% Link x-axes for synchronized zooming
ax = findall(fig2, 'Type', 'axes');
linkaxes(ax, 'x');

print(fig2, fullfile(output_dir, 'quaternion_history'), '-dpng', '-r150');
fprintf('  Saved: quaternion_history.png\n');

%% ========================================================================
%  Figure 3: Angular Velocity (3 subplots)
%  ========================================================================

fprintf('[PLOT 3] Angular velocity history...\n');

fig3 = figure('Name', 'Angular Velocity', ...
    'Position', [140 140 1000 600], 'Color', 'w');

w_data   = [omega_x, omega_y, omega_z];
w_labels = {'$\omega_x$ [mrad/s]', '$\omega_y$ [mrad/s]', '$\omega_z$ [mrad/s]'};
w_colors = {[0.0 0.45 0.74], [0.85 0.33 0.10], [0.47 0.67 0.19]};

for k = 1:3
    subplot(3, 1, k);
    plot(t/3600, w_data(:, k)*1e3, '-', 'Color', w_colors{k}, 'LineWidth', 1.2);
    ylabel(w_labels{k}, 'Interpreter', 'latex', 'FontSize', 11);
    grid on;
    if k == 1
        title('Body Angular Velocity History', ...
            'Interpreter', 'latex', 'FontSize', 14);
    end
    if k < 3
        set(gca, 'XTickLabel', []);
    end
    xlim([t(1) t(end)] / 3600);
    set(gca, 'TickLabelInterpreter', 'latex');
end
xlabel('Time [hours]', 'Interpreter', 'latex', 'FontSize', 12);

ax3 = findall(fig3, 'Type', 'axes');
linkaxes(ax3, 'x');

print(fig3, fullfile(output_dir, 'angular_velocity'), '-dpng', '-r150');
fprintf('  Saved: angular_velocity.png\n');

%% ========================================================================
%  Figure 4: Pointing Error with Requirement Line
%  ========================================================================

fprintf('[PLOT 4] Pointing error...\n');

fig4 = figure('Name', 'Pointing Error', ...
    'Position', [160 160 900 500], 'Color', 'w');

% Compute pointing error from quaternion (angle from reference)
% Reference quaternion: q_ref = [0, 0, sin(target/2), cos(target/2)]
% Error angle = 2 * acos(|q_ref . q_actual|) for small errors
% Simplified: use deviation of q4 from its steady-state value
q4_ss = q4(end);  % Steady-state scalar component
q_err_angle = 2 * acos(min(abs(q4), 1)) * 180 / pi;  % Total rotation [deg]

% For post-slew: compute error relative to final orientation
pointing_error = zeros(N, 1);
for i = 1:N
    % Quaternion error = q_ref^(-1) * q_actual
    % For rotation about z, the error angle is approximately:
    dot_prod = abs(q1(i)*q1(end) + q2(i)*q2(end) + ...
                   q3(i)*q3(end) + q4(i)*q4(end));
    dot_prod = min(dot_prod, 1.0);
    pointing_error(i) = 2 * acos(dot_prod) * 180 / pi;
end

% Pointing accuracy requirement
req_deg = 0.1;  % 0.1 degree requirement

semilogy(t/3600, pointing_error, '-', ...
    'Color', [0.0 0.45 0.74], 'LineWidth', 1.0);
hold on;
yline(req_deg, 'r--', 'LineWidth', 2.0);
text(t(end)/3600 * 0.6, req_deg * 1.5, ...
    sprintf('Requirement: %.1f$^\\circ$', req_deg), ...
    'Color', 'r', 'FontSize', 11, 'Interpreter', 'latex');

% Shade compliant region
fill([t(1)/3600, t(end)/3600, t(end)/3600, t(1)/3600], ...
     [1e-5, 1e-5, req_deg, req_deg], ...
     [0.47 0.67 0.19], 'FaceAlpha', 0.1, 'EdgeColor', 'none');
hold off;

xlabel('Time [hours]', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Pointing Error [$^\circ$]', 'Interpreter', 'latex', 'FontSize', 12);
title('Attitude Pointing Error vs Time', ...
    'Interpreter', 'latex', 'FontSize', 14);
legend({'Pointing Error', 'Requirement', 'Compliant Region'}, ...
    'Interpreter', 'latex', 'Location', 'northeast');
grid on;
xlim([t(1) t(end)] / 3600);
ylim([1e-4 max(pointing_error)*2]);
set(gca, 'TickLabelInterpreter', 'latex');

print(fig4, fullfile(output_dir, 'pointing_error'), '-dpng', '-r150');
fprintf('  Saved: pointing_error.png\n');

%% ========================================================================
%  Figure 5: Control Torques (3 subplots)
%  ========================================================================

fprintf('[PLOT 5] Control torque history...\n');

fig5 = figure('Name', 'Control Torques', ...
    'Position', [180 180 1000 600], 'Color', 'w');

tau_data   = [tau_x, tau_y, tau_z];
tau_labels = {'$\tau_x$ [N$\cdot$m]', '$\tau_y$ [N$\cdot$m]', ...
              '$\tau_z$ [N$\cdot$m]'};
tau_colors = {[0.0 0.45 0.74], [0.85 0.33 0.10], [0.47 0.67 0.19]};

for k = 1:3
    subplot(3, 1, k);
    plot(t/3600, tau_data(:, k), '-', 'Color', tau_colors{k}, 'LineWidth', 1.0);
    ylabel(tau_labels{k}, 'Interpreter', 'latex', 'FontSize', 11);
    grid on;
    if k == 1
        title('Control Torque Commands', ...
            'Interpreter', 'latex', 'FontSize', 14);
    end
    if k < 3
        set(gca, 'XTickLabel', []);
    end
    xlim([t(1) t(end)] / 3600);
    set(gca, 'TickLabelInterpreter', 'latex');
end
xlabel('Time [hours]', 'Interpreter', 'latex', 'FontSize', 12);

ax5 = findall(fig5, 'Type', 'axes');
linkaxes(ax5, 'x');

print(fig5, fullfile(output_dir, 'control_torques'), '-dpng', '-r150');
fprintf('  Saved: control_torques.png\n');

%% ========================================================================
%  Figure 6: Orbital Radius vs Time
%  ========================================================================

fprintf('[PLOT 6] Orbital radius vs time...\n');

fig6 = figure('Name', 'Orbital Radius', ...
    'Position', [200 200 900 450], 'Color', 'w');

% Compute orbital radius magnitude
r_mag = sqrt(pos_x.^2 + pos_y.^2 + pos_z.^2);

plot(t/3600, r_mag/1e3, '-', 'Color', [0.0 0.45 0.74], 'LineWidth', 1.5);
hold on;
yline(R_earth/1e3, 'k--', 'LineWidth', 1.0);
text(t(end)/3600 * 0.02, R_earth/1e3 - 20, 'Earth Surface', ...
    'Interpreter', 'latex', 'FontSize', 9, 'Color', [0.3 0.3 0.3]);

% Reference orbit altitude line
yline(r_orbit/1e3, 'r:', 'LineWidth', 1.0);
text(t(end)/3600 * 0.02, r_orbit/1e3 + 15, 'Reference Orbit', ...
    'Interpreter', 'latex', 'FontSize', 9, 'Color', 'r');
hold off;

xlabel('Time [hours]', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Orbital Radius [km]', 'Interpreter', 'latex', 'FontSize', 12);
title('Orbital Radius vs Time', 'Interpreter', 'latex', 'FontSize', 14);
grid on;
xlim([t(1) t(end)] / 3600);
set(gca, 'TickLabelInterpreter', 'latex');

print(fig6, fullfile(output_dir, 'orbital_radius'), '-dpng', '-r150');
fprintf('  Saved: orbital_radius.png\n');

%% ========================================================================
%  Figure 7: Spacecraft Mass vs Time
%  ========================================================================

fprintf('[PLOT 7] Spacecraft mass vs time...\n');

fig7 = figure('Name', 'Mass vs Time', ...
    'Position', [220 220 900 450], 'Color', 'w');

plot(t/3600, mass, '-', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.5);
hold on;

% Shade propellant consumed region
fill([t(1)/3600, t(end)/3600, t(end)/3600, t(1)/3600], ...
     [mass(end), mass(end), mass(1), mass(1)], ...
     [0.85 0.33 0.10], 'FaceAlpha', 0.1, 'EdgeColor', 'none');

% Mark initial and final mass
plot(t(1)/3600, mass(1), 'b^', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
plot(t(end)/3600, mass(end), 'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

text(t(end)/3600 * 0.5, mass(1) + 20, ...
    sprintf('$m_0 = %.1f$ kg', mass(1)), ...
    'Interpreter', 'latex', 'FontSize', 10, 'HorizontalAlignment', 'center');
text(t(end)/3600 * 0.5, mass(end) - 30, ...
    sprintf('$m_f = %.1f$ kg', mass(end)), ...
    'Interpreter', 'latex', 'FontSize', 10, 'HorizontalAlignment', 'center');
hold off;

xlabel('Time [hours]', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Spacecraft Mass [kg]', 'Interpreter', 'latex', 'FontSize', 12);
title('Spacecraft Mass vs Time', 'Interpreter', 'latex', 'FontSize', 14);
legend({'Mass', 'Propellant Consumed', '$m_0$', '$m_f$'}, ...
    'Interpreter', 'latex', 'Location', 'northeast');
grid on;
xlim([t(1) t(end)] / 3600);
set(gca, 'TickLabelInterpreter', 'latex');

print(fig7, fullfile(output_dir, 'mass_vs_time'), '-dpng', '-r150');
fprintf('  Saved: mass_vs_time.png\n');

%% ========================================================================
%  Figure 8: Velocity Magnitude vs Time
%  ========================================================================

fprintf('[PLOT 8] Velocity magnitude vs time...\n');

fig8 = figure('Name', 'Velocity Magnitude', ...
    'Position', [240 240 900 450], 'Color', 'w');

% Compute velocity magnitude
v_mag = sqrt(vel_x.^2 + vel_y.^2 + vel_z.^2);

plot(t/3600, v_mag/1e3, '-', 'Color', [0.49 0.18 0.56], 'LineWidth', 1.5);
hold on;

% Reference circular velocity
v_circ_ref = sqrt(mu_earth / r_orbit);
yline(v_circ_ref/1e3, 'r:', 'LineWidth', 1.0);
text(t(end)/3600 * 0.6, v_circ_ref/1e3 + 0.01, ...
    sprintf('$v_{\\rm circ} = %.3f$ km/s', v_circ_ref/1e3), ...
    'Interpreter', 'latex', 'FontSize', 10, 'Color', 'r');
hold off;

xlabel('Time [hours]', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Velocity [km/s]', 'Interpreter', 'latex', 'FontSize', 12);
title('Velocity Magnitude vs Time', 'Interpreter', 'latex', 'FontSize', 14);
legend({'$|v|$', 'Circular Reference'}, ...
    'Interpreter', 'latex', 'Location', 'best');
grid on;
xlim([t(1) t(end)] / 3600);
set(gca, 'TickLabelInterpreter', 'latex');

print(fig8, fullfile(output_dir, 'velocity_magnitude'), '-dpng', '-r150');
fprintf('  Saved: velocity_magnitude.png\n');

%% ========================================================================
%  Reset Global Defaults and Finish
%  ========================================================================

% Restore default interpreters to avoid side effects on other scripts
set(0, 'DefaultAxesTickLabelInterpreter', 'tex');
set(0, 'DefaultTextInterpreter', 'tex');
set(0, 'DefaultLegendInterpreter', 'tex');

fprintf('\n=============================================================\n');
fprintf('  Plot generation complete.\n');
fprintf('  Data source: %s\n', data_source);
fprintf('  All 8 figures saved to: %s\n', output_dir);
fprintf('=============================================================\n');
