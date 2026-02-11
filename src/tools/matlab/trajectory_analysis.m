%% =========================================================================
%  GNC Mission - Trajectory Analysis and Validation
%  Validates trajectory computations from Python with independent MATLAB calc
%  =========================================================================
clear; clc; close all;

%% ------------------------------------------------------------------------
%  Constants and Celestial Body Parameters
%  ------------------------------------------------------------------------

% Gravitational parameters [m^3/s^2]
mu_earth   = 3.986e14;     % Earth
mu_moon    = 4.905e12;     % Moon
mu_jupiter = 1.267e17;     % Jupiter
mu_sun     = 1.327e20;     % Sun

% Mean radii [m]
R_earth   = 6371e3;        % Earth
R_moon    = 1737.4e3;      % Moon
R_jupiter = 69911e3;       % Jupiter

% Orbital semi-major axes [m]
moon_sma      = 384400e3;  % Moon around Earth
jupiter_sma   = 778.57e9;  % Jupiter around Sun

% Heliocentric orbital radii [m]
r_earth_sun   = 1.496e11;  % Earth around Sun (1 AU)
r_jupiter_sun = 778.57e9;  % Jupiter around Sun (5.2 AU)

% Spacecraft and propulsion parameters
g0  = 9.80665;             % Standard gravitational acceleration [m/s^2]
Isp = 316;                 % Specific impulse (bi-propellant engine) [s]
m0  = 6000;                % Initial wet mass [kg]

fprintf('=============================================================\n');
fprintf('  GNC MISSION - TRAJECTORY ANALYSIS AND VALIDATION\n');
fprintf('=============================================================\n\n');

%% ========================================================================
%  Hohmann Transfer: LEO to Moon
%  ========================================================================
fprintf('--- Hohmann Transfer: LEO to Moon ---\n\n');

% Departure orbit: Low Earth Orbit at 200 km altitude
r1 = R_earth + 200e3;                  % LEO orbital radius [m]

% Arrival: Moon's orbital radius from Earth
r2 = moon_sma;                         % Moon orbit radius [m]

% Transfer ellipse semi-major axis
a_transfer = (r1 + r2) / 2;

% Circular velocity at LEO (vis-viva with a = r for circular orbit)
v1_circular = sqrt(mu_earth / r1);

% Velocity at perigee of the transfer ellipse (vis-viva equation)
v1_transfer = sqrt(mu_earth * (2/r1 - 1/a_transfer));

% Trans-Lunar Injection (TLI) delta-V
dv_tli = v1_transfer - v1_circular;

% Velocity at apogee of the transfer ellipse (at Moon distance)
v2_transfer = sqrt(mu_earth * (2/r2 - 1/a_transfer));

% Circular velocity at Moon distance from Earth (for reference)
v2_circular_earth = sqrt(mu_earth / r2);

% Circular velocity in a 100 km low lunar orbit
v2_circular_moon = sqrt(mu_moon / (R_moon + 100e3));

% Transfer time: half the orbital period of the transfer ellipse
T_transfer_moon = pi * sqrt(a_transfer^3 / mu_earth);

fprintf('  LEO altitude             : %.0f km\n', (r1 - R_earth)/1e3);
fprintf('  LEO circular velocity    : %.2f m/s\n', v1_circular);
fprintf('  Transfer perigee vel     : %.2f m/s\n', v1_transfer);
fprintf('  TLI delta-V              : %.2f m/s\n', dv_tli);
fprintf('  Transfer apogee vel      : %.2f m/s\n', v2_transfer);
fprintf('  Moon circular orbit vel  : %.2f m/s\n', v2_circular_moon);
fprintf('  Transfer time            : %.2f hours (%.2f days)\n\n', ...
    T_transfer_moon/3600, T_transfer_moon/86400);

%% ========================================================================
%  Lunar Orbit Maneuvers
%  ========================================================================
fprintf('--- Lunar Orbit Maneuvers ---\n\n');

% Low Lunar Orbit at 100 km altitude
r_lunar_orbit = R_moon + 100e3;
v_lunar_orbit = sqrt(mu_moon / r_lunar_orbit);

% Lunar Orbit Insertion (LOI)
% Approximate the arrival velocity at the Moon as the difference between
% the transfer apogee velocity and the Moon's circular velocity around
% Earth (excess hyperbolic velocity at the Moon's sphere of influence).
v_inf_moon     = abs(v2_transfer - v2_circular_earth);
v_arrival_peri = sqrt(v_inf_moon^2 + 2*mu_moon / r_lunar_orbit);
dv_loi         = v_arrival_peri - v_lunar_orbit;

% Inclination change maneuver: 45-degree plane change
inc_change     = 45 * pi / 180;           % [rad]
dv_inc_change  = 2 * v_lunar_orbit * sin(inc_change / 2);

% Lunar escape: from circular orbit to parabolic (v_escape = sqrt(2)*v_circ)
dv_lunar_escape = v_lunar_orbit * (sqrt(2) - 1);

fprintf('  Lunar orbit altitude     : 100 km\n');
fprintf('  Lunar circular velocity  : %.2f m/s\n', v_lunar_orbit);
fprintf('  V-infinity at Moon       : %.2f m/s\n', v_inf_moon);
fprintf('  LOI delta-V              : %.2f m/s\n', dv_loi);
fprintf('  Inclination change (45d) : %.2f m/s\n', dv_inc_change);
fprintf('  Lunar escape delta-V     : %.2f m/s\n\n', dv_lunar_escape);

%% ========================================================================
%  Earth to Jupiter Transfer (Simplified Heliocentric Hohmann)
%  ========================================================================
fprintf('--- Earth to Jupiter Transfer (Heliocentric Hohmann) ---\n\n');

% Hohmann transfer ellipse from Earth orbit to Jupiter orbit around the Sun
a_transfer_jup = (r_earth_sun + r_jupiter_sun) / 2;

% Earth heliocentric circular velocity
v_earth_helio = sqrt(mu_sun / r_earth_sun);

% Departure velocity on the transfer ellipse (at periapsis / Earth)
v_dep_transfer = sqrt(mu_sun * (2/r_earth_sun - 1/a_transfer_jup));

% Heliocentric delta-V at Earth departure
dv_earth_dep = v_dep_transfer - v_earth_helio;

% Jupiter heliocentric circular velocity
v_jupiter_helio = sqrt(mu_sun / r_jupiter_sun);

% Arrival velocity on the transfer ellipse (at apoapsis / Jupiter)
v_arr_transfer = sqrt(mu_sun * (2/r_jupiter_sun - 1/a_transfer_jup));

% Heliocentric delta-V at Jupiter arrival
dv_jupiter_arr = v_jupiter_helio - v_arr_transfer;

% Jupiter Orbit Insertion (JOI) into a 1000 km altitude orbit
r_jup_orbit    = R_jupiter + 1000e3;
v_jup_circ     = sqrt(mu_jupiter / r_jup_orbit);
v_inf_jupiter  = abs(dv_jupiter_arr);
v_hyp_peri_jup = sqrt(v_inf_jupiter^2 + 2*mu_jupiter / r_jup_orbit);
dv_joi         = v_hyp_peri_jup - v_jup_circ;

% Jupiter escape delta-V (from circular orbit to parabolic trajectory)
dv_jupiter_escape = v_jup_circ * (sqrt(2) - 1);

% Transfer time: half the orbital period of the heliocentric transfer ellipse
T_transfer_jup = pi * sqrt(a_transfer_jup^3 / mu_sun);

% Return transfer delta-V (symmetric Hohmann, same magnitudes)
dv_return = dv_earth_dep + abs(dv_jupiter_arr);

fprintf('  Earth heliocentric vel   : %.2f km/s\n', v_earth_helio/1e3);
fprintf('  Departure transfer vel   : %.2f km/s\n', v_dep_transfer/1e3);
fprintf('  Delta-V at Earth dep     : %.2f km/s (%.2f m/s)\n', ...
    dv_earth_dep/1e3, dv_earth_dep);
fprintf('  Arrival transfer vel     : %.2f km/s\n', v_arr_transfer/1e3);
fprintf('  Delta-V at Jupiter arr   : %.2f km/s (%.2f m/s)\n', ...
    abs(dv_jupiter_arr)/1e3, abs(dv_jupiter_arr));
fprintf('  JOI delta-V              : %.2f km/s (%.2f m/s)\n', ...
    dv_joi/1e3, dv_joi);
fprintf('  Jupiter escape delta-V   : %.2f km/s (%.2f m/s)\n', ...
    dv_jupiter_escape/1e3, dv_jupiter_escape);
fprintf('  Transfer time            : %.2f days (%.2f years)\n\n', ...
    T_transfer_jup/86400, T_transfer_jup/(86400*365.25));

%% ========================================================================
%  Complete Delta-V Budget
%  ========================================================================
fprintf('--- Complete Delta-V Budget ---\n\n');

% Collect all mission phase labels and delta-V values [m/s]
phases = {'TLI', 'LOI', 'Inc Change', 'Lunar Esc', ...
          'Jup Transfer', 'JOI', 'Jup Esc', 'Return'};

dvs = [dv_tli, ...
       dv_loi, ...
       dv_inc_change, ...
       dv_lunar_escape, ...
       dv_earth_dep, ...
       dv_joi, ...
       dv_jupiter_escape, ...
       dv_return];

dv_total = sum(dvs);

fprintf('  %-20s  %14s\n', 'Mission Phase', 'Delta-V [m/s]');
fprintf('  %s\n', repmat('-', 1, 38));
for k = 1:length(phases)
    fprintf('  %-20s  %14.2f\n', phases{k}, dvs(k));
end
fprintf('  %s\n', repmat('-', 1, 38));
fprintf('  %-20s  %14.2f\n\n', 'TOTAL', dv_total);

%% ========================================================================
%  Spacecraft Mass Budget (Tsiolkovsky Rocket Equation)
%  ========================================================================
fprintf('--- Spacecraft Mass Budget ---\n\n');

% Compute mass after each sequential burn:
%   m_f = m_i * exp( -dv / (Isp * g0) )
mass_history    = zeros(1, length(dvs) + 1);
mass_history(1) = m0;

for k = 1:length(dvs)
    mass_history(k+1) = mass_history(k) * exp(-dvs(k) / (Isp * g0));
end

prop_consumed = mass_history(1:end-1) - mass_history(2:end);
total_prop    = m0 - mass_history(end);

fprintf('  %-16s  %10s  %14s  %14s\n', ...
    'Phase', 'dV [m/s]', 'Mass After [kg]', 'Prop Used [kg]');
fprintf('  %s\n', repmat('-', 1, 58));
fprintf('  %-16s  %10s  %14.2f  %14s\n', 'Initial', '-', m0, '-');
for k = 1:length(phases)
    fprintf('  %-16s  %10.2f  %14.2f  %14.2f\n', ...
        phases{k}, dvs(k), mass_history(k+1), prop_consumed(k));
end
fprintf('  %s\n', repmat('-', 1, 58));
fprintf('  %-16s  %10.2f  %14.2f  %14.2f\n', ...
    'TOTALS', dv_total, mass_history(end), total_prop);
fprintf('  Propellant mass fraction : %.1f%%\n\n', 100*total_prop/m0);

%% ========================================================================
%  Generate Plots
%  ========================================================================

% Resolve the output directory relative to this script's location
output_dir = fullfile(fileparts(mfilename('fullpath')), ...
    '..', '..', '..', 'output', 'matlab');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

theta = linspace(0, 2*pi, 360);

% ---- Figure 1: Earth-Moon Transfer Trajectory (2D) ----
fig1 = figure('Name', 'Earth-Moon Transfer Trajectory', ...
    'Position', [50 50 900 700], 'Color', 'w');

% Earth surface circle
earth_x = R_earth * cos(theta);
earth_y = R_earth * sin(theta);

% LEO circle
leo_x = r1 * cos(theta);
leo_y = r1 * sin(theta);

% Moon orbit circle
moon_orb_x = r2 * cos(theta);
moon_orb_y = r2 * sin(theta);

% Transfer ellipse from perigee (r1) to apogee (r2)
e_moon = 1 - r1 / a_transfer;                 % Eccentricity
theta_half = linspace(0, pi, 180);             % Half-ellipse
r_trans = a_transfer * (1 - e_moon^2) ./ (1 + e_moon * cos(theta_half));
trans_x = r_trans .* cos(theta_half);
trans_y = r_trans .* sin(theta_half);

% Scaling to megameters for readability
scale = 1e6;

fill(earth_x/scale, earth_y/scale, [0.2 0.4 0.8], 'EdgeColor', 'none');
hold on;
plot(leo_x/scale, leo_y/scale, 'g--', 'LineWidth', 1);
plot(moon_orb_x/scale, moon_orb_y/scale, 'Color', [0.5 0.5 0.5], ...
    'LineStyle', '--', 'LineWidth', 1);
plot(trans_x/scale, trans_y/scale, 'r', 'LineWidth', 2);

% Mark TLI departure and arrival points
plot(r1/scale, 0, 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot(trans_x(end)/scale, trans_y(end)/scale, 'rs', ...
    'MarkerSize', 10, 'MarkerFaceColor', 'r');

% Draw Moon at apogee location
moon_cx = r2 * cos(pi);
moon_cy = r2 * sin(pi);
moon_surf_x = moon_cx + R_moon * cos(theta);
moon_surf_y = moon_cy + R_moon * sin(theta);
fill(moon_surf_x/scale, moon_surf_y/scale, [0.7 0.7 0.7], 'EdgeColor', 'k');

xlabel('X [$10^3$ km]', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Y [$10^3$ km]', 'Interpreter', 'latex', 'FontSize', 12);
title('Earth--Moon Hohmann Transfer Trajectory', ...
    'Interpreter', 'latex', 'FontSize', 14);
legend({'Earth', 'LEO (200 km)', 'Moon Orbit', 'Transfer Ellipse', ...
    'Departure (TLI)', 'Arrival', 'Moon'}, ...
    'Interpreter', 'latex', 'Location', 'best', 'FontSize', 9);
grid on; axis equal;
set(gca, 'FontSize', 11, 'TickLabelInterpreter', 'latex');
hold off;

print(fig1, fullfile(output_dir, 'trajectory_earth_moon'), '-dpng', '-r150');
fprintf('  Saved: trajectory_earth_moon.png\n');

% ---- Figure 2: Earth-Jupiter Heliocentric Transfer (2D Ecliptic) ----
fig2 = figure('Name', 'Earth-Jupiter Heliocentric Transfer', ...
    'Position', [100 100 900 700], 'Color', 'w');

% Earth orbit around Sun
earth_orb_x = r_earth_sun * cos(theta);
earth_orb_y = r_earth_sun * sin(theta);

% Jupiter orbit around Sun
jup_orb_x = r_jupiter_sun * cos(theta);
jup_orb_y = r_jupiter_sun * sin(theta);

% Transfer ellipse (periapsis at Earth, apoapsis at Jupiter)
e_jup = 1 - r_earth_sun / a_transfer_jup;
theta_jup_half = linspace(0, pi, 180);
r_jup_trans = a_transfer_jup * (1 - e_jup^2) ./ ...
    (1 + e_jup * cos(theta_jup_half));
jtrans_x = r_jup_trans .* cos(theta_jup_half);
jtrans_y = r_jup_trans .* sin(theta_jup_half);

% Scaling to billions of meters (Gm)
gm = 1e9;

plot(earth_orb_x/gm, earth_orb_y/gm, 'b', 'LineWidth', 1.5); hold on;
plot(jup_orb_x/gm, jup_orb_y/gm, 'Color', [0.8 0.5 0.1], 'LineWidth', 1.5);
plot(jtrans_x/gm, jtrans_y/gm, 'r--', 'LineWidth', 2);

% Sun, Earth departure, Jupiter arrival markers
plot(0, 0, 'yo', 'MarkerSize', 18, 'MarkerFaceColor', [1 0.9 0]);
plot(r_earth_sun/gm, 0, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
plot(jtrans_x(end)/gm, jtrans_y(end)/gm, 'rs', ...
    'MarkerSize', 10, 'MarkerFaceColor', 'r');

xlabel('X [$10^6$ km]', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Y [$10^6$ km]', 'Interpreter', 'latex', 'FontSize', 12);
title('Heliocentric Earth--Jupiter Hohmann Transfer', ...
    'Interpreter', 'latex', 'FontSize', 14);
legend({'Earth Orbit', 'Jupiter Orbit', 'Transfer Ellipse', ...
    'Sun', 'Earth Departure', 'Jupiter Arrival'}, ...
    'Interpreter', 'latex', 'Location', 'best', 'FontSize', 9);
grid on; axis equal;
set(gca, 'FontSize', 11, 'TickLabelInterpreter', 'latex');
hold off;

print(fig2, fullfile(output_dir, 'trajectory_earth_jupiter'), '-dpng', '-r150');
fprintf('  Saved: trajectory_earth_jupiter.png\n');

% ---- Figure 3: Delta-V Budget Bar Chart ----
fig3 = figure('Name', 'Delta-V Budget', ...
    'Position', [150 150 900 600], 'Color', 'w');

barh_colors = [0.2 0.6 0.9;   % TLI - blue
               0.3 0.7 0.4;   % LOI - green
               0.9 0.6 0.2;   % Inc Change - orange
               0.8 0.3 0.3;   % Lunar Esc - red
               0.5 0.3 0.8;   % Jup Transfer - purple
               0.2 0.8 0.7;   % JOI - teal
               0.9 0.4 0.6;   % Jup Esc - pink
               0.6 0.6 0.6];  % Return - gray

cats = categorical(phases, phases);  % Preserve ordering
bh   = barh(cats, dvs, 'FaceColor', 'flat');
bh.CData = barh_colors;

% Value labels on bars
for k = 1:length(dvs)
    text(dvs(k) + max(dvs)*0.02, k, sprintf('%.0f m/s', dvs(k)), ...
        'VerticalAlignment', 'middle', 'FontSize', 10, ...
        'Interpreter', 'latex');
end

xlabel('$\Delta V$ [m/s]', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Mission Phase', 'Interpreter', 'latex', 'FontSize', 12);
title('Mission $\Delta V$ Budget by Phase', ...
    'Interpreter', 'latex', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 11, 'TickLabelInterpreter', 'latex');
xlim([0 max(dvs)*1.2]);

print(fig3, fullfile(output_dir, 'deltav_budget'), '-dpng', '-r150');
fprintf('  Saved: deltav_budget.png\n');

% ---- Figure 4: Spacecraft Mass vs Mission Phase (Stairs Plot) ----
fig4 = figure('Name', 'Spacecraft Mass Budget', ...
    'Position', [200 200 900 550], 'Color', 'w');

phase_idx    = 0:length(dvs);
phase_labels = [{'Initial'}, phases];

stairs(phase_idx, mass_history, 'b-o', 'LineWidth', 2, ...
    'MarkerFaceColor', 'b', 'MarkerSize', 6);
hold on;

% Shade propellant-consumed region
fill([phase_idx, fliplr(phase_idx)], ...
     [mass_history, mass_history(end)*ones(size(mass_history))], ...
     [1 0.8 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'none');

% Mark final dry mass
yline(mass_history(end), 'r--', 'LineWidth', 1.5);
text(length(dvs)*0.5, mass_history(end) - 80, ...
    sprintf('Final Mass: %.1f kg', mass_history(end)), ...
    'FontSize', 10, 'Color', 'r', 'Interpreter', 'latex', ...
    'HorizontalAlignment', 'center');

xticks(phase_idx);
xticklabels(phase_labels);
xtickangle(30);
xlabel('Mission Phase', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Spacecraft Mass [kg]', 'Interpreter', 'latex', 'FontSize', 12);
title('Spacecraft Mass Depletion Through Mission Phases', ...
    'Interpreter', 'latex', 'FontSize', 14);
legend({'Mass History', 'Propellant Consumed', 'Final Dry Mass'}, ...
    'Interpreter', 'latex', 'Location', 'northeast', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10, 'TickLabelInterpreter', 'latex');
ylim([0 m0*1.1]);
hold off;

print(fig4, fullfile(output_dir, 'mass_budget'), '-dpng', '-r150');
fprintf('  Saved: mass_budget.png\n');

% ---- Figure 5: Velocity Magnitude Over Mission Timeline ----
fig5 = figure('Name', 'Velocity Profile', ...
    'Position', [250 250 1000 500], 'Color', 'w');

% Approximate cumulative mission time for each phase boundary [days]
phase_durations = [0, ...                              % Initial (t = 0)
                   T_transfer_moon/86400, ...          % TLI coast
                   1, ...                              % LOI (short burn)
                   7, ...                              % Lunar operations
                   2, ...                              % Lunar escape
                   T_transfer_jup/86400, ...           % Jupiter cruise
                   1, ...                              % JOI
                   30, ...                             % Jupiter operations
                   T_transfer_jup/86400];              % Return cruise
cum_time = cumsum(phase_durations);

% Approximate velocities at each phase boundary [m/s]
velocities = [v1_circular, ...              % LEO
              v1_transfer, ...              % After TLI
              v_lunar_orbit, ...            % Lunar orbit (post-LOI)
              v_lunar_orbit, ...            % After inclination change (same speed)
              v_lunar_orbit*sqrt(2), ...    % Lunar escape
              v_dep_transfer, ...           % Heliocentric cruise outbound
              v_jup_circ, ...              % Jupiter orbit (post-JOI)
              v_jup_circ*sqrt(2), ...      % Jupiter escape
              v_earth_helio];              % Return near Earth

plot(cum_time, velocities/1e3, 'b-s', 'LineWidth', 2, ...
    'MarkerFaceColor', [0.2 0.5 0.9], 'MarkerSize', 8);
hold on;

% Annotate each point with its velocity value
for k = 1:length(cum_time)
    text(cum_time(k), velocities(k)/1e3 + 0.8, ...
        sprintf('%.1f', velocities(k)/1e3), ...
        'FontSize', 8, 'HorizontalAlignment', 'center', ...
        'Interpreter', 'latex');
end

xlabel('Mission Time [days]', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Velocity Magnitude [km/s]', 'Interpreter', 'latex', 'FontSize', 12);
title('Spacecraft Velocity Profile Over Mission Timeline', ...
    'Interpreter', 'latex', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 11, 'TickLabelInterpreter', 'latex');
hold off;

print(fig5, fullfile(output_dir, 'velocity_profile'), '-dpng', '-r150');
fprintf('  Saved: velocity_profile.png\n');

%% ========================================================================
%  Print Final Summary Table
%  ========================================================================
fprintf('\n');
fprintf('=============================================================\n');
fprintf('  MISSION SUMMARY\n');
fprintf('=============================================================\n\n');
fprintf('  %-16s  %12s  %14s  %14s\n', ...
    'Phase', 'dV [m/s]', 'Mass After [kg]', 'Prop Used [kg]');
fprintf('  %s\n', repmat('=', 1, 60));
for k = 1:length(phases)
    fprintf('  %-16s  %12.2f  %14.2f  %14.2f\n', ...
        phases{k}, dvs(k), mass_history(k+1), prop_consumed(k));
end
fprintf('  %s\n', repmat('-', 1, 60));
fprintf('  %-16s  %12.2f  %14.2f  %14.2f\n', ...
    'TOTAL', dv_total, mass_history(end), total_prop);
fprintf('\n');
fprintf('  Initial mass              : %.1f kg\n', m0);
fprintf('  Final mass                : %.1f kg\n', mass_history(end));
fprintf('  Total propellant          : %.1f kg\n', total_prop);
fprintf('  Propellant mass fraction  : %.1f%%\n', 100*total_prop/m0);
fprintf('  Earth-Moon transfer time  : %.2f days\n', T_transfer_moon/86400);
fprintf('  Earth-Jupiter xfer time   : %.2f years\n', ...
    T_transfer_jup/(86400*365.25));
fprintf('  Engine Isp                : %.0f s\n', Isp);
fprintf('\n');
fprintf('  All plots saved to: %s\n', output_dir);
fprintf('=============================================================\n');
fprintf('  Trajectory analysis complete.\n');
fprintf('=============================================================\n');
