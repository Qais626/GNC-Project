function omega_dot = euler_dynamics(tau, omega, I)
%EULER_DYNAMICS Euler's rotational equations of motion
%
%   omega_dot = EULER_DYNAMICS(tau, omega, I) computes the angular
%   acceleration of a rigid body using Euler's equations:
%
%       I * omega_dot = tau - omega x (I * omega)
%
%   Inputs:
%       tau   - Applied torque in body frame [N*m] [3x1]
%       omega - Angular velocity in body frame [rad/s] [3x1]
%       I     - Inertia tensor [kg*m^2] [3x3]
%
%   Output:
%       omega_dot - Angular acceleration [rad/s^2] [3x1]
%
%   The equation accounts for:
%       - Applied external torques
%       - Gyroscopic coupling between axes
%
%   Example:
%       I = diag([1200, 1350, 980]);
%       omega = [0.01; 0.02; -0.01];
%       tau = [0.1; -0.05; 0.02];
%       omega_dot = euler_dynamics(tau, omega, I);
%
%   Author: GNC Engineering Team
%   Date: 2026

%#codegen

% Angular momentum
H = I * omega;

% Gyroscopic torque (omega x H)
tau_gyro = cross(omega, H);

% Euler's equation: I * omega_dot = tau - omega x (I * omega)
omega_dot = I \ (tau - tau_gyro);

end
