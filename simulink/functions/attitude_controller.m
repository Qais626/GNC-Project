function tau = attitude_controller(q_err, omega_err, mode, params)
%ATTITUDE_CONTROLLER Multi-mode attitude control law
%
%   tau = ATTITUDE_CONTROLLER(q_err, omega_err, mode, params) computes
%   the control torque based on attitude and rate errors
%
%   Inputs:
%       q_err     - Attitude error quaternion [4x1] (error = q_target^-1 * q_actual)
%       omega_err - Angular rate error [rad/s] [3x1]
%       mode      - Controller mode: 1=PID, 2=LQR, 3=SMC
%       params    - Structure containing controller parameters
%
%   Output:
%       tau - Control torque command [N*m] [3x1]
%
%   Controller Modes:
%       1 (PID): Proportional-Integral-Derivative with anti-windup
%       2 (LQR): Linear Quadratic Regulator (optimal)
%       3 (SMC): Sliding Mode Control (robust)
%
%   Example:
%       params.Kp = diag([12.5, 14, 11]);
%       params.Ki = diag([0.08, 0.1, 0.06]);
%       params.Kd = diag([45, 50, 40]);
%       params.K_lqr = [Kp, Kd];  % 3x6 gain matrix
%       params.smc_lambda = 3;
%       params.smc_eta = 0.8;
%       params.smc_phi = 0.01;
%       params.I = diag([1200, 1350, 980]);
%       params.max_torque = 0.6;
%
%       tau = attitude_controller(q_err, omega_err, 2, params);
%
%   Author: GNC Engineering Team
%   Date: 2026

%#codegen

% Convert quaternion error to axis-angle error (small angle approximation)
% For small errors: theta_err ≈ 2 * q_vec
theta_err = 2 * q_err(1:3);

% Persistent variable for integral term (PID)
persistent int_err;
if isempty(int_err)
    int_err = zeros(3, 1);
end

% Controller selection
switch mode
    case 1
        %% ========== PID Controller ==========
        % Extract gains
        Kp = params.Kp;
        Ki = params.Ki;
        Kd = params.Kd;

        % Proportional term
        tau_p = -Kp * theta_err;

        % Integral term with anti-windup
        int_err = int_err + theta_err * params.dt;

        % Anti-windup: limit integral when saturated
        int_limit = params.max_torque / max(diag(Ki));
        int_err = max(min(int_err, int_limit), -int_limit);

        tau_i = -Ki * int_err;

        % Derivative term
        tau_d = -Kd * omega_err;

        % Total PID torque
        tau = tau_p + tau_i + tau_d;

    case 2
        %% ========== LQR Controller ==========
        % State vector: x = [theta_err; omega_err]
        x = [theta_err; omega_err];

        % LQR control law: u = -K * x
        K_lqr = params.K_lqr;
        tau = -K_lqr * x;

        % Reset integral (not used in LQR)
        int_err = zeros(3, 1);

    case 3
        %% ========== Sliding Mode Controller ==========
        lambda = params.smc_lambda;  % Sliding surface slope
        eta = params.smc_eta;        % Switching gain
        phi = params.smc_phi;        % Boundary layer
        I = params.I;                % Inertia tensor

        % Sliding surface: s = omega_err + lambda * theta_err
        s = omega_err + lambda * theta_err;

        % Saturation function (smooth switching)
        sat_s = zeros(3, 1);
        for i = 1:3
            if abs(s(i)) > phi
                sat_s(i) = sign(s(i));
            else
                sat_s(i) = s(i) / phi;
            end
        end

        % Control law: tau = -I * (lambda * omega_err + eta * sat(s))
        tau_eq = -lambda * omega_err;   % Equivalent control
        tau_sw = -eta * sat_s;          % Switching control
        tau = I * (tau_eq + tau_sw);

        % Reset integral (not used in SMC)
        int_err = zeros(3, 1);

    otherwise
        % Default to PD control
        tau = -params.Kp * theta_err - params.Kd * omega_err;
end

% Torque saturation
tau = max(min(tau, params.max_torque), -params.max_torque);

end
