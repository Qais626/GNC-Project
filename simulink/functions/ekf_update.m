function [x_est, P_est] = ekf_update(z, x_prev, P_prev, Q, R, dt, measurement_type)
%EKF_UPDATE Extended Kalman Filter prediction and update
%
%   [x_est, P_est] = EKF_UPDATE(z, x_prev, P_prev, Q, R, dt, measurement_type)
%   performs one cycle of EKF prediction and measurement update
%
%   State Vector (16 states):
%       x = [pos(3), vel(3), q(4), omega(3), bias_g(3)]
%
%   Inputs:
%       z               - Measurement vector
%       x_prev          - Previous state estimate [16x1]
%       P_prev          - Previous covariance [16x16]
%       Q               - Process noise covariance [16x16]
%       R               - Measurement noise covariance
%       dt              - Time step [s]
%       measurement_type - 1=GPS, 2=StarTracker, 3=Gyro
%
%   Outputs:
%       x_est - Updated state estimate [16x1]
%       P_est - Updated covariance [16x16]
%
%   Example:
%       x0 = [r0; v0; q0; omega0; bias0];
%       P0 = diag([100^2*ones(1,3), 1^2*ones(1,3), ...
%                  0.01^2*ones(1,4), 0.001^2*ones(1,3), 1e-6*ones(1,3)]);
%       Q = diag([1e-6*ones(1,6), 1e-10*ones(1,4), 1e-8*ones(1,3), 1e-12*ones(1,3)]);
%       R_gps = diag([1.5^2, 1.5^2, 1.5^2, 0.01^2, 0.01^2, 0.01^2]);
%
%       [x_new, P_new] = ekf_update(z_gps, x0, P0, Q, R_gps, 0.1, 1);
%
%   Author: GNC Engineering Team
%   Date: 2026

%#codegen

% Earth parameters
mu = 3.986004418e14;  % [m^3/s^2]

% Extract state components
pos = x_prev(1:3);
vel = x_prev(4:6);
q = x_prev(7:10);
omega = x_prev(11:13);
bias_g = x_prev(14:16);

%% ==================== PREDICTION STEP ====================

% --- Propagate Position & Velocity ---
r = norm(pos);
a_grav = -mu * pos / r^3;  % Two-body gravity

% Simple Euler integration (for demonstration)
pos_pred = pos + vel * dt + 0.5 * a_grav * dt^2;
vel_pred = vel + a_grav * dt;

% --- Propagate Quaternion ---
omega_body = omega;  % Use estimated omega (gyro would be processed separately)

% Quaternion derivative: q_dot = 0.5 * Omega(omega) * q
omega_norm = norm(omega_body);
if omega_norm > 1e-12
    dtheta = omega_norm * dt;
    axis = omega_body / omega_norm;
    dq = [axis * sin(dtheta/2); cos(dtheta/2)];
else
    dq = [0; 0; 0; 1];
end

% Quaternion multiplication
q_pred = quat_mult_local(q, dq);
q_pred = q_pred / norm(q_pred);  % Normalize

% --- Propagate Angular Rate & Biases ---
omega_pred = omega;      % Assume constant (no dynamics model)
bias_pred = bias_g;      % Random walk

% Predicted state
x_pred = [pos_pred; vel_pred; q_pred; omega_pred; bias_pred];

% --- Propagate Covariance ---
% State transition Jacobian (simplified)
F = eye(16);
F(1:3, 4:6) = eye(3) * dt;  % Position depends on velocity

% Covariance prediction
P_pred = F * P_prev * F' + Q;

%% ==================== UPDATE STEP ====================

switch measurement_type
    case 1  % GPS (position and velocity)
        % Measurement model: z = [pos; vel]
        H = zeros(6, 16);
        H(1:3, 1:3) = eye(3);  % Position
        H(4:6, 4:6) = eye(3);  % Velocity

        % Predicted measurement
        z_pred = [pos_pred; vel_pred];

        % Innovation
        y = z - z_pred;

    case 2  % Star Tracker (attitude quaternion)
        % Measurement model: z = q
        H = zeros(4, 16);
        H(1:4, 7:10) = eye(4);

        % Predicted measurement
        z_pred = q_pred;

        % Innovation (quaternion error)
        y = z - z_pred;

        % Handle quaternion sign ambiguity
        if dot(z, z_pred) < 0
            y = z + z_pred;
        end

    case 3  % Gyroscope (angular rate with bias)
        % Measurement model: z = omega + bias + noise
        H = zeros(3, 16);
        H(1:3, 11:13) = eye(3);  % Angular rate
        H(1:3, 14:16) = eye(3);  % Bias

        % Predicted measurement
        z_pred = omega_pred + bias_pred;

        % Innovation
        y = z - z_pred;

    otherwise
        error('Unknown measurement type');
end

% Innovation covariance
S = H * P_pred * H' + R;

% Kalman gain
K = P_pred * H' / S;

% State update
x_est = x_pred + K * y;

% Covariance update (Joseph form for numerical stability)
I_KH = eye(16) - K * H;
P_est = I_KH * P_pred * I_KH' + K * R * K';

% Ensure quaternion normalization
x_est(7:10) = x_est(7:10) / norm(x_est(7:10));

% Ensure covariance symmetry
P_est = 0.5 * (P_est + P_est');

end

%% ==================== LOCAL FUNCTIONS ====================

function qp = quat_mult_local(q1, q2)
    % Quaternion multiplication (scalar-last convention)
    v1 = q1(1:3);
    s1 = q1(4);
    v2 = q2(1:3);
    s2 = q2(4);

    qp = [s1*v2 + s2*v1 + cross(v1, v2);
          s1*s2 - dot(v1, v2)];
end
