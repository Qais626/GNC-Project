function qp = quat_mult(q1, q2)
%QUAT_MULT Quaternion multiplication (Hamilton product)
%
%   qp = QUAT_MULT(q1, q2) computes the quaternion product q1 * q2
%
%   Convention: scalar-last [qx, qy, qz, qw]
%
%   The product represents the composition of rotations:
%   R(qp) = R(q1) * R(q2)
%
%   Inputs:
%       q1 - First quaternion [4x1]
%       q2 - Second quaternion [4x1]
%
%   Output:
%       qp - Product quaternion [4x1]
%
%   Example:
%       q1 = [0; 0; sin(pi/4); cos(pi/4)];  % 90 deg about z
%       q2 = [0; sin(pi/4); 0; cos(pi/4)];  % 90 deg about y
%       qp = quat_mult(q1, q2);
%
%   Author: GNC Engineering Team
%   Date: 2026

%#codegen

% Extract components
v1 = q1(1:3);   % Vector part
s1 = q1(4);     % Scalar part
v2 = q2(1:3);
s2 = q2(4);

% Hamilton product
qp = [s1*v2 + s2*v1 + cross(v1, v2);
      s1*s2 - dot(v1, v2)];

% Ensure unit norm (numerical stability)
qp = qp / norm(qp);

end
