function DCM = quat_to_dcm(q)
%QUAT_TO_DCM Convert quaternion to Direction Cosine Matrix (DCM)
%
%   DCM = QUAT_TO_DCM(q) converts a unit quaternion to a 3x3 rotation matrix
%
%   Convention: scalar-last [qx, qy, qz, qw]
%   The DCM transforms vectors from the reference frame to the body frame:
%       v_body = DCM * v_ref
%
%   Inputs:
%       q - Unit quaternion [4x1]
%
%   Output:
%       DCM - Direction Cosine Matrix [3x3]
%
%   Example:
%       q = [0; 0; sin(pi/4); cos(pi/4)];  % 90 deg about z
%       DCM = quat_to_dcm(q);
%       % DCM should be [0 1 0; -1 0 0; 0 0 1]
%
%   Author: GNC Engineering Team
%   Date: 2026

%#codegen

% Normalize quaternion
q = q / norm(q);

% Extract components
qx = q(1);
qy = q(2);
qz = q(3);
qw = q(4);

% Compute DCM elements
DCM = [1 - 2*(qy^2 + qz^2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw);
       2*(qx*qy + qz*qw),     1 - 2*(qx^2 + qz^2),     2*(qy*qz - qx*qw);
       2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),   1 - 2*(qx^2 + qy^2)];

end
