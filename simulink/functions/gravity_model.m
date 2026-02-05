function [a_grav, a_j2] = gravity_model(r_eci, mu, Re, J2)
%GRAVITY_MODEL Compute gravitational acceleration with J2 perturbation
%
%   [a_grav, a_j2] = GRAVITY_MODEL(r_eci, mu, Re, J2) computes the total
%   gravitational acceleration including two-body and J2 oblateness effects
%
%   Inputs:
%       r_eci - Position vector in ECI frame [m] [3x1]
%       mu    - Gravitational parameter [m^3/s^2]
%       Re    - Equatorial radius [m]
%       J2    - J2 zonal harmonic coefficient
%
%   Outputs:
%       a_grav - Total gravitational acceleration [m/s^2] [3x1]
%       a_j2   - J2 perturbation component only [m/s^2] [3x1]
%
%   The J2 perturbation models the oblateness of the central body.
%   For Earth: J2 = 1.08263e-3
%
%   Example:
%       r = [7000e3; 0; 0];  % 7000 km altitude on x-axis
%       mu = 3.986004418e14;  % Earth mu
%       Re = 6378137;         % Earth radius
%       J2 = 1.08263e-3;
%       a = gravity_model(r, mu, Re, J2);
%
%   Author: GNC Engineering Team
%   Date: 2026

%#codegen

% Default Earth parameters if not provided
if nargin < 2
    mu = 3.986004418e14;  % Earth gravitational parameter [m^3/s^2]
end
if nargin < 3
    Re = 6378137;         % Earth equatorial radius [m]
end
if nargin < 4
    J2 = 1.08263e-3;      % Earth J2
end

% Distance from center
r = norm(r_eci);
x = r_eci(1);
y = r_eci(2);
z = r_eci(3);

% Two-body gravitational acceleration
a_2body = -mu * r_eci / r^3;

% J2 perturbation acceleration
% a_J2 = -3/2 * J2 * mu * Re^2 / r^5 * [x*(1-5*z^2/r^2);
%                                        y*(1-5*z^2/r^2);
%                                        z*(3-5*z^2/r^2)]
factor = 1.5 * J2 * mu * Re^2 / r^5;
z2_r2 = (z/r)^2;

a_j2 = -factor * [x * (1 - 5*z2_r2);
                  y * (1 - 5*z2_r2);
                  z * (3 - 5*z2_r2)];

% Total gravitational acceleration
a_grav = a_2body + a_j2;

end
