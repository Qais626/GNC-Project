"""
===============================================================================
GNC PROJECT - Attitude Dynamics Engine
===============================================================================
High-fidelity rotational dynamics model for spacecraft attitude propagation.

This module propagates the rigid-body rotational equations of motion (Euler's
equations) coupled with quaternion kinematics, plus non-ideal effects that
dominate real spacecraft behavior:

    - Structural flexibility (flex modes coupled to the rigid body)
    - Inertia uncertainty and time-varying inertia (propellant slosh/burn)
    - CG offset from geometric center (creates torque when thrusting)
    - Momentum exchange devices (reaction wheels, CMGs)
    - Gravity-gradient torque
    - Residual magnetic dipole torque

Governing equations:

    Euler's equation:
        I * omega_dot = -omega x (I * omega) + T_external + T_disturbance

    Quaternion kinematics (scalar-last convention q = [qx, qy, qz, qw]):
        q_dot = 0.5 * Omega(omega) * q

    where Omega(omega) is the 4x4 skew-symmetric matrix constructed from
    the angular velocity vector.

State vector layout:
    [q0, q1, q2, q3, wx, wy, wz, flex_0_pos, flex_0_vel, ...]

References:
    - Wie, B. "Space Vehicle Dynamics and Control," 2nd ed., AIAA, 2008.
    - Sidi, M.J. "Spacecraft Dynamics and Control," Cambridge, 1997.
    - Hughes, P.C. "Spacecraft Attitude Dynamics," Dover, 2004.
===============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from core.quaternion import Quaternion
from core.constants import *


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FlexMode:
    """
    Structural flexibility mode coupled to the rigid body.

    Each flex mode is a second-order oscillator:
        eta_ddot + 2*zeta*omega_n*eta_dot + omega_n^2 * eta = coupling^T * theta_ddot

    The flex displacement eta feeds back a parasitic torque on the rigid body:
        T_flex = -coupling * (omega_n^2 * eta + 2*zeta*omega_n * eta_dot)

    Attributes:
        frequency_hz: Natural frequency of the flex mode [Hz].
        damping_ratio: Viscous damping ratio zeta (typically 0.001 to 0.05).
        coupling_vector: 3-element coupling coefficient vector [Nm/rad] that
                         maps flex displacement to torque on each body axis.
    """
    frequency_hz: float
    damping_ratio: float
    coupling_vector: np.ndarray  # shape (3,)

    @property
    def omega_n(self) -> float:
        """Natural frequency in rad/s."""
        return 2.0 * PI * self.frequency_hz

    @property
    def omega_d(self) -> float:
        """Damped natural frequency in rad/s."""
        return self.omega_n * np.sqrt(1.0 - self.damping_ratio ** 2)


@dataclass
class MomentumExchangeDevice:
    """
    Reaction wheel or CMG model for angular momentum bookkeeping.

    Attributes:
        spin_axis: Unit vector of wheel spin axis in body frame (3,).
        inertia: Spin-axis moment of inertia [kg*m^2].
        speed: Current wheel speed [rad/s].
        max_speed: Saturation speed [rad/s].
        max_torque: Maximum torque the wheel can apply [Nm].
    """
    spin_axis: np.ndarray   # shape (3,)
    inertia: float          # kg*m^2
    speed: float = 0.0      # rad/s
    max_speed: float = 6000.0 * TWO_PI / 60.0   # ~6000 RPM default
    max_torque: float = 0.2  # Nm


@dataclass
class AttitudeConfig:
    """
    Configuration for the attitude dynamics model.

    Attributes:
        inertia: 3x3 inertia tensor in body frame [kg*m^2]. Includes
                 off-diagonal products of inertia from structural asymmetry.
        inertia_uncertainty: Fractional 1-sigma uncertainty on each element
                            of the inertia tensor (e.g. 0.05 for 5%).
        cg_offset: Center of gravity offset from geometric center [m] (3,).
        flex_modes: List of structural flex modes coupled to the rigid body.
        wheels: List of momentum exchange devices (reaction wheels / CMGs).
        residual_magnetic_moment: Spacecraft magnetic dipole moment [A*m^2] (3,).
                                  Creates torque in Earth's magnetic field.
        magnetic_field_body: Approximate local magnetic field in body frame [T] (3,).
                            Used for residual magnetic torque computation.
    """
    inertia: np.ndarray                                # shape (3, 3)
    inertia_uncertainty: float = 0.05                  # 5% default
    cg_offset: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )
    flex_modes: List[FlexMode] = field(default_factory=list)
    wheels: List[MomentumExchangeDevice] = field(default_factory=list)
    residual_magnetic_moment: np.ndarray = field(
        default_factory=lambda: np.array([0.1, 0.05, 0.02])  # A*m^2
    )
    magnetic_field_body: np.ndarray = field(
        default_factory=lambda: np.array([2.0e-5, 0.0, -4.0e-5])  # Tesla, ~LEO
    )


# =============================================================================
# ATTITUDE DYNAMICS ENGINE
# =============================================================================

class AttitudeDynamics:
    """
    Spacecraft attitude dynamics propagator with non-ideal effects.

    This class integrates the coupled system of:
      1. Euler's rotational equation for rigid-body angular velocity.
      2. Quaternion kinematics for orientation.
      3. Second-order flex mode oscillators for structural flexibility.

    The state vector is laid out as:
        x = [q0, q1, q2, q3, wx, wy, wz, eta0, eta0_dot, eta1, eta1_dot, ...]

    where q is the attitude quaternion (scalar-last: [qx, qy, qz, qw]),
    omega is the angular velocity in the body frame [rad/s], and each flex
    mode contributes two states (displacement eta and rate eta_dot).

    Parameters:
        config: AttitudeConfig with inertia, flex modes, wheels, etc.
    """

    # Indices into state vector
    Q_START = 0
    Q_END = 4
    W_START = 4
    W_END = 7
    FLEX_START = 7

    def __init__(self, config: AttitudeConfig):
        self.config = config
        self.inertia = np.array(config.inertia, dtype=np.float64)
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.cg_offset = np.array(config.cg_offset, dtype=np.float64)
        self.flex_modes = list(config.flex_modes)
        self.wheels = list(config.wheels)

        # Derived quantities
        self.num_flex_modes = len(self.flex_modes)
        self.state_size = 7 + 2 * self.num_flex_modes  # 4 quat + 3 omega + 2 per flex

        # Propellant tracking for time-varying inertia
        self._propellant_fraction = 1.0  # 1.0 = full, 0.0 = dry
        self._initial_inertia = self.inertia.copy()

    # -------------------------------------------------------------------------
    # STATE VECTOR HELPERS
    # -------------------------------------------------------------------------

    def build_initial_state(
        self,
        quaternion: np.ndarray,
        angular_velocity: np.ndarray,
        flex_initial: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Construct the full state vector from components.

        Args:
            quaternion: Attitude quaternion [qx, qy, qz, qw] (4,).
            angular_velocity: Body angular velocity [rad/s] (3,).
            flex_initial: Initial flex states [eta0, eta0_dot, ...].
                          Defaults to zeros if not provided.

        Returns:
            Full state vector of shape (state_size,).
        """
        q = np.array(quaternion, dtype=np.float64)
        q = q / np.linalg.norm(q)  # ensure unit quaternion

        state = np.zeros(self.state_size)
        state[self.Q_START:self.Q_END] = q
        state[self.W_START:self.W_END] = angular_velocity

        if flex_initial is not None:
            state[self.FLEX_START:] = flex_initial
        return state

    def _unpack_state(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Unpack state vector into quaternion, angular velocity, and flex states.

        Returns:
            (quaternion(4,), omega(3,), flex_states(2*N,))
        """
        q = state[self.Q_START:self.Q_END]
        omega = state[self.W_START:self.W_END]
        flex = state[self.FLEX_START:]
        return q, omega, flex

    # -------------------------------------------------------------------------
    # QUATERNION KINEMATICS
    # -------------------------------------------------------------------------

    @staticmethod
    def omega_matrix(omega: np.ndarray) -> np.ndarray:
        """
        Build the 4x4 Omega matrix for quaternion kinematics.

        For scalar-last quaternion q = [qx, qy, qz, qw], the kinematic
        equation is:
            q_dot = 0.5 * Omega(omega) * q

        The Omega matrix is:
            | 0    wz  -wy   wx |
            |-wz   0    wx   wy |
            | wy  -wx   0    wz |
            |-wx  -wy  -wz   0  |

        Args:
            omega: Angular velocity [wx, wy, wz] in body frame (3,).

        Returns:
            4x4 Omega matrix.
        """
        wx, wy, wz = omega
        return np.array([
            [ 0.0,   wz,  -wy,   wx],
            [-wz,    0.0,  wx,   wy],
            [ wy,   -wx,   0.0,  wz],
            [-wx,   -wy,  -wz,   0.0],
        ])

    @staticmethod
    def euler_to_quaternion_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        Compute quaternion time derivative from current quaternion and angular
        velocity using the kinematic differential equation.

            q_dot = 0.5 * Omega(omega) * q

        This propagates the orientation quaternion given the body-frame angular
        velocity. The result must be integrated (e.g. via RK4) and the quaternion
        re-normalized at each step to stay on the unit sphere.

        Args:
            q: Current attitude quaternion [qx, qy, qz, qw] (4,).
            omega: Body angular velocity [rad/s] (3,).

        Returns:
            q_dot: Time derivative of quaternion (4,).
        """
        Omega = AttitudeDynamics.omega_matrix(omega)
        return 0.5 * Omega @ q

    # -------------------------------------------------------------------------
    # DISTURBANCE TORQUES
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_gravity_gradient_torque(
        r_body: np.ndarray,
        inertia: np.ndarray,
        mu: float,
    ) -> np.ndarray:
        """
        Compute gravity-gradient torque on the spacecraft.

        For a spacecraft at position r from the central body, the gravity
        gradient torque in the body frame is:

            T_gg = (3 * mu / r^3) * (r_hat x (I * r_hat))

        where r_hat is the unit position vector resolved in the body frame,
        I is the inertia tensor, and mu is the gravitational parameter.

        This torque tends to align the minimum inertia axis with the local
        vertical (gravity-gradient stabilization). It is significant for
        large spacecraft (e.g. ISS-class) in LEO.

        Args:
            r_body: Position vector from central body to spacecraft, expressed
                    in the body frame [m] (3,).
            inertia: 3x3 inertia tensor in body frame [kg*m^2].
            mu: Gravitational parameter of central body [m^3/s^2].

        Returns:
            Gravity-gradient torque in body frame [Nm] (3,).
        """
        r_mag = np.linalg.norm(r_body)
        if r_mag < 1.0:
            return np.zeros(3)

        r_hat = r_body / r_mag
        factor = 3.0 * mu / (r_mag ** 3)
        return factor * np.cross(r_hat, inertia @ r_hat)

    def compute_residual_magnetic_torque(self) -> np.ndarray:
        """
        Compute residual magnetic dipole torque.

        A spacecraft with magnetic dipole moment m in an external magnetic
        field B experiences a torque:

            T_mag = m x B

        This is typically a small but persistent disturbance (order of
        ~1e-5 to ~1e-3 Nm in LEO) from residual magnetization of structural
        elements, current loops in wiring harnesses, and electronic components.

        Returns:
            Magnetic disturbance torque in body frame [Nm] (3,).
        """
        m = self.config.residual_magnetic_moment
        B = self.config.magnetic_field_body
        return np.cross(m, B)

    def compute_cg_offset_torque(self, thrust_body: np.ndarray) -> np.ndarray:
        """
        Compute torque from thrust acting through an offset center of gravity.

        When the CG is not at the geometric center (where the thrust vector
        nominally passes), any thrust produces a parasitic torque:

            T_cg = cg_offset x F_thrust

        This is a major disturbance during powered flight and must be
        compensated by the attitude control system (e.g., TVC or RCS).

        Args:
            thrust_body: Thrust force vector in body frame [N] (3,).

        Returns:
            Parasitic torque from CG offset [Nm] (3,).
        """
        return np.cross(self.cg_offset, thrust_body)

    def compute_flex_torque(
        self,
        flex_state: np.ndarray,
        flex_params: Optional[List[FlexMode]] = None,
    ) -> np.ndarray:
        """
        Compute parasitic torque from structural flex modes.

        Each flex mode is a damped harmonic oscillator. The coupling between
        the flex displacement and the rigid body creates a feedback torque:

            T_flex_i = -L_i * (omega_n_i^2 * eta_i + 2*zeta_i*omega_n_i * eta_dot_i)

        where L_i is the coupling vector for mode i, eta_i is the modal
        displacement, and eta_dot_i is the modal velocity.

        This torque can excite controller-structure interaction if the control
        bandwidth overlaps with flex mode frequencies. It is the primary
        reason spacecraft controllers use notch filters and roll-off.

        Args:
            flex_state: Flex portion of state vector [eta0, eta0_dot, eta1, ...].
            flex_params: List of FlexMode objects. If None, uses self.flex_modes.

        Returns:
            Total parasitic torque from all flex modes [Nm] (3,).
        """
        if flex_params is None:
            flex_params = self.flex_modes

        torque = np.zeros(3)
        for i, mode in enumerate(flex_params):
            eta = flex_state[2 * i]
            eta_dot = flex_state[2 * i + 1]
            wn = mode.omega_n
            zeta = mode.damping_ratio
            L = np.array(mode.coupling_vector, dtype=np.float64)

            # Restoring + damping force fed back as torque
            torque -= L * (wn ** 2 * eta + 2.0 * zeta * wn * eta_dot)

        return torque

    # -------------------------------------------------------------------------
    # MOMENTUM BOOKKEEPING
    # -------------------------------------------------------------------------

    def get_total_angular_momentum(
        self,
        omega: np.ndarray,
        wheel_speeds: Optional[np.ndarray] = None,
        inertia: Optional[np.ndarray] = None,
        wheel_inertias: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute total system angular momentum (body + wheels).

        The total angular momentum is conserved in the absence of external
        torques (conservation of angular momentum):

            H_total = I * omega + sum_i (J_i * Omega_i * a_i)

        where J_i is the spin-axis inertia, Omega_i is the spin speed, and
        a_i is the spin axis unit vector of wheel i.

        Monitoring H_total is critical for:
          - Detecting wheel saturation (H_wheels growing, H_body shrinking)
          - Scheduling momentum desaturation maneuvers (magnetorquer or thruster)
          - Verifying simulation conservation laws

        Args:
            omega: Body angular velocity [rad/s] (3,).
            wheel_speeds: Array of wheel speeds [rad/s]. If None, uses
                          current wheel speeds from self.wheels.
            inertia: 3x3 inertia tensor. If None, uses self.inertia.
            wheel_inertias: Array of wheel spin-axis inertias [kg*m^2].
                           If None, uses inertias from self.wheels.

        Returns:
            Total angular momentum vector in body frame [Nms] (3,).
        """
        if inertia is None:
            inertia = self.inertia
        if wheel_speeds is None:
            wheel_speeds = np.array([w.speed for w in self.wheels])
        if wheel_inertias is None:
            wheel_inertias = np.array([w.inertia for w in self.wheels])

        # Body angular momentum
        h_body = inertia @ omega

        # Wheel angular momentum
        h_wheels = np.zeros(3)
        for i, wheel in enumerate(self.wheels):
            axis = np.array(wheel.spin_axis, dtype=np.float64)
            axis = axis / np.linalg.norm(axis)
            h_wheels += wheel_inertias[i] * wheel_speeds[i] * axis

        return h_body + h_wheels

    def check_wheel_saturation(self) -> List[bool]:
        """
        Check if any reaction wheels are near saturation.

        A wheel is considered near saturation if its speed exceeds 90%
        of its maximum rated speed. Saturated wheels can no longer absorb
        angular momentum and must be desaturated using an external torque
        (magnetorquers or thrusters).

        Returns:
            List of booleans, True if wheel i is near saturation.
        """
        return [
            abs(w.speed) > 0.9 * w.max_speed for w in self.wheels
        ]

    # -------------------------------------------------------------------------
    # INERTIA UPDATES
    # -------------------------------------------------------------------------

    def update_propellant_fraction(self, fraction: float) -> None:
        """
        Update the effective inertia tensor for propellant consumption.

        As propellant is consumed, the spacecraft mass decreases and the
        inertia tensor changes. The inertia is interpolated linearly between
        the fully-loaded and dry configurations. Products of inertia (off-
        diagonal terms) also shift as the CG moves.

        Args:
            fraction: Remaining propellant fraction [0.0, 1.0].
        """
        self._propellant_fraction = np.clip(fraction, 0.0, 1.0)

        # Simple linear interpolation: dry mass contributes ~60% of full inertia
        # (this ratio depends on spacecraft design; 0.6 is representative)
        dry_fraction = 0.6
        scale = dry_fraction + (1.0 - dry_fraction) * self._propellant_fraction
        self.inertia = scale * self._initial_inertia

        # Off-diagonal terms shift slightly with CG motion
        cg_shift = 0.01 * (1.0 - self._propellant_fraction)  # meters of CG travel
        mass_approx = np.trace(self.inertia) / 2.0  # rough mass estimate
        self.inertia[0, 1] += mass_approx * cg_shift * 0.001
        self.inertia[1, 0] = self.inertia[0, 1]  # maintain symmetry

        self.inertia_inv = np.linalg.inv(self.inertia)

    def get_perturbed_inertia(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Return a Monte-Carlo-perturbed inertia tensor for dispersion analysis.

        Each element of the inertia tensor is perturbed by a Gaussian random
        variable with standard deviation equal to the configured uncertainty
        fraction. The result is symmetrized to remain physically valid.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Perturbed 3x3 inertia tensor [kg*m^2].
        """
        rng = np.random.default_rng(seed)
        sigma = self.config.inertia_uncertainty
        perturbation = rng.normal(0.0, sigma, size=(3, 3))
        I_perturbed = self.inertia * (1.0 + perturbation)
        # Symmetrize
        I_perturbed = 0.5 * (I_perturbed + I_perturbed.T)
        return I_perturbed

    # -------------------------------------------------------------------------
    # EQUATIONS OF MOTION
    # -------------------------------------------------------------------------

    def _state_derivative(
        self,
        state: np.ndarray,
        external_torque: np.ndarray,
        thrust_body: np.ndarray,
        r_body_for_gg: Optional[np.ndarray] = None,
        mu: float = EARTH_MU,
    ) -> np.ndarray:
        """
        Compute time derivative of the full attitude state vector.

        This evaluates the right-hand side of the coupled ODEs:

        1. Quaternion kinematics:
            q_dot = 0.5 * Omega(omega) * q

        2. Euler's equation (rigid body + flex feedback):
            omega_dot = I^{-1} * [-omega x (I*omega) + T_ext + T_dist]

        3. Flex mode oscillators (for each mode i):
            eta_ddot = -omega_n^2 * eta - 2*zeta*omega_n*eta_dot + L^T * omega_dot

        Args:
            state: Full state vector (state_size,).
            external_torque: Commanded control torque in body frame [Nm] (3,).
            thrust_body: Thrust vector in body frame [N] (3,) for CG offset torque.
            r_body_for_gg: Position vector in body frame [m] (3,) for gravity
                           gradient. None to skip gravity gradient.
            mu: Gravitational parameter [m^3/s^2] for gravity gradient.

        Returns:
            State derivative vector (state_size,).
        """
        q, omega, flex = self._unpack_state(state)

        # --- Disturbance torques ---
        T_dist = np.zeros(3)

        # Gravity gradient torque
        if r_body_for_gg is not None:
            T_dist += self.compute_gravity_gradient_torque(r_body_for_gg, self.inertia, mu)

        # CG offset torque from thrust
        if np.linalg.norm(thrust_body) > 0.0:
            T_dist += self.compute_cg_offset_torque(thrust_body)

        # Residual magnetic torque
        T_dist += self.compute_residual_magnetic_torque()

        # Structural flex parasitic torque
        if self.num_flex_modes > 0:
            T_dist += self.compute_flex_torque(flex)

        # --- Euler's equation ---
        # I * omega_dot = -omega x (I*omega) + T_external + T_disturbance
        I_omega = self.inertia @ omega
        gyroscopic = np.cross(omega, I_omega)
        omega_dot = self.inertia_inv @ (-gyroscopic + external_torque + T_dist)

        # --- Quaternion kinematics ---
        q_dot = self.euler_to_quaternion_derivative(q, omega)

        # --- Flex mode dynamics ---
        flex_dot = np.zeros(2 * self.num_flex_modes)
        for i, mode in enumerate(self.flex_modes):
            eta = flex[2 * i]
            eta_dot_val = flex[2 * i + 1]
            wn = mode.omega_n
            zeta = mode.damping_ratio
            L = np.array(mode.coupling_vector, dtype=np.float64)

            # eta_dot = eta_dot  (position derivative = velocity)
            flex_dot[2 * i] = eta_dot_val

            # eta_ddot = -wn^2 * eta - 2*zeta*wn*eta_dot + L^T * omega_dot
            eta_ddot = -wn ** 2 * eta - 2.0 * zeta * wn * eta_dot_val + L @ omega_dot
            flex_dot[2 * i + 1] = eta_ddot

        # --- Assemble full derivative ---
        x_dot = np.zeros(self.state_size)
        x_dot[self.Q_START:self.Q_END] = q_dot
        x_dot[self.W_START:self.W_END] = omega_dot
        x_dot[self.FLEX_START:] = flex_dot

        return x_dot

    # -------------------------------------------------------------------------
    # INTEGRATION
    # -------------------------------------------------------------------------

    def propagate(
        self,
        state: np.ndarray,
        dt: float,
        external_torque: np.ndarray = None,
        thrust_body: np.ndarray = None,
        r_body_for_gg: Optional[np.ndarray] = None,
        mu: float = EARTH_MU,
    ) -> np.ndarray:
        """
        Propagate the attitude state forward by one timestep using RK4.

        The classical 4th-order Runge-Kutta method is used for its balance
        of accuracy and computational cost. After integration, the quaternion
        is re-normalized to remain on the unit sphere (numerical drift from
        integration would otherwise cause the quaternion norm to grow).

        Args:
            state: Current state vector (state_size,).
            dt: Timestep [s].
            external_torque: Control torque in body frame [Nm] (3,).
                            Defaults to zero.
            thrust_body: Thrust vector in body frame [N] (3,).
                        Defaults to zero.
            r_body_for_gg: Position in body frame for gravity gradient [m] (3,).
                          Defaults to None (no gravity gradient).
            mu: Gravitational parameter [m^3/s^2].

        Returns:
            Propagated state vector at t + dt (state_size,).
        """
        if external_torque is None:
            external_torque = np.zeros(3)
        if thrust_body is None:
            thrust_body = np.zeros(3)

        # RK4 integration
        def f(s):
            return self._state_derivative(s, external_torque, thrust_body,
                                          r_body_for_gg, mu)

        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)

        state_new = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Re-normalize quaternion to unit magnitude
        q = state_new[self.Q_START:self.Q_END]
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-10:
            state_new[self.Q_START:self.Q_END] = q / q_norm
        else:
            # Degenerate quaternion -- reset to identity
            state_new[self.Q_START:self.Q_END] = np.array([0.0, 0.0, 0.0, 1.0])

        return state_new

    def propagate_interval(
        self,
        state: np.ndarray,
        t_span: float,
        dt: float,
        external_torque: np.ndarray = None,
        thrust_body: np.ndarray = None,
        r_body_for_gg: Optional[np.ndarray] = None,
        mu: float = EARTH_MU,
        record_history: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Propagate over a time interval with fixed timestep.

        Args:
            state: Initial state vector.
            t_span: Total propagation time [s].
            dt: Integration timestep [s].
            external_torque: Control torque [Nm] (3,). Held constant.
            thrust_body: Thrust vector [N] (3,). Held constant.
            r_body_for_gg: Position for gravity gradient [m] (3,).
            mu: Gravitational parameter [m^3/s^2].
            record_history: If True, return full state history.

        Returns:
            (final_state, history) where history is (N, state_size) or None.
        """
        n_steps = int(np.ceil(t_span / dt))
        current = state.copy()

        if record_history:
            history = np.zeros((n_steps + 1, self.state_size))
            history[0] = current

        for step in range(n_steps):
            step_dt = min(dt, t_span - step * dt)
            current = self.propagate(
                current, step_dt, external_torque, thrust_body,
                r_body_for_gg, mu,
            )
            if record_history:
                history[step + 1] = current

        if record_history:
            return current, history
        return current, None

    # -------------------------------------------------------------------------
    # UTILITY
    # -------------------------------------------------------------------------

    @staticmethod
    def quaternion_to_dcm(q: np.ndarray) -> np.ndarray:
        """
        Convert unit quaternion (scalar-last) to direction cosine matrix.

        Args:
            q: Quaternion [qx, qy, qz, qw] (4,).

        Returns:
            3x3 rotation matrix (body-from-inertial).
        """
        qx, qy, qz, qw = q / np.linalg.norm(q)

        return np.array([
            [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),    2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),      1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)],
        ])

    @staticmethod
    def angular_velocity_from_quaternions(
        q1: np.ndarray, q2: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        Estimate body angular velocity from two quaternion samples.

        Uses the finite-difference approximation:
            omega ~ 2 * (q2 * q1_conj).vec / dt

        This is accurate for small rotation angles (high sample rates).

        Args:
            q1: Quaternion at time t (4,).
            q2: Quaternion at time t+dt (4,).
            dt: Time between samples [s].

        Returns:
            Estimated angular velocity in body frame [rad/s] (3,).
        """
        # Conjugate of q1 (negate vector part for scalar-last)
        q1_conj = np.array([-q1[0], -q1[1], -q1[2], q1[3]])

        # Quaternion multiplication: q_delta = q2 * q1_conj
        # Using scalar-last convention
        v1 = q1_conj[:3]
        s1 = q1_conj[3]
        v2 = q2[:3]
        s2 = q2[3]
        q_delta = np.zeros(4)
        q_delta[:3] = s2 * v1 + s1 * v2 + np.cross(v2, v1)
        q_delta[3] = s2 * s1 - np.dot(v2, v1)

        # omega ~ 2 * vector_part / dt
        return 2.0 * q_delta[:3] / dt

    def energy(self, state: np.ndarray) -> float:
        """
        Compute rotational kinetic energy of the spacecraft.

            T = 0.5 * omega^T * I * omega

        Args:
            state: Full state vector.

        Returns:
            Rotational kinetic energy [J].
        """
        _, omega, _ = self._unpack_state(state)
        return 0.5 * omega @ self.inertia @ omega

    def __repr__(self) -> str:
        return (
            f"AttitudeDynamics("
            f"state_size={self.state_size}, "
            f"flex_modes={self.num_flex_modes}, "
            f"wheels={len(self.wheels)}, "
            f"Ixx={self.inertia[0,0]:.1f}, "
            f"Iyy={self.inertia[1,1]:.1f}, "
            f"Izz={self.inertia[2,2]:.1f})"
        )
