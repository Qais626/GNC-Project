"""
===============================================================================
GNC PROJECT - Actuator Models with Realistic Physics
===============================================================================
High-fidelity models for spacecraft actuators including reaction wheels,
control moment gyroscopes (CMGs), and thrusters. Each model captures the
dominant physics: saturation limits, friction, noise, quantization, minimum
impulse bits, rise/tail-off transients, and misalignment errors.

All units are SI: Newtons, meters, seconds, kilograms, radians.
===============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# REACTION WHEEL
# =============================================================================

class ReactionWheel:
    """
    Single reaction wheel actuator with realistic physics.

    The wheel spins about a fixed body-frame axis and exchanges angular
    momentum with the spacecraft.  The model includes:

      - Torque saturation at +/- max_torque
      - Momentum saturation at +/- max_momentum
      - Bearing friction composed of Coulomb (constant opposing sign) and
        viscous (proportional to speed) terms
      - Torque quantization (finite command resolution)
      - Additive Gaussian torque noise

    State variables:
        current_speed       -- wheel spin rate (rad/s)
        accumulated_momentum -- I_w * current_speed (N m s)

    Parameters
    ----------
    max_torque : float
        Maximum deliverable torque magnitude (N m).
    max_momentum : float
        Maximum storable angular momentum magnitude (N m s).
    wheel_inertia : float
        Rotor moment of inertia about spin axis (kg m^2).
    friction_coulomb : float
        Coulomb (dry) friction torque magnitude (N m).
    friction_viscous : float
        Viscous friction coefficient (N m s / rad).
    noise_std : float
        1-sigma torque noise (N m).
    quantization_step : float
        Minimum torque command resolution (N m).  Commands are rounded to
        the nearest multiple of this value.
    spin_axis : np.ndarray
        Unit vector of the wheel spin axis in body frame (3,).
    """

    def __init__(
        self,
        max_torque: float = 0.2,
        max_momentum: float = 50.0,
        wheel_inertia: float = 0.05,
        friction_coulomb: float = 1.0e-4,
        friction_viscous: float = 5.0e-6,
        noise_std: float = 1.0e-5,
        quantization_step: float = 1.0e-4,
        spin_axis: Optional[np.ndarray] = None,
    ):
        self.max_torque = max_torque
        self.max_momentum = max_momentum
        self.wheel_inertia = wheel_inertia
        self.friction_coulomb = friction_coulomb
        self.friction_viscous = friction_viscous
        self.noise_std = noise_std
        self.quantization_step = quantization_step

        if spin_axis is not None:
            self.spin_axis = np.array(spin_axis, dtype=float)
            self.spin_axis /= np.linalg.norm(self.spin_axis)
        else:
            self.spin_axis = np.array([0.0, 0.0, 1.0])

        # State
        self.current_speed = 0.0  # rad/s
        self.accumulated_momentum = 0.0  # N m s

    # -----------------------------------------------------------------
    def command_torque(self, torque_cmd: float) -> float:
        """
        Apply a torque command and return the actual delivered torque.

        Processing pipeline:
          1. Quantize the command to the nearest resolution step.
          2. Saturate at +/- max_torque.
          3. Subtract bearing friction (Coulomb + viscous).
          4. Add Gaussian measurement/execution noise.
          5. Check momentum saturation -- clamp if at limit.

        Parameters
        ----------
        torque_cmd : float
            Desired torque (N m).  Positive accelerates wheel in the
            positive spin direction.

        Returns
        -------
        float
            Actual torque delivered to the spacecraft (equal and opposite
            to the torque on the wheel).
        """
        # 1. Quantize
        if self.quantization_step > 0:
            torque_cmd = (
                round(torque_cmd / self.quantization_step) * self.quantization_step
            )

        # 2. Saturate
        torque_cmd = np.clip(torque_cmd, -self.max_torque, self.max_torque)

        # 3. Bearing friction: opposes wheel motion
        friction = 0.0
        if abs(self.current_speed) > 1.0e-10:
            sign_omega = np.sign(self.current_speed)
            friction = (
                self.friction_coulomb * sign_omega
                + self.friction_viscous * self.current_speed
            )
        # Friction reduces the torque that actually reaches the spacecraft.
        # The motor must overcome friction first.
        actual_torque = torque_cmd - friction

        # 4. Noise
        actual_torque += np.random.normal(0.0, self.noise_std)

        # 5. Momentum saturation guard
        prospective_momentum = self.accumulated_momentum + actual_torque * 1.0
        if abs(prospective_momentum) > self.max_momentum:
            # Clamp: only allow torque that keeps us at the limit
            if prospective_momentum > 0:
                actual_torque = max(0.0, self.max_momentum - self.accumulated_momentum)
            else:
                actual_torque = min(0.0, -self.max_momentum - self.accumulated_momentum)

        return actual_torque

    # -----------------------------------------------------------------
    def get_momentum(self) -> float:
        """
        Return the current angular momentum stored in the wheel.

        h = I_w * omega_w

        Returns
        -------
        float
            Angular momentum (N m s).
        """
        self.accumulated_momentum = self.wheel_inertia * self.current_speed
        return self.accumulated_momentum

    # -----------------------------------------------------------------
    def is_saturated(self) -> bool:
        """
        Check whether the wheel has reached its momentum storage limit.

        Returns
        -------
        bool
            True if |h| >= max_momentum.
        """
        return abs(self.get_momentum()) >= self.max_momentum

    # -----------------------------------------------------------------
    def reset_speed(self, new_speed: float = 0.0) -> None:
        """
        Set the wheel speed to a new value (e.g. after a desaturation
        maneuver using external torque from thrusters).

        Parameters
        ----------
        new_speed : float
            New wheel spin rate (rad/s).
        """
        self.current_speed = new_speed
        self.accumulated_momentum = self.wheel_inertia * new_speed

    # -----------------------------------------------------------------
    def update_state(self, actual_torque: float, dt: float) -> None:
        """
        Integrate wheel speed forward by dt given an applied torque.

        Parameters
        ----------
        actual_torque : float
            Net torque on the wheel (N m).
        dt : float
            Time step (s).
        """
        angular_accel = actual_torque / self.wheel_inertia
        self.current_speed += angular_accel * dt
        self.accumulated_momentum = self.wheel_inertia * self.current_speed

    def __repr__(self) -> str:
        return (
            f"ReactionWheel(speed={self.current_speed:.2f} rad/s, "
            f"h={self.get_momentum():.4f} Nms, "
            f"saturated={self.is_saturated()})"
        )


# =============================================================================
# REACTION WHEEL ARRAY (4-wheel pyramid)
# =============================================================================

class ReactionWheelArray:
    """
    Four reaction wheels arranged in a pyramid configuration.

    Three wheels are aligned roughly with the body X, Y, Z axes and a
    fourth is canted equally toward all three, providing single-fault
    tolerance.  A distribution (torque allocation) matrix maps a desired
    3-axis torque command into four individual wheel torque commands using
    a pseudo-inverse.

    Pyramid geometry
    ----------------
    Wheel 1 spin axis:  close to +X, canted by beta toward +Z
    Wheel 2 spin axis:  close to +Y, canted by beta toward +Z
    Wheel 3 spin axis:  close to -X, canted by beta toward +Z
    Wheel 4 spin axis:  (skew) equally canted toward +X, +Y, +Z

    The cant angle beta defaults to ~20 deg for good redundancy coverage.

    Parameters
    ----------
    max_torque : float
        Per-wheel maximum torque (N m).
    max_momentum : float
        Per-wheel maximum momentum (N m s).
    wheel_inertia : float
        Per-wheel rotor inertia (kg m^2).
    beta_deg : float
        Pyramid cant angle (degrees).
    """

    def __init__(
        self,
        max_torque: float = 0.2,
        max_momentum: float = 50.0,
        wheel_inertia: float = 0.05,
        beta_deg: float = 20.0,
    ):
        beta = np.radians(beta_deg)
        cb, sb = np.cos(beta), np.sin(beta)

        # Spin-axis unit vectors in body frame (columns of the distribution matrix)
        axes = np.array([
            [cb,  0.0, -cb, sb],   # X components
            [0.0, cb,   0.0, sb],  # Y components
            [sb,  sb,   sb,  sb],  # Z components
        ])
        # Normalize each column
        for j in range(4):
            axes[:, j] /= np.linalg.norm(axes[:, j])

        self._distribution_matrix = axes  # 3x4
        # Pseudo-inverse for torque allocation: (4x3)
        self._allocation_matrix = np.linalg.pinv(axes)

        # Create four individual wheels
        self.wheels = []
        for j in range(4):
            self.wheels.append(
                ReactionWheel(
                    max_torque=max_torque,
                    max_momentum=max_momentum,
                    wheel_inertia=wheel_inertia,
                    spin_axis=axes[:, j],
                )
            )

    # -----------------------------------------------------------------
    def command_torque_3axis(self, torque_cmd_3: np.ndarray) -> np.ndarray:
        """
        Distribute a 3-axis torque command to the four wheels and apply.

        Uses the pseudo-inverse of the distribution matrix to compute
        individual wheel torques, then passes each through the wheel
        model (saturation, friction, noise).

        Parameters
        ----------
        torque_cmd_3 : np.ndarray
            Desired body-frame torque vector [Tx, Ty, Tz] (N m).

        Returns
        -------
        np.ndarray
            Actual torques delivered by each wheel (4,) (N m).
        """
        torque_cmd_3 = np.asarray(torque_cmd_3, dtype=float)
        wheel_cmds = self._allocation_matrix @ torque_cmd_3  # (4,)

        actual = np.zeros(4)
        for j, wheel in enumerate(self.wheels):
            actual[j] = wheel.command_torque(wheel_cmds[j])
        return actual

    # -----------------------------------------------------------------
    def get_total_momentum(self) -> np.ndarray:
        """
        Return the total angular momentum stored in the array, projected
        onto body-frame axes.

        Returns
        -------
        np.ndarray
            3-vector of angular momentum [hx, hy, hz] (N m s).
        """
        h_total = np.zeros(3)
        for j, wheel in enumerate(self.wheels):
            h_total += wheel.get_momentum() * wheel.spin_axis
        return h_total

    # -----------------------------------------------------------------
    def is_any_saturated(self) -> bool:
        """
        Check whether any wheel in the array has reached momentum
        saturation.

        Returns
        -------
        bool
            True if at least one wheel is saturated.
        """
        return any(w.is_saturated() for w in self.wheels)

    # -----------------------------------------------------------------
    def desaturate(self, external_torque: np.ndarray, dt: float) -> None:
        """
        Reduce wheel speeds using an externally supplied torque (e.g.
        from thrusters or magnetic torquers).

        The external torque is applied to the spacecraft; the controller
        simultaneously commands wheel torque in the opposite direction so
        that the net spacecraft torque is zero while the wheels slow down.

        Parameters
        ----------
        external_torque : np.ndarray
            Torque supplied by thrusters / magnetorquers in body frame (N m).
        dt : float
            Time step (s).
        """
        external_torque = np.asarray(external_torque, dtype=float)
        # Desired wheel deceleration torque = -external_torque projected onto wheels
        desat_cmds = -self._allocation_matrix @ external_torque
        for j, wheel in enumerate(self.wheels):
            wheel.update_state(desat_cmds[j], dt)

    def __repr__(self) -> str:
        h = self.get_total_momentum()
        return (
            f"ReactionWheelArray(h_total=[{h[0]:.3f}, {h[1]:.3f}, {h[2]:.3f}] Nms, "
            f"any_saturated={self.is_any_saturated()})"
        )


# =============================================================================
# CONTROL MOMENT GYROSCOPE (single-gimbal)
# =============================================================================

class CMG:
    """
    Single-gimbal Control Moment Gyroscope (CMG).

    A CMG stores angular momentum in a constant-speed rotor and produces
    output torque by rotating (gimbaling) the rotor's spin axis.  The
    output torque is the cross product of the rotor angular momentum
    vector and the gimbal rate vector:

        tau = h_rotor x gimbal_rate_direction * gimbal_rate

    Near certain gimbal configurations the Jacobian (mapping from gimbal
    rates to output torques) becomes singular -- the dreaded "CMG
    singularity".  The singularity measure is the determinant (or
    minimum singular value) of the Jacobian.

    Parameters
    ----------
    rotor_momentum : float
        Constant angular momentum magnitude of the rotor (N m s).
    max_gimbal_rate : float
        Maximum gimbal angular rate (rad/s).
    gimbal_axis : np.ndarray
        Unit vector of the gimbal rotation axis in body frame (3,).
    initial_gimbal_angle : float
        Starting gimbal angle (rad).
    gimbal_friction : float
        Viscous friction coefficient on the gimbal bearing (N m s / rad).
    gimbal_noise_std : float
        1-sigma noise on gimbal rate execution (rad/s).
    """

    def __init__(
        self,
        rotor_momentum: float = 100.0,
        max_gimbal_rate: float = 1.0,
        gimbal_axis: Optional[np.ndarray] = None,
        initial_gimbal_angle: float = 0.0,
        gimbal_friction: float = 0.001,
        gimbal_noise_std: float = 1.0e-4,
    ):
        self.rotor_momentum = rotor_momentum
        self.max_gimbal_rate = max_gimbal_rate
        self.gimbal_friction = gimbal_friction
        self.gimbal_noise_std = gimbal_noise_std

        if gimbal_axis is not None:
            self.gimbal_axis = np.array(gimbal_axis, dtype=float)
            self.gimbal_axis /= np.linalg.norm(self.gimbal_axis)
        else:
            self.gimbal_axis = np.array([0.0, 1.0, 0.0])

        self.gimbal_angle = initial_gimbal_angle

        # The rotor spin axis is perpendicular to the gimbal axis and
        # rotates as the gimbal turns.  At gimbal_angle = 0 the spin axis
        # is along the third orthogonal direction.
        self._compute_rotor_axis()

    # -----------------------------------------------------------------
    def _compute_rotor_axis(self) -> None:
        """Recompute the rotor spin-axis direction from the current gimbal angle."""
        g = self.gimbal_axis
        # Build an orthonormal frame: g, e1, e2
        if abs(g[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(g, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(g, e1)
        e2 /= np.linalg.norm(e2)

        # Rotor axis rotates in the plane (e1, e2) with gimbal angle
        self.rotor_axis = (
            np.cos(self.gimbal_angle) * e1 + np.sin(self.gimbal_angle) * e2
        )

    # -----------------------------------------------------------------
    def command_gimbal_rate(self, rate_cmd: float) -> float:
        """
        Command a gimbal rate and return the actual achieved rate.

        Parameters
        ----------
        rate_cmd : float
            Desired gimbal angular rate (rad/s).

        Returns
        -------
        float
            Actual gimbal rate after saturation, friction, and noise.
        """
        # Saturate
        rate = np.clip(rate_cmd, -self.max_gimbal_rate, self.max_gimbal_rate)
        # Viscous friction opposes gimbal motion
        rate -= self.gimbal_friction * rate
        # Noise
        rate += np.random.normal(0.0, self.gimbal_noise_std)
        return rate

    # -----------------------------------------------------------------
    def get_output_torque(self, gimbal_rate: float) -> np.ndarray:
        """
        Compute the torque produced by the CMG at the given gimbal rate.

        The torque is:
            tau = d/dt (h_rotor_vector) = h_rotor * (gimbal_axis x rotor_axis) * gimbal_rate

        which simplifies to:
            tau = h_rotor * gimbal_rate * (gimbal_axis x rotor_axis)

        Parameters
        ----------
        gimbal_rate : float
            Current gimbal angular rate (rad/s).

        Returns
        -------
        np.ndarray
            Output torque vector in body frame (N m) (3,).
        """
        cross = np.cross(self.gimbal_axis, self.rotor_axis)
        return self.rotor_momentum * gimbal_rate * cross

    # -----------------------------------------------------------------
    def get_singularity_measure(self) -> float:
        """
        Compute a scalar measure of proximity to a CMG singularity.

        For a single CMG the measure is the magnitude of the cross
        product (gimbal_axis x rotor_axis).  When this approaches zero
        the CMG cannot produce torque in certain directions -- it is at
        a singular configuration.

        Returns
        -------
        float
            Singularity measure in [0, 1].  Values < 0.1 indicate
            proximity to singularity.
        """
        cross = np.cross(self.gimbal_axis, self.rotor_axis)
        return np.linalg.norm(cross)

    # -----------------------------------------------------------------
    def update_state(self, gimbal_rate: float, dt: float) -> None:
        """
        Integrate the gimbal angle forward by dt.

        Parameters
        ----------
        gimbal_rate : float
            Current gimbal rate (rad/s).
        dt : float
            Time step (s).
        """
        self.gimbal_angle += gimbal_rate * dt
        self._compute_rotor_axis()

    def __repr__(self) -> str:
        return (
            f"CMG(gimbal_angle={np.degrees(self.gimbal_angle):.1f} deg, "
            f"singularity={self.get_singularity_measure():.3f})"
        )


# =============================================================================
# THRUSTER (single on/off thruster)
# =============================================================================

class Thruster:
    """
    On/off thruster with minimum impulse bit and realistic transients.

    Models:
      - Finite rise time: thrust ramps from zero to full in t_rise.
      - Tail-off: thrust decays exponentially after valve close.
      - Thrust uncertainty: actual thrust = nominal * (1 + bias + noise).
      - Angular misalignment: thrust direction is offset from nominal.
      - Minimum impulse bit: shortest allowable firing duration.

    Parameters
    ----------
    nominal_thrust : float
        Steady-state thrust level (N).
    isp : float
        Specific impulse (s).
    min_impulse_bit : float
        Minimum on-time (s).
    rise_time : float
        Time for thrust to ramp from 0 to nominal (s).
    tail_off_tau : float
        Exponential decay time constant after shutoff (s).
    thrust_bias : float
        Fractional systematic thrust error (dimensionless, e.g. 0.01 = 1 %).
    thrust_noise_std : float
        1-sigma fractional random thrust error (dimensionless).
    misalignment_deg : float
        1-sigma angular misalignment of thrust vector (degrees).
    thrust_direction : np.ndarray
        Nominal thrust unit vector in body frame (3,).
    position : np.ndarray
        Thruster position relative to spacecraft center of mass (m) (3,).
    """

    G0 = 9.80665  # standard gravity (m/s^2) for Isp conversion

    def __init__(
        self,
        nominal_thrust: float = 22.0,
        isp: float = 230.0,
        min_impulse_bit: float = 0.020,
        rise_time: float = 0.005,
        tail_off_tau: float = 0.010,
        thrust_bias: float = 0.0,
        thrust_noise_std: float = 0.01,
        misalignment_deg: float = 0.1,
        thrust_direction: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
    ):
        self.nominal_thrust = nominal_thrust
        self.isp = isp
        self.min_impulse_bit = min_impulse_bit
        self.rise_time = rise_time
        self.tail_off_tau = tail_off_tau
        self.thrust_bias = thrust_bias
        self.thrust_noise_std = thrust_noise_std
        self.misalignment_rad = np.radians(misalignment_deg)

        if thrust_direction is not None:
            self.thrust_direction = np.array(thrust_direction, dtype=float)
            self.thrust_direction /= np.linalg.norm(self.thrust_direction)
        else:
            self.thrust_direction = np.array([1.0, 0.0, 0.0])

        if position is not None:
            self.position = np.array(position, dtype=float)
        else:
            self.position = np.zeros(3)

        # Derived
        self.mass_flow_rate = nominal_thrust / (isp * self.G0)  # kg/s

        # Running total of propellant consumed
        self.total_propellant_used = 0.0  # kg

    # -----------------------------------------------------------------
    def fire(self, duration: float) -> np.ndarray:
        """
        Fire the thruster for the given duration and return the actual
        impulse vector delivered (body frame).

        The impulse accounts for:
          - Minimum impulse bit enforcement (duration is clamped upward).
          - Rise-time and tail-off transients.
          - Thrust magnitude uncertainty (bias + noise).
          - Thrust direction misalignment.

        Parameters
        ----------
        duration : float
            Commanded on-time (s).  Will be increased to min_impulse_bit
            if shorter.

        Returns
        -------
        np.ndarray
            Actual impulse vector (N s) in body frame (3,).
        """
        # Enforce minimum impulse bit
        duration = max(duration, self.min_impulse_bit)

        # Effective impulse accounting for rise and tail-off transients.
        # During rise-time the average thrust is about half nominal.
        effective_on_time = duration
        if duration > self.rise_time:
            effective_on_time = duration - 0.5 * self.rise_time
        else:
            effective_on_time = 0.5 * duration  # still ramping

        # Tail-off adds a small residual impulse ~ F * tau
        tail_off_impulse = self.nominal_thrust * self.tail_off_tau

        # Thrust magnitude with uncertainty
        thrust_scale = 1.0 + self.thrust_bias + np.random.normal(
            0.0, self.thrust_noise_std
        )
        actual_thrust = self.nominal_thrust * thrust_scale

        # Total impulse magnitude
        impulse_mag = actual_thrust * effective_on_time + tail_off_impulse

        # Misaligned direction
        direction = self._apply_misalignment(self.thrust_direction)

        # Track propellant
        self.total_propellant_used += self.get_propellant_consumed(duration)

        return impulse_mag * direction

    # -----------------------------------------------------------------
    def get_propellant_consumed(self, duration: float) -> float:
        """
        Mass of propellant consumed for a burn of the given duration.

        Parameters
        ----------
        duration : float
            Burn duration (s).

        Returns
        -------
        float
            Propellant mass consumed (kg).
        """
        return self.mass_flow_rate * max(duration, self.min_impulse_bit)

    # -----------------------------------------------------------------
    def _apply_misalignment(self, direction: np.ndarray) -> np.ndarray:
        """
        Perturb the thrust direction by a small random misalignment.

        Parameters
        ----------
        direction : np.ndarray
            Nominal thrust unit vector (3,).

        Returns
        -------
        np.ndarray
            Perturbed unit vector (3,).
        """
        # Generate two small-angle perturbations perpendicular to direction
        angle1 = np.random.normal(0.0, self.misalignment_rad)
        angle2 = np.random.normal(0.0, self.misalignment_rad)

        # Build perpendicular basis
        if abs(direction[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        perp1 = np.cross(direction, ref)
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)

        perturbed = direction + angle1 * perp1 + angle2 * perp2
        return perturbed / np.linalg.norm(perturbed)

    # -----------------------------------------------------------------
    def get_torque_about_com(self) -> np.ndarray:
        """
        Return the torque produced about the spacecraft center of mass
        for a unit-thrust firing: tau = r x F_hat.

        Returns
        -------
        np.ndarray
            Torque per unit thrust (N m / N) (3,).
        """
        return np.cross(self.position, self.thrust_direction)

    def __repr__(self) -> str:
        return (
            f"Thruster(F={self.nominal_thrust:.1f} N, "
            f"Isp={self.isp:.0f} s, "
            f"propellant_used={self.total_propellant_used:.4f} kg)"
        )


# =============================================================================
# THRUSTER ARRAY (16 thrusters for 6-DOF control)
# =============================================================================

class ThrusterArray:
    """
    Sixteen thrusters arranged for full 6-DOF control (3 force + 3 torque).

    The configuration uses four pods of four thrusters each, located at
    the +X, -X, +Y, -Y faces of the spacecraft bus.  Within each pod
    two thrusters point along +/- the face-normal direction (for
    translation) and two point tangentially (for rotation).

    An allocation matrix (6 x 16) maps the desired 6-DOF wrench
    [Fx, Fy, Fz, Tx, Ty, Tz] to 16 thruster on-times.  The allocation
    is computed via a non-negative least-squares (NNLS) approach since
    thrusters can only push, not pull.

    Parameters
    ----------
    nominal_thrust : float
        Per-thruster nominal thrust (N).
    isp : float
        Per-thruster specific impulse (s).
    arm_length : float
        Distance from spacecraft center of mass to thruster pod (m).
    """

    def __init__(
        self,
        nominal_thrust: float = 22.0,
        isp: float = 230.0,
        arm_length: float = 1.0,
    ):
        self.nominal_thrust = nominal_thrust
        self.arm_length = arm_length

        # Build thruster configuration
        self.thrusters = []
        self._build_configuration(nominal_thrust, isp, arm_length)
        self._build_allocation_matrix()

    # -----------------------------------------------------------------
    def _build_configuration(
        self, thrust: float, isp: float, arm: float
    ) -> None:
        """
        Create 16 thrusters in 4 pods of 4.

        Pod layout (body frame):
            Pod 0: position +X face, thrusters along +X, -X, +Y, -Y
            Pod 1: position -X face, thrusters along +X, -X, +Y, -Y
            Pod 2: position +Y face, thrusters along +Y, -Y, +X, -X
            Pod 3: position -Y face, thrusters along +Y, -Y, +X, -X
        """
        pod_configs = [
            # (position,                    thrust_directions)
            (np.array([arm, 0, 0]),   [[ 1, 0, 0], [-1, 0, 0], [0, 1, 0], [0,-1, 0]]),
            (np.array([-arm, 0, 0]),  [[ 1, 0, 0], [-1, 0, 0], [0, 1, 0], [0,-1, 0]]),
            (np.array([0, arm, 0]),   [[ 0, 1, 0], [0,-1, 0],  [1, 0, 0], [-1, 0, 0]]),
            (np.array([0,-arm, 0]),   [[ 0, 1, 0], [0,-1, 0],  [1, 0, 0], [-1, 0, 0]]),
        ]

        for pos, dirs in pod_configs:
            for d in dirs:
                self.thrusters.append(
                    Thruster(
                        nominal_thrust=thrust,
                        isp=isp,
                        thrust_direction=np.array(d, dtype=float),
                        position=pos.copy(),
                    )
                )

    # -----------------------------------------------------------------
    def _build_allocation_matrix(self) -> None:
        """
        Build the 6 x 16 wrench matrix B where column j is the unit
        wrench produced by thruster j:

            B[:, j] = [f_j;  r_j x f_j]

        The allocation is: on_times = pinv(B) * wrench_desired / F_nom
        """
        n = len(self.thrusters)
        B = np.zeros((6, n))
        for j, thr in enumerate(self.thrusters):
            f = thr.thrust_direction  # unit force
            tau = np.cross(thr.position, f)  # unit torque
            B[:3, j] = f
            B[3:, j] = tau

        self._wrench_matrix = B
        # Pseudo-inverse for allocation (will be clamped to non-negative)
        self._allocation_pinv = np.linalg.pinv(B)

    # -----------------------------------------------------------------
    def command_force_torque(
        self, force_3: np.ndarray, torque_3: np.ndarray
    ) -> np.ndarray:
        """
        Allocate a desired force and torque to thruster on-times and fire.

        The allocation solves for on-times proportional to the desired
        wrench, clamped to non-negative values (thrusters push only),
        and scaled so that each thruster fires at its nominal thrust.

        Parameters
        ----------
        force_3 : np.ndarray
            Desired force in body frame (N) (3,).
        torque_3 : np.ndarray
            Desired torque in body frame (N m) (3,).

        Returns
        -------
        np.ndarray
            Thruster on-times (s) for each of 16 thrusters (16,).
        """
        force_3 = np.asarray(force_3, dtype=float)
        torque_3 = np.asarray(torque_3, dtype=float)
        wrench = np.concatenate([force_3, torque_3])

        # Compute raw on-time fractions
        raw = self._allocation_pinv @ wrench / self.nominal_thrust
        # Clamp to non-negative (thrusters cannot produce negative thrust)
        on_times = np.clip(raw, 0.0, None)

        return on_times

    # -----------------------------------------------------------------
    def get_total_propellant_used(self) -> float:
        """
        Sum propellant consumed across all thrusters.

        Returns
        -------
        float
            Total propellant mass consumed (kg).
        """
        return sum(t.total_propellant_used for t in self.thrusters)

    def __repr__(self) -> str:
        return (
            f"ThrusterArray(n={len(self.thrusters)}, "
            f"propellant_used={self.get_total_propellant_used():.3f} kg)"
        )
