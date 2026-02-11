"""
===============================================================================
GNC PROJECT - Signal Model for Deep Space Communication
===============================================================================

Models the real-world imperfections of spacecraft-to-ground communication links
for a deep-space mission (Earth -> Jupiter and back). In an actual mission,
communication is never perfect: signals weaken with distance, experience
random dropouts, suffer from light-time delay, and degrade during planetary
reentry. This module quantifies all of those effects.

Key Models
----------
1. **SNR (Signal-to-Noise Ratio)**: Free-space path loss model based on the
   Friis transmission equation. Accounts for transmit power, antenna gains,
   frequency, and distance. At Jupiter distances (~5 AU), the received signal
   is billions of times weaker than the transmitted one.

2. **Signal Availability**: Probabilistic dropout model using a sigmoid
   function of distance. Near Jupiter (~778 million km), we model a ~10%
   dropout rate, reflecting real DSN experience with deep-space probes.

3. **Light-Time Delay**: One-way signal propagation time. At Jupiter opposition
   the delay is ~33 minutes; at conjunction, ~54 minutes. This means the
   spacecraft must operate autonomously for extended periods.

4. **Bit Error Rate (BER)**: BPSK modulation BER from SNR using the
   complementary error function. This tells us how many bits are corrupted
   per unit data, informing forward error correction requirements.

5. **Reentry Blackout**: During atmospheric reentry, the hypersonic shock
   wave ionizes the surrounding air, creating a plasma sheath that blocks
   all radio communication. This occurs at altitudes below ~80 km and
   velocities above ~7 km/s, lasting 4-10 minutes for typical reentry
   trajectories.

6. **Measurement Degradation**: Adds realistic noise to sensor measurements
   proportional to the inverse of SNR. When the link is weak, nav solutions
   degrade gracefully rather than failing abruptly.

7. **Communication Window**: Line-of-sight check between the spacecraft and
   a ground station. The Earth itself can block the signal when the station
   is on the far side.

Physical Background
-------------------
The Friis transmission equation gives received power:

    P_r = P_t * G_t * G_r * (lambda / (4 * pi * d))^2

where lambda = c / f is the wavelength. Converting to dB:

    FSPL (dB) = 20*log10(4*pi*d*f/c)

For deep-space missions on X-band (8.4 GHz) at Jupiter distance (~5 AU):
    FSPL ~ 290 dB, requiring extremely sensitive receivers (like DSN's 70-m dishes).

References
----------
    [1] Yuen, "Deep Space Telecommunications Systems Engineering", JPL, 1983.
    [2] Taylor et al., "Mars Exploration Rover Telecommunications", JPL, 2005.
    [3] Rybak & Churchill, "Progress in Reentry Communications", IEEE, 1971.
    [4] DSN Telecommunications Link Design Handbook, JPL 810-005.

===============================================================================
"""

import numpy as np
from core.constants import SPEED_OF_LIGHT, EARTH_RADIUS


class SignalModel:
    """
    Models deep-space communication link characteristics including SNR,
    dropouts, delays, bit error rates, reentry blackout, and measurement
    degradation.

    This class is designed to be used by the navigation filter to determine
    when measurements are available and how much additional noise to inject
    based on link conditions. In a real mission, the telecom subsystem would
    provide these estimates; here we compute them analytically.

    Attributes
    ----------
    _rng : np.random.Generator
        Random number generator for stochastic dropout modeling.
        Using a dedicated generator allows reproducible Monte Carlo runs
        when seeded.

    Examples
    --------
    >>> model = SignalModel(seed=42)
    >>> snr = model.get_snr(distance_m=5.0 * 1.496e11)  # 5 AU
    >>> print(f"SNR at Jupiter: {snr:.1f} dB")
    >>> delay = model.get_delay(distance_m=5.0 * 1.496e11)
    >>> print(f"One-way delay: {delay / 60:.1f} minutes")
    """

    # =========================================================================
    # Boltzmann constant (J/K) - used for thermal noise floor calculation.
    # The noise power in a receiver is N = k * T * B, where T is the system
    # noise temperature and B is the bandwidth.
    # =========================================================================
    BOLTZMANN_CONSTANT = 1.380649e-23  # J/K

    # =========================================================================
    # Default system noise temperature (Kelvin). A typical DSN receiver
    # achieves ~20-25 K system temperature with cryogenic cooling. We use
    # 25 K as a representative value for X-band reception.
    # =========================================================================
    DEFAULT_SYSTEM_TEMP = 25.0  # K

    # =========================================================================
    # Default receiver bandwidth (Hz). For deep-space telemetry, bandwidths
    # range from 1 Hz (for carrier tracking) to several MHz (for high-rate
    # data). We use 1 MHz as a baseline for data reception.
    # =========================================================================
    DEFAULT_BANDWIDTH = 1.0e6  # Hz (1 MHz)

    # =========================================================================
    # Reentry blackout thresholds. These come from empirical data on
    # Apollo, Shuttle, and Stardust reentry missions. The plasma sheath
    # forms when kinetic energy is high enough to ionize atmospheric
    # molecules (N2, O2) in the shock layer.
    # =========================================================================
    BLACKOUT_ALTITUDE_THRESHOLD = 80000.0  # m (80 km)
    BLACKOUT_VELOCITY_THRESHOLD = 7000.0   # m/s (7 km/s)

    def __init__(self, seed: int = None):
        """
        Initialize the signal model.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducible dropout/noise behavior.
            If None, uses entropy from the OS for non-deterministic behavior.
            For Monte Carlo analysis, set this to a known value.
        """
        self._rng = np.random.default_rng(seed)

    # =========================================================================
    # SNR COMPUTATION
    # =========================================================================

    def get_snr(self, distance_m: float, transmit_power_W: float = 20.0,
                antenna_gain_dBi: float = 40.0, freq_Hz: float = 8.4e9) -> float:
        """
        Compute the Signal-to-Noise Ratio (SNR) for a deep-space link.

        Uses the Friis transmission equation to compute free-space path loss
        (FSPL), then derives SNR from the link budget:

            FSPL = (4 * pi * d * f / c)^2       [linear]
            SNR  = P_t * G_t * G_r / (FSPL * k * T * B)

        The result is returned in decibels (dB).

        Parameters
        ----------
        distance_m : float
            Distance between transmitter and receiver in meters.
            For Earth-Jupiter, this ranges from ~588e9 m to ~968e9 m.
        transmit_power_W : float, optional
            Transmitter RF power in Watts. Default 20 W is typical for a
            deep-space probe (Voyager used 23 W, Juno uses 25 W).
        antenna_gain_dBi : float, optional
            Antenna gain in dBi (decibels relative to isotropic). Default
            40 dBi corresponds to a ~2-3 m high-gain antenna on the spacecraft.
            The DSN 70-m dish achieves ~74 dBi at X-band.
        freq_Hz : float, optional
            Carrier frequency in Hz. Default 8.4 GHz (X-band), the standard
            deep-space downlink frequency. Higher frequencies (Ka-band, 32 GHz)
            offer more bandwidth but are more affected by weather.

        Returns
        -------
        float
            SNR in decibels (dB). Typical values:
            - LEO (400 km): ~80-100 dB (strong signal)
            - Lunar distance (384,000 km): ~50-60 dB (good signal)
            - Mars (~1.5 AU): ~20-30 dB (adequate)
            - Jupiter (~5 AU): ~5-15 dB (marginal, but workable with coding)

        Notes
        -----
        We assume symmetric link: same antenna gain for transmit (spacecraft)
        and receive (ground). In practice, the DSN ground antenna is much
        larger, so the real link budget is asymmetric. This simplification
        is acceptable for navigation simulation purposes.

        The noise floor is computed as N = k * T * B:
        - k = 1.38e-23 J/K (Boltzmann constant)
        - T = 25 K (cryogenic receiver)
        - B = 1 MHz (telemetry bandwidth)
        => N = 3.45e-16 W = -154.6 dBW

        Raises
        ------
        ValueError
            If distance_m is non-positive.
        """
        if distance_m <= 0.0:
            raise ValueError(
                f"Distance must be positive, got {distance_m:.2e} m. "
                "Check spacecraft position computation."
            )

        # --- Step 1: Free-Space Path Loss (FSPL) ---
        # FSPL = (4 * pi * d * f / c)^2 in linear scale
        # This represents the geometric spreading of the signal over a sphere
        # of radius d. At Jupiter distances, FSPL ~ 10^29 (290 dB).
        wavelength = SPEED_OF_LIGHT / freq_Hz
        fspl_linear = (4.0 * np.pi * distance_m / wavelength) ** 2

        # --- Step 2: Convert antenna gain from dBi to linear ---
        # G_linear = 10^(G_dBi / 10)
        # 40 dBi -> 10,000x gain (a 2-3 m parabolic dish at X-band)
        antenna_gain_linear = 10.0 ** (antenna_gain_dBi / 10.0)

        # --- Step 3: Compute thermal noise power ---
        # N = k * T * B (Watts)
        # This is the fundamental noise floor of the receiver, set by
        # quantum mechanics (Johnson-Nyquist noise). Cryogenic cooling
        # reduces T to minimize this.
        noise_power = (self.BOLTZMANN_CONSTANT * self.DEFAULT_SYSTEM_TEMP
                       * self.DEFAULT_BANDWIDTH)

        # --- Step 4: Compute received signal power ---
        # P_r = P_t * G_t * G_r / FSPL
        # We use the same gain for transmit and receive (symmetric assumption)
        received_power = (transmit_power_W * antenna_gain_linear
                          * antenna_gain_linear / fspl_linear)

        # --- Step 5: SNR = P_r / N, convert to dB ---
        # SNR_dB = 10 * log10(P_r / N)
        # Guard against division by zero (impossible physically, but
        # defensively coded)
        if noise_power <= 0.0:
            return np.inf

        snr_linear = received_power / noise_power
        snr_dB = 10.0 * np.log10(max(snr_linear, 1e-30))

        return snr_dB

    # =========================================================================
    # SIGNAL AVAILABILITY (DROPOUT MODEL)
    # =========================================================================

    def is_signal_available(self, distance_m: float,
                            environment: str = 'deep_space') -> bool:
        """
        Determine if the communication link is currently available.

        Models probabilistic signal dropouts that increase with distance.
        Uses a sigmoid function to transition smoothly from near-certain
        availability at LEO distances to significant dropout probability
        at Jupiter distances.

        The dropout model:
            P(dropout) = sigmoid_max / (1 + exp(-k * (d - d_mid)))

        where:
        - sigmoid_max = 0.10 for deep_space (10% max dropout at Jupiter)
        - d_mid = midpoint distance where dropout = sigmoid_max / 2
        - k = sigmoid steepness

        Parameters
        ----------
        distance_m : float
            Distance to ground station in meters.
        environment : str, optional
            Operating environment. Options:
            - 'deep_space': Gradual dropout increase with distance (default)
            - 'near_earth': Very low dropout probability (< 1%)
            - 'atmospheric': Higher dropout due to atmospheric effects

        Returns
        -------
        bool
            True if the signal link is currently active, False if dropped out.

        Notes
        -----
        This is a STOCHASTIC model. Each call draws a new random sample,
        simulating the unpredictable nature of real signal dropouts caused by:
        - Pointing errors in the spacecraft antenna
        - Solar scintillation (signal passing near the Sun)
        - Receiver lock loss
        - Atmospheric scintillation at the ground station
        - Equipment anomalies

        At Jupiter distance (~5 AU = 7.48e11 m), the dropout rate is ~10%,
        reflecting real DSN experience with missions like Juno and Galileo.
        """
        if environment == 'near_earth':
            # In LEO or GEO, dropouts are rare (<1%), mostly from
            # ground station handover gaps
            dropout_probability = 0.005
        elif environment == 'atmospheric':
            # During atmospheric flight (launch, reentry), signal
            # degradation from plasma and vibration causes ~15% dropout
            dropout_probability = 0.15
        else:
            # Deep-space sigmoid dropout model
            # Parameters tuned so that:
            #   - At 1 AU (Earth-Sun distance): ~1% dropout
            #   - At 3 AU (asteroid belt): ~5% dropout
            #   - At 5 AU (Jupiter): ~10% dropout
            #   - At 10 AU (Saturn): ~15% dropout
            au_in_meters = 1.496e11  # 1 Astronomical Unit
            distance_au = distance_m / au_in_meters

            # Sigmoid parameters
            max_dropout = 0.20       # Asymptotic max dropout rate (20%)
            midpoint_au = 5.0        # 50% of max dropout at Jupiter distance
            steepness = 0.8          # Controls transition sharpness

            # Sigmoid: smooth S-curve from 0 to max_dropout
            exponent = -steepness * (distance_au - midpoint_au)
            # Clip exponent to prevent overflow in exp()
            exponent = np.clip(exponent, -50.0, 50.0)
            dropout_probability = max_dropout / (1.0 + np.exp(exponent))

        # Draw a random sample to determine if dropout occurs this instant
        # This models the stochastic nature of real communication links
        random_draw = self._rng.uniform(0.0, 1.0)

        # Signal is available if the random draw exceeds the dropout probability
        return random_draw > dropout_probability

    # =========================================================================
    # LIGHT-TIME DELAY
    # =========================================================================

    def get_delay(self, distance_m: float) -> float:
        """
        Compute one-way light-time delay.

        The finite speed of light means that commands sent from Earth take
        significant time to reach the spacecraft, and telemetry takes equally
        long to return. This delay is the fundamental reason deep-space
        spacecraft must be highly autonomous.

        Parameters
        ----------
        distance_m : float
            Distance between Earth and spacecraft in meters.

        Returns
        -------
        float
            One-way signal propagation time in seconds.

        Examples
        --------
        Typical delays:
        - Moon:    ~1.28 seconds
        - Mars:    ~4-24 minutes (depending on orbital geometry)
        - Jupiter: ~33-54 minutes
        - Saturn:  ~68-84 minutes
        - Pluto:   ~4.5-6.5 hours

        Notes
        -----
        This assumes propagation through vacuum at exactly c. In practice,
        the solar wind plasma and Earth's ionosphere introduce small
        additional delays (~nanoseconds), which matter for precision ranging
        but not for communication timing.
        """
        return distance_m / SPEED_OF_LIGHT

    # =========================================================================
    # BIT ERROR RATE
    # =========================================================================

    def get_bit_error_rate(self, snr_dB: float) -> float:
        """
        Compute the Bit Error Rate (BER) for BPSK modulation.

        Binary Phase-Shift Keying (BPSK) is the standard modulation scheme
        for deep-space communication due to its simplicity and optimal
        power efficiency. The BER for BPSK in AWGN (Additive White Gaussian
        Noise) is:

            BER = 0.5 * erfc(sqrt(Eb/N0))

        where Eb/N0 is the energy-per-bit to noise spectral density ratio,
        which equals the linear SNR for BPSK.

        Parameters
        ----------
        snr_dB : float
            Signal-to-Noise Ratio in decibels.

        Returns
        -------
        float
            Bit Error Rate (probability of a single bit being incorrect).
            Ranges from 0.5 (pure noise, no signal) to ~0 (strong signal).

        Notes
        -----
        Typical BER requirements:
        - Uncoded data: BER < 1e-5 (1 error per 100,000 bits)
        - With turbo coding (standard for deep space): can tolerate
          BER ~ 0.1 at the decoder input and still achieve < 1e-6 output BER
        - With coding gain of ~10 dB, the effective SNR threshold drops
          significantly

        At Jupiter distance with 20 W transmitter:
        - SNR ~ 10 dB => BER ~ 3.8e-6 (marginal but workable)
        - SNR ~ 5 dB  => BER ~ 6.0e-4 (needs coding)
        - SNR ~ 0 dB  => BER ~ 7.9e-2 (very noisy, heavy coding needed)

        References
        ----------
        Proakis, "Digital Communications", 5th ed., McGraw-Hill, 2008, Ch. 4.
        """
        from scipy.special import erfc

        # Convert SNR from dB to linear scale
        # SNR_linear = 10^(SNR_dB / 10)
        snr_linear = 10.0 ** (snr_dB / 10.0)

        # Ensure non-negative (physically, SNR >= 0 in linear)
        snr_linear = max(snr_linear, 0.0)

        # BPSK BER formula: BER = 0.5 * erfc(sqrt(SNR_linear))
        # erfc(x) = 1 - erf(x) = (2/sqrt(pi)) * integral from x to inf of exp(-t^2) dt
        # For large SNR, erfc decays exponentially, giving very low BER
        ber = 0.5 * erfc(np.sqrt(snr_linear))

        return float(ber)

    # =========================================================================
    # REENTRY BLACKOUT DETECTION
    # =========================================================================

    def is_blackout(self, altitude_m: float, velocity_ms: float) -> bool:
        """
        Determine if the spacecraft is in a communication blackout due to
        reentry plasma.

        During atmospheric reentry at hypersonic velocities, the bow shock
        compresses and heats the air to temperatures exceeding 10,000 K.
        At these temperatures, atmospheric molecules (primarily N2 and O2)
        dissociate and ionize, forming a plasma sheath around the vehicle.
        This plasma has a critical frequency above which radio signals
        can penetrate; below it, all communication is blocked.

        For X-band (8.4 GHz), the plasma electron density during peak
        heating exceeds the critical density (~8.7e17 electrons/m^3),
        causing complete signal reflection.

        Parameters
        ----------
        altitude_m : float
            Altitude above the surface in meters.
        velocity_ms : float
            Velocity magnitude in meters per second.

        Returns
        -------
        bool
            True if the vehicle is in communication blackout, False otherwise.

        Notes
        -----
        Blackout conditions (empirical, based on Apollo/Shuttle data):
        - Altitude < 80 km: dense enough atmosphere for significant ionization
        - Velocity > 7 km/s: sufficient kinetic energy for plasma formation

        The blackout typically lasts:
        - LEO reentry (~7.8 km/s): ~4-6 minutes
        - Lunar return (~11 km/s): ~6-8 minutes
        - Mars/Jupiter return (~12-15 km/s): ~8-12 minutes

        Modern techniques to mitigate blackout:
        - Magnetic window (applied magnetic field to create a gap in plasma)
        - Raman scattering-based communication
        - Relay satellites above the plasma (TDRS)
        - Ablative antenna materials

        References
        ----------
        Rybak & Churchill, "Progress in Reentry Communications", IEEE, 1971.
        Hartunian et al., "Cause and Mitigation of Radio Blackout", Aerospace Corp, 2007.
        """
        return (altitude_m < self.BLACKOUT_ALTITUDE_THRESHOLD and
                velocity_ms > self.BLACKOUT_VELOCITY_THRESHOLD)

    # =========================================================================
    # MEASUREMENT DEGRADATION
    # =========================================================================

    def degrade_measurement(self, measurement: np.ndarray,
                            snr_dB: float) -> np.ndarray:
        """
        Add realistic noise to a measurement based on current link SNR.

        In a real navigation system, measurements received over a weak
        communication link are noisier than those received over a strong
        link. This method models that effect by adding Gaussian noise
        whose standard deviation is inversely proportional to the SNR.

        The noise model:
            sigma = |measurement| / (SNR_linear)
            degraded = measurement + N(0, sigma^2)

        At high SNR (strong signal), the added noise is negligible.
        At low SNR (weak signal), the noise can be comparable to the
        measurement itself, significantly degrading navigation accuracy.

        Parameters
        ----------
        measurement : np.ndarray
            Original measurement vector (e.g., position, velocity, range).
        snr_dB : float
            Current SNR in decibels. Lower SNR means more noise added.

        Returns
        -------
        np.ndarray
            Degraded measurement with additional noise. Same shape as input.

        Notes
        -----
        The noise scaling is designed so that:
        - SNR > 30 dB: noise is < 0.1% of signal (negligible)
        - SNR ~ 20 dB: noise is ~ 1% of signal (minor degradation)
        - SNR ~ 10 dB: noise is ~ 10% of signal (noticeable)
        - SNR ~ 0 dB:  noise is ~ 100% of signal (severe degradation)

        The navigation filter should increase its measurement noise covariance
        (R matrix) when SNR is low to properly weight these degraded
        measurements.
        """
        measurement = np.asarray(measurement, dtype=np.float64)

        # Convert SNR from dB to linear scale
        snr_linear = 10.0 ** (snr_dB / 10.0)

        # Prevent division by zero or negative SNR (physically impossible,
        # but handle defensively)
        snr_linear = max(snr_linear, 1e-10)

        # Compute noise standard deviation inversely proportional to SNR
        # Use the magnitude of the measurement as the reference scale
        measurement_magnitude = np.linalg.norm(measurement)
        if measurement_magnitude < 1e-30:
            # Measurement is essentially zero; no meaningful noise to add
            return measurement.copy()

        # Noise sigma: larger when SNR is low, smaller when SNR is high
        # The sqrt(SNR) scaling gives the standard deviation of the noise
        # relative to the signal, matching the definition of SNR = S^2 / N^2
        sigma = measurement_magnitude / np.sqrt(snr_linear)

        # Generate Gaussian noise with the computed standard deviation
        noise = self._rng.normal(0.0, sigma, size=measurement.shape)

        return measurement + noise

    # =========================================================================
    # COMMUNICATION WINDOW (LINE-OF-SIGHT CHECK)
    # =========================================================================

    def get_communication_window(self, sc_position: np.ndarray,
                                 ground_station_ecef: np.ndarray,
                                 earth_radius: float = EARTH_RADIUS) -> bool:
        """
        Check if line-of-sight exists between spacecraft and ground station.

        The Earth can block the communication path when the ground station
        is on the far side of the planet relative to the spacecraft. This
        method computes whether a straight line from the spacecraft to the
        ground station intersects the Earth (modeled as a sphere).

        The geometric test uses the closest-point-on-line method:
        1. Parameterize the line from spacecraft (S) to station (G): P(t) = S + t*(G-S)
        2. Find t* that minimizes |P(t) - Earth_center|^2
        3. If the minimum distance < earth_radius AND 0 < t* < 1, the
           line of sight is blocked.

        Parameters
        ----------
        sc_position : np.ndarray
            Spacecraft position in ECI or ECEF coordinates (meters).
            Must be a 3-element vector [x, y, z].
        ground_station_ecef : np.ndarray
            Ground station position in ECEF coordinates (meters).
            Must be a 3-element vector [x, y, z].
        earth_radius : float, optional
            Radius of the Earth in meters. Default uses the mean Earth
            radius from constants. A slightly larger value can be used to
            account for atmospheric refraction effects.

        Returns
        -------
        bool
            True if line-of-sight exists (communication is possible),
            False if the Earth blocks the signal path.

        Notes
        -----
        This is a simplified model that treats the Earth as a perfect sphere.
        In practice:
        - The Earth is an oblate spheroid (equatorial radius ~21 km larger
          than polar radius)
        - The atmosphere refracts signals, extending the radio horizon
          slightly beyond the geometric horizon
        - Terrain (mountains) can block signals for stations not at sea level

        For deep-space missions, the Earth-blocking geometry changes as the
        planet rotates, creating periodic communication windows for each
        ground station. The DSN uses three sites ~120 degrees apart in
        longitude (Goldstone CA, Madrid, Canberra) to maximize coverage.
        """
        sc_position = np.asarray(sc_position, dtype=np.float64)
        ground_station_ecef = np.asarray(ground_station_ecef, dtype=np.float64)

        # Earth center is at the origin in both ECI and ECEF
        earth_center = np.zeros(3)

        # Direction vector from spacecraft to ground station
        # d = G - S
        direction = ground_station_ecef - sc_position

        # --- Closest point on line segment to Earth center ---
        # The line is parameterized as: P(t) = S + t * d, where t in [0, 1]
        # The point on the line closest to the origin minimizes:
        #   |S + t*d|^2 = |S|^2 + 2*t*(S.d) + t^2*|d|^2
        # Taking derivative and setting to zero:
        #   t* = -(S . d) / |d|^2

        d_dot_d = np.dot(direction, direction)

        if d_dot_d < 1e-20:
            # Spacecraft and station are at the same point (degenerate case)
            # Check if that point is above the surface
            return np.linalg.norm(sc_position) > earth_radius

        t_closest = -np.dot(sc_position, direction) / d_dot_d

        # Clamp t to [0, 1] to stay on the segment between S and G
        t_closest = np.clip(t_closest, 0.0, 1.0)

        # Compute the closest point on the segment to Earth center
        closest_point = sc_position + t_closest * direction

        # Distance from Earth center to the closest point on the line segment
        min_distance = np.linalg.norm(closest_point - earth_center)

        # If the minimum distance is greater than Earth's radius,
        # the line of sight is clear (no Earth blockage)
        has_line_of_sight = min_distance > earth_radius

        return has_line_of_sight
