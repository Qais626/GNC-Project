"""
===============================================================================
GNC PROJECT - Environment Models
===============================================================================
High-fidelity models of the space environment encountered during an
Earth-to-Jupiter mission, including:

    - ExponentialAtmosphere : Simple exponential density model for Earth
    - GravityField          : Central body gravity with J2/J3 zonal harmonics
    - SolarRadiationPressure: Cannonball SRP acceleration model
    - RadiationEnvironment  : Jovian radiation belt dose-rate model
    - ThirdBodyPerturbation : Point-mass third-body gravitational acceleration

All vectors are assumed to be in an Earth-Centered Inertial (ECI) frame unless
stated otherwise.  SI units throughout (m, s, kg, rad).
===============================================================================
"""

import numpy as np
from numpy.typing import NDArray

from core.constants import (
    # Earth
    EARTH_MU, EARTH_EQUATORIAL_RADIUS, EARTH_J2, EARTH_J3,
    EARTH_SEA_LEVEL_DENSITY, EARTH_SCALE_HEIGHT, EARTH_ATMOSPHERE_LIMIT,
    EARTH_RADIUS,
    # Moon
    MOON_MU, MOON_RADIUS, MOON_J2,
    # Jupiter
    JUPITER_MU, JUPITER_RADIUS, JUPITER_J2,
    JUPITER_RAD_BELT_INNER, JUPITER_RAD_BELT_OUTER, JUPITER_RAD_DOSE_RATE,
    # Sun / SRP
    SOLAR_PRESSURE_AT_1AU, AU, SPEED_OF_LIGHT,
    # Generic
    GRAVITATIONAL_CONSTANT,
)


# ============================================================================
#  EXPONENTIAL ATMOSPHERE MODEL
# ============================================================================

class ExponentialAtmosphere:
    """
    Simplified exponential atmosphere model for Earth.

    The density is modelled as:

        rho(h) = rho_0 * exp(-h / H)

    where
        rho_0 = sea-level density  (1.225 kg/m^3)
        H     = scale height       (8500 m for Earth)
        h     = altitude above the reference ellipsoid (m)

    Above the atmosphere limit (default 1 000 km) the density is set to zero.

    Parameters
    ----------
    rho_0 : float, optional
        Sea-level atmospheric density in kg/m^3 (default: EARTH_SEA_LEVEL_DENSITY).
    scale_height : float, optional
        Atmospheric scale height in metres (default: EARTH_SCALE_HEIGHT).
    atmosphere_limit : float, optional
        Altitude above which density is treated as zero (m)
        (default: EARTH_ATMOSPHERE_LIMIT).
    body_radius : float, optional
        Mean radius of the central body (m) (default: EARTH_RADIUS).
    """

    def __init__(
        self,
        rho_0: float = EARTH_SEA_LEVEL_DENSITY,
        scale_height: float = EARTH_SCALE_HEIGHT,
        atmosphere_limit: float = EARTH_ATMOSPHERE_LIMIT,
        body_radius: float = EARTH_RADIUS,
    ) -> None:
        self.rho_0 = rho_0
        self.scale_height = scale_height
        self.atmosphere_limit = atmosphere_limit
        self.body_radius = body_radius

    # ------------------------------------------------------------------ #
    def get_density(self, altitude: float) -> float:
        """
        Return atmospheric density at the given geometric altitude.

        Parameters
        ----------
        altitude : float
            Geometric altitude above the surface in metres.

        Returns
        -------
        float
            Atmospheric density in kg/m^3.  Returns 0.0 for altitudes
            below 0 or above the atmosphere limit.
        """
        if altitude < 0.0 or altitude > self.atmosphere_limit:
            return 0.0

        # rho = rho_0 * exp(-h / H)
        density = self.rho_0 * np.exp(-altitude / self.scale_height)
        return float(density)

    # ------------------------------------------------------------------ #
    def get_density_from_position(self, position_eci: NDArray) -> float:
        """
        Convenience wrapper: extract altitude from an ECI position vector
        and return the atmospheric density.

        Parameters
        ----------
        position_eci : ndarray, shape (3,)
            Position in the ECI frame (m).

        Returns
        -------
        float
            Atmospheric density in kg/m^3.
        """
        r = np.linalg.norm(position_eci)
        altitude = r - self.body_radius
        return self.get_density(altitude)


# ============================================================================
#  GRAVITY FIELD (Central body + J2 + J3 zonal harmonics)
# ============================================================================

class GravityField:
    """
    Gravitational acceleration model incorporating zonal harmonics J2 and J3.

    The acceleration is computed in the body-centred inertial (BCI) frame.
    The z-axis is assumed to be aligned with the body's spin axis (pole).

    The J2 perturbation captures the equatorial bulge; J3 captures the
    north-south pear-shape asymmetry.

    For the Moon and Jupiter, only J2 is used (J3 is set to zero).
    """

    def __init__(
        self,
        mu: float,
        body_radius: float,
        j2: float = 0.0,
        j3: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        mu : float
            Gravitational parameter of the central body (m^3/s^2).
        body_radius : float
            Equatorial (reference) radius of the body (m).
        j2 : float, optional
            Second-degree zonal harmonic coefficient (dimensionless).
        j3 : float, optional
            Third-degree zonal harmonic coefficient (dimensionless).
        """
        self.mu = mu
        self.R = body_radius
        self.j2 = j2
        self.j3 = j3

    # ------------------------------------------------------------------ #
    #  Factory class-methods for specific bodies
    # ------------------------------------------------------------------ #
    @classmethod
    def earth(cls) -> "GravityField":
        """Return a GravityField configured for Earth (J2 + J3)."""
        return cls(
            mu=EARTH_MU,
            body_radius=EARTH_EQUATORIAL_RADIUS,
            j2=EARTH_J2,
            j3=EARTH_J3,
        )

    @classmethod
    def moon(cls) -> "GravityField":
        """Return a GravityField configured for the Moon (J2 only)."""
        return cls(
            mu=MOON_MU,
            body_radius=MOON_RADIUS,
            j2=MOON_J2,
            j3=0.0,
        )

    @classmethod
    def jupiter(cls) -> "GravityField":
        """Return a GravityField configured for Jupiter (J2 only)."""
        return cls(
            mu=JUPITER_MU,
            body_radius=JUPITER_RADIUS,
            j2=JUPITER_J2,
            j3=0.0,
        )

    # ------------------------------------------------------------------ #
    def acceleration(self, position_eci: NDArray) -> NDArray:
        """
        Compute the gravitational acceleration at *position_eci*, including
        J2 and J3 perturbation terms.

        The derivation follows Vallado, *Fundamentals of Astrodynamics and
        Applications*, 4th ed., eqs 8-25 through 8-27.

        Parameters
        ----------
        position_eci : ndarray, shape (3,)
            Position vector in the body-centred inertial frame (m).

        Returns
        -------
        ndarray, shape (3,)
            Total gravitational acceleration vector (m/s^2).
        """
        x, y, z = position_eci
        r = np.linalg.norm(position_eci)

        if r < 1.0:
            raise ValueError("Position is at the origin; gravity is undefined.")

        # ---- Central (point-mass / Keplerian) term ----
        # a_central = -mu / r^3 * r_vec
        r2 = r * r
        r3 = r2 * r
        a_central = -self.mu / r3 * position_eci

        # ---- J2 perturbation term ----
        # From the gradient of the J2 potential:
        #   U_J2 = -(mu / r) * J2 * (R/r)^2 * (3 sin^2(phi) - 1) / 2
        # where sin(phi) = z / r.
        #
        # Partial derivatives give (see Vallado eqs. 8-25):
        #   a_x = -mu*x/r^3 * [1 - (3/2)*J2*(R/r)^2 * (5*z^2/r^2 - 1)]
        #   a_y = -mu*y/r^3 * [1 - (3/2)*J2*(R/r)^2 * (5*z^2/r^2 - 1)]
        #   a_z = -mu*z/r^3 * [1 - (3/2)*J2*(R/r)^2 * (5*z^2/r^2 - 3)]
        #
        # We compute the perturbation *delta* relative to the central term.
        a_j2 = np.zeros(3)
        if self.j2 != 0.0:
            R_over_r_sq = (self.R / r) ** 2
            z2_over_r2 = (z / r) ** 2
            factor_j2 = 1.5 * self.j2 * R_over_r_sq

            # Radial common factor for x and y components
            common_xy = factor_j2 * (5.0 * z2_over_r2 - 1.0)
            common_z  = factor_j2 * (5.0 * z2_over_r2 - 3.0)

            # The full acceleration (central + J2) for each component is:
            #   a_i = -mu * pos_i / r^3 * (1 - common_i)
            # So the J2 *perturbation* delta is:
            #   delta_i = +mu * pos_i / r^3 * common_i
            a_j2[0] = (self.mu * x / r3) * common_xy
            a_j2[1] = (self.mu * y / r3) * common_xy
            a_j2[2] = (self.mu * z / r3) * common_z

        # ---- J3 perturbation term ----
        # From the gradient of the J3 potential:
        #   U_J3 = -(mu / r) * J3 * (R/r)^3 * (5 sin^3(phi) - 3 sin(phi)) / 2
        #
        # Partial derivatives (see Montenbruck & Gill, Satellite Orbits, eq 3.28):
        #   a_x = -mu*x/(2*r^3) * J3*(R/r)^3 * (  5*(7*z^3/(r^3) - 3*z/r) / r )
        #   ... etc.
        # Re-written in a cleaner form (Vallado 8-27):
        a_j3 = np.zeros(3)
        if self.j3 != 0.0:
            R_over_r_cu = (self.R / r) ** 3
            z_over_r = z / r
            z2_over_r2 = z_over_r ** 2

            factor_j3 = 0.5 * self.j3 * R_over_r_cu * (self.mu / r3)

            # Components from the J3 gradient:
            #   a_x = factor_j3 * x/r * (35*z^3/r^3 - 30*z/r + 3*r/z)  ... but z=0
            # A safer formulation (avoiding divide-by-zero when z~0):
            #   a_x = factor_j3 * x * (5*(7*z2_over_r2 - 3) * z_over_r)
            #   a_y = factor_j3 * y * (5*(7*z2_over_r2 - 3) * z_over_r)
            #   a_z = factor_j3 * (3 - 30*z2_over_r2 + 35*z2_over_r2 * z2_over_r2)
            # (derived from the potential gradient, keeping only the perturbation)

            term_xy = 5.0 * z_over_r * (7.0 * z2_over_r2 - 3.0)
            term_z = (35.0 * z2_over_r2 * z2_over_r2
                      - 30.0 * z2_over_r2
                      + 3.0)

            a_j3[0] = factor_j3 * x * term_xy
            a_j3[1] = factor_j3 * y * term_xy
            a_j3[2] = factor_j3 * z * term_z       # note: z, not r

        # ---- Sum up ----
        return a_central + a_j2 + a_j3


# ============================================================================
#  SOLAR RADIATION PRESSURE (Cannonball model)
# ============================================================================

class SolarRadiationPressure:
    """
    Cannonball solar-radiation-pressure (SRP) acceleration model.

    Assumes the spacecraft is a flat plate (or sphere) with projected area A
    and a bulk reflectivity coefficient C_r.

    The SRP acceleration magnitude at distance *d* from the Sun is:

        a_srp = P_1AU * (AU / d)^2 * C_r * A / m

    where
        P_1AU = solar radiation pressure at 1 AU  (4.56e-6 N/m^2)
        C_r   = 1 + reflectivity  (1 = fully absorbing, 2 = fully reflecting)
        A     = projected cross-section area (m^2)
        m     = spacecraft mass (kg)

    The acceleration vector points *away from the Sun* (anti-sunward).

    Parameters
    ----------
    pressure_1au : float, optional
        Solar radiation pressure at 1 AU (N/m^2).  Default from constants.
    """

    def __init__(self, pressure_1au: float = SOLAR_PRESSURE_AT_1AU) -> None:
        self.pressure_1au = pressure_1au

    # ------------------------------------------------------------------ #
    def acceleration(
        self,
        pos_sun_relative: NDArray,
        area: float,
        reflectivity: float,
        mass: float,
    ) -> NDArray:
        """
        Compute the SRP acceleration on the spacecraft.

        Parameters
        ----------
        pos_sun_relative : ndarray, shape (3,)
            Position of the spacecraft *relative to the Sun* (m).
            The direction from Sun to S/C defines the SRP push direction.
        area : float
            Effective cross-section area exposed to sunlight (m^2).
        reflectivity : float
            Surface reflectivity in [0, 1].
            0 = fully absorbing, 1 = perfectly reflecting.
            The radiation pressure coefficient C_r = 1 + reflectivity.
        mass : float
            Spacecraft mass (kg).

        Returns
        -------
        ndarray, shape (3,)
            SRP acceleration vector (m/s^2), pointing away from the Sun.
        """
        # Distance from the Sun
        d = np.linalg.norm(pos_sun_relative)
        if d < 1.0:
            return np.zeros(3)

        # Unit vector from Sun to S/C (direction of SRP push)
        sun_to_sc_hat = pos_sun_relative / d

        # Radiation pressure coefficient
        #   C_r = 1.0  for a perfectly absorbing body
        #   C_r = 2.0  for a perfectly reflecting body
        C_r = 1.0 + reflectivity

        # Pressure scales as inverse-square of distance from Sun
        # P(d) = P_1AU * (AU / d)^2
        pressure = self.pressure_1au * (AU / d) ** 2

        # a_srp = P(d) * C_r * A / m   (directed away from Sun)
        a_mag = pressure * C_r * area / mass

        return a_mag * sun_to_sc_hat


# ============================================================================
#  RADIATION ENVIRONMENT (Jupiter radiation belts)
# ============================================================================

class RadiationEnvironment:
    """
    Simplified model of Jupiter's intense radiation environment.

    Jupiter's magnetosphere traps high-energy protons and electrons in
    radiation belts extending from ~1.5 R_J to ~20 R_J.  The dose rate
    is modelled as an empirical power-law that peaks at the inner edge
    and falls off with distance:

        dose_rate(r) = D0 * (r_inner / r_norm)^3      r in [r_inner, r_outer]

    where
        D0       = reference dose rate at the inner belt boundary  (rad/s)
        r_norm   = distance from Jupiter in Jupiter radii
        r_inner  = inner boundary of the belt  (Jupiter radii)
        r_outer  = outer boundary of the belt  (Jupiter radii)

    Outside [r_inner, r_outer] the dose rate drops to a small background.

    The class also tracks *cumulative* absorbed dose over time and maps it
    to degradation factors for solar panels and electronics.

    Parameters
    ----------
    inner_boundary_rj : float
        Inner edge of the radiation belt in Jupiter radii.
    outer_boundary_rj : float
        Outer edge of the radiation belt in Jupiter radii.
    reference_dose_rate : float
        Dose rate at the inner boundary in rad/s.
    jupiter_radius : float
        Jupiter's equatorial radius in metres.
    """

    def __init__(
        self,
        inner_boundary_rj: float = JUPITER_RAD_BELT_INNER,
        outer_boundary_rj: float = JUPITER_RAD_BELT_OUTER,
        reference_dose_rate: float = JUPITER_RAD_DOSE_RATE,
        jupiter_radius: float = JUPITER_RADIUS,
    ) -> None:
        self.inner_rj = inner_boundary_rj
        self.outer_rj = outer_boundary_rj
        self.D0 = reference_dose_rate
        self.R_J = jupiter_radius

        # Cumulative tracking
        self.cumulative_dose: float = 0.0          # total absorbed dose (rad)
        self._panel_degradation: float = 1.0       # 1.0 = no degradation
        self._electronics_degradation: float = 1.0

        # Background dose rate outside the belt  (very small)
        self.background_dose_rate = 1.0e-8  # rad/s

    # ------------------------------------------------------------------ #
    def get_dose_rate(self, position_jupiter_relative: NDArray) -> float:
        """
        Instantaneous radiation dose rate at the given position relative
        to Jupiter.

        Parameters
        ----------
        position_jupiter_relative : ndarray, shape (3,)
            Position of the spacecraft relative to Jupiter's centre (m).

        Returns
        -------
        float
            Dose rate in rad/s.
        """
        d = np.linalg.norm(position_jupiter_relative)
        r_norm = d / self.R_J  # distance in Jupiter radii

        if r_norm < self.inner_rj:
            # Inside the inner boundary -- assume dose rate at max
            # (physically the S/C would not survive here long)
            return self.D0

        if r_norm > self.outer_rj:
            # Outside the belt -- background only
            return self.background_dose_rate

        # Within the belt: power-law fall-off (inverse cube of distance)
        #   dose_rate = D0 * (r_inner / r_norm)^3
        dose = self.D0 * (self.inner_rj / r_norm) ** 3
        return float(dose)

    # ------------------------------------------------------------------ #
    def accumulate_dose(
        self,
        position_jupiter_relative: NDArray,
        dt: float,
    ) -> float:
        """
        Accumulate radiation dose over a timestep *dt* and update the
        degradation factors.

        Parameters
        ----------
        position_jupiter_relative : ndarray, shape (3,)
            S/C position relative to Jupiter (m).
        dt : float
            Timestep duration (s).

        Returns
        -------
        float
            Dose accumulated during this timestep (rad).
        """
        rate = self.get_dose_rate(position_jupiter_relative)
        delta_dose = rate * dt
        self.cumulative_dose += delta_dose

        # Update degradation factors
        self._update_degradation()

        return delta_dose

    # ------------------------------------------------------------------ #
    def _update_degradation(self) -> None:
        """
        Map cumulative dose to degradation factors using empirical curves.

        Solar panels:
            Modelled as an exponential decay:
                factor = exp(-dose / dose_half)
            where dose_half = 50 krad  (dose at which output drops to ~37 %).

        Electronics:
            Similar exponential but more radiation-hard:
                factor = exp(-dose / dose_half)
            where dose_half = 150 krad.
        """
        # Dose thresholds (rad) -- representative of radiation-hardened parts
        panel_dose_half = 50_000.0       # 50 krad
        electronics_dose_half = 150_000.0  # 150 krad

        self._panel_degradation = float(
            np.exp(-self.cumulative_dose / panel_dose_half)
        )
        self._electronics_degradation = float(
            np.exp(-self.cumulative_dose / electronics_dose_half)
        )

    # ------------------------------------------------------------------ #
    @property
    def panel_degradation_factor(self) -> float:
        """Current solar-panel efficiency multiplier in [0, 1]."""
        return self._panel_degradation

    @property
    def electronics_degradation_factor(self) -> float:
        """Current electronics health multiplier in [0, 1]."""
        return self._electronics_degradation

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Reset cumulative dose and degradation state (e.g. for a new run)."""
        self.cumulative_dose = 0.0
        self._panel_degradation = 1.0
        self._electronics_degradation = 1.0


# ============================================================================
#  THIRD-BODY PERTURBATION
# ============================================================================

class ThirdBodyPerturbation:
    """
    Point-mass gravitational perturbation from a third body (Sun, Moon,
    Jupiter, etc.) acting on a spacecraft orbiting a central body.

    The perturbing acceleration on the spacecraft in the central-body frame
    is given by (Battin, 1999):

        a_pert = mu_3 * ( (r_3 - r_sc) / |r_3 - r_sc|^3  -  r_3 / |r_3|^3 )

    The first term is the direct attraction of the third body on the S/C.
    The second term subtracts the attraction of the third body on the
    central body (since we work in the central-body-centred frame).

    Parameters
    ----------
    mu_third_body : float, optional
        Gravitational parameter of the third body (m^3/s^2).
        Can also be supplied per-call to ``acceleration()``.
    """

    def __init__(self, mu_third_body: float = 0.0) -> None:
        self.mu = mu_third_body

    # ------------------------------------------------------------------ #
    def acceleration(
        self,
        pos_sc: NDArray,
        pos_third_body: NDArray,
        mu_third_body: float | None = None,
    ) -> NDArray:
        """
        Compute the perturbing acceleration due to a third body.

        Parameters
        ----------
        pos_sc : ndarray, shape (3,)
            Position of the spacecraft in the central-body inertial frame (m).
        pos_third_body : ndarray, shape (3,)
            Position of the third body in the *same* frame (m).
        mu_third_body : float, optional
            Gravitational parameter of the third body.  If not provided the
            value stored at construction is used.

        Returns
        -------
        ndarray, shape (3,)
            Perturbing acceleration on the S/C (m/s^2).
        """
        mu = mu_third_body if mu_third_body is not None else self.mu
        if mu <= 0.0:
            raise ValueError("mu_third_body must be positive.")

        # Vector from S/C to third body
        r_sc_to_3 = pos_third_body - pos_sc
        d_sc_to_3 = np.linalg.norm(r_sc_to_3)

        # Distance of third body from the central body origin
        d_3 = np.linalg.norm(pos_third_body)

        if d_sc_to_3 < 1.0 or d_3 < 1.0:
            # Degenerate case -- bodies overlap; return zero to avoid NaN
            return np.zeros(3)

        # Battin formulation:
        #   a = mu * [ (r_3 - r_sc) / |r_3 - r_sc|^3  -  r_3 / |r_3|^3 ]
        a = mu * (
            r_sc_to_3 / d_sc_to_3 ** 3
            - pos_third_body / d_3 ** 3
        )

        return a
