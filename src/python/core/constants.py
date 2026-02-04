"""
===============================================================================
GNC PROJECT - Physical and Astronomical Constants
===============================================================================
Central repository for all physical constants used throughout the mission
simulation. Using SI units throughout (meters, seconds, kilograms, radians).

These values come from IAU 2012 / IERS standards where applicable.
===============================================================================
"""

import numpy as np


# =============================================================================
# MATHEMATICAL CONSTANTS
# =============================================================================
PI = np.pi
TWO_PI = 2.0 * np.pi
DEG2RAD = PI / 180.0
RAD2DEG = 180.0 / PI
ARCSEC2RAD = DEG2RAD / 3600.0

# =============================================================================
# FUNDAMENTAL PHYSICAL CONSTANTS
# =============================================================================
SPEED_OF_LIGHT = 299792458.0           # m/s
GRAVITATIONAL_CONSTANT = 6.67430e-11   # m^3 / (kg * s^2)
BOLTZMANN = 1.380649e-23               # J/K
STEFAN_BOLTZMANN = 5.670374419e-8      # W / (m^2 * K^4)
PLANCK = 6.62607015e-34                # J*s
AU = 1.495978707e11                    # Astronomical Unit in meters

# =============================================================================
# EARTH PARAMETERS
# =============================================================================
EARTH_MU = 3.986004418e14              # Gravitational parameter (m^3/s^2)
EARTH_RADIUS = 6371000.0               # Mean radius (m)
EARTH_EQUATORIAL_RADIUS = 6378137.0    # WGS84 equatorial radius (m)
EARTH_POLAR_RADIUS = 6356752.314       # WGS84 polar radius (m)
EARTH_MASS = 5.97237e24                # kg
EARTH_J2 = 1.08263e-3                  # J2 oblateness coefficient
EARTH_J3 = -2.5327e-6                  # J3 coefficient
EARTH_ROTATION_RATE = 7.2921159e-5     # rad/s (sidereal)
EARTH_FLATTENING = 1.0 / 298.257223563  # WGS84 flattening
EARTH_SOI_RADIUS = 9.24e8             # Sphere of Influence radius (m)

# Earth atmosphere model constants (exponential model)
EARTH_SEA_LEVEL_DENSITY = 1.225        # kg/m^3
EARTH_SCALE_HEIGHT = 8500.0            # m
EARTH_ATMOSPHERE_LIMIT = 1000000.0     # m (Karman line extended)

# =============================================================================
# MOON PARAMETERS
# =============================================================================
MOON_MU = 4.9048695e12                 # m^3/s^2
MOON_RADIUS = 1737400.0               # Mean radius (m)
MOON_MASS = 7.342e22                   # kg
MOON_SMA = 384400000.0                # Semi-major axis of lunar orbit (m)
MOON_ORBITAL_PERIOD = 2360591.0        # Sidereal orbital period (s) ~27.3 days
MOON_ECCENTRICITY = 0.0549            # Orbital eccentricity
MOON_INCLINATION = 5.145 * DEG2RAD    # Inclination to ecliptic (rad)
MOON_J2 = 2.027e-4                    # Moon J2
MOON_SOI_RADIUS = 6.61e7             # ~66,100 km

# =============================================================================
# JUPITER PARAMETERS
# =============================================================================
JUPITER_MU = 1.26686534e17            # m^3/s^2
JUPITER_RADIUS = 69911000.0           # Mean equatorial radius (m)
JUPITER_MASS = 1.89819e27             # kg
JUPITER_SMA = 778.57e9               # Semi-major axis from Sun (m)
JUPITER_ORBITAL_PERIOD = 3.7435577e8  # Sidereal period (s) ~11.86 years
JUPITER_ECCENTRICITY = 0.0489        # Orbital eccentricity
JUPITER_INCLINATION = 1.303 * DEG2RAD  # Inclination to ecliptic
JUPITER_J2 = 0.01475                  # Very oblate planet
JUPITER_ROTATION_RATE = 1.7585e-4    # rad/s (~9.9 hr rotation)
JUPITER_SOI_RADIUS = 4.82e10        # ~48.2 million km

# Jupiter radiation environment
JUPITER_RAD_BELT_INNER = 1.5         # Inner boundary in Jupiter radii
JUPITER_RAD_BELT_OUTER = 20.0        # Outer boundary in Jupiter radii
JUPITER_RAD_DOSE_RATE = 0.01         # rad/s at inner belt boundary

# =============================================================================
# SUN PARAMETERS
# =============================================================================
SUN_MU = 1.32712440018e20             # m^3/s^2
SUN_MASS = 1.989e30                   # kg
SUN_RADIUS = 6.957e8                  # m
SUN_LUMINOSITY = 3.828e26             # Watts
SOLAR_PRESSURE_AT_1AU = 4.56e-6       # N/m^2 (solar radiation pressure)

# =============================================================================
# MIAMI LAUNCH SITE
# =============================================================================
MIAMI_LATITUDE = 25.7617 * DEG2RAD    # rad
MIAMI_LONGITUDE = -80.1918 * DEG2RAD  # rad
MIAMI_ALTITUDE = 0.0                   # m (sea level)

# =============================================================================
# USEFUL DERIVED QUANTITIES
# =============================================================================
# Circular orbital velocity at Earth surface
V_CIRCULAR_EARTH_SURFACE = np.sqrt(EARTH_MU / EARTH_RADIUS)

# Escape velocity at Earth surface
V_ESCAPE_EARTH_SURFACE = np.sqrt(2.0 * EARTH_MU / EARTH_RADIUS)

# LEO (200 km) orbital parameters
LEO_ALTITUDE = 200000.0               # m
LEO_RADIUS = EARTH_RADIUS + LEO_ALTITUDE
LEO_VELOCITY = np.sqrt(EARTH_MU / LEO_RADIUS)
LEO_PERIOD = TWO_PI * np.sqrt(LEO_RADIUS**3 / EARTH_MU)

# Light-time delays
LIGHT_TIME_EARTH_MOON = MOON_SMA / SPEED_OF_LIGHT      # ~1.28 s
LIGHT_TIME_EARTH_JUPITER_MIN = 588.0e9 / SPEED_OF_LIGHT  # ~33 min (closest)
LIGHT_TIME_EARTH_JUPITER_MAX = 968.0e9 / SPEED_OF_LIGHT  # ~54 min (farthest)


def get_body_mu(body_name: str) -> float:
    """
    Look up gravitational parameter by body name.

    Args:
        body_name: One of 'earth', 'moon', 'jupiter', 'sun'

    Returns:
        Gravitational parameter mu in m^3/s^2

    Raises:
        ValueError: If body_name is not recognized
    """
    lookup = {
        'earth': EARTH_MU,
        'moon': MOON_MU,
        'jupiter': JUPITER_MU,
        'sun': SUN_MU,
    }
    if body_name.lower() not in lookup:
        raise ValueError(f"Unknown body: {body_name}. Valid: {list(lookup.keys())}")
    return lookup[body_name.lower()]


def get_body_radius(body_name: str) -> float:
    """
    Look up mean radius by body name.

    Args:
        body_name: One of 'earth', 'moon', 'jupiter', 'sun'

    Returns:
        Mean radius in meters
    """
    lookup = {
        'earth': EARTH_RADIUS,
        'moon': MOON_RADIUS,
        'jupiter': JUPITER_RADIUS,
        'sun': SUN_RADIUS,
    }
    if body_name.lower() not in lookup:
        raise ValueError(f"Unknown body: {body_name}. Valid: {list(lookup.keys())}")
    return lookup[body_name.lower()]
