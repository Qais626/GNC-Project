"""
===============================================================================
GNC PROJECT - Reference Frame Transformations
===============================================================================
Supports: ECI (J2000), ECEF, Geodetic, Body, LVLH, Perifocal,
          Moon-centered, Jupiter-centered frames.

A spacecraft mission from Earth to Jupiter traverses multiple gravitational
domains.  At each phase the natural computational frame changes:

    Launch          -> ECEF / geodetic  (ground-track, atmosphere model)
    LEO parking     -> ECI              (inertial propagation, J2 perturbations)
    Transfer orbit  -> Perifocal / ECI  (orbit geometry, Lambert targeting)
    Proximity ops   -> LVLH             (relative navigation, rendezvous)
    Attitude ctrl   -> Body <-> ECI     (torque commands, sensor fusion)
    Lunar flyby     -> Moon-centred     (patched-conic SOI transition)
    Jupiter arrival -> Jupiter-centred  (JOI targeting, radiation belts)

This module provides the rotation matrices and coordinate conversions that
stitch these frames together.  All functions operate on NumPy arrays and
return NumPy arrays.  Angles are in radians unless noted otherwise.

References
----------
    [1] Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed.
    [2] Montenbruck & Gill, "Satellite Orbits", Springer, 2000.
    [3] Wertz, "Space Mission Engineering: The New SMAD", Ch. 5-6.
    [4] Bowring, "Transformation from spatial to geographical coordinates",
        Survey Review, 1976.

===============================================================================
"""

import numpy as np
from core.constants import (
    EARTH_ROTATION_RATE,
    EARTH_EQUATORIAL_RADIUS,
    EARTH_FLATTENING,
    EARTH_POLAR_RADIUS,
    DEG2RAD,
    RAD2DEG,
    AU,
    TWO_PI,
    MOON_SMA,
    MOON_ORBITAL_PERIOD,
    MOON_INCLINATION,
    JUPITER_SMA,
    JUPITER_ORBITAL_PERIOD,
    JUPITER_INCLINATION,
    SUN_MU,
)


# =============================================================================
# ELEMENTARY ROTATION MATRICES
# =============================================================================

def Rx(angle: float) -> np.ndarray:
    """
    Elementary rotation matrix about the X-axis.

    Rotates a vector by *angle* radians about the X-axis using the
    right-hand rule:

        Rx(a) = | 1    0       0     |
                | 0   cos(a)  sin(a)  |
                | 0  -sin(a)  cos(a)  |

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [1.0,  0.0,  0.0],
        [0.0,    c,    s],
        [0.0,   -s,    c],
    ], dtype=np.float64)


def Ry(angle: float) -> np.ndarray:
    """
    Elementary rotation matrix about the Y-axis.

    Rotates a vector by *angle* radians about the Y-axis using the
    right-hand rule:

        Ry(a) = | cos(a)  0  -sin(a) |
                |   0     1     0     |
                | sin(a)  0   cos(a)  |

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [  c,  0.0,   -s],
        [0.0,  1.0,  0.0],
        [  s,  0.0,    c],
    ], dtype=np.float64)


def Rz(angle: float) -> np.ndarray:
    """
    Elementary rotation matrix about the Z-axis.

    Rotates a vector by *angle* radians about the Z-axis using the
    right-hand rule:

        Rz(a) = |  cos(a)  sin(a)  0 |
                | -sin(a)  cos(a)  0 |
                |    0       0     1 |

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [  c,    s,  0.0],
        [ -s,    c,  0.0],
        [0.0,  0.0,  1.0],
    ], dtype=np.float64)


# =============================================================================
# ECEF <-> ECI CONVERSIONS
# =============================================================================

def ecef_to_eci(r_ecef: np.ndarray, time_s: float) -> np.ndarray:
    """
    Convert a position vector from ECEF to ECI (J2000) frame.

    The ECEF frame co-rotates with the Earth.  At epoch t = 0 the two
    frames are assumed to be aligned (Greenwich Sidereal Time = 0).
    The Earth rotation angle at time *time_s* is:

        theta = EARTH_ROTATION_RATE * time_s   (rad)

    The ECI position is obtained by rotating the ECEF vector *backwards*
    by theta about the Z-axis (i.e. undoing the Earth's rotation):

        r_eci = Rz(-theta) * r_ecef

    Parameters
    ----------
    r_ecef : np.ndarray
        3-element position vector in ECEF (m).
    time_s : float
        Elapsed time since epoch in seconds.

    Returns
    -------
    np.ndarray
        3-element position vector in ECI (m).
    """
    theta = EARTH_ROTATION_RATE * time_s
    return Rz(-theta) @ np.asarray(r_ecef, dtype=np.float64)


def eci_to_ecef(r_eci: np.ndarray, time_s: float) -> np.ndarray:
    """
    Convert a position vector from ECI (J2000) to ECEF frame.

    Inverse of ecef_to_eci.  Applies a forward rotation by the Earth
    rotation angle about the Z-axis:

        r_ecef = Rz(+theta) * r_eci

    Parameters
    ----------
    r_eci : np.ndarray
        3-element position vector in ECI (m).
    time_s : float
        Elapsed time since epoch in seconds.

    Returns
    -------
    np.ndarray
        3-element position vector in ECEF (m).
    """
    theta = EARTH_ROTATION_RATE * time_s
    return Rz(theta) @ np.asarray(r_eci, dtype=np.float64)


# =============================================================================
# GEODETIC <-> ECEF CONVERSIONS (WGS84)
# =============================================================================

def geodetic_to_ecef(lat_rad: float, lon_rad: float, alt_m: float) -> np.ndarray:
    """
    Convert geodetic coordinates to ECEF using the WGS84 ellipsoid.

    The geodetic latitude is the angle between the ellipsoid normal and
    the equatorial plane (NOT the geocentric latitude).  The conversion
    uses the prime vertical radius of curvature N:

        N = a / sqrt(1 - e^2 * sin^2(lat))

    where a = equatorial radius and e^2 = 2f - f^2 (first eccentricity
    squared, f = flattening).

    The ECEF coordinates are:

        x = (N + h) * cos(lat) * cos(lon)
        y = (N + h) * cos(lat) * sin(lon)
        z = (N * (1 - e^2) + h) * sin(lat)

    Parameters
    ----------
    lat_rad : float
        Geodetic latitude in radians [-pi/2, pi/2].
    lon_rad : float
        Geodetic longitude in radians [-pi, pi].
    alt_m : float
        Altitude above the WGS84 ellipsoid in metres.

    Returns
    -------
    np.ndarray
        3-element ECEF position vector (m).

    References
    ----------
    Vallado (2013), Algorithm 51.
    """
    a = EARTH_EQUATORIAL_RADIUS
    f = EARTH_FLATTENING
    e2 = 2.0 * f - f * f  # first eccentricity squared

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Prime vertical radius of curvature
    N = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)

    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1.0 - e2) + alt_m) * sin_lat

    return np.array([x, y, z], dtype=np.float64)


def ecef_to_geodetic(r_ecef: np.ndarray) -> tuple:
    """
    Convert ECEF position to geodetic coordinates using Bowring's
    iterative method on the WGS84 ellipsoid.

    Bowring's method converges in 2-3 iterations to sub-millimetre
    accuracy for all terrestrial and LEO altitudes.  It iterates on
    the parametric (reduced) latitude beta:

        tan(beta) = (1 - f) * tan(lat)

    The algorithm:
        1. Initial guess: tan(lat) ~ z / (p * (1 - e^2))
        2. Iterate:
            beta  = atan2((1-f)*sin(lat), cos(lat))
            lat   = atan2(z + e'^2 * b * sin^3(beta),
                          p - e^2 * a * cos^3(beta))
           where e'^2 = (a^2 - b^2)/b^2 is the second eccentricity squared.
        3. Compute altitude from the converged latitude.

    Parameters
    ----------
    r_ecef : np.ndarray
        3-element ECEF position vector (m).

    Returns
    -------
    tuple of (float, float, float)
        (lat_rad, lon_rad, alt_m) -- geodetic latitude (rad),
        longitude (rad), altitude above ellipsoid (m).

    References
    ----------
    Bowring, "Transformation from spatial to geographical coordinates",
    Survey Review, 1976.
    """
    r = np.asarray(r_ecef, dtype=np.float64)
    x, y, z = r[0], r[1], r[2]

    a = EARTH_EQUATORIAL_RADIUS
    b = EARTH_POLAR_RADIUS
    f = EARTH_FLATTENING

    e2 = 2.0 * f - f * f           # first eccentricity squared
    ep2 = (a * a - b * b) / (b * b)  # second eccentricity squared

    # Longitude is straightforward
    lon = np.arctan2(y, x)

    # Distance from the Z-axis
    p = np.sqrt(x * x + y * y)

    # Initial latitude guess (geocentric approximation)
    lat = np.arctan2(z, p * (1.0 - e2))

    # Bowring iteration (3 iterations is more than enough)
    for _ in range(5):
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        beta = np.arctan2((1.0 - f) * sin_lat, cos_lat)
        sin_beta = np.sin(beta)
        cos_beta = np.cos(beta)

        lat = np.arctan2(
            z + ep2 * b * sin_beta * sin_beta * sin_beta,
            p - e2 * a * cos_beta * cos_beta * cos_beta,
        )

    # Compute altitude
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)

    # Avoid division by zero near the poles
    if abs(cos_lat) > 1e-10:
        alt = p / cos_lat - N
    else:
        alt = abs(z) - b

    return (float(lat), float(lon), float(alt))


# =============================================================================
# ECI <-> PERIFOCAL FRAME
# =============================================================================

def eci_to_perifocal(r_eci: np.ndarray, v_eci: np.ndarray,
                     mu: float) -> np.ndarray:
    """
    Transform an ECI position vector into the perifocal (PQW) frame.

    The perifocal frame is defined by the orbit geometry:
        p-hat = unit vector pointing toward periapsis
        w-hat = unit angular momentum vector (h / |h|)
        q-hat = w-hat x p-hat  (completes the right-hand triad)

    The transformation builds the PQW basis from r and v:
        h = r x v                          (specific angular momentum)
        w = h / |h|                        (orbit normal)
        n_vec = (-h_y, h_x, 0) / |...|    (ascending node direction)
        e_vec = (v x h)/mu - r/|r|         (eccentricity vector -> periapsis)
        p = e_vec / |e_vec|                (toward periapsis)
        q = w x p

    The perifocal position is then:
        r_pqw = [r . p,  r . q,  r . w]

    Parameters
    ----------
    r_eci : np.ndarray
        3-element ECI position vector (m).
    v_eci : np.ndarray
        3-element ECI velocity vector (m/s).
    mu : float
        Gravitational parameter of the central body (m^3/s^2).

    Returns
    -------
    np.ndarray
        3-element position vector in the perifocal frame (m).
    """
    r = np.asarray(r_eci, dtype=np.float64)
    v = np.asarray(v_eci, dtype=np.float64)
    r_mag = np.linalg.norm(r)

    # Angular momentum
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    # Orbit normal (w-hat)
    w_hat = h / h_mag

    # Eccentricity vector (points toward periapsis)
    e_vec = np.cross(v, h) / mu - r / r_mag
    e_mag = np.linalg.norm(e_vec)

    if e_mag < 1e-12:
        # Circular orbit: periapsis direction is undefined.
        # Use the current position direction as a stand-in.
        p_hat = r / r_mag
    else:
        p_hat = e_vec / e_mag

    # Complete the right-hand triad
    q_hat = np.cross(w_hat, p_hat)

    # Build rotation matrix (rows are the PQW basis vectors)
    R_pqw = np.array([p_hat, q_hat, w_hat], dtype=np.float64)

    return R_pqw @ r


def perifocal_to_eci(r_pqw: np.ndarray, RAAN: float,
                     inc: float, omega: float) -> np.ndarray:
    """
    Rotate a vector from the perifocal (PQW) frame to ECI using
    the classical 3-1-3 Euler rotation sequence.

    The perifocal frame is related to ECI by three successive rotations:
        1. Rotate about Z by -RAAN            (undo right ascension)
        2. Rotate about X by -inc             (undo inclination)
        3. Rotate about Z by -omega           (undo argument of periapsis)

    The combined rotation from PQW to ECI is:

        R_eci_pqw = Rz(-RAAN) * Rx(-inc) * Rz(-omega)

    so that:
        r_eci = R_eci_pqw * r_pqw

    Parameters
    ----------
    r_pqw : np.ndarray
        3-element position vector in perifocal frame (m).
    RAAN : float
        Right Ascension of the Ascending Node (rad).
    inc : float
        Orbital inclination (rad).
    omega : float
        Argument of periapsis (rad).

    Returns
    -------
    np.ndarray
        3-element position vector in ECI (m).

    References
    ----------
    Vallado (2013), Algorithm 11.
    """
    r = np.asarray(r_pqw, dtype=np.float64)

    # 3-1-3 rotation: PQW -> ECI
    R = Rz(-RAAN) @ Rx(-inc) @ Rz(-omega)

    return R @ r


# =============================================================================
# ECI <-> LVLH (Local Vertical Local Horizontal)
# =============================================================================

def eci_to_lvlh(r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    """
    Build the rotation matrix from ECI to the LVLH frame.

    The LVLH (Local Vertical Local Horizontal) frame -- also known as
    the RSW or Hill frame -- is defined as:

        R-hat (radial)       = r / |r|          (local vertical, outward)
        W-hat (cross-track)  = (r x v) / |r x v|  (orbit normal)
        S-hat (along-track)  = W x R            (approximately velocity dir)

    This frame is natural for relative navigation (e.g., rendezvous and
    proximity operations) because perturbations separate neatly into
    radial, along-track, and cross-track components.

    The returned matrix R_lvlh satisfies:

        v_lvlh = R_lvlh @ v_eci

    for any vector v.

    Parameters
    ----------
    r_eci : np.ndarray
        3-element ECI position vector (m).
    v_eci : np.ndarray
        3-element ECI velocity vector (m/s).

    Returns
    -------
    np.ndarray
        3x3 rotation matrix from ECI to LVLH.
    """
    r = np.asarray(r_eci, dtype=np.float64)
    v = np.asarray(v_eci, dtype=np.float64)

    r_mag = np.linalg.norm(r)
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    # LVLH basis vectors expressed in ECI
    R_hat = r / r_mag                  # radial (outward)
    W_hat = h / h_mag                  # cross-track (orbit normal)
    S_hat = np.cross(W_hat, R_hat)     # along-track

    # Rotation matrix: rows are the LVLH basis vectors
    R_lvlh = np.array([R_hat, S_hat, W_hat], dtype=np.float64)

    return R_lvlh


# =============================================================================
# BODY <-> ECI via QUATERNION
# =============================================================================

def body_to_eci(v_body: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """
    Rotate a vector from the spacecraft body frame to ECI using a
    unit quaternion.

    The quaternion convention is scalar-first: q = [w, x, y, z].

    The rotation is performed via the optimised Rodrigues formula:

        t = 2 * (q_vec x v)
        v_eci = v + w * t + (q_vec x t)

    where q_vec = [x, y, z] and w is the scalar part.

    Parameters
    ----------
    v_body : np.ndarray
        3-element vector in the body frame.
    quaternion : np.ndarray
        4-element unit quaternion [w, x, y, z] representing the
        rotation from body to ECI.

    Returns
    -------
    np.ndarray
        3-element vector in the ECI frame.
    """
    v = np.asarray(v_body, dtype=np.float64)
    q = np.asarray(quaternion, dtype=np.float64)

    w = q[0]
    q_vec = q[1:4]

    # Rodrigues rotation formula (optimised for unit quaternions)
    t = 2.0 * np.cross(q_vec, v)
    return v + w * t + np.cross(q_vec, t)


def eci_to_body(v_eci: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """
    Rotate a vector from ECI to the spacecraft body frame using a
    unit quaternion.

    This is the inverse of body_to_eci.  For a unit quaternion, the
    inverse rotation is obtained by conjugating the quaternion:

        q_conj = [w, -x, -y, -z]

    and then applying the same Rodrigues formula.

    Parameters
    ----------
    v_eci : np.ndarray
        3-element vector in the ECI frame.
    quaternion : np.ndarray
        4-element unit quaternion [w, x, y, z] representing the
        rotation from body to ECI.

    Returns
    -------
    np.ndarray
        3-element vector in the body frame.
    """
    q = np.asarray(quaternion, dtype=np.float64)

    # Conjugate: reverse the rotation direction
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

    return body_to_eci(v_eci, q_conj)


# =============================================================================
# ECI <-> MOON-CENTERED / JUPITER-CENTERED (TRANSLATIONS)
# =============================================================================

def eci_to_moon_centered(r_eci: np.ndarray,
                         moon_position_eci: np.ndarray) -> np.ndarray:
    """
    Transform an ECI position to a Moon-centred frame by simple
    translation.

    In the patched-conic approximation, when a spacecraft enters the
    Moon's sphere of influence, the equations of motion are reformulated
    about the Moon.  As a first-order step the position is translated:

        r_moon_centered = r_eci - moon_position_eci

    A full treatment would also rotate into the Moon's body-fixed or
    inertial frame, but for trajectory-level work this translation
    suffices.

    Parameters
    ----------
    r_eci : np.ndarray
        3-element ECI position of the spacecraft (m).
    moon_position_eci : np.ndarray
        3-element ECI position of the Moon (m).

    Returns
    -------
    np.ndarray
        3-element position vector centred on the Moon (m).
    """
    return np.asarray(r_eci, dtype=np.float64) - np.asarray(
        moon_position_eci, dtype=np.float64
    )


def eci_to_jupiter_centered(r_eci: np.ndarray,
                            jupiter_position_eci: np.ndarray) -> np.ndarray:
    """
    Transform an ECI position to a Jupiter-centred frame by simple
    translation.

    Analogous to eci_to_moon_centered but for the Jupiter sphere of
    influence.  Used during the interplanetary cruise phase when the
    spacecraft crosses into Jupiter's gravitational domain.

        r_jupiter_centered = r_eci - jupiter_position_eci

    Parameters
    ----------
    r_eci : np.ndarray
        3-element ECI position of the spacecraft (m).
    jupiter_position_eci : np.ndarray
        3-element ECI position of Jupiter (m).

    Returns
    -------
    np.ndarray
        3-element position vector centred on Jupiter (m).
    """
    return np.asarray(r_eci, dtype=np.float64) - np.asarray(
        jupiter_position_eci, dtype=np.float64
    )


# =============================================================================
# SIMPLIFIED EPHEMERIDES
# =============================================================================

def compute_sun_position(time_s: float) -> np.ndarray:
    """
    Compute a simplified Sun position in the ECI frame.

    The Sun is modelled on a circular orbit in the ecliptic plane at
    a distance of 1 AU from the Earth.  The ecliptic is tilted by
    23.4393 degrees relative to the equatorial plane (ECI Z-axis).

    The mean motion is:

        n_sun = 2*pi / T_year

    where T_year = 365.25 * 86400 s.  The Sun's position in the
    ecliptic is:

        r_ecliptic = AU * [cos(n*t), sin(n*t), 0]

    Rotated into the equatorial (ECI) frame by Rx(-obliquity):

        r_eci = Rx(-epsilon) @ r_ecliptic

    (Note: the Earth orbits the Sun, so in the geocentric ECI frame
    the Sun appears to orbit the Earth in the opposite sense.  We
    reverse the sign of the angle for convenience -- the important
    thing is that |r| = 1 AU and the Sun traces the ecliptic.)

    Parameters
    ----------
    time_s : float
        Elapsed time since epoch in seconds.

    Returns
    -------
    np.ndarray
        3-element ECI position of the Sun (m).
    """
    T_year = 365.25 * 86400.0  # sidereal year in seconds
    n_sun = TWO_PI / T_year
    obliquity = 23.4393 * DEG2RAD  # axial tilt

    # Sun position in ecliptic coordinates (geocentric)
    angle = n_sun * time_s
    r_ecliptic = AU * np.array([
        np.cos(angle),
        np.sin(angle),
        0.0,
    ], dtype=np.float64)

    # Rotate from ecliptic to equatorial (ECI)
    return Rx(-obliquity) @ r_ecliptic


def compute_moon_position(time_s: float) -> np.ndarray:
    """
    Compute a simplified Moon position in the ECI frame.

    The Moon is modelled on a circular orbit at the mean Earth-Moon
    distance (MOON_SMA) with period MOON_ORBITAL_PERIOD.  The orbital
    plane is inclined by MOON_INCLINATION to the ecliptic, which is
    itself tilted by 23.44 degrees to the equator.  For this simplified
    model we approximate the Moon's orbit as inclined to the equatorial
    plane by the sum of the ecliptic obliquity and the lunar inclination
    (a rough but serviceable approximation for trajectory-level work).

    The mean motion is:
        n_moon = 2*pi / MOON_ORBITAL_PERIOD

    Position in the lunar orbital plane:
        r_orbit = MOON_SMA * [cos(n*t), sin(n*t), 0]

    Rotated into ECI by Rx(-i_total):
        r_eci = Rx(-i_total) @ r_orbit

    Parameters
    ----------
    time_s : float
        Elapsed time since epoch in seconds.

    Returns
    -------
    np.ndarray
        3-element ECI position of the Moon (m).
    """
    n_moon = TWO_PI / MOON_ORBITAL_PERIOD
    obliquity = 23.4393 * DEG2RAD
    # Total inclination to equator (approximate)
    i_total = obliquity + MOON_INCLINATION

    angle = n_moon * time_s
    r_orbit = MOON_SMA * np.array([
        np.cos(angle),
        np.sin(angle),
        0.0,
    ], dtype=np.float64)

    return Rx(-i_total) @ r_orbit


def compute_jupiter_position(time_s: float) -> np.ndarray:
    """
    Compute a simplified Jupiter position in the ECI frame.

    Jupiter is modelled on a circular heliocentric orbit at JUPITER_SMA
    with period JUPITER_ORBITAL_PERIOD.  Since ECI is geocentric, we
    compute Jupiter's heliocentric position and the Earth's heliocentric
    position, then subtract to get the geocentric (ECI) position of
    Jupiter.

    Jupiter's heliocentric orbital plane is inclined by
    JUPITER_INCLINATION to the ecliptic.  The ecliptic is tilted by
    23.44 degrees to the equator.

    Procedure:
        1. Compute Sun position in ECI (= -Earth heliocentric in ECI)
        2. Compute Jupiter heliocentric in ecliptic, rotate to ECI
        3. Jupiter_ECI = Jupiter_helio_ECI - Earth_helio_ECI
                       = Jupiter_helio_ECI + Sun_ECI

    Parameters
    ----------
    time_s : float
        Elapsed time since epoch in seconds.

    Returns
    -------
    np.ndarray
        3-element ECI position of Jupiter (m).
    """
    obliquity = 23.4393 * DEG2RAD
    n_jup = TWO_PI / JUPITER_ORBITAL_PERIOD
    i_total = obliquity + JUPITER_INCLINATION

    angle = n_jup * time_s
    r_jup_helio_ecliptic = JUPITER_SMA * np.array([
        np.cos(angle),
        np.sin(angle),
        0.0,
    ], dtype=np.float64)

    # Rotate Jupiter heliocentric from ecliptic to equatorial
    r_jup_helio_eci = Rx(-i_total) @ r_jup_helio_ecliptic

    # Sun position in ECI is the negative of Earth's heliocentric position
    # Jupiter_ECI = Jupiter_helio - Earth_helio = Jupiter_helio + Sun_ECI
    sun_pos = compute_sun_position(time_s)

    return r_jup_helio_eci + sun_pos
