"""
===============================================================================
GNC PROJECT - Quaternion Mathematics Library
===============================================================================

Comprehensive quaternion implementation for spacecraft attitude representation,
propagation, and control. Quaternions are the preferred attitude representation
in GNC because they avoid the gimbal lock singularity inherent in Euler angles
and require only 4 parameters (vs 9 for a DCM), at the cost of a single
normalization constraint.

Convention
----------
We use the scalar-first convention:

    q = [q_w, q_x, q_y, q_z] = q_w + q_x*i + q_y*j + q_z*k

where q_w is the scalar (real) part and [q_x, q_y, q_z] is the vector
(imaginary) part. This convention follows Hamilton's original formulation
and is standard in the aerospace/GNC community (JPL, NASA, ESA).

The quaternion represents a rotation from frame A to frame B:

    v_B = q * v_A * q_conjugate

where v is a pure quaternion embedding of a 3-vector (v_w = 0).

Unit quaternion constraint: |q| = sqrt(q_w^2 + q_x^2 + q_y^2 + q_z^2) = 1

Euler Angle Convention
----------------------
We use the aerospace-standard 3-2-1 (ZYX) rotation sequence:
    1. Yaw   (psi)   about the Z-axis
    2. Pitch (theta) about the new Y-axis
    3. Roll  (phi)   about the new X-axis

This is the most common convention for aircraft and spacecraft, mapping
directly to heading/elevation/bank angles.

References
----------
    [1] Markley & Crassidis, "Fundamentals of Spacecraft Attitude
        Determination and Control", Springer, 2014.
    [2] Wertz, "Space Mission Engineering: The New SMAD", Ch. 18-19.
    [3] Kuipers, "Quaternions and Rotation Sequences", Princeton, 1999.
    [4] Shuster, "A Survey of Attitude Representations", JASS, 1993.

===============================================================================
"""

import numpy as np
from typing import Tuple, Optional, Union


class Quaternion:
    """
    Unit quaternion class for 3D rotation representation.

    A unit quaternion q = [w, x, y, z] parameterizes a rotation by angle theta
    about unit axis n as:

        q = [cos(theta/2), sin(theta/2) * n_x, sin(theta/2) * n_y, sin(theta/2) * n_z]

    This half-angle encoding is what makes quaternion composition equivalent to
    rotation composition: the product of two unit quaternions corresponds to the
    composition of the two rotations.

    Attributes
    ----------
    w : float
        Scalar (real) component of the quaternion.
    x : float
        First imaginary component (i-axis).
    y : float
        Second imaginary component (j-axis).
    z : float
        Third imaginary component (k-axis).

    Examples
    --------
    >>> q = Quaternion(1.0, 0.0, 0.0, 0.0)  # Identity rotation
    >>> q_rot = Quaternion.from_euler(0.0, 0.0, np.pi / 2)  # 90-deg yaw
    >>> v_rotated = q_rot.rotate_vector(np.array([1.0, 0.0, 0.0]))
    """

    # =========================================================================
    # Tolerance for floating-point comparisons and normalization checks.
    # In spacecraft GNC, quaternion normalization drift must be caught early
    # to prevent attitude determination errors from accumulating.
    # =========================================================================
    _NORM_TOLERANCE = 1e-10
    _COMPARISON_TOLERANCE = 1e-9

    def __init__(self, w: float, x: float, y: float, z: float,
                 normalize: bool = True) -> None:
        """
        Initialize a quaternion with scalar-first convention.

        Parameters
        ----------
        w : float
            Scalar part (cos(theta/2) for a rotation by angle theta).
        x : float
            i-component of the vector part.
        y : float
            j-component of the vector part.
        z : float
            k-component of the vector part.
        normalize : bool, optional
            If True (default), normalize the quaternion to unit magnitude.
            Set to False only when you are certain the input is already
            unit-length (e.g., inside internal methods that guarantee this).

        Notes
        -----
        The constructor enforces the scalar-positive convention by default.
        Two quaternions q and -q represent the same rotation, so we pick
        the one with w >= 0 for uniqueness. This avoids discontinuities
        in telemetry and control loops.
        """
        self._q = np.array([w, x, y, z], dtype=np.float64)

        if normalize:
            self._normalize_in_place()

    # =========================================================================
    # PROPERTIES - Read access to quaternion components
    # =========================================================================

    @property
    def w(self) -> float:
        """Scalar (real) part of the quaternion."""
        return float(self._q[0])

    @property
    def x(self) -> float:
        """First imaginary component (i-axis)."""
        return float(self._q[1])

    @property
    def y(self) -> float:
        """Second imaginary component (j-axis)."""
        return float(self._q[2])

    @property
    def z(self) -> float:
        """Third imaginary component (k-axis)."""
        return float(self._q[3])

    @property
    def scalar(self) -> float:
        """Scalar part of the quaternion (alias for w)."""
        return self.w

    @property
    def vector(self) -> np.ndarray:
        """
        Vector (imaginary) part of the quaternion as a 3-element array.

        Returns
        -------
        np.ndarray
            Array [x, y, z] representing the imaginary components.
        """
        return self._q[1:4].copy()

    @property
    def components(self) -> np.ndarray:
        """
        Full quaternion as a 4-element numpy array [w, x, y, z].

        Returns
        -------
        np.ndarray
            Copy of the internal quaternion array.
        """
        return self._q.copy()

    @property
    def norm(self) -> float:
        """
        L2 norm (magnitude) of the quaternion.

        For a valid rotation quaternion, this should always be 1.0
        (within floating-point tolerance).

        Returns
        -------
        float
            Euclidean norm sqrt(w^2 + x^2 + y^2 + z^2).
        """
        return float(np.linalg.norm(self._q))

    @property
    def rotation_angle(self) -> float:
        """
        Total rotation angle in radians [0, pi].

        For a unit quaternion q = [cos(theta/2), sin(theta/2)*n],
        the rotation angle is theta = 2 * arccos(|w|).

        Returns
        -------
        float
            Rotation angle in radians, always in [0, pi].
        """
        # Clamp to [-1, 1] to protect against floating-point overshoot in arccos
        return 2.0 * np.arccos(np.clip(abs(self.w), -1.0, 1.0))

    @property
    def rotation_axis(self) -> np.ndarray:
        """
        Unit rotation axis.

        For a unit quaternion q = [cos(theta/2), sin(theta/2)*n],
        the rotation axis is n = vector / |vector|.

        Returns
        -------
        np.ndarray
            Unit 3-vector along the rotation axis. Returns [0, 0, 1]
            for the identity quaternion (where the axis is undefined).
        """
        vec = self.vector
        vec_norm = np.linalg.norm(vec)

        if vec_norm < self._NORM_TOLERANCE:
            # Identity rotation: axis is undefined, return Z by convention
            return np.array([0.0, 0.0, 1.0])

        return vec / vec_norm

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _normalize_in_place(self) -> None:
        """
        Normalize the quaternion to unit magnitude in-place.

        This is called by the constructor and after any operation that may
        cause the quaternion to drift from unit norm (e.g., numerical
        integration of the kinematic equation).

        Raises
        ------
        ValueError
            If the quaternion has near-zero norm (degenerate case).
        """
        n = np.linalg.norm(self._q)

        if n < self._NORM_TOLERANCE:
            raise ValueError(
                f"Cannot normalize near-zero quaternion (norm = {n:.2e}). "
                "This indicates a degenerate state in the attitude solution."
            )

        self._q /= n

        # Enforce scalar-positive convention for uniqueness.
        # q and -q represent the same rotation, so we always pick w >= 0.
        if self._q[0] < 0.0:
            self._q = -self._q

    # =========================================================================
    # STATIC FACTORY METHODS
    # =========================================================================

    @staticmethod
    def identity() -> 'Quaternion':
        """
        Create the identity quaternion [1, 0, 0, 0].

        The identity quaternion represents zero rotation (no change in
        orientation). It is the multiplicative identity element:
        q * identity = q for any quaternion q.

        Returns
        -------
        Quaternion
            The identity quaternion.
        """
        return Quaternion(1.0, 0.0, 0.0, 0.0, normalize=False)

    @staticmethod
    def from_euler(phi: float, theta: float, psi: float) -> 'Quaternion':
        """
        Create a quaternion from 3-2-1 (ZYX) Euler angles.

        The 3-2-1 rotation sequence applies rotations in this order:
            1. Yaw   (psi)   about Z-axis
            2. Pitch (theta) about the rotated Y-axis
            3. Roll  (phi)   about the twice-rotated X-axis

        The resulting quaternion transforms vectors from the original frame
        to the rotated body frame.

        Parameters
        ----------
        phi : float
            Roll angle about the X-axis (radians).
        theta : float
            Pitch angle about the Y-axis (radians). Must be in
            (-pi/2, pi/2) to avoid gimbal lock.
        psi : float
            Yaw angle about the Z-axis (radians).

        Returns
        -------
        Quaternion
            Unit quaternion equivalent to the Euler angle sequence.

        Notes
        -----
        The closed-form conversion is derived by multiplying the three
        single-axis quaternions:

            q = q_z(psi) * q_y(theta) * q_x(phi)

        where each single-axis quaternion is:

            q_x(a) = [cos(a/2), sin(a/2), 0, 0]
            q_y(a) = [cos(a/2), 0, sin(a/2), 0]
            q_z(a) = [cos(a/2), 0, 0, sin(a/2)]

        References
        ----------
        Diebel, "Representing Attitude: Euler Angles, Unit Quaternions, and
        Rotation Vectors", Stanford, 2006, Eq. 290.
        """
        # Half-angles for efficiency (each trig function called once)
        c_phi   = np.cos(phi / 2.0)
        s_phi   = np.sin(phi / 2.0)
        c_theta = np.cos(theta / 2.0)
        s_theta = np.sin(theta / 2.0)
        c_psi   = np.cos(psi / 2.0)
        s_psi   = np.sin(psi / 2.0)

        # Quaternion components from the 3-2-1 product
        w = c_phi * c_theta * c_psi + s_phi * s_theta * s_psi
        x = s_phi * c_theta * c_psi - c_phi * s_theta * s_psi
        y = c_phi * s_theta * c_psi + s_phi * c_theta * s_psi
        z = c_phi * c_theta * s_psi - s_phi * s_theta * c_psi

        return Quaternion(w, x, y, z)

    @staticmethod
    def from_dcm(dcm: np.ndarray) -> 'Quaternion':
        """
        Create a quaternion from a Direction Cosine Matrix (DCM).

        Uses Shepperd's method, which is numerically robust for all
        rotations. The naive method (based on trace alone) fails when the
        rotation angle is near 180 degrees because the trace approaches -1
        and the square root becomes numerically unstable. Shepperd's method
        avoids this by always extracting the largest quaternion component
        first.

        Parameters
        ----------
        dcm : np.ndarray
            3x3 proper orthogonal matrix (det = +1, R^T R = I).
            Represents a rotation from frame A to frame B.

        Returns
        -------
        Quaternion
            Unit quaternion equivalent to the given DCM.

        Raises
        ------
        ValueError
            If dcm is not a valid 3x3 rotation matrix.

        Notes
        -----
        Shepperd's method (JGCD, 1978):
            1. Compute the four diagonal quantities:
               d0 = 1 + trace(R)           -> proportional to 4*w^2
               d1 = 1 + 2*R[0,0] - trace   -> proportional to 4*x^2
               d2 = 1 + 2*R[1,1] - trace   -> proportional to 4*y^2
               d3 = 1 + 2*R[2,2] - trace   -> proportional to 4*z^2
            2. Choose the largest d_k to compute q_k = 0.5 * sqrt(d_k)
            3. Recover the remaining components from the off-diagonal elements
        """
        dcm = np.asarray(dcm, dtype=np.float64)

        if dcm.shape != (3, 3):
            raise ValueError(f"DCM must be 3x3, got shape {dcm.shape}")

        # Verify orthogonality (R^T R should be identity)
        orthogonality_error = np.linalg.norm(dcm.T @ dcm - np.eye(3))
        if orthogonality_error > 1e-6:
            raise ValueError(
                f"Input matrix is not orthogonal (error = {orthogonality_error:.2e}). "
                "Ensure the DCM satisfies R^T R = I."
            )

        trace = np.trace(dcm)

        # Shepperd's method: compute all four diagonal quantities
        d0 = 1.0 + trace                          # 4*w^2
        d1 = 1.0 + 2.0 * dcm[0, 0] - trace       # 4*x^2
        d2 = 1.0 + 2.0 * dcm[1, 1] - trace       # 4*y^2
        d3 = 1.0 + 2.0 * dcm[2, 2] - trace       # 4*z^2

        # Choose the largest to maximize numerical stability
        d_max = max(d0, d1, d2, d3)

        if d_max == d0:
            # w is largest
            w = 0.5 * np.sqrt(d0)
            scale = 0.25 / w
            x = (dcm[2, 1] - dcm[1, 2]) * scale
            y = (dcm[0, 2] - dcm[2, 0]) * scale
            z = (dcm[1, 0] - dcm[0, 1]) * scale
        elif d_max == d1:
            # x is largest
            x = 0.5 * np.sqrt(d1)
            scale = 0.25 / x
            w = (dcm[2, 1] - dcm[1, 2]) * scale
            y = (dcm[0, 1] + dcm[1, 0]) * scale
            z = (dcm[0, 2] + dcm[2, 0]) * scale
        elif d_max == d2:
            # y is largest
            y = 0.5 * np.sqrt(d2)
            scale = 0.25 / y
            w = (dcm[0, 2] - dcm[2, 0]) * scale
            x = (dcm[0, 1] + dcm[1, 0]) * scale
            z = (dcm[1, 2] + dcm[2, 1]) * scale
        else:
            # z is largest
            z = 0.5 * np.sqrt(d3)
            scale = 0.25 / z
            w = (dcm[1, 0] - dcm[0, 1]) * scale
            x = (dcm[0, 2] + dcm[2, 0]) * scale
            y = (dcm[1, 2] + dcm[2, 1]) * scale

        return Quaternion(w, x, y, z)

    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle: float) -> 'Quaternion':
        """
        Create a quaternion from an axis-angle representation.

        The axis-angle representation parameterizes a rotation as a unit
        vector n (the axis) and a scalar angle theta. The corresponding
        quaternion is:

            q = [cos(theta/2), sin(theta/2) * n]

        This is the most intuitive connection between quaternions and
        rotations: a rotation by theta radians about axis n.

        Parameters
        ----------
        axis : np.ndarray
            3-element rotation axis vector. Will be normalized internally.
        angle : float
            Rotation angle in radians.

        Returns
        -------
        Quaternion
            Unit quaternion representing the specified rotation.

        Raises
        ------
        ValueError
            If axis has near-zero magnitude.
        """
        axis = np.asarray(axis, dtype=np.float64)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-12:
            raise ValueError(
                "Rotation axis has near-zero magnitude. "
                "Cannot define a rotation about a zero vector."
            )

        # Normalize the axis to unit length
        n = axis / axis_norm

        # Half-angle encoding
        half_angle = angle / 2.0
        w = np.cos(half_angle)
        sin_half = np.sin(half_angle)

        return Quaternion(w, sin_half * n[0], sin_half * n[1], sin_half * n[2])

    @staticmethod
    def from_rotation_vector(rot_vec: np.ndarray) -> 'Quaternion':
        """
        Create a quaternion from a rotation vector (Rodrigues vector).

        A rotation vector packs the axis and angle into a single 3-vector:
            rot_vec = theta * n
        where theta = |rot_vec| is the angle and n = rot_vec / |rot_vec|
        is the axis.

        This representation is convenient for small-angle perturbations
        (e.g., attitude error in Kalman filters) because for small theta:
            rot_vec ~ 2 * q_vector (the imaginary part of the quaternion)

        Parameters
        ----------
        rot_vec : np.ndarray
            3-element rotation vector (radians). Direction is the axis,
            magnitude is the angle.

        Returns
        -------
        Quaternion
            Unit quaternion representing the rotation.
        """
        rot_vec = np.asarray(rot_vec, dtype=np.float64)
        angle = np.linalg.norm(rot_vec)

        if angle < 1e-12:
            # Near-zero rotation: return identity
            return Quaternion.identity()

        axis = rot_vec / angle
        return Quaternion.from_axis_angle(axis, angle)

    # =========================================================================
    # QUATERNION ARITHMETIC
    # =========================================================================

    def conjugate(self) -> 'Quaternion':
        """
        Return the quaternion conjugate.

        For q = [w, x, y, z], the conjugate is q* = [w, -x, -y, -z].

        For unit quaternions, the conjugate equals the inverse and represents
        the reverse rotation. If q rotates from frame A to frame B, then
        q* rotates from frame B to frame A.

        Returns
        -------
        Quaternion
            The conjugate quaternion.
        """
        return Quaternion(self.w, -self.x, -self.y, -self.z, normalize=False)

    def inverse(self) -> 'Quaternion':
        """
        Return the quaternion inverse.

        For a unit quaternion, the inverse equals the conjugate:
            q^{-1} = q* / |q|^2 = q*  (when |q| = 1)

        For non-unit quaternions (which shouldn't arise in attitude work),
        the full formula is used.

        Returns
        -------
        Quaternion
            The multiplicative inverse q^{-1} such that q * q^{-1} = identity.
        """
        norm_sq = np.dot(self._q, self._q)
        conj = np.array([self.w, -self.x, -self.y, -self.z])
        inv_q = conj / norm_sq
        return Quaternion(inv_q[0], inv_q[1], inv_q[2], inv_q[3])

    def normalize(self) -> 'Quaternion':
        """
        Return a new normalized (unit-magnitude) quaternion.

        During numerical propagation (e.g., Runge-Kutta integration of the
        attitude kinematic equation), the quaternion norm drifts from unity
        due to floating-point accumulation. This method should be called
        after every integration step to maintain the unit constraint.

        Returns
        -------
        Quaternion
            A new quaternion with |q| = 1.
        """
        return Quaternion(self.w, self.x, self.y, self.z, normalize=True)

    def multiply(self, other: 'Quaternion') -> 'Quaternion':
        """
        Multiply this quaternion by another (Hamilton product).

        Quaternion multiplication is NOT commutative: q1 * q2 != q2 * q1
        in general. The product represents sequential rotation: first by
        other, then by self. That is:

            q_result = self * other

        rotates a vector first by 'other' and then by 'self'.

        The Hamilton product formula is:

            (a1 + b1*i + c1*j + d1*k) * (a2 + b2*i + c2*j + d2*k) =

            (a1*a2 - b1*b2 - c1*c2 - d1*d2) +
            (a1*b2 + b1*a2 + c1*d2 - d1*c2) i +
            (a1*c2 - b1*d2 + c1*a2 + d1*b2) j +
            (a1*d2 + b1*c2 - c1*b2 + d1*a2) k

        Parameters
        ----------
        other : Quaternion
            The right-hand quaternion in the product.

        Returns
        -------
        Quaternion
            The Hamilton product self * other.
        """
        # Extract components for clarity
        a1, b1, c1, d1 = self.w, self.x, self.y, self.z
        a2, b2, c2, d2 = other.w, other.x, other.y, other.z

        # Hamilton product
        w = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
        x = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
        y = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
        z = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2

        return Quaternion(w, x, y, z)

    # =========================================================================
    # ROTATION OPERATIONS
    # =========================================================================

    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Rotate a 3D vector by this quaternion.

        Applies the rotation using the sandwich product:

            v' = q * v_pure * q*

        where v_pure = [0, v_x, v_y, v_z] is the vector embedded as a
        pure quaternion. This is equivalent to the matrix operation v' = R * v
        where R is the DCM corresponding to this quaternion.

        For computational efficiency, we use the optimized Rodrigues form
        instead of the full quaternion triple product:

            v' = v + 2*w*(t) + 2*(u x t)

        where u = [x, y, z] (vector part of q) and t = u x v.
        This requires only 2 cross products and some additions (15 multiplies,
        15 adds) vs the naive method's 28 multiplies.

        Parameters
        ----------
        v : np.ndarray
            3-element vector to rotate.

        Returns
        -------
        np.ndarray
            Rotated 3-element vector.

        References
        ----------
        This optimization is described in:
        Markley & Crassidis (2014), Eq. 2.89.
        """
        v = np.asarray(v, dtype=np.float64)
        u = self.vector  # [x, y, z]

        # Rodrigues optimized rotation
        t = 2.0 * np.cross(u, v)
        return v + self.w * t + np.cross(u, t)

    # =========================================================================
    # CONVERSION METHODS
    # =========================================================================

    def to_euler(self) -> Tuple[float, float, float]:
        """
        Convert to 3-2-1 (ZYX) Euler angles.

        Extracts roll (phi), pitch (theta), and yaw (psi) from the
        quaternion using the standard aerospace convention:

            phi   = atan2(2*(w*x + y*z), 1 - 2*(x^2 + y^2))
            theta = arcsin(2*(w*y - z*x))
            psi   = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))

        Returns
        -------
        tuple of (float, float, float)
            (phi, theta, psi) = (roll, pitch, yaw) in radians.
            phi in [-pi, pi], theta in [-pi/2, pi/2], psi in [-pi, pi].

        Warnings
        --------
        Gimbal lock occurs when theta = +/-pi/2 (pitch = +/-90 degrees).
        At this singularity, roll and yaw become coupled and only their
        sum/difference is determinable. If |theta| > 89.9 degrees, consider
        using quaternions directly in your control law.

        Notes
        -----
        The atan2 function is used instead of atan to correctly handle all
        four quadrants and avoid division by zero.
        """
        w, x, y, z = self.w, self.x, self.y, self.z

        # Roll (phi) - rotation about X-axis
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        phi = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (theta) - rotation about Y-axis
        # Clamp to [-1, 1] to prevent NaN from arcsin due to float rounding
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        theta = np.arcsin(sinp)

        # Yaw (psi) - rotation about Z-axis
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        psi = np.arctan2(siny_cosp, cosy_cosp)

        return (phi, theta, psi)

    def to_dcm(self) -> np.ndarray:
        """
        Convert to a Direction Cosine Matrix (DCM / rotation matrix).

        The DCM R is a 3x3 proper orthogonal matrix (R^T R = I, det(R) = +1)
        that rotates vectors from the reference frame to the body frame:

            v_body = R * v_reference

        The elements of R in terms of quaternion components are:

            R = | 1-2(y^2+z^2)    2(xy-wz)      2(xz+wy)   |
                | 2(xy+wz)      1-2(x^2+z^2)    2(yz-wx)   |
                | 2(xz-wy)      2(yz+wx)      1-2(x^2+y^2) |

        Returns
        -------
        np.ndarray
            3x3 rotation matrix (Direction Cosine Matrix).

        Notes
        -----
        This is derived by expanding the sandwich product v' = q * v * q*
        and collecting terms into matrix form. See Markley & Crassidis (2014),
        Eq. 2.90.
        """
        w, x, y, z = self.w, self.x, self.y, self.z

        # Pre-compute products that appear multiple times
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        # Assemble the DCM
        dcm = np.array([
            [1.0 - 2.0 * (yy + zz),  2.0 * (xy - wz),        2.0 * (xz + wy)],
            [2.0 * (xy + wz),         1.0 - 2.0 * (xx + zz),  2.0 * (yz - wx)],
            [2.0 * (xz - wy),         2.0 * (yz + wx),         1.0 - 2.0 * (xx + yy)]
        ], dtype=np.float64)

        return dcm

    def to_axis_angle(self) -> Tuple[np.ndarray, float]:
        """
        Convert to axis-angle representation.

        Extracts the rotation axis (unit vector) and rotation angle from
        the quaternion:

            angle = 2 * arccos(w)
            axis  = [x, y, z] / sin(angle/2)

        Returns
        -------
        tuple of (np.ndarray, float)
            (axis, angle) where axis is a 3-element unit vector and angle
            is in radians in [0, pi].

        Notes
        -----
        For the identity quaternion (zero rotation), the axis is undefined.
        We return [0, 0, 1] by convention (Z-axis) with angle = 0.
        """
        return (self.rotation_axis, self.rotation_angle)

    def to_rotation_vector(self) -> np.ndarray:
        """
        Convert to rotation vector (Rodrigues vector).

        The rotation vector is:  rot_vec = angle * axis

        This is the logarithmic map from SO(3) to its Lie algebra so(3),
        useful for small-angle representations in Kalman filters and
        for computing attitude errors.

        Returns
        -------
        np.ndarray
            3-element rotation vector (radians).
        """
        axis, angle = self.to_axis_angle()
        return angle * axis

    # =========================================================================
    # ATTITUDE PROPAGATION
    # =========================================================================

    def derivative(self, omega: np.ndarray) -> np.ndarray:
        """
        Compute the quaternion time-derivative for attitude propagation.

        The quaternion kinematic differential equation relates the time
        derivative of the attitude quaternion to the angular velocity:

            dq/dt = 0.5 * q (*) omega_q

        where omega_q = [0, omega_x, omega_y, omega_z] is the angular
        velocity vector embedded as a pure quaternion, and (*) denotes
        the Hamilton product.

        In component form:

            | dw/dt |       | -x  -y  -z |   | omega_x |
            | dx/dt | = 0.5 |  w  -z   y | * | omega_y |
            | dy/dt |       |  z   w  -x |   | omega_z |
            | dz/dt |       | -y   x   w |

        This is the fundamental equation used by the attitude propagation
        module. It is integrated using RK4 or similar methods, with
        renormalization after each step.

        Parameters
        ----------
        omega : np.ndarray
            3-element angular velocity vector [omega_x, omega_y, omega_z]
            in rad/s, expressed in the body frame.

        Returns
        -------
        np.ndarray
            4-element array [dw/dt, dx/dt, dy/dt, dz/dt], the time
            derivative of the quaternion components.

        Notes
        -----
        The angular velocity is assumed to be expressed in the BODY frame
        (as measured by on-board gyroscopes). If omega is in the inertial
        frame, use the left-multiply form instead:

            dq/dt = 0.5 * omega_q (*) q

        References
        ----------
        Wertz, "Space Mission Engineering: The New SMAD", Eq. 18-14.
        Markley & Crassidis (2014), Eq. 3.20.
        """
        omega = np.asarray(omega, dtype=np.float64)
        w, x, y, z = self.w, self.x, self.y, self.z
        ox, oy, oz = omega[0], omega[1], omega[2]

        # Omega matrix (4x3) from the kinematic equation
        # dq/dt = 0.5 * Omega_matrix * omega
        q_dot = 0.5 * np.array([
            -x * ox - y * oy - z * oz,
             w * ox - z * oy + y * oz,
             z * ox + w * oy - x * oz,
            -y * ox + x * oy + w * oz
        ], dtype=np.float64)

        return q_dot

    def propagate(self, omega: np.ndarray, dt: float) -> 'Quaternion':
        """
        Propagate the quaternion forward in time using a first-order update.

        For small time steps, the quaternion can be updated via:

            q(t + dt) = q(t) + dq/dt * dt

        followed by renormalization. For higher accuracy, use RK4 integration
        with the derivative() method directly.

        This simple Euler step is adequate for dt << 1/|omega| (i.e., the
        time step is much smaller than the rotation period).

        Parameters
        ----------
        omega : np.ndarray
            Angular velocity vector [omega_x, omega_y, omega_z] in rad/s
            (body frame).
        dt : float
            Time step in seconds.

        Returns
        -------
        Quaternion
            The propagated quaternion at time t + dt.
        """
        q_dot = self.derivative(omega)
        new_q = self._q + q_dot * dt
        return Quaternion(new_q[0], new_q[1], new_q[2], new_q[3], normalize=True)

    # =========================================================================
    # ATTITUDE CONTROL SUPPORT
    # =========================================================================

    def error_quaternion(self, q_desired: 'Quaternion') -> 'Quaternion':
        """
        Compute the error quaternion between the current and desired attitude.

        The error quaternion q_err represents the rotation needed to go from
        the current attitude to the desired attitude:

            q_err = q_desired * q_current^{-1}

        such that:

            q_desired = q_err * q_current

        In a PD attitude controller, the error quaternion's vector part
        [q_err_x, q_err_y, q_err_z] is proportional to the small-angle
        attitude error and drives the control torque:

            tau = -K_p * q_err_vector - K_d * omega_err

        Parameters
        ----------
        q_desired : Quaternion
            The target (commanded) attitude quaternion.

        Returns
        -------
        Quaternion
            The error quaternion. For small errors, the vector part
            approximates the rotation vector error (in radians) divided by 2.

        Notes
        -----
        The error quaternion is always returned with w >= 0 (short-rotation
        convention) to ensure the controller takes the shortest path.
        """
        return q_desired.multiply(self.inverse())

    def angle_to(self, other: 'Quaternion') -> float:
        """
        Compute the rotation angle between this quaternion and another.

        This is the geodesic distance on the unit quaternion hypersphere,
        equivalent to the minimum rotation angle needed to get from one
        attitude to the other.

            angle = 2 * arccos(|q1 . q2|)

        where q1 . q2 is the 4D dot product.

        Parameters
        ----------
        other : Quaternion
            Another unit quaternion.

        Returns
        -------
        float
            Rotation angle in radians, in [0, pi].
        """
        # Inner product, clamped for numerical safety
        dot = np.clip(abs(np.dot(self._q, other._q)), 0.0, 1.0)
        return 2.0 * np.arccos(dot)

    # =========================================================================
    # INTERPOLATION
    # =========================================================================

    @staticmethod
    def slerp(q1: 'Quaternion', q2: 'Quaternion', t: float) -> 'Quaternion':
        """
        Spherical Linear Interpolation (SLERP) between two quaternions.

        SLERP produces the shortest-path interpolation on the unit quaternion
        hypersphere (S^3). It is the quaternion analog of great-circle
        interpolation on a sphere, maintaining constant angular velocity.

        The formula is:

            slerp(q1, q2, t) = q1 * sin((1-t)*Omega) / sin(Omega)
                              + q2 * sin(t*Omega) / sin(Omega)

        where Omega = arccos(q1 . q2) is the angle between the quaternions.

        Parameters
        ----------
        q1 : Quaternion
            Starting quaternion (at t=0).
        q2 : Quaternion
            Ending quaternion (at t=1).
        t : float
            Interpolation parameter in [0, 1].
            t=0 returns q1, t=1 returns q2, t=0.5 returns the midpoint.

        Returns
        -------
        Quaternion
            Interpolated quaternion at parameter t.

        Notes
        -----
        - Always interpolates along the SHORT arc. If the dot product
          q1 . q2 < 0, we negate q2 before interpolating to ensure the
          short path.
        - For very small angles (Omega ~ 0), falls back to normalized
          linear interpolation (NLERP) to avoid division by zero.

        Applications
        ------------
        - Smooth attitude maneuver planning (slew commands)
        - Interpolating between keyframe attitudes in trajectory design
        - Telemetry data resampling at different time rates
        """
        # Ensure t is in valid range
        t = np.clip(t, 0.0, 1.0)

        # Compute the cosine of the angle between the quaternions
        dot = np.dot(q1._q, q2._q)

        # If the dot product is negative, negate q2 to take the short path.
        # (q and -q represent the same rotation)
        q2_q = q2._q.copy()
        if dot < 0.0:
            q2_q = -q2_q
            dot = -dot

        # Clamp for numerical safety
        dot = np.clip(dot, 0.0, 1.0)

        if dot > 0.9995:
            # Quaternions are nearly identical: use NLERP to avoid
            # numerical instability in sin(Omega) ~ 0
            result = q1._q + t * (q2_q - q1._q)
            result /= np.linalg.norm(result)
            return Quaternion(result[0], result[1], result[2], result[3],
                              normalize=False)

        # Standard SLERP formula
        omega = np.arccos(dot)          # Angle between quaternions
        sin_omega = np.sin(omega)

        scale1 = np.sin((1.0 - t) * omega) / sin_omega
        scale2 = np.sin(t * omega) / sin_omega

        result = scale1 * q1._q + scale2 * q2_q
        return Quaternion(result[0], result[1], result[2], result[3])

    # =========================================================================
    # OPERATOR OVERLOADS
    # =========================================================================

    def __mul__(self, other: Union['Quaternion', float, int]) -> 'Quaternion':
        """
        Multiplication operator.

        - Quaternion * Quaternion -> Hamilton product (rotation composition)
        - Quaternion * scalar -> component-wise scaling (rarely used;
          the result is not a unit quaternion)

        Parameters
        ----------
        other : Quaternion or float
            Right-hand operand.

        Returns
        -------
        Quaternion
            Product quaternion.
        """
        if isinstance(other, Quaternion):
            return self.multiply(other)
        elif isinstance(other, (int, float)):
            q = self._q * float(other)
            return Quaternion(q[0], q[1], q[2], q[3])
        return NotImplemented

    def __rmul__(self, other: Union[float, int]) -> 'Quaternion':
        """Right-multiplication by a scalar: scalar * Quaternion."""
        if isinstance(other, (int, float)):
            q = self._q * float(other)
            return Quaternion(q[0], q[1], q[2], q[3])
        return NotImplemented

    def __add__(self, other: 'Quaternion') -> 'Quaternion':
        """
        Component-wise addition of two quaternions.

        This is NOT a rotation operation -- it is used in numerical
        integration schemes (e.g., Euler step: q_new = q + q_dot * dt)
        and in SLERP/NLERP interpolation. The result should generally
        be renormalized.

        Parameters
        ----------
        other : Quaternion
            Quaternion to add.

        Returns
        -------
        Quaternion
            Sum quaternion (automatically normalized).
        """
        if isinstance(other, Quaternion):
            q = self._q + other._q
            return Quaternion(q[0], q[1], q[2], q[3])
        return NotImplemented

    def __sub__(self, other: 'Quaternion') -> 'Quaternion':
        """
        Component-wise subtraction of two quaternions.

        Used in numerical methods (e.g., finite-difference derivatives).

        Parameters
        ----------
        other : Quaternion
            Quaternion to subtract.

        Returns
        -------
        Quaternion
            Difference quaternion (automatically normalized).
        """
        if isinstance(other, Quaternion):
            q = self._q - other._q
            return Quaternion(q[0], q[1], q[2], q[3])
        return NotImplemented

    def __neg__(self) -> 'Quaternion':
        """
        Negate all components.

        Note: -q represents the same rotation as q. This is included
        for mathematical completeness.
        """
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def __eq__(self, other: object) -> bool:
        """
        Equality comparison with tolerance.

        Two quaternions are considered equal if they represent the same
        rotation, accounting for the q/-q ambiguity and floating-point
        tolerance.

        Parameters
        ----------
        other : Quaternion
            Quaternion to compare against.

        Returns
        -------
        bool
            True if the quaternions represent the same rotation.
        """
        if not isinstance(other, Quaternion):
            return NotImplemented

        # Check both q and -q (same rotation)
        diff_pos = np.linalg.norm(self._q - other._q)
        diff_neg = np.linalg.norm(self._q + other._q)
        return min(diff_pos, diff_neg) < self._COMPARISON_TOLERANCE

    def __repr__(self) -> str:
        """
        Unambiguous string representation for debugging.

        Format: Quaternion(w=..., x=..., y=..., z=...)
        """
        return (f"Quaternion(w={self.w:+.8f}, x={self.x:+.8f}, "
                f"y={self.y:+.8f}, z={self.z:+.8f})")

    def __str__(self) -> str:
        """
        Human-readable string representation.

        Shows the quaternion components and the equivalent rotation
        angle for quick interpretation.
        """
        angle_deg = np.degrees(self.rotation_angle)
        return (f"[{self.w:+.6f}, {self.x:+.6f}, {self.y:+.6f}, "
                f"{self.z:+.6f}] (rot={angle_deg:.2f} deg)")

    def __hash__(self) -> int:
        """Hash based on rounded components for use in sets/dicts."""
        # Round to avoid floating-point hash collisions
        rounded = tuple(np.round(self._q, decimals=8))
        return hash(rounded)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def is_unit(self, tolerance: float = 1e-8) -> bool:
        """
        Check if this quaternion has unit norm.

        Parameters
        ----------
        tolerance : float
            Acceptable deviation from 1.0.

        Returns
        -------
        bool
            True if |q| is within tolerance of 1.0.
        """
        return abs(self.norm - 1.0) < tolerance

    def as_attitude_string(self) -> str:
        """
        Format the quaternion as a human-readable attitude description.

        Converts to Euler angles for display, showing roll/pitch/yaw
        in degrees. Useful for telemetry and logging.

        Returns
        -------
        str
            Formatted string with Euler angles in degrees.
        """
        phi, theta, psi = self.to_euler()
        return (f"Roll={np.degrees(phi):+7.2f} deg, "
                f"Pitch={np.degrees(theta):+7.2f} deg, "
                f"Yaw={np.degrees(psi):+7.2f} deg")

    def copy(self) -> 'Quaternion':
        """Return a deep copy of this quaternion."""
        return Quaternion(self.w, self.x, self.y, self.z, normalize=False)

    @staticmethod
    def random() -> 'Quaternion':
        """
        Generate a uniformly random unit quaternion.

        Uses the subgroup algorithm (Shoemake, 1992) to produce a quaternion
        uniformly distributed over SO(3). Simply normalizing a random
        4-vector does NOT produce a uniform rotation distribution.

        Returns
        -------
        Quaternion
            A random unit quaternion representing a uniformly distributed
            random rotation.

        References
        ----------
        Shoemake, "Uniform Random Rotations", Graphics Gems III, 1992.
        """
        # Three uniform random numbers in [0, 1)
        u1, u2, u3 = np.random.random(3)

        # Shoemake's method
        sqrt_u1 = np.sqrt(u1)
        sqrt_1_minus_u1 = np.sqrt(1.0 - u1)

        w = sqrt_1_minus_u1 * np.sin(2.0 * np.pi * u2)
        x = sqrt_1_minus_u1 * np.cos(2.0 * np.pi * u2)
        y = sqrt_u1 * np.sin(2.0 * np.pi * u3)
        z = sqrt_u1 * np.cos(2.0 * np.pi * u3)

        return Quaternion(w, x, y, z)
