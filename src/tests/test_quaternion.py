"""
===============================================================================
GNC PROJECT - Quaternion Mathematics Test Suite
===============================================================================
Thorough tests for the Quaternion class covering identity, normalization,
conjugate, multiplication, rotation, Euler/DCM conversions, SLERP
interpolation, kinematic derivatives, and axis-angle construction.

All floating-point comparisons use numpy.testing.assert_allclose with
explicit tolerances appropriate for double-precision arithmetic.
===============================================================================
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from numpy.testing import assert_allclose

from core.quaternion import Quaternion


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def identity_quat():
    """Return the identity quaternion [1, 0, 0, 0]."""
    return Quaternion.identity()


@pytest.fixture
def quat_90z():
    """Return a quaternion representing 90-degree rotation about Z axis."""
    return Quaternion.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)


@pytest.fixture
def quat_45x():
    """Return a quaternion representing 45-degree rotation about X axis."""
    return Quaternion.from_axis_angle(np.array([1.0, 0.0, 0.0]), np.pi / 4)


@pytest.fixture
def random_quat():
    """Return a deterministic 'random' quaternion for reproducible tests."""
    np.random.seed(42)
    return Quaternion.random()


# =============================================================================
# Test: Identity quaternion
# =============================================================================

class TestIdentity:
    """Tests for the identity quaternion."""

    def test_identity(self, identity_quat):
        """Quaternion.identity() should be [1, 0, 0, 0]."""
        assert_allclose(identity_quat.components, [1.0, 0.0, 0.0, 0.0], atol=1e-15)

    def test_identity_is_unit(self, identity_quat):
        """Identity quaternion must have unit norm."""
        assert identity_quat.is_unit()

    def test_identity_rotation_angle_is_zero(self, identity_quat):
        """Identity quaternion represents zero rotation."""
        assert_allclose(identity_quat.rotation_angle, 0.0, atol=1e-15)


# =============================================================================
# Test: Normalization
# =============================================================================

class TestNormalize:
    """Tests for quaternion normalization."""

    def test_normalize(self):
        """An unnormalized quaternion should be automatically normalized."""
        q = Quaternion(2.0, 0.0, 0.0, 0.0)
        assert_allclose(q.norm, 1.0, atol=1e-15)
        assert_allclose(q.components, [1.0, 0.0, 0.0, 0.0], atol=1e-15)

    def test_normalize_general(self):
        """Normalization of a general quaternion preserves direction."""
        q = Quaternion(1.0, 1.0, 1.0, 1.0)
        assert_allclose(q.norm, 1.0, atol=1e-15)
        expected = np.array([1.0, 1.0, 1.0, 1.0]) / 2.0
        assert_allclose(q.components, expected, atol=1e-15)

    @pytest.mark.parametrize("w,x,y,z", [
        (3.0, 4.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 2.0, 3.0, 4.0),
    ])
    def test_normalize_parametrized(self, w, x, y, z):
        """After construction, norm must be 1 for various inputs."""
        q = Quaternion(w, x, y, z)
        assert_allclose(q.norm, 1.0, atol=1e-14)


# =============================================================================
# Test: Conjugate
# =============================================================================

class TestConjugate:
    """Tests for quaternion conjugate."""

    def test_conjugate(self, quat_90z):
        """q.conjugate() should flip the vector part signs."""
        q = quat_90z
        qc = q.conjugate()
        assert_allclose(qc.w, q.w, atol=1e-15)
        assert_allclose(qc.x, -q.x, atol=1e-15)
        assert_allclose(qc.y, -q.y, atol=1e-15)
        assert_allclose(qc.z, -q.z, atol=1e-15)

    def test_conjugate_identity(self, identity_quat):
        """Conjugate of identity is identity."""
        qc = identity_quat.conjugate()
        assert_allclose(qc.components, identity_quat.components, atol=1e-15)

    def test_double_conjugate(self, random_quat):
        """Double conjugate returns the original quaternion."""
        q = random_quat
        qcc = q.conjugate().conjugate()
        assert_allclose(qcc.components, q.components, atol=1e-14)


# =============================================================================
# Test: Multiply with identity
# =============================================================================

class TestMultiplyIdentity:
    """Tests for multiplication with identity."""

    def test_multiply_identity(self, quat_90z, identity_quat):
        """q * identity = q for any quaternion q."""
        result = quat_90z.multiply(identity_quat)
        assert_allclose(result.components, quat_90z.components, atol=1e-14)

    def test_identity_multiply(self, quat_90z, identity_quat):
        """identity * q = q for any quaternion q."""
        result = identity_quat.multiply(quat_90z)
        assert_allclose(result.components, quat_90z.components, atol=1e-14)


# =============================================================================
# Test: Multiply with inverse
# =============================================================================

class TestMultiplyInverse:
    """Tests for multiplication with inverse."""

    def test_multiply_inverse(self, quat_90z):
        """q * q.inverse() should yield the identity quaternion."""
        result = quat_90z.multiply(quat_90z.inverse())
        assert_allclose(result.components, [1.0, 0.0, 0.0, 0.0], atol=1e-14)

    def test_multiply_inverse_random(self, random_quat):
        """q * q.inverse() = identity for a random quaternion."""
        result = random_quat.multiply(random_quat.inverse())
        assert_allclose(result.components, [1.0, 0.0, 0.0, 0.0], atol=1e-14)


# =============================================================================
# Test: Rotate vector
# =============================================================================

class TestRotateVector:
    """Tests for vector rotation by a quaternion."""

    def test_rotate_vector_90z(self, quat_90z):
        """90-degree rotation about Z should rotate x-hat to y-hat."""
        x_hat = np.array([1.0, 0.0, 0.0])
        y_hat = np.array([0.0, 1.0, 0.0])
        result = quat_90z.rotate_vector(x_hat)
        assert_allclose(result, y_hat, atol=1e-14)

    def test_rotate_identity_no_change(self, identity_quat):
        """Rotation by identity should not change the vector."""
        v = np.array([1.0, 2.0, 3.0])
        result = identity_quat.rotate_vector(v)
        assert_allclose(result, v, atol=1e-14)

    def test_rotate_preserves_magnitude(self, random_quat):
        """Rotation should not change the vector magnitude."""
        v = np.array([3.0, -4.0, 5.0])
        result = random_quat.rotate_vector(v)
        assert_allclose(np.linalg.norm(result), np.linalg.norm(v), atol=1e-14)

    @pytest.mark.parametrize("axis,angle,v_in,v_expected", [
        ([0, 0, 1], np.pi, [1, 0, 0], [-1, 0, 0]),
        ([0, 1, 0], np.pi / 2, [1, 0, 0], [0, 0, -1]),
        ([1, 0, 0], np.pi / 2, [0, 1, 0], [0, 0, 1]),
    ])
    def test_rotate_parametrized(self, axis, angle, v_in, v_expected):
        """Parametrized rotation tests with known results."""
        q = Quaternion.from_axis_angle(np.array(axis, dtype=float), angle)
        result = q.rotate_vector(np.array(v_in, dtype=float))
        assert_allclose(result, np.array(v_expected, dtype=float), atol=1e-14)


# =============================================================================
# Test: Euler angle round-trip
# =============================================================================

class TestEulerRoundTrip:
    """Tests for Euler angle <-> quaternion conversion."""

    def test_to_euler_and_back(self):
        """Converting to Euler and back should yield the same quaternion."""
        q_orig = Quaternion.from_euler(0.1, 0.2, 0.3)
        phi, theta, psi = q_orig.to_euler()
        q_recovered = Quaternion.from_euler(phi, theta, psi)
        assert_allclose(q_recovered.components, q_orig.components, atol=1e-14)

    @pytest.mark.parametrize("phi,theta,psi", [
        (0.0, 0.0, 0.0),
        (0.5, -0.3, 1.2),
        (-0.8, 0.1, -0.6),
        (np.pi / 4, np.pi / 6, np.pi / 3),
    ])
    def test_euler_roundtrip_parametrized(self, phi, theta, psi):
        """Parametrized Euler angle round-trip test."""
        q = Quaternion.from_euler(phi, theta, psi)
        phi2, theta2, psi2 = q.to_euler()
        q2 = Quaternion.from_euler(phi2, theta2, psi2)
        assert_allclose(q2.components, q.components, atol=1e-13)

    def test_euler_identity(self, identity_quat):
        """Identity quaternion should map to zero Euler angles."""
        phi, theta, psi = identity_quat.to_euler()
        assert_allclose([phi, theta, psi], [0.0, 0.0, 0.0], atol=1e-15)


# =============================================================================
# Test: DCM round-trip
# =============================================================================

class TestDCMRoundTrip:
    """Tests for DCM <-> quaternion conversion."""

    def test_to_dcm_and_back(self):
        """Converting to DCM and back should yield the same quaternion."""
        q_orig = Quaternion.from_euler(0.1, 0.2, 0.3)
        dcm = q_orig.to_dcm()
        q_recovered = Quaternion.from_dcm(dcm)
        # q and -q represent the same rotation, so compare with sign tolerance
        dot = abs(np.dot(q_orig.components, q_recovered.components))
        assert_allclose(dot, 1.0, atol=1e-13)

    def test_dcm_orthogonality(self, random_quat):
        """DCM from quaternion must be orthogonal (R^T R = I)."""
        dcm = random_quat.to_dcm()
        assert_allclose(dcm.T @ dcm, np.eye(3), atol=1e-14)

    def test_dcm_determinant(self, random_quat):
        """DCM from quaternion must have determinant +1 (proper rotation)."""
        dcm = random_quat.to_dcm()
        assert_allclose(np.linalg.det(dcm), 1.0, atol=1e-14)

    @pytest.mark.parametrize("phi,theta,psi", [
        (0.0, 0.0, 0.0),
        (0.3, -0.15, 0.9),
        (-1.0, 0.4, 2.1),
    ])
    def test_dcm_roundtrip_parametrized(self, phi, theta, psi):
        """Parametrized DCM round-trip test."""
        q = Quaternion.from_euler(phi, theta, psi)
        dcm = q.to_dcm()
        q2 = Quaternion.from_dcm(dcm)
        dot = abs(np.dot(q.components, q2.components))
        assert_allclose(dot, 1.0, atol=1e-13)


# =============================================================================
# Test: SLERP endpoints
# =============================================================================

class TestSLERP:
    """Tests for Spherical Linear Interpolation."""

    def test_slerp_endpoints(self, quat_90z, quat_45x):
        """slerp(q1, q2, 0) = q1 and slerp(q1, q2, 1) = q2."""
        q1 = quat_90z
        q2 = quat_45x

        result_0 = Quaternion.slerp(q1, q2, 0.0)
        result_1 = Quaternion.slerp(q1, q2, 1.0)

        # At t=0, should equal q1
        dot0 = abs(np.dot(result_0.components, q1.components))
        assert_allclose(dot0, 1.0, atol=1e-14)

        # At t=1, should equal q2
        dot1 = abs(np.dot(result_1.components, q2.components))
        assert_allclose(dot1, 1.0, atol=1e-14)

    def test_slerp_midpoint_is_unit(self, quat_90z, quat_45x):
        """Midpoint of SLERP must be a unit quaternion."""
        mid = Quaternion.slerp(quat_90z, quat_45x, 0.5)
        assert_allclose(mid.norm, 1.0, atol=1e-14)

    def test_slerp_identical_quaternions(self, quat_90z):
        """SLERP between identical quaternions returns the same quaternion."""
        result = Quaternion.slerp(quat_90z, quat_90z, 0.5)
        dot = abs(np.dot(result.components, quat_90z.components))
        assert_allclose(dot, 1.0, atol=1e-14)

    @pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_slerp_unit_norm(self, quat_90z, quat_45x, t):
        """SLERP at any parameter must yield a unit quaternion."""
        result = Quaternion.slerp(quat_90z, quat_45x, t)
        assert_allclose(result.norm, 1.0, atol=1e-14)


# =============================================================================
# Test: Quaternion derivative
# =============================================================================

class TestDerivative:
    """Tests for the quaternion kinematic derivative."""

    def test_derivative(self, identity_quat):
        """Quaternion derivative has the correct structure for identity + omega."""
        omega = np.array([0.1, 0.2, 0.3])
        q_dot = identity_quat.derivative(omega)

        # For q = [1, 0, 0, 0] the derivative should be:
        # dq/dt = 0.5 * [0, omega_x, omega_y, omega_z] composed with q
        # = 0.5 * [-0*ox - 0*oy - 0*oz,
        #           1*ox - 0*oy + 0*oz,
        #           0*ox + 1*oy - 0*oz,
        #          -0*ox + 0*oy + 1*oz]
        # = 0.5 * [0, ox, oy, oz]
        expected = 0.5 * np.array([0.0, omega[0], omega[1], omega[2]])
        assert_allclose(q_dot, expected, atol=1e-15)

    def test_derivative_zero_omega(self, random_quat):
        """Zero angular velocity should produce zero derivative."""
        omega = np.zeros(3)
        q_dot = random_quat.derivative(omega)
        assert_allclose(q_dot, np.zeros(4), atol=1e-15)

    def test_derivative_perpendicular_to_q(self, random_quat):
        """For a unit quaternion, dq/dt should be perpendicular to q (dot = 0)."""
        omega = np.array([0.5, -0.3, 0.1])
        q_dot = random_quat.derivative(omega)
        # dq/dt . q = 0 for unit quaternions (rate of norm change is zero)
        dot_product = np.dot(q_dot, random_quat.components)
        assert_allclose(dot_product, 0.0, atol=1e-14)


# =============================================================================
# Test: From axis angle
# =============================================================================

class TestFromAxisAngle:
    """Tests for constructing a quaternion from axis-angle."""

    def test_from_axis_angle(self):
        """90-degree rotation about Z should give [cos(45), 0, 0, sin(45)]."""
        q = Quaternion.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
        expected_w = np.cos(np.pi / 4)
        expected_z = np.sin(np.pi / 4)
        assert_allclose(q.w, expected_w, atol=1e-14)
        assert_allclose(q.x, 0.0, atol=1e-15)
        assert_allclose(q.y, 0.0, atol=1e-15)
        assert_allclose(q.z, expected_z, atol=1e-14)

    def test_from_axis_angle_zero_rotation(self):
        """Zero-angle rotation should give the identity quaternion."""
        q = Quaternion.from_axis_angle(np.array([1.0, 0.0, 0.0]), 0.0)
        assert_allclose(q.components, [1.0, 0.0, 0.0, 0.0], atol=1e-15)

    def test_from_axis_angle_180_degrees(self):
        """180-degree rotation about X should give [0, 1, 0, 0]."""
        q = Quaternion.from_axis_angle(np.array([1.0, 0.0, 0.0]), np.pi)
        assert_allclose(abs(q.x), 1.0, atol=1e-14)
        assert_allclose(q.w, 0.0, atol=1e-14)

    @pytest.mark.parametrize("axis,angle", [
        ([1, 0, 0], np.pi / 6),
        ([0, 1, 0], np.pi / 3),
        ([0, 0, 1], np.pi / 2),
        ([1, 1, 0], np.pi / 4),
        ([1, 1, 1], 2 * np.pi / 3),
    ])
    def test_from_axis_angle_roundtrip(self, axis, angle):
        """Constructing from axis-angle and converting back should match."""
        axis_arr = np.array(axis, dtype=float)
        q = Quaternion.from_axis_angle(axis_arr, angle)
        recovered_axis, recovered_angle = q.to_axis_angle()
        # Angles should match
        assert_allclose(recovered_angle, angle, atol=1e-13)
        # Axes should be parallel (normalized)
        axis_normalized = axis_arr / np.linalg.norm(axis_arr)
        dot = abs(np.dot(recovered_axis, axis_normalized))
        assert_allclose(dot, 1.0, atol=1e-13)

    def test_from_axis_angle_zero_axis_raises(self):
        """Zero-length axis should raise ValueError."""
        with pytest.raises(ValueError):
            Quaternion.from_axis_angle(np.array([0.0, 0.0, 0.0]), np.pi / 2)
