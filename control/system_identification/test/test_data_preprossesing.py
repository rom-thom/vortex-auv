"""Check gravity compensation, frame conventions, and invalid measurements."""

import numpy as np
import pytest

from system_identification.src.data_preprossesing import remove_gravity


@pytest.mark.parametrize("gravity_z", [9.81, -9.81])
def test_stationary_identity_for_both_world_conventions(gravity_z):
    result = remove_gravity([0, 0, -gravity_z], [0, 0, 0, 1], [0, 0, gravity_z])
    np.testing.assert_allclose(result, [0, 0, 0], atol=1e-12)
    assert result.shape == (3,)


def test_tilted_batch_preserves_motion_and_inputs():
    # With +90 degrees about y, world +z points along sensor -x.
    half = np.sqrt(0.5)
    quaternions = np.array([[0, 0, 0, 1], [0, half, 0, half]])
    measured = np.array([[1, 2, 3 - 9.81], [4 + 9.81, 5, 6]])
    before = measured.copy(), quaternions.copy()
    result = remove_gravity(measured, quaternions)
    np.testing.assert_allclose(result, [[1, 2, 3], [4, 5, 6]], atol=1e-12)
    np.testing.assert_array_equal(measured, before[0])
    np.testing.assert_array_equal(quaternions, before[1])


@pytest.mark.parametrize("scale", [2.0, -1.0, 1e-300, 1e300])
def test_single_unnormalized_orientation_for_batch(scale):
    result = remove_gravity([[0, 0, -9.81], [2, 0, -9.81]], [0, 0, 0, scale])
    np.testing.assert_allclose(result, [[0, 0, 0], [2, 0, 0]], atol=1e-12)


@pytest.mark.parametrize("name, value", [
    ("accelerations", []),
    ("accelerations", [1, 2]),
    ("accelerations", [[1, 2]]),
    ("accelerations", [1, np.nan, 3]),
    ("accelerations", [1, np.inf, 3]),
    ("accelerations", [1j, 2, 3]),
    ("accelerations", ["invalid", 2, 3]),
    ("accelerations", [[1, 2, 3], [1]]),
    ("orientation_quaternion", [0, 0, 0, 0]),
    ("orientation_quaternion", [0, 0, 1]),
    ("orientation_quaternion", [0, 0, np.nan, 1]),
    ("orientation_quaternion", [[0, 0, 0, 1], [0, 0, 0, 1]]),
    ("gravity_world", [0, 9.81]),
    ("gravity_world", [[0, 0, 9.81]]),
    ("gravity_world", [0, 0, np.inf]),
])
def test_invalid_inputs(name, value):
    arguments = {
        "accelerations": [0, 0, -9.81],
        "orientation_quaternion": [0, 0, 0, 1],
    }
    arguments[name] = value
    with pytest.raises(ValueError, match=name):
        remove_gravity(**arguments)


def test_batch_rejects_missing_or_zero_orientations():
    measurements = np.zeros((2, 3))
    for orientations in ([[0, 0, 0, 1]], [[0, 0, 0, 1], [0, 0, 0, 0]]):
        with pytest.raises(ValueError, match="orientation_quaternion"):
            remove_gravity(measurements, orientations)
