"""Tests for splitting ordered samples and shared constants."""

import numpy as np
import pytest

from system_identification.src.system_identification import SystemIdentification


def test_split_preserves_alignment_constants_and_input():
    """Keep samples aligned and ensure editing a split leaves the input intact."""
    data = {
        "time": np.arange(5),
        "velocity": [10, 11, 12, 13, 14],
        "vectors": np.arange(10).reshape(5, 2),
        "mass": 53.7,
    }
    training, validation = SystemIdentification().split_data(data, 0.5)

    assert training.keys() == validation.keys() == data.keys()
    assert training["mass"] == validation["mass"] == 53.7
    for name in ("time", "velocity", "vectors"):
        np.testing.assert_array_equal(training[name], data[name][:2])
        np.testing.assert_array_equal(validation[name], data[name][2:])
        training[name][0] = -100
        validation[name][0] = -200
    np.testing.assert_array_equal(data["time"], np.arange(5))
    assert data["velocity"] == [10, 11, 12, 13, 14]
    np.testing.assert_array_equal(data["vectors"], np.arange(10).reshape(5, 2))


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1, np.nan, np.inf])
def test_split_rejects_invalid_fraction(fraction):
    """Reject fractions that cannot describe two nonempty datasets."""
    with pytest.raises(ValueError, match="training_fraction"):
        SystemIdentification().split_data({"velocity": [1, 2, 3]}, fraction)


@pytest.mark.parametrize("data, fraction, message", [
    ({}, 0.5, "sample array"),
    ({"mass": 53.7}, 0.5, "sample array"),
    ({"velocity": [1, 2], "force": [1]}, 0.5, "matching lengths"),
    ({"velocity": []}, 0.5, "at least one sample"),
    ({"velocity": [1]}, 0.5, "at least one sample"),
    ({"velocity": [1, 2]}, 0.1, "at least one sample"),
])
def test_split_rejects_unusable_samples(data, fraction, message):
    """Reject misaligned columns or a split that would produce an empty set."""
    with pytest.raises(ValueError, match=message):
        SystemIdentification().split_data(data, fraction)
