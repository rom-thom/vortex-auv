"""Check timestamp selection, interpolation, and invalid merge inputs."""

import numpy as np
import pytest

from system_identification.src.data_io import load_data, merge_data, write_to_file


def test_default_uses_fastest_spacing_and_clips_to_overlap(tmp_path):
    # The slower signal has more samples because its recording is longer.
    slow_time = np.arange(10.0)
    fast_time = np.array([1.25, 1.5, 1.75, 2.0])
    signals = {
        "velocity": (slow_time, 2 * slow_time + 1),
        "acceleration": (fast_time, -fast_time),
        "force": (np.array([1.4, 1.9]), np.array([14.0, 19.0])),
    }
    merged = merge_data(signals)
    np.testing.assert_allclose(merged["time"], [1.5, 1.75])
    np.testing.assert_allclose(merged["velocity"], [4.0, 4.5])
    np.testing.assert_allclose(merged["acceleration"], [-1.5, -1.75])
    np.testing.assert_allclose(merged["force"], [15.0, 17.5])

    path = tmp_path / "merged.csv"
    write_to_file(merged, path)
    for name, values in load_data(path).items():
        np.testing.assert_allclose(values, merged[name])


def test_explicit_targets_include_endpoints_and_leave_inputs_unchanged():
    times = np.array([0.0, 1.0, 2.0])
    values = np.array([0.0, 4.0, 2.0])
    target = np.array([-1.0, 0.0, 0.5, 1.5, 2.0, 3.0])
    merged = merge_data({"value": (times, values)}, target)
    np.testing.assert_allclose(merged["time"], [0.0, 0.5, 1.5, 2.0])
    np.testing.assert_allclose(merged["value"], [0.0, 2.0, 3.0, 2.0])
    merged["time"][:] = -10
    merged["value"][:] = -10
    np.testing.assert_array_equal(times, [0, 1, 2])
    np.testing.assert_array_equal(values, [0, 4, 2])
    np.testing.assert_array_equal(target, [-1, 0, 0.5, 1.5, 2, 3])


@pytest.mark.parametrize("times, values", [
    ([], []),
    ([0], [1]),
    ([0, 1], [1]),
    ([[0, 1]], [[1, 2]]),
    ([0, 0], [1, 2]),
    ([1, 0], [1, 2]),
    ([0, np.nan], [1, 2]),
    ([0, 1], [1, np.inf]),
])
def test_invalid_signal_arrays(times, values):
    with pytest.raises(ValueError):
        merge_data({"value": (times, values)})


@pytest.mark.parametrize("target", [[], [[0, 1]], [np.nan], [np.inf], [1, 0], [0, 0], [3, 4]])
def test_invalid_or_unusable_targets(target):
    with pytest.raises(ValueError):
        merge_data({"value": ([0, 1], [0, 2])}, target)


@pytest.mark.parametrize("name", ["time", "", " value ", 1])
def test_invalid_signal_names(name):
    with pytest.raises(ValueError, match="Signal names"):
        merge_data({name: ([0, 1], [1, 2])})


def test_empty_input_and_nonoverlapping_recordings():
    with pytest.raises(ValueError, match="At least one signal"):
        merge_data({})
    with pytest.raises(ValueError, match="no overlapping"):
        merge_data({"a": ([0, 1], [0, 1]), "b": ([2, 3], [2, 3])})
