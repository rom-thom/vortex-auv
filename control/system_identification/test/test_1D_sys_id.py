"""Run a basic x-direction fit using the local recorded CSVs."""

from pathlib import Path

import numpy as np
import pytest

from system_identification.src.data_handeling import merge_file_data
from system_identification.src.data_preprossesing import remove_gravity
from system_identification.src.system_identification import SystemIdentification


def test_1D_sys_id():
    directory = Path(__file__).resolve().parents[1] / "system_identification" / "data"
    paths = [directory / name for name in (
        "imu_test.csv", "odom_test.csv", "wrench_input_test.csv",
    )]
    if any(not path.is_file() for path in paths):
        pytest.skip("Requires local IMU, odometry and wrench test recordings.")

    data = merge_file_data(paths, exclude_comon_headers=True)
    accelerations = np.column_stack([
        data[f"linear_acceleration.{axis}"] for axis in "xyz"
    ])
    quaternions = np.column_stack([
        data[f"orientation.{axis}"] for axis in "xyzw"
    ])

    # Assumes a z-down world and IMU axes aligned with the vehicle axes.
    corrected = remove_gravity(accelerations, quaternions, gravity_world=(0, 0, 9.81))
    fit_data = {
        "velocity": data["twist.twist.linear.x"],
        "acceleration": corrected[:, 0],
        "force": data["wrench.force.x"],
    }

    model = SystemIdentification()
    training, validation = model.split_data(fit_data, training_fraction=0.7)
    parameters = model.fit_parameters(training)
    score = model.score(validation)

    # These check that the pipeline runs, not the physical accuracy of the fit.
    assert parameters.shape == (3,)
    assert np.all(np.isfinite(parameters))
    assert np.isfinite(score)

    print("Samples:", len(data["time"]))
    print("Estimated [d1, d2, mass]:", parameters)
    print("Validation force RMSE [N]:", score)


if __name__ == "__main__":
    test_1D_sys_id()
