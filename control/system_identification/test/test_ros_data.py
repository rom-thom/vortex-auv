"""Exercise generic ROS conversion and incremental CSV recording without ROS."""

from array import array
from types import SimpleNamespace

import numpy as np
import pytest

from system_identification.ros.ros_file_converting import (
    ros_message_to_data,
    write_ros_data,
)
from system_identification.src.data_io import load_data, write_to_file


class Message(SimpleNamespace):
    """Implement the field introspection interface of generated ROS messages."""

    def get_fields_and_field_types(self):
        return {name: "unused" for name in vars(self)}


def header(sec=0):
    return Message(stamp=Message(sec=sec, nanosec=10_000_000), frame_id="base_link")


def test_nested_odometry_appends_without_repeated_header(tmp_path):
    path = tmp_path / "odom.csv"
    for sec in (0, 1):
        msg = Message(
            header=header(sec), child_frame_id="base_link",
            twist=Message(twist=Message(linear=Message(x=2.0 + sec, y=0.0, z=0.0))),
        )
        write_ros_data(msg, path)
    data = load_data(path)
    np.testing.assert_allclose(data["time"], [0.01, 1.01])
    np.testing.assert_array_equal(data["header.stamp.sec"], [0, 1])
    np.testing.assert_array_equal(data["header.stamp.nanosec"], [10_000_000] * 2)
    np.testing.assert_allclose(data["twist.twist.linear.x"], [2, 3])
    assert "header.frame_id" not in data
    assert len(path.read_text().splitlines()) == 3


def test_imu_arrays_and_wrench_go_to_separate_files(tmp_path):
    imu = Message(
        header=header(), linear_acceleration=Message(x=1.0, y=2.0, z=3.0),
        linear_acceleration_covariance=np.arange(9, dtype=float),
    )
    wrench = Message(header=header(), wrench=Message(force=Message(x=12.0)))
    write_ros_data(imu, tmp_path / "imu.csv")
    write_ros_data(wrench, tmp_path / "force.csv")
    data = load_data(tmp_path / "imu.csv")
    assert data["linear_acceleration.z"][0] == 3
    assert data["linear_acceleration_covariance[8]"][0] == 8
    assert load_data(tmp_path / "force.csv")["wrench.force.x"][0] == 12


def test_string_filename_returned_columns_and_explicit_overwrite(tmp_path):
    filename = str(tmp_path / "measurements.csv")
    first = Message(header=header(1), value=2.0)
    columns = write_ros_data(first, filename)
    assert all(isinstance(name, str) for name in columns)
    assert all(isinstance(values, np.ndarray) for values in columns.values())
    assert all(values.shape == (1,) for values in columns.values())
    for name, values in load_data(filename).items():
        np.testing.assert_array_equal(values, columns[name])

    write_ros_data(Message(header=header(2), value=3.0), filename)
    np.testing.assert_array_equal(load_data(filename)["value"], [2.0, 3.0])
    write_ros_data(Message(header=header(3), value=4.0), filename, append=False)
    np.testing.assert_array_equal(load_data(filename)["value"], [4.0])


def test_headerless_thruster_message_and_explicit_time(tmp_path):
    msg = Message(thrust=array("d", [1, 2, 3]), enabled=True)
    path = tmp_path / "thrusters.csv"
    with pytest.raises(ValueError, match="timestamp"):
        write_ros_data(msg, path)
    assert not path.exists()
    data = write_ros_data(msg, path, timestamp=2.5)
    assert data["time"][0] == 2.5
    assert data["thrust[2]"][0] == 3
    assert data["enabled"][0] == 1


def test_schema_change_does_not_modify_file(tmp_path):
    path = tmp_path / "samples.csv"
    write_ros_data(Message(header=header(), values=[1, 2]), path)
    previous = path.read_bytes()
    for msg in (
        Message(header=header(), values=[1, 2, 3]),
        Message(header=header(), force=1.0),
    ):
        with pytest.raises(ValueError, match="header"):
            write_ros_data(msg, path)
        assert path.read_bytes() == previous


@pytest.mark.parametrize("timestamp", [np.nan, np.inf, "now"])
def test_invalid_timestamp(timestamp):
    with pytest.raises(ValueError, match="timestamp"):
        ros_message_to_data(Message(header=header()), timestamp=timestamp)


def test_invalid_input_and_time_field_collision():
    with pytest.raises(TypeError, match="ROS 2 message"):
        ros_message_to_data([Message(header=header())])
    data = ros_message_to_data(Message(time=7.0), timestamp=2.0)
    assert data["time"][0] == 2.0
    assert data["message.time"][0] == 7.0


def test_append_header_validation_and_overwrite(tmp_path):
    path = tmp_path / "samples.csv"
    path.touch()
    write_to_file({"time": [0], "velocity": [1]}, path, append=True)
    previous = path.read_bytes()
    with pytest.raises(ValueError, match="header"):
        write_to_file({"velocity": [2], "time": [1]}, path, append=True)
    assert path.read_bytes() == previous
    write_to_file({"other": [42]}, path)
    np.testing.assert_array_equal(load_data(path)["other"], [42])


@pytest.mark.parametrize("contents", ["value\n1", "value\n"])
def test_append_handles_missing_final_newline(tmp_path, contents):
    path = tmp_path / "samples.csv"
    path.write_text(contents)
    write_to_file({"value": [2]}, path, append=True)
    expected = [1, 2] if contents.endswith("1") else [2]
    np.testing.assert_array_equal(load_data(path)["value"], expected)
