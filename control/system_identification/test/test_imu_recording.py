"""Test the IMU callback with real ROS messages when ROS is installed."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rclpy", reason="Run in a sourced ROS 2 environment.")
Imu = pytest.importorskip("sensor_msgs.msg").Imu
pytest.importorskip("vortex_msgs.msg")

from system_identification.ros.system_identification_node import SystemIdentificationNode
from system_identification.src.data_io import load_data


def test_imu_callback_appends_real_messages(tmp_path):
    path = tmp_path / "imu.csv"
    recorder = SimpleNamespace(imu_output_file=str(path))

    # Exercise the callback without creating a live subscription or ROS context.
    for index in range(2):
        msg = Imu()
        msg.header.stamp.sec = 10 + index
        msg.header.stamp.nanosec = 250_000_000
        msg.header.frame_id = "imu_link"
        msg.orientation.w = 1.0
        msg.angular_velocity.z = 0.1 + index
        msg.linear_acceleration.x = 0.5 + index
        msg.linear_acceleration.z = 9.81
        msg.linear_acceleration_covariance = [0.01] * 9
        SystemIdentificationNode.imu_callback(recorder, msg)

    data = load_data(str(path))
    np.testing.assert_allclose(data["time"], [10.25, 11.25])
    np.testing.assert_allclose(data["linear_acceleration.x"], [0.5, 1.5])
    np.testing.assert_allclose(data["linear_acceleration.z"], [9.81, 9.81])
    np.testing.assert_allclose(data["angular_velocity.z"], [0.1, 1.1])
    np.testing.assert_allclose(data["orientation.w"], [1.0, 1.0])
    np.testing.assert_allclose(data["linear_acceleration_covariance[8]"], [0.01, 0.01])
    assert len(path.read_text().splitlines()) == 3


def test_imu_recording_disabled_by_empty_filename():
    recorder = SimpleNamespace(imu_output_file="")
    SystemIdentificationNode.imu_callback(recorder, Imu())
