"""Check recording subscriptions and routing without a ROS installation."""

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


@pytest.fixture
def recorder_module(monkeypatch):
    parameters = {}

    class Node:
        def __init__(self, name):
            self.parameters = dict(parameters)
            self.subscribed = []

        def declare_parameter(self, name, default):
            self.parameters.setdefault(name, default)

        def get_parameter(self, name):
            return SimpleNamespace(value=self.parameters[name])

        def get_logger(self):
            return SimpleNamespace(info=lambda text: None)

        def create_subscription(self, message_type, topic, callback, qos):
            subscription = SimpleNamespace(
                message_type=message_type, topic=topic, callback=callback,
            )
            self.subscribed.append(subscription)
            return subscription

    modules = {
        "rclpy": {}, "rclpy.node": {"Node": Node},
        "rclpy.qos": {"qos_profile_sensor_data": object()},
        "sensor_msgs": {}, "sensor_msgs.msg": {"Imu": type("Imu", (), {})},
        "nav_msgs": {}, "nav_msgs.msg": {"Odometry": type("Odometry", (), {})},
        "geometry_msgs": {}, "geometry_msgs.msg": {
            name: type(name, (), {})
            for name in ("TwistWithCovarianceStamped", "WrenchStamped")
        },
        "vortex_msgs": {}, "vortex_msgs.msg": {
            "ThrusterForces": type("ThrusterForces", (), {}),
        },
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    path = Path(__file__).parents[1] / "system_identification/ros/system_identification_node.py"
    spec = importlib.util.spec_from_file_location("recording_node_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, parameters


def test_each_stream_uses_its_topic_and_separate_file(recorder_module, tmp_path, monkeypatch):
    module, parameters = recorder_module
    names = ("imu", "odom", "dvl_twist", "twist", "wrench_input", "thruster_forces")
    for name in names:
        parameters[f"topics.{name}"] = f"/test/{name}"
        parameters[f"{name}_output_file"] = str(tmp_path / "recordings" / f"{name}.csv")

    written = []
    monkeypatch.setattr(module, "write_ros_data", lambda msg, file_path: written.append((msg, file_path)))
    node = module.SystemIdentificationNode()
    assert len(node.subscribed) == 6
    assert (tmp_path / "recordings").is_dir()
    for name, subscription in zip(names, node.subscribed):
        assert subscription.topic == f"/test/{name}"
        message = object()
        subscription.callback(message)
        assert written[-1] == (message, parameters[f"{name}_output_file"])


def test_empty_filenames_disable_additional_streams(recorder_module):
    module, parameters = recorder_module
    parameters["imu_output_file"] = ""
    node = module.SystemIdentificationNode()
    assert len(node.subscribed) == 1
    assert node.recording_subscriptions == []


def test_duplicate_output_paths_rejected(recorder_module, tmp_path):
    module, parameters = recorder_module
    parameters["imu_output_file"] = str(tmp_path / "same.csv")
    parameters["odom_output_file"] = str(tmp_path / "same.csv")
    with pytest.raises(ValueError, match="separate output file"):
        module.SystemIdentificationNode()
