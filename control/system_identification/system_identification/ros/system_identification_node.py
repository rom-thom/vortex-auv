"""Record independent sensor and force-command streams to CSV."""

from functools import partial
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistWithCovarianceStamped, WrenchStamped
from vortex_msgs.msg import ThrusterForces

from system_identification.ros.ros_file_converting import write_ros_data


class SystemIdentificationNode(Node):
    def __init__(self):
        super().__init__("system_identification_node")

        self.declare_parameter("topics.imu", "/imu/data")
        self.declare_parameter("imu_output_file", "system_identification/data/imu_test.csv")

        imu_topic = self.get_parameter("topics.imu").value
        self.imu_output_file = self.get_parameter("imu_output_file").value
        if self.imu_output_file:
            # Relative paths are resolved from the directory used to launch ROS.
            output_path = Path(self.imu_output_file).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.imu_output_file = str(output_path)
            self.get_logger().info(f"Recording IMU from {imu_topic} to {self.imu_output_file}")

        self.imu_subscription = self.create_subscription(
            Imu, imu_topic, self.imu_callback, qos_profile_sensor_data
        )

        self.recording_subscriptions = []
        output_paths = {self.imu_output_file} if self.imu_output_file else set()
        for name, message_type, default_topic in (
            ("odom", Odometry, "/odom"),
            ("dvl_twist", TwistWithCovarianceStamped, "/dvl/twist"),
            ("twist", TwistWithCovarianceStamped, "/twist"),
            ("wrench_input", WrenchStamped, "/wrench_input"),
            ("thruster_forces", ThrusterForces, "/thruster_forces"),
        ):
            self.declare_parameter(f"topics.{name}", default_topic)
            # An empty filename disables recording for this topic.
            self.declare_parameter(f"{name}_output_file", "")
            topic = self.get_parameter(f"topics.{name}").value
            filename = self.get_parameter(f"{name}_output_file").value
            if not filename:
                continue

            output_path = Path(filename).expanduser().resolve()
            if str(output_path) in output_paths:
                raise ValueError(f"Each topic needs a separate output file: {output_path}")
            output_paths.add(str(output_path))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f"Recording {topic} to {output_path}")
            self.recording_subscriptions.append(self.create_subscription(
                message_type,
                topic,
                partial(write_ros_data, file_path=str(output_path)),
                qos_profile_sensor_data,))

    def imu_callback(self, msg):
        if self.imu_output_file:
            write_ros_data(msg, self.imu_output_file)


def main(args=None):
    rclpy.init(args=args)
    node = SystemIdentificationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
