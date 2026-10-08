import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import numpy as np

from system_identification.src.system_identification import SystemIdentification


class SystemIdentificationNode(Node):
    def __init__(self):
        super().__init__("system_identification_node")

        self.declare_parameter("topics.odom", "/odom")
        self.declare_parameter("topics.imu", "/imu/data")
        self.declare_parameter("mass", 53.7)

        self.odom_buffer = []
        self.imu_buffer = []
        self.model = SystemIdentification()

        odom_topic = self.get_parameter("topics.odom").value
        imu_topic = self.get_parameter("topics.imu").value
        self.mass = self.get_parameter("mass").value

        self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self.create_subscription(Imu, imu_topic, self.imu_callback, 10)

    def odom_callback(self, msg):
        self.odom_buffer.append(msg.twist.twist.linear.x)
        self.get_logger().info(f"Odometry x: {msg.twist.twist.linear.x}")

    def imu_callback(self, msg):
        self.imu_buffer.append(msg.linear_acceleration.x)
        self.get_logger().info(f"IMU accel x: {msg.linear_acceleration.x}")

    def build_model_input(self):
        if len(self.odom_buffer) == 0 or len(self.imu_buffer) == 0:
            return None

        velocity = np.asarray(self.odom_buffer, dtype=float)
        acceleration = np.asarray(self.imu_buffer, dtype=float)

        if len(velocity) != len(acceleration):
            return None

        force = self.mass * acceleration

        return {
            "velocity": velocity,
            "acceleration": acceleration,
            "force": force,
            "mass": self.mass,
        }
def main(args=None):
    rclpy.init(args=args)
    node = SystemIdentificationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()