from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="system_identification",
            executable="system_identification_node",
            name="system_identification_node",
            parameters=[
                "/ros_ws/install/system_identification/share/system_identification/config/config.yaml"
            ],
        )
    ])