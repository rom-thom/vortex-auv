from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="system_identification",
            executable="system_identification_node",
            name="system_identification_node",
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare("system_identification"), "config", "config.yaml"
                ])
            ],
        )
    ])
