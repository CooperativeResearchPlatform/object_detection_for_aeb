import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("perception")
    models = os.path.join(share, "models")

    return LaunchDescription([
        DeclareLaunchArgument("pointcloud_topic", default_value="/lidar/points"),
        DeclareLaunchArgument("ego_topic", default_value="/ego"),
        DeclareLaunchArgument("pfe_onnx", default_value=os.path.join(models, "centerpoint_pfe.onnx")),
        DeclareLaunchArgument("head_onnx", default_value=os.path.join(models, "centerpoint_head.onnx")),
        Node(
            package="perception",
            executable="perception_node",
            name="perception_node",
            output="screen",
            parameters=[
                os.path.join(share, "config", "perception.yaml"),
                {
                    "model.pfe_onnx": LaunchConfiguration("pfe_onnx"),
                    "model.head_onnx": LaunchConfiguration("head_onnx"),
                },
            ],
            remappings=[
                ("pointcloud", LaunchConfiguration("pointcloud_topic")),
                ("ego", LaunchConfiguration("ego_topic")),
            ],
        ),
    ])
