"""Launch Q2 localization in an asymmetric scene with optional grading node."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Generate launch description for Q2 localization evaluation."""
    q1_share = Path(get_package_share_directory("occupancy_grid"))
    q2_share = Path(get_package_share_directory("bayes_localizer"))

    default_model = q2_share / "models" / "turtlebot_scene_q2_asymmetric.xml"
    q1_bringup_launch = q1_share / "launch" / "turtlebot_bringup.launch.py"

    args = [
        DeclareLaunchArgument("model_path", default_value=str(default_model)),
        DeclareLaunchArgument("use_viewer", default_value="true"),
        DeclareLaunchArgument("use_map_viz", default_value="true"),
        DeclareLaunchArgument("paused", default_value="false"),
        DeclareLaunchArgument("realtime_factor", default_value="2.0"),
        DeclareLaunchArgument("run_test", default_value="false"),
        DeclareLaunchArgument(
            "bayes_localizer_module", default_value="bayes_localizer.py"
        ),
    ]

    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(q1_bringup_launch)),
        launch_arguments={
            "model_path": LaunchConfiguration("model_path"),
            "use_viewer": LaunchConfiguration("use_viewer"),
            "use_map_viz": LaunchConfiguration("use_map_viz"),
            "paused": LaunchConfiguration("paused"),
            "realtime_factor": LaunchConfiguration("realtime_factor"),
            "lidar_noise_sigma0_m": "0.005",
            "lidar_noise_scale": "0.005",
        }.items(),
    )

    localizer = Node(
        package="q2",
        executable=LaunchConfiguration("bayes_localizer_module"),
        name="bayes_localizer",
        output="screen",
        parameters=[
            {
                "scene_path": LaunchConfiguration("model_path"),
                "height_cells": 96,  # Covers y from -2.0 to +2.8 m (extended north wall)
            }
        ],
    )

    grader = Node(
        package="q2",
        executable="test_bayes_localizer.py",
        name="test_bayes_localizer",
        output="screen",
        condition=IfCondition(LaunchConfiguration("run_test")),
    )

    return LaunchDescription(args + [bringup, localizer, grader])
