"""Launch MuJoCo to simulate a TurtleBot with a LiDAR sensor."""

import os
import platform
import shutil
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Generate a ROS 2 launch description to bring up MuJoCo with a simulated TurtleBot."""
    bridge_prefix = Path(get_package_prefix("mujoco_bridge"))
    q1_pkg = Path(get_package_share_directory("occupancy_grid"))

    default_model = q1_pkg / "models" / "turtlebot_scene.xml"
    sim_config = q1_pkg / "config" / "sim_params.yaml"

    args = [
        DeclareLaunchArgument("model_path", default_value=str(default_model)),
        DeclareLaunchArgument("use_viewer", default_value="true"),
        DeclareLaunchArgument("use_map_viz", default_value="true"),
        DeclareLaunchArgument("use_pure_pursuit", default_value="true"),
        DeclareLaunchArgument("paused", default_value="false"),
        DeclareLaunchArgument("realtime_factor", default_value="1.0"),
        DeclareLaunchArgument("lidar_noise_sigma0_m", default_value="0.01"),
        DeclareLaunchArgument("lidar_noise_scale", default_value="0.01"),
    ]

    sim_script = bridge_prefix / "lib" / "mujoco_bridge" / "mujoco_sim_node.py"

    mjpython = shutil.which("mjpython")
    if not mjpython:
        conda = Path(os.environ.get("CONDA_PREFIX", ""))
        cand = conda / "bin" / "mjpython"
        if cand.is_file():
            mjpython = str(cand)

    is_mac = platform.system() == "Darwin"

    if is_mac and mjpython:
        sim_node = ExecuteProcess(
            cmd=[
                mjpython,
                str(sim_script),
                "--ros-args",
                "-r",
                "__node:=mujoco_sim_node",
                "--params-file",
                str(sim_config),
                "-p",
                ["model_path:=", LaunchConfiguration("model_path")],
                "-p",
                ["use_viewer:=", LaunchConfiguration("use_viewer")],
                "-p",
                ["paused:=", LaunchConfiguration("paused")],
                "-p",
                ["realtime_factor:=", LaunchConfiguration("realtime_factor")],
            ],
            output="screen",
        )
    else:
        if is_mac and not mjpython:
            import warnings

            warnings.warn(
                "macOS detected but 'mjpython' not found! Viewer will NOT work.\n"
                "Install: pip install mujoco   then check: which mjpython"
            )
        sim_node = Node(
            package="mujoco_bridge",
            executable="mujoco_sim_node.py",
            name="mujoco_sim_node",
            output="screen",
            parameters=[
                str(sim_config),
                {
                    "model_path": LaunchConfiguration("model_path"),
                    "use_viewer": LaunchConfiguration("use_viewer"),
                    "paused": LaunchConfiguration("paused"),
                    "realtime_factor": LaunchConfiguration("realtime_factor"),
                },
            ],
        )

    bridge_node = Node(
        package="mujoco_bridge",
        executable="turtlebot_bridge_node.py",
        name="turtlebot_bridge_node",
        output="screen",
        parameters=[
            {
                "lidar_noise_sigma0_m": LaunchConfiguration("lidar_noise_sigma0_m"),
                "lidar_noise_scale": LaunchConfiguration("lidar_noise_scale"),
            }
        ],
    )

    map_viz_node = Node(
        package="q1",
        executable="rviz_marker_node.py",
        name="rviz_marker_node",
        output="screen",
        parameters=[{"scene_path": LaunchConfiguration("model_path")}],
        condition=IfCondition(LaunchConfiguration("use_map_viz")),
    )

    rviz_config = q1_pkg / "config" / "occupancy_mapper.rviz"
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", str(rviz_config)] if rviz_config.exists() else [],
        condition=IfCondition(LaunchConfiguration("use_map_viz")),
    )

    pure_pursuit_node = Node(
        package="q1",
        executable="pure_pursuit_node.py",
        name="pure_pursuit_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_pure_pursuit")),
    )

    return LaunchDescription(
        args + [sim_node, bridge_node, map_viz_node, rviz_node, pure_pursuit_node]
    )
