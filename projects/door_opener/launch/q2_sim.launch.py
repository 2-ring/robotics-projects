"""Launch MuJoCo simulation for Q2: Door Manipulation."""

import os
import platform
import shutil
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    bridge_prefix = Path(get_package_prefix("mujoco_bridge"))
    q2_pkg = Path(get_package_share_directory("door_opener"))

    default_model = q2_pkg / "models" / "door_manipulation_scene.xml"
    default_rviz = q2_pkg / "rviz" / "q2_viz.rviz"
    sim_config = q2_pkg / "config" / "q2_params.yaml"

    args = [
        DeclareLaunchArgument("model_path", default_value=str(default_model)),
        DeclareLaunchArgument("use_viewer", default_value="true"),
        DeclareLaunchArgument("paused", default_value="false"),
        DeclareLaunchArgument("realtime_factor", default_value="1.0"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=str(default_rviz)),
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
                mjpython, str(sim_script),
                "--ros-args", "-r", "__node:=mujoco_sim_node",
                "--params-file", str(sim_config),
                "-p", ["model_path:=", LaunchConfiguration("model_path")],
                "-p", ["use_viewer:=", LaunchConfiguration("use_viewer")],
                "-p", ["paused:=", LaunchConfiguration("paused")],
                "-p", ["realtime_factor:=", LaunchConfiguration("realtime_factor")],
            ],
            output="screen",
        )
    else:
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

    # Panda bridge
    bridge_node = Node(
        package="mujoco_bridge",
        executable="bridge_node.py",
        name="bridge_node",
        output="screen",
        parameters=[str(sim_config)],
    )

    # Q2 sensor node
    sensor_node = TimerAction(
        period=1.0,
        actions=[Node(
            package="q2",
            executable="sensor_node.py",
            name="q2_sensor_node",
            output="screen",
            parameters=[str(sim_config)],
        )],
    )

    # Goal classifier
    goal_node = TimerAction(
        period=1.5,
        actions=[Node(
            package="q2",
            executable="goal_classifier_node.py",
            name="goal_classifier_node",
            output="screen",
            parameters=[str(sim_config)],
        )],
    )

    # Grasp controller
    grasp_node = TimerAction(
        period=2.0,
        actions=[Node(
            package="q2",
            executable="grasp_controller.py",
            name="grasp_controller",
            output="screen",
        )],
    )

    # RViz visualizer node
    viz_node = TimerAction(
        period=2.0,
        actions=[Node(
            package="q2",
            executable="rviz_visualizer.py",
            name="q2_rviz_visualizer",
            output="screen",
        )],
    )

    # RViz2
    rviz_node = TimerAction(
        period=2.5,
        actions=[Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", LaunchConfiguration("rviz_config")],
            condition=IfCondition(LaunchConfiguration("use_rviz")),
            output="screen",
        )],
    )

    return LaunchDescription(
        args + [sim_node, bridge_node, sensor_node, goal_node, grasp_node, viz_node, rviz_node]
    )
