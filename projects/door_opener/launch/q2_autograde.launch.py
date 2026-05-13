"""Launch Q2 autograder: simulation + all nodes + student planner + test."""

import os
import platform
import shutil
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    bridge_prefix = Path(get_package_prefix("mujoco_bridge"))
    q2_pkg = Path(get_package_share_directory("door_opener"))

    default_model = q2_pkg / "models" / "door_manipulation_scene.xml"
    sim_config = q2_pkg / "config" / "q2_params.yaml"

    args = [
        DeclareLaunchArgument("model_path", default_value=str(default_model)),
        DeclareLaunchArgument("use_viewer", default_value="false"),
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
            cmd=[mjpython, str(sim_script), "--ros-args", "-r", "__node:=mujoco_sim_node",
                 "--params-file", str(sim_config),
                 "-p", ["model_path:=", LaunchConfiguration("model_path")],
                 "-p", "use_viewer:=false"],
            output="screen")
    else:
        sim_node = Node(package="mujoco_bridge", executable="mujoco_sim_node.py",
                        name="mujoco_sim_node", output="screen",
                        parameters=[str(sim_config), {"model_path": LaunchConfiguration("model_path"), "use_viewer": False}])

    bridge_node = Node(package="mujoco_bridge", executable="bridge_node.py",
                       name="bridge_node", output="screen", parameters=[str(sim_config)])

    sensor_node = TimerAction(period=1.0, actions=[Node(
        package="q2", executable="sensor_node.py", name="q2_sensor_node",
        output="screen", parameters=[str(sim_config)])])

    goal_node = TimerAction(period=1.5, actions=[Node(
        package="q2", executable="goal_classifier_node.py", name="goal_classifier_node",
        output="screen", parameters=[str(sim_config)])])

    grasp_node = TimerAction(period=2.0, actions=[Node(
        package="q2", executable="grasp_controller.py", name="grasp_controller", output="screen")])

    # test_manipulation must start before manipulation_planner so it is
    # subscribed in time to catch the planner's one-shot grasp handshake.
    test_node = TimerAction(period=2.5, actions=[Node(
        package="q2", executable="test_manipulation.py", name="q2_tester",
        output="screen", parameters=[str(sim_config)])])

    student_node = TimerAction(period=3.5, actions=[Node(
        package="q2", executable="manipulation_planner.py", name="manipulation_planner", output="screen")])

    return LaunchDescription(
        args + [sim_node, bridge_node, sensor_node, goal_node, grasp_node, test_node, student_node])
