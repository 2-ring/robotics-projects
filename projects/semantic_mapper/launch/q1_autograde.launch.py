"""Launch Q1 autograder: simulation + all nodes + student mapper + test.

Randomizes object positions on tables before launching the simulation.
"""

import os
import platform
import random
import re
import shutil
import tempfile
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _randomize_scene_xml(base_xml_path: Path) -> Path:
    """Generate a randomized scene XML in a temp file, return its path."""
    # Import the randomizer logic inline to avoid install-path issues
    import sys
    scripts_dir = Path(get_package_prefix("q1")) / "lib" / "q1"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from randomize_scene import randomize

    xml_in = base_xml_path.read_text()
    seed = random.randint(0, 2**31)
    xml_out, placements = randomize(xml_in, seed)

    tmp = tempfile.NamedTemporaryFile(
        suffix=".xml", prefix="q1_scene_", delete=False, mode="w")
    tmp.write(xml_out)
    tmp.close()

    print(f"[Q1 Autograde] Randomized scene (seed={seed}):")
    for obj, (px, py, pz) in placements.items():
        print(f"  {obj}: ({px:.3f}, {py:.3f})")

    return Path(tmp.name)


def generate_launch_description() -> LaunchDescription:
    bridge_prefix = Path(get_package_prefix("mujoco_bridge"))
    q1_pkg = Path(get_package_share_directory("semantic_mapper"))

    base_model = q1_pkg / "models" / "semantic_map_scene.xml"
    sim_config = q1_pkg / "config" / "q1_params.yaml"

    # Randomize object positions
    randomized_model = _randomize_scene_xml(base_model)

    args = [
        DeclareLaunchArgument("model_path", default_value=str(randomized_model)),
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
            cmd=[
                mjpython, str(sim_script),
                "--ros-args", "-r", "__node:=mujoco_sim_node",
                "--params-file", str(sim_config),
                "-p", ["model_path:=", LaunchConfiguration("model_path")],
                "-p", "use_viewer:=false",
            ],
            output="screen",
        )
    else:
        sim_node = Node(
            package="mujoco_bridge",
            executable="mujoco_sim_node.py",
            name="mujoco_sim_node",
            output="screen",
            parameters=[str(sim_config), {"model_path": LaunchConfiguration("model_path"), "use_viewer": False}],
        )

    bridge_node = Node(
        package="mujoco_bridge",
        executable="turtlebot_bridge_node.py",
        name="turtlebot_bridge_node",
        output="screen",
        parameters=[str(sim_config)],
    )

    sensor_node = TimerAction(period=1.0, actions=[Node(
        package="q1", executable="sensor_node.py", name="q1_sensor_node", output="screen")])

    detector_node = TimerAction(period=1.5, actions=[Node(
        package="q1", executable="object_detector_node.py", name="object_detector_node",
        output="screen", parameters=[str(sim_config)])])

    navigator_node = TimerAction(period=1.5, actions=[Node(
        package="q1", executable="navigator_node.py", name="navigator_node",
        output="screen", parameters=[str(sim_config)])])

    student_node = TimerAction(period=2.0, actions=[Node(
        package="q1", executable="semantic_mapper.py", name="semantic_mapper", output="screen")])

    test_node = TimerAction(period=3.0, actions=[Node(
        package="q1", executable="test_semantic_mapper.py", name="q1_tester", output="screen")])

    return LaunchDescription(
        args + [sim_node, bridge_node, sensor_node, detector_node,
                navigator_node, student_node, test_node]
    )
