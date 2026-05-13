"""
Launch the MuJoCo–ROS2 Panda simulation.

Starts two nodes:
  1. mujoco_sim_node  — physics + viewer  (from mujoco_bridge, via mjpython on macOS)
  2. bridge_node      — MuJoCo ↔ ROS2 translation  (from mujoco_bridge package)

After this is running, connect with:
  ros2 run q0 joint_pose_commander.py
  ros2 run q0 keyboard_teleop.py
"""

import os
import platform
import shutil

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    q0_pkg     = get_package_share_directory("panda_teleop")
    bridge_pkg = get_package_share_directory("mujoco_bridge")

    default_model  = os.path.join(q0_pkg, 'models', 'panda', 'ros2_scene.xml')
    sim_config     = os.path.join(q0_pkg, 'config', 'sim_params.yaml')
    bridge_config  = os.path.join(q0_pkg, 'config', 'bridge_params.yaml')

    # ── Launch arguments ────────────────────────────────────────────
    args = [
        DeclareLaunchArgument('model_path',      default_value=default_model),
        DeclareLaunchArgument('use_viewer',       default_value='true'),
        DeclareLaunchArgument('paused',           default_value='false'),
        DeclareLaunchArgument('realtime_factor',  default_value='1.0'),
    ]

    # ── Sim node (from mujoco_bridge) ─────────────────────────────
    # On macOS, MuJoCo viewer needs mjpython for OpenGL.
    sim_script = os.path.join(bridge_pkg, '..', '..', 'lib', 'mujoco_bridge', 'mujoco_sim_node.py')

    mjpython = shutil.which('mjpython')
    if not mjpython:
        conda = os.environ.get('CONDA_PREFIX', '')
        cand  = os.path.join(conda, 'bin', 'mjpython')
        if os.path.isfile(cand):
            mjpython = cand

    is_mac = platform.system() == 'Darwin'

    if is_mac and mjpython:
        sim_node = ExecuteProcess(
            cmd=[
                mjpython, sim_script,
                '--ros-args',
                '-r', '__node:=mujoco_sim_node',
                '--params-file', sim_config,
                '-p', ['model_path:=', LaunchConfiguration('model_path')],
                '-p', ['use_viewer:=', LaunchConfiguration('use_viewer')],
                '-p', ['paused:=',     LaunchConfiguration('paused')],
                '-p', ['realtime_factor:=', LaunchConfiguration('realtime_factor')],
            ],
            output='screen',
        )
    else:
        if is_mac and not mjpython:
            import warnings
            warnings.warn(
                "macOS detected but 'mjpython' not found! Viewer will NOT work.\n"
                "Install: pip install mujoco   then check: which mjpython")
        sim_node = Node(
            package='mujoco_bridge',
            executable='mujoco_sim_node.py',
            name='mujoco_sim_node',
            output='screen',
            parameters=[
                sim_config,
                {
                    'model_path':      LaunchConfiguration('model_path'),
                    'use_viewer':      LaunchConfiguration('use_viewer'),
                    'paused':          LaunchConfiguration('paused'),
                    'realtime_factor': LaunchConfiguration('realtime_factor'),
                }
            ],
        )

    # ── Bridge node (from mujoco_bridge, delayed 1 s) ─────────
    bridge_node = TimerAction(
        period=1.0,
        actions=[
            Node(
                package='mujoco_bridge',
                executable='bridge_node.py',
                name='bridge_node',
                output='screen',
                parameters=[bridge_config],
            )
        ]
    )

    return LaunchDescription(args + [sim_node, bridge_node])
