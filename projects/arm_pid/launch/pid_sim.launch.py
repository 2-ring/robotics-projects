"""
Launch the PID Controller simulation (Q2).

Starts two processes:
  1. mujoco_sim_node  — physics + viewer  (from mujoco_bridge, via mjpython on macOS)
  2. sensor_node      — sensor bridge     (from q2)

After launch, the student runs their controller separately:
    ros2 run q2 pid_controller.py

To test:
    ros2 run q2 test_pid.py
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
    bridge_pkg = get_package_share_directory("mujoco_bridge")
    q2_pkg = get_package_share_directory("arm_pid")

    default_model = os.path.join(q2_pkg, 'models', 'panda', 'pid_scene.xml')
    sim_config    = os.path.join(q2_pkg, 'config', 'sim_params.yaml')

    # ── Launch arguments ────────────────────────────────────────────
    args = [
        DeclareLaunchArgument('model_path',      default_value=default_model),
        DeclareLaunchArgument('use_viewer',       default_value='true'),
        DeclareLaunchArgument('paused',           default_value='false'),
        DeclareLaunchArgument('realtime_factor',  default_value='1.0'),
    ]

    # ── Sim node (reuse mujoco_bridge's mujoco_sim_node.py) ───
    sim_script = os.path.join(bridge_pkg, '..', '..', 'lib',
                              'mujoco_bridge', 'mujoco_sim_node.py')

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

    # ── Sensor node (delayed 1 s for sim to start) ───────────────────
    sensor_node = TimerAction(
        period=1.0,
        actions=[
            Node(
                package='q2',
                executable='sensor_node.py',
                name='q2_sensor_node',
                output='screen',
            )
        ]
    )

    return LaunchDescription(args + [sim_node, sensor_node])
