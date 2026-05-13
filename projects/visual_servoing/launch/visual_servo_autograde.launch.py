"""
Autograding launch for Q3 (Visual Servoing) — CI/headless.

Starts all four processes in sequence:
  t=0s   mujoco_sim_node          — headless physics (no viewer)
  t=2s   sensor_node              — sensor bridge
  t=4s   visual_servo_controller  — student code
  t=6s   test_visual_servo        — runs test, exits 0 (pass) or 1 (fail)

The test node's exit code is what GitHub Actions checks.
"""

import os

from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    q3_pkg = get_package_share_directory("visual_servoing")

    model_path = os.path.join(q3_pkg, 'models', 'visual_servo_scene.xml')
    ci_config  = os.path.join(q3_pkg, 'config', 'sim_params_ci.yaml')

    # ── 1. Sim node — headless, max speed ───────────────────────────
    sim_node = Node(
        package='mujoco_bridge',
        executable='mujoco_sim_node.py',
        name='mujoco_sim_node',
        output='screen',
        parameters=[
            ci_config,
            {'model_path': model_path},
        ],
    )

    # ── 2. Sensor node (after sim is up) ────────────────────────────
    sensor_node = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='q3',
                executable='sensor_node.py',
                name='q3_sensor_node',
                output='screen',
            )
        ],
    )

    # ── 3. Student controller (after sensor is up) ──────────────────
    controller_node = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='q3',
                executable='visual_servo_controller.py',
                name='visual_servo_controller',
                output='screen',
            )
        ],
    )

    # ── 4. Test node — exits with 0/1 for pass/fail ─────────────────
    test_node = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='q3',
                executable='test_visual_servo.py',
                name='test_visual_servo',
                output='screen',
            )
        ],
    )

    return LaunchDescription([sim_node, sensor_node, controller_node, test_node])
