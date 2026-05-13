#!/usr/bin/env python3
"""
Joint Pose Commander — set joint positions and control the gripper.

Interactive CLI tool for commanding the Panda robot.
Publishes to /panda/position_targets and /panda/gripper_command.

Usage:
  ros2 run q0 joint_pose_commander.py
"""

import sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

# ── Joint names (Menagerie) ─────────────────────────────────────────
ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4',
              'joint5', 'joint6', 'joint7']

# ── Predefined poses (7 values in radians) ─────────────────────────
POSES = {
    'home':       [0.0,  0.0,    0.0, -1.571,  0.0,  1.571, -0.785],
    'ready':      [0.0, -0.785,  0.0, -2.356,  0.0,  1.571,  0.785],
    'up':         [0.0, -1.2,    0.0, -0.5,    0.0,  0.7,    0.785],
    'left':       [1.0, -0.785,  0.0, -2.356,  0.0,  1.571,  0.785],
    'right':      [-1.0, -0.785, 0.0, -2.356,  0.0,  1.571,  0.785],
    'forward':    [0.0,  0.0,    0.0, -1.0,    0.0,  1.0,    0.785],
    'tuck':       [0.0, -1.5,    0.0, -2.8,    0.0,  1.2,    0.785],
    'zero':       [0.0,  0.0,    0.0,  0.0,    0.0,  0.0,    0.0],
    'pick_ready': [0.0, -0.3,    0.0, -2.0,    0.0,  2.0,    0.785],
}


class JointPoseCommander(Node):

    def __init__(self):
        super().__init__('joint_pose_commander')

        fast = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST, depth=1)

        self.pos_pub     = self.create_publisher(Float64MultiArray, '/panda/position_targets', 10)
        self.gripper_pub = self.create_publisher(Float64MultiArray, '/panda/gripper_command',  10)

        self.state_sub = self.create_subscription(
            JointState, '/panda/joint_states', self._state_cb, fast)
        self.current_state = None

    def _state_cb(self, msg):
        self.current_state = msg

    def send_position(self, positions):
        msg = Float64MultiArray()
        msg.data = list(positions)
        self.pos_pub.publish(msg)
        self.get_logger().info(f'Sent: {[f"{p:.3f}" for p in positions]}')

    def send_gripper(self, value):
        """value: 0.0 = closed, 1.0 = open (normalised)."""
        msg = Float64MultiArray()
        msg.data = [value]
        self.gripper_pub.publish(msg)
        self.get_logger().info(f'Gripper: {"OPEN" if value > 0.5 else "CLOSED"} ({value:.2f})')


def print_help():
    print('\n' + '=' * 60)
    print('  Panda Joint Pose Commander')
    print('=' * 60)
    print('\n  Predefined poses:')
    for name, joints in POSES.items():
        print(f'    {name:12s}  {[f"{j:.3f}" for j in joints]}')
    print()
    print('  Commands:')
    print('    <pose_name>       Send a predefined pose')
    print('    set <j> <val>     Set joint j (1-7) to val radians')
    print('    open              Open gripper')
    print('    close             Close gripper')
    print('    status            Print current joint state')
    print('    custom            Enter 7 joint angles manually')
    print('    quit / q          Exit')
    print('=' * 60)


def main(args=None):
    rclpy.init(args=args)
    node = JointPoseCommander()

    # Track the last-sent position (start with home)
    current_target = list(POSES['home'])

    print_help()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            try:
                cmd = input('\nCommand > ').strip().lower()
            except EOFError:
                break

            if not cmd:
                continue

            # ── Predefined pose ──
            if cmd in POSES:
                current_target = list(POSES[cmd])
                node.send_position(current_target)

            # ── Set individual joint ──
            elif cmd.startswith('set'):
                parts = cmd.split()
                if len(parts) == 3:
                    try:
                        j = int(parts[1])
                        val = float(parts[2])
                        if 1 <= j <= 7:
                            current_target[j - 1] = val
                            node.send_position(current_target)
                        else:
                            print('Joint must be 1–7')
                    except ValueError:
                        print('Usage: set <joint 1-7> <radians>')
                else:
                    print('Usage: set <joint 1-7> <radians>')

            # ── Gripper ──
            elif cmd == 'open':
                node.send_gripper(1.0)
            elif cmd == 'close':
                node.send_gripper(0.0)

            # ── Status ──
            elif cmd == 'status':
                if node.current_state:
                    print(f'\n  {"Joint":>20s}  {"Position":>10s}  {"Velocity":>10s}  {"Effort":>10s}')
                    print('  ' + '-' * 55)
                    for i, name in enumerate(node.current_state.name):
                        p = node.current_state.position[i] if i < len(node.current_state.position) else 0
                        v = node.current_state.velocity[i] if i < len(node.current_state.velocity) else 0
                        e = node.current_state.effort[i]   if i < len(node.current_state.effort)   else 0
                        print(f'  {name:>20s}  {p:10.4f}  {v:10.4f}  {e:10.4f}')
                    print(f'\n  Target: {[f"{t:.3f}" for t in current_target]}')
                else:
                    print('  No joint state received yet.')

            # ── Custom ──
            elif cmd == 'custom':
                try:
                    s = input('  Enter 7 joint angles (space-separated, radians): ')
                    vals = [float(x) for x in s.split()]
                    if len(vals) == 7:
                        current_target = vals
                        node.send_position(current_target)
                    else:
                        print(f'  Expected 7 values, got {len(vals)}')
                except ValueError:
                    print('  Invalid input.')

            # ── Quit ──
            elif cmd in ('quit', 'exit', 'q'):
                break

            # ── Help ──
            elif cmd == 'help':
                print_help()

            else:
                print(f'  Unknown: {cmd}  (type "help" for commands)')

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        print('\nBye!')


if __name__ == '__main__':
    main()
