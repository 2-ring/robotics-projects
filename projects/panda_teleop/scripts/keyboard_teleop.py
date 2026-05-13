#!/usr/bin/env python3
"""
Keyboard Teleop for Panda Robot.

Control individual joints using keyboard keys in real time.
Publishes to /panda/position_targets and /panda/gripper_command.

Usage:
  ros2 run q0 keyboard_teleop.py
"""

import sys
import tty
import termios
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

# ── Constants ───────────────────────────────────────────────────────
ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4',
              'joint5', 'joint6', 'joint7']
HOME  = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
READY = np.array([0.0,  0.0,   0.0, -1.571, 0.0, 1.571, 0.0])
ZERO  = np.zeros(7)

JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

HELP = """
╔═══════════════════════════════════════════════════════╗
║            Panda Keyboard Teleop                      ║
╠═══════════════════════════════════════════════════════╣
║  Joint selection:   1-7  arm joints, 8  gripper       ║
║  Joint control:     w/s  ±step,  a/d  ±5×step         ║
║  Step size:         +/-  double / halve                ║
║  Presets:           h home  r ready  z zero            ║
║  Gripper:           o open  c close                    ║
║  Print state:       p                                  ║
║  Quit:              q                                  ║
╚═══════════════════════════════════════════════════════╝
"""


class KeyboardTeleop(Node):

    def __init__(self):
        super().__init__('keyboard_teleop')

        fast = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST, depth=1)

        self.pos_pub     = self.create_publisher(Float64MultiArray, '/panda/position_targets', 10)
        self.gripper_pub = self.create_publisher(Float64MultiArray, '/panda/gripper_command',  10)

        self.state_sub = self.create_subscription(
            JointState, '/panda/joint_states', self._state_cb, fast)

        self.targets       = HOME.copy()
        self.gripper_open  = True              # True = open (1.0), False = closed (0.0)
        self.selected      = 0                 # 0-6 arm, 7 gripper
        self.step          = 0.05              # radians
        self.current_state = None

    def _state_cb(self, msg):
        self.current_state = msg

    def send(self):
        m = Float64MultiArray(); m.data = self.targets.tolist()
        self.pos_pub.publish(m)
        g = Float64MultiArray(); g.data = [1.0 if self.gripper_open else 0.0]
        self.gripper_pub.publish(g)

    def print_state(self):
        print('\n' + '-' * 55)
        print(f'  Selected: {"J" + str(self.selected+1) if self.selected < 7 else "Gripper"}  '
              f'| Step: {self.step:.4f} rad')
        print(f'  {"Joint":>10}  {"Target":>10}  {"Actual":>10}  {"Error":>10}')
        print('  ' + '-' * 45)
        for i in range(7):
            actual = 0.0
            if self.current_state and f'joint{i+1}' in self.current_state.name:
                idx = self.current_state.name.index(f'joint{i+1}')
                actual = self.current_state.position[idx]
            mark = ' <<' if i == self.selected else ''
            print(f'  {"J"+str(i+1):>10}  {self.targets[i]:10.4f}  {actual:10.4f}  '
                  f'{self.targets[i]-actual:10.4f}{mark}')
        grip_actual = 0.0
        if self.current_state and 'finger_joint1' in self.current_state.name:
            idx = self.current_state.name.index('finger_joint1')
            grip_actual = self.current_state.position[idx]
        mark = ' <<' if self.selected == 7 else ''
        g_val = 1.0 if self.gripper_open else 0.0
        print(f'  {"Grip":>10}  {g_val:10.4f}  {grip_actual:10.4f}  '
              f'{g_val-grip_actual:10.4f}{mark}')
        print('-' * 55)


def _get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleop()
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

    print(HELP)
    node.send()

    try:
        while rclpy.ok():
            k = _get_key()

            if k == 'q':
                break
            elif k in '1234567':
                node.selected = int(k) - 1
                print(f'\r  Selected: J{node.selected+1}          ', end='', flush=True)
            elif k == '8':
                node.selected = 7
                print(f'\r  Selected: Gripper       ', end='', flush=True)
            elif k in ('w', 's', 'a', 'd'):
                delta = node.step * (5 if k in ('a', 'd') else 1)
                if k in ('s', 'a'):
                    delta = -delta
                if node.selected < 7:
                    node.targets[node.selected] += delta
                    node.targets[node.selected] = np.clip(
                        node.targets[node.selected],
                        JOINT_LOWER[node.selected],
                        JOINT_UPPER[node.selected])
                    node.send()
                    print(f'\r  J{node.selected+1}: {node.targets[node.selected]:.4f}       ',
                          end='', flush=True)
                else:
                    node.gripper_open = not node.gripper_open
                    node.send()
                    print(f'\r  Gripper: {"open" if node.gripper_open else "closed"}       ',
                          end='', flush=True)
            elif k in ('+', '='):
                node.step = min(node.step * 2, 0.5)
                print(f'\r  Step: {node.step:.4f}          ', end='', flush=True)
            elif k == '-':
                node.step = max(node.step / 2, 0.001)
                print(f'\r  Step: {node.step:.4f}          ', end='', flush=True)
            elif k == 'h':
                node.targets = HOME.copy(); node.send()
                print('\r  Home                     ', end='', flush=True)
            elif k == 'r':
                node.targets = READY.copy(); node.send()
                print('\r  Ready                    ', end='', flush=True)
            elif k == 'z':
                node.targets = ZERO.copy(); node.send()
                print('\r  Zero                     ', end='', flush=True)
            elif k == 'o':
                node.gripper_open = True; node.send()
                print('\r  Gripper open             ', end='', flush=True)
            elif k == 'c':
                node.gripper_open = False; node.send()
                print('\r  Gripper closed           ', end='', flush=True)
            elif k == 'p':
                node.print_state()

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        print('\nBye!')


if __name__ == '__main__':
    main()
