#!/usr/bin/env python3
"""
Q3: Visual Servo Controller Test

Moves the red ball along a smooth path on the table and checks whether
the end-effector tracks it in XY (ignoring Z).

Prerequisites — these must already be running:
    ros2 launch q3 visual_servo_sim.launch.py
    ros2 run q3 visual_servo_controller.py   (your code!)

Run:
    ros2 run q3 test_visual_servo.py
"""

import time
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64MultiArray


# ── Test parameters ─────────────────────────────────────────────────
TEST_DURATION    = 30.0     # total test time (seconds)
SETTLE_TIME      = 5.0      # initial settling time before scoring starts
TOLERANCE_XY     = 0.05     # metres — EE must be within this of ball in XY
PASS_FRACTION    = 0.70     # fraction of samples that must be within tolerance
SAMPLE_RATE      = 0.05     # seconds between samples

# Ball motion parameters
BALL_Z           = 0.445    # ball height (table top at z=0.42 + ball radius 0.025)
CENTER_X         = 0.5      # center of ball path (X) — table center
CENTER_Y         = 0.0      # center of ball path (Y)
RADIUS           = 0.15     # radius of figure-8 path
SPEED            = 0.3      # angular speed (rad/s)


def _ball_path(t):
    """Figure-8 path on the table surface."""
    x = CENTER_X + RADIUS * math.sin(SPEED * t)
    y = CENTER_Y + RADIUS * math.sin(SPEED * t * 2) * 0.5
    return [x, y, BALL_Z]


class VisualServoTester(Node):

    def __init__(self):
        super().__init__('visual_servo_tester')

        self.ee_position = np.zeros(3)
        self.ball_position = np.array([CENTER_X, CENTER_Y, BALL_Z])

        fast = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(
            Float64MultiArray, '/q3/ee_position', self._ee_cb, fast)

        self.ball_pub = self.create_publisher(
            Float64MultiArray, '/q3/set_ball_position', 10)

        self.get_logger().info('Visual servo tester started.')

    def _ee_cb(self, msg):
        self.ee_position = np.array(msg.data[:3])

    def _set_ball(self, pos):
        msg = Float64MultiArray()
        msg.data = list(pos)
        self.ball_pub.publish(msg)
        self.ball_position = np.array(pos)  # track locally

    def run_tests(self):
        print('\n' + '=' * 65)
        print('  Q3: Visual Servoing Test')
        print('=' * 65)
        print(f'  Duration:       {TEST_DURATION}s (+ {SETTLE_TIME}s settling)')
        print(f'  XY tolerance:   {TOLERANCE_XY}m')
        print(f'  Pass threshold: {PASS_FRACTION*100:.0f}% of samples within tolerance')
        print(f'  Ball path:      figure-8, radius={RADIUS}m, speed={SPEED} rad/s')
        print('=' * 65)

        # Wait for sensor data
        print('\nWaiting for sensor data...')
        time.sleep(2.0)
        rclpy.spin_once(self, timeout_sec=1.0)

        # Phase 1: Settle — move ball to start and let controller converge
        print(f'\nSettling phase ({SETTLE_TIME}s)...')
        start_pos = _ball_path(0.0)
        self._set_ball(start_pos)
        t_start = time.time()
        last_print_s = -1
        while time.time() - t_start < SETTLE_TIME:
            rclpy.spin_once(self, timeout_sec=SAMPLE_RATE)
            self._set_ball(start_pos)  # keep publishing
            dist_xy = np.linalg.norm(self.ee_position[:2] - np.array(start_pos[:2]))
            elapsed = time.time() - t_start
            current_s = int(elapsed)
            if current_s != last_print_s:
                last_print_s = current_s
                print(f'  t={elapsed:.0f}s  EE=[{self.ee_position[0]:.3f}, '
                      f'{self.ee_position[1]:.3f}, {self.ee_position[2]:.3f}]  '
                      f'XY dist={dist_xy:.4f}m')

        # Phase 2: Moving ball — score tracking
        print(f'\nTracking phase ({TEST_DURATION}s)...')
        t_start = time.time()
        n_samples = 0
        n_within = 0
        distances = []
        last_print_s = -1

        while time.time() - t_start < TEST_DURATION:
            elapsed = time.time() - t_start
            ball_pos = _ball_path(elapsed)
            self._set_ball(ball_pos)

            rclpy.spin_once(self, timeout_sec=SAMPLE_RATE)

            # XY distance only (ignore Z)
            dist_xy = np.linalg.norm(self.ee_position[:2] - np.array(ball_pos[:2]))
            distances.append(dist_xy)
            n_samples += 1
            if dist_xy < TOLERANCE_XY:
                n_within += 1

            current_s = int(elapsed / 2) * 2  # every 2 seconds
            if current_s != last_print_s:
                last_print_s = current_s
                pct = (n_within / n_samples * 100) if n_samples > 0 else 0
                print(f'  t={elapsed:.1f}s  ball=[{ball_pos[0]:.3f}, {ball_pos[1]:.3f}]  '
                      f'EE=[{self.ee_position[0]:.3f}, {self.ee_position[1]:.3f}]  '
                      f'XY err={dist_xy:.4f}m  tracking={pct:.0f}%')

        # Results
        distances = np.array(distances)
        pass_rate = n_within / n_samples if n_samples > 0 else 0
        passed = pass_rate >= PASS_FRACTION

        print('\n' + '=' * 65)
        print(f'  Samples:           {n_samples}')
        print(f'  Within {TOLERANCE_XY}m (XY):  {n_within}/{n_samples} ({pass_rate*100:.1f}%)')
        print(f'  Mean XY error:     {np.mean(distances):.4f}m')
        print(f'  Max XY error:      {np.max(distances):.4f}m')
        print(f'  Median XY error:   {np.median(distances):.4f}m')
        print(f'  Result:            {"PASS ✓" if passed else "FAIL ✗"}')
        print('=' * 65 + '\n')

        # Stop the robot
        stop_msg = Float64MultiArray()
        stop_msg.data = [0.0] * 7
        # publish to velocity command to stop
        stop_pub = self.create_publisher(Float64MultiArray, '/q3/joint_velocity_command', 10)
        stop_pub.publish(stop_msg)

        return passed


def main(args=None):
    rclpy.init(args=args)
    node = VisualServoTester()
    try:
        success = node.run_tests()
        raise SystemExit(0 if success else 1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
