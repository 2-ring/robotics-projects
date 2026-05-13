#!/usr/bin/env python3
"""
Q2: PID Controller Test

Moves the target to several positions and checks whether the end-effector
reaches each one within a time limit and distance tolerance.

Usage:
    ros2 run q2 test_pid.py
"""

import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64MultiArray

# ── Test parameters ─────────────────────────────────────────────────
NUM_TARGETS    = 10     # number of random targets to sample
SETTLE_TIME    = 5.0    # seconds to wait for each target
TOLERANCE      = 0.05   # metres — EE must be within this of target
SAMPLE_RATE    = 0.05   # seconds between samples
PASS_THRESHOLD = 7      # must reach at least this many targets

# Oscillation detection
OSCILLATION_WINDOW   = 2.0    # check last N seconds for oscillation
OSCILLATION_STD_MAX  = 0.015  # max std-dev of distance in window (metres)
SMOOTH_THRESHOLD     = 7      # must be smooth on at least this many targets

# Panda workspace bounds (reachable region in front of the arm)
WS_X = (0.2, 0.6)      # forward
WS_Y = (-0.4, 0.4)     # left-right
WS_Z = (0.2, 0.8)      # up-down
MIN_REACH = 0.25        # minimum distance from base to avoid self-collision
MAX_REACH = 0.75        # maximum distance from base


def _sample_reachable_targets(n):
    """Sample n random Cartesian targets within the Panda's reachable workspace."""
    rng = np.random.default_rng()
    targets = []
    while len(targets) < n:
        x = rng.uniform(*WS_X)
        y = rng.uniform(*WS_Y)
        z = rng.uniform(*WS_Z)
        dist = np.sqrt(x**2 + y**2 + z**2)
        if MIN_REACH < dist < MAX_REACH:
            targets.append([x, y, z])
    return targets


class PIDTester(Node):

    def __init__(self):
        super().__init__('pid_tester')

        self.ee_position     = np.zeros(3)
        self.target_position = np.zeros(3)

        fast = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(
            Float64MultiArray, '/q2/ee_position', self._ee_cb, fast)
        self.create_subscription(
            Float64MultiArray, '/q2/target_position', self._target_cb, fast)

        self.target_pub = self.create_publisher(
            Float64MultiArray, '/q2/set_target_position', 10)

        self.get_logger().info('PID tester started.')

    def _ee_cb(self, msg):
        self.ee_position = np.array(msg.data[:3])

    def _target_cb(self, msg):
        self.target_position = np.array(msg.data[:3])

    def _set_target(self, pos):
        msg = Float64MultiArray()
        msg.data = list(pos)
        self.target_pub.publish(msg)

    def run_tests(self):
        TARGETS = _sample_reachable_targets(NUM_TARGETS)

        print('\n' + '=' * 60)
        print('  Q2: PID Controller Test')
        print('=' * 60)
        print(f'  Targets:        {len(TARGETS)} (randomly sampled)')
        print(f'  Settle time:    {SETTLE_TIME}s per target')
        print(f'  Tolerance:      {TOLERANCE}m')
        print(f'  Pass threshold: {PASS_THRESHOLD}/{len(TARGETS)}')
        print('=' * 60)

        print('\nWaiting for sensor data...')
        time.sleep(2.0)
        rclpy.spin_once(self, timeout_sec=1.0)

        results = []
        smooth_results = []

        for i, target in enumerate(TARGETS):
            target = np.array(target)
            print(f'\n── Target {i+1}/{len(TARGETS)}: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}] ──')

            # Send target
            self._set_target(target)

            # Wait for settling — collect distance samples for oscillation check
            t_start = time.time()
            best_dist = float('inf')
            reached = False
            dist_history = []  # (elapsed, distance) pairs
            last_print_s = -1

            while time.time() - t_start < SETTLE_TIME:
                rclpy.spin_once(self, timeout_sec=SAMPLE_RATE)

                dist = np.linalg.norm(self.ee_position - target)
                best_dist = min(best_dist, dist)
                elapsed = time.time() - t_start
                dist_history.append((elapsed, dist))

                current_s = int(elapsed)
                if current_s != last_print_s:  # exactly once per second
                    last_print_s = current_s
                    print(f'  t={elapsed:.1f}s  EE=[{self.ee_position[0]:.3f}, '
                          f'{self.ee_position[1]:.3f}, {self.ee_position[2]:.3f}]  '
                          f'dist={dist:.4f}m  best={best_dist:.4f}m')

                if dist < TOLERANCE:
                    reached = True

            # ── Oscillation check ───────────────────────────────────
            # Measure std-dev of distance in the last OSCILLATION_WINDOW seconds
            window_start = SETTLE_TIME - OSCILLATION_WINDOW
            tail_dists = [d for t, d in dist_history if t >= window_start]
            if len(tail_dists) >= 3:
                osc_std = np.std(tail_dists)
                smooth = osc_std < OSCILLATION_STD_MAX
            else:
                osc_std = 0.0
                smooth = True

            status = 'PASS ✓' if reached else 'FAIL ✗'
            osc_status = 'smooth' if smooth else f'oscillating (σ={osc_std:.4f}m)'
            print(f'  Result: {status}  (best: {best_dist:.4f}m)  [{osc_status}]')
            results.append(reached)
            smooth_results.append(smooth)

        # Summary
        passed = sum(results)
        total = len(results)
        num_smooth = sum(smooth_results)
        reach_ok = passed >= PASS_THRESHOLD
        smooth_ok = num_smooth >= SMOOTH_THRESHOLD
        overall = 'PASS ✓' if (reach_ok and smooth_ok) else 'FAIL ✗'

        print('\n' + '=' * 60)
        print(f'  Accuracy:    {passed}/{total} targets reached'
              f'  (need {PASS_THRESHOLD})  {"✓" if reach_ok else "✗"}')
        print(f'  Smoothness:  {num_smooth}/{total} targets without oscillation'
              f'  (need {SMOOTH_THRESHOLD})  {"✓" if smooth_ok else "✗"}')
        print(f'  Overall: {overall}')
        print('=' * 60 + '\n')

        return reach_ok and smooth_ok


def main(args=None):
    rclpy.init(args=args)
    node = PIDTester()
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
