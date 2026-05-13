#!/usr/bin/env python3
"""Q3 Autograder — tests Learning from Demonstration puck pushing.

Test procedure:
  For each of N sampled test configurations:
    1. Reset simulation and sample a puck start / goal pose.
    2. Wait for student policy to push puck to goal.
    3. Check if puck reached within tolerance.

Grading:
  - Puck must reach within 0.05 m of goal center.
  - Must complete within 30 seconds per episode.
  - Pass: at least 8 out of 10 test configurations succeed.

This script is PROVIDED and should NOT be modified.
"""

import argparse
import math
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray, String
from std_srvs.srv import Trigger

from skills_utils import FAST_QoS

GOAL_TOLERANCE_M = 0.05
EPISODE_TIMEOUT_S = 30.0
TABLE_HEIGHT = 0.44
NUM_TEST_EPISODES = 10
PUCK_Y_RANGE_M = (-0.15, 0.15)
GOAL_Y_RANGE_M = (-0.25, 0.25)
HOME_ARM_POSITION = np.array(
    [0.0000225, -1.4632756, 0.0000459, -2.6155017, 0.0000499, 1.1522261, 0.7855477],
    dtype=float,
)
PUCK_RESET_POSITION = np.array([0.38, 0.0, 0.432], dtype=float)
GOAL_RESET_POSITION = np.array([0.80, 0.10, 0.422], dtype=float)
OBSTACLE_Z_M = 0.432


class Q3Tester(Node):

    def __init__(self):
        super().__init__("q3_tester")

        self.puck_position = None
        self.goal_position = None
        self.joint_positions = None

        self.create_subscription(
            Float64MultiArray, "/q3/joint_positions", self._joint_cb, FAST_QoS)
        self.create_subscription(
            Float64MultiArray, "/q3/puck_position", self._puck_cb, FAST_QoS)
        self.create_subscription(
            Float64MultiArray, "/q3/goal_position", self._goal_cb, FAST_QoS)

        self.mocap_pub = self.create_publisher(
            Float64MultiArray, "/mujoco/mocap_pos", FAST_QoS)
        self.body_pose_pub = self.create_publisher(
            String, "/mujoco/body_pose", FAST_QoS)
        self.target_pub = self.create_publisher(
            Float64MultiArray, "/panda/position_targets", FAST_QoS)
        self.episode_ready_pub = self.create_publisher(
            Bool, "/q3/episode_ready", FAST_QoS)

        self.reset_client = self.create_client(Trigger, "/mujoco/reset")

    def _joint_cb(self, msg):
        self.joint_positions = np.array(msg.data, dtype=float)

    def _puck_cb(self, msg):
        self.puck_position = np.array(msg.data)

    def _goal_cb(self, msg):
        self.goal_position = np.array(msg.data)

    def reset_sim(self):
        """Reset the MuJoCo simulation."""
        if not self.reset_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("Reset service not available.")
            return False
        future = self.reset_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        return future.result() is not None and future.result().success

    def wait_for_state(self, timeout_sec=3.0):
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if (
                self.puck_position is not None
                and self.goal_position is not None
                and self.joint_positions is not None
            ):
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return False

    def wait_for_scene_target(self, puck_target, goal_target,
                              obstacle_pos=None, timeout_sec=5.0, tol=0.03):
        deadline = time.time() + timeout_sec
        puck_target = np.asarray(puck_target, dtype=float)
        goal_target = np.asarray(goal_target, dtype=float)
        while time.time() < deadline:
            # Republish placement commands each iteration so dropped
            # BEST_EFFORT messages are retried automatically.
            self.set_puck_position(*puck_target)
            if obstacle_pos is not None:
                self.set_goal_and_obstacle_position(
                    goal_target[0], goal_target[1], goal_target[2],
                    obstacle_pos[0], obstacle_pos[1], obstacle_pos[2],
                )
            else:
                self.set_goal_and_obstacle_position(
                    goal_target[0], goal_target[1], goal_target[2],
                    goal_target[0], goal_target[1], goal_target[2],
                )
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.puck_position is None or self.goal_position is None:
                continue
            puck_err = np.linalg.norm(np.asarray(self.puck_position[:3], dtype=float) - puck_target)
            goal_err = np.linalg.norm(np.asarray(self.goal_position[:3], dtype=float) - goal_target)
            if puck_err <= tol and goal_err <= tol:
                return True
        return False

    def wait_for_home(self, timeout_sec=8.0, pos_tol=0.05):
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.joint_positions is None:
                continue
            if np.linalg.norm(self.joint_positions[:7] - HOME_ARM_POSITION) <= pos_tol:
                return True
            home_msg = Float64MultiArray()
            home_msg.data = HOME_ARM_POSITION.tolist()
            self.target_pub.publish(home_msg)
        return False

    def set_goal_and_obstacle_position(self, gx, gy, gz, ox, oy, oz):
        msg = Float64MultiArray()
        msg.data = [gx, gy, gz, ox, oy, oz]
        self.mocap_pub.publish(msg)

    def publish_episode_ready(self, ready: bool):
        msg = Bool()
        msg.data = bool(ready)
        self.episode_ready_pub.publish(msg)

    def set_puck_position(self, px, py, pz):
        msg = String()
        msg.data = (
            '{"body":"puck","pos":['
            f'{float(px)},{float(py)},{float(pz)}'
            '],"quat":[1.0,0.0,0.0,0.0]}'
        )
        self.body_pose_pub.publish(msg)

    def run_tests(self, plot=False):
        self.get_logger().info("=" * 50)
        self.get_logger().info("Q3 AUTOGRADE: Learning from Demonstration")
        self.get_logger().info("=" * 50)

        # Wait for sensor data
        t0 = time.time()
        while self.puck_position is None and time.time() - t0 < 10.0:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.puck_position is None:
            self.get_logger().error("No puck position received. Is the simulation running?")
            return False

        score = 0
        episode_data = []

        rng = np.random.default_rng(7)

        for i in range(NUM_TEST_EPISODES):
            self.get_logger().info(f"\n--- Test {i+1}/{NUM_TEST_EPISODES} ---")
            self.publish_episode_ready(False)
            time.sleep(0.15)  # let policy process stop signal

            if not self.reset_sim():
                self.get_logger().error("Failed to reset simulation.")
                return False
            if not self.wait_for_state():
                self.get_logger().error("Timed out waiting for state after reset.")
                return False

            # After reset the bridge may still be driving the arm toward the
            # last policy target. In headless mode the bridge's 500 Hz state
            # callbacks can starve the position-target callback, so pump home
            # targets aggressively while spinning before the home check runs
            # — this guarantees the bridge has processed the new target
            # before wait_for_home starts polling.
            home_msg = Float64MultiArray()
            home_msg.data = HOME_ARM_POSITION.tolist()
            settle_deadline = time.time() + 0.5
            while time.time() < settle_deadline:
                self.target_pub.publish(home_msg)
                rclpy.spin_once(self, timeout_sec=0.01)

            if not self.wait_for_home():
                self.get_logger().error("Robot did not return to home pose before placing objects.")
                return False
            time.sleep(0.1)  # let arm stabilize at home

            px = float(PUCK_RESET_POSITION[0])
            py = float(rng.uniform(*PUCK_Y_RANGE_M))
            pz = float(PUCK_RESET_POSITION[2])
            gx = float(GOAL_RESET_POSITION[0])
            gy = float(rng.uniform(*GOAL_Y_RANGE_M))
            gz = float(GOAL_RESET_POSITION[2])
            oz = OBSTACLE_Z_M

            self.get_logger().info(f"  Puck start: ({px:.2f}, {py:.2f})")
            self.get_logger().info(f"  Goal:       ({gx:.2f}, {gy:.2f})")

            ox = px + (gx - px) / 3.0
            oy = py + (gy - py) / 3.0

            if not self.wait_for_scene_target(
                [px, py, pz], [gx, gy, gz], obstacle_pos=[ox, oy, oz]
            ):
                self.get_logger().error("Timed out waiting for sampled puck/goal placement.")
                return False
            self.publish_episode_ready(True)

            # Wait for episode — record puck trajectory
            t_start = time.time()
            reached = False
            min_dist = float("inf")
            puck_traj = []

            while time.time() - t_start < EPISODE_TIMEOUT_S:
                rclpy.spin_once(self, timeout_sec=0.05)

                if self.puck_position is not None:
                    puck_traj.append(self.puck_position[:3].copy())
                    dx = self.puck_position[0] - gx
                    dy = self.puck_position[1] - gy
                    dist = math.sqrt(dx*dx + dy*dy)
                    min_dist = min(min_dist, dist)

                    if dist < GOAL_TOLERANCE_M:
                        reached = True
                        break

            elapsed = time.time() - t_start
            score += int(reached)
            self.publish_episode_ready(False)
            status = "PASS" if reached else "FAIL"
            self.get_logger().info(
                f"  Result: {status} (min_dist={min_dist:.3f} m, time={elapsed:.1f} s)")

            episode_data.append({
                "puck_start": np.array([px, py, pz]),
                "goal": np.array([gx, gy, gz]),
                "obstacle": np.array([ox, oy, oz]),
                "puck_traj": np.array(puck_traj) if puck_traj else np.zeros((0, 3)),
                "reached": reached,
                "min_dist": min_dist,
            })

        pass_threshold = 8
        overall = score >= pass_threshold

        self.get_logger().info("\n" + "=" * 50)
        self.get_logger().info(f"Score: {score}/{NUM_TEST_EPISODES} episodes succeeded")
        self.get_logger().info(f"Required: {pass_threshold}/{NUM_TEST_EPISODES}")
        self.get_logger().info(f"OVERALL: {'PASS' if overall else 'FAIL'}")
        self.get_logger().info("=" * 50)

        if plot:
            self._plot_episodes(episode_data, score)

        return overall

    def _plot_episodes(self, episode_data, score):
        """Show a figure with one subplot per episode (top-down XY view)."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        n = len(episode_data)
        cols = 5
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        fig.suptitle(
            f"Q3 LfD Results — {score}/{n} passed",
            fontsize=16, fontweight="bold",
        )
        axes = np.asarray(axes).flatten()

        for i, ep in enumerate(episode_data):
            ax = axes[i]
            ps = ep["puck_start"]
            gl = ep["goal"]
            ob = ep["obstacle"]
            traj = ep["puck_traj"]

            # Puck trajectory
            if len(traj) > 0:
                ax.plot(traj[:, 0], traj[:, 1], "-", color="steelblue",
                        linewidth=1.5, label="puck trajectory", zorder=2)

            # Puck start
            ax.plot(ps[0], ps[1], "o", color="red", markersize=10,
                    label="puck start", zorder=4)

            # Goal
            goal_circle = plt.Circle(
                (gl[0], gl[1]), GOAL_TOLERANCE_M, color="green",
                alpha=0.25, zorder=1,
            )
            ax.add_patch(goal_circle)
            ax.plot(gl[0], gl[1], "*", color="green", markersize=14,
                    label="goal", zorder=4)

            # Obstacle
            ax.plot(ob[0], ob[1], "s", color="orange", markersize=9,
                    label="obstacle", zorder=4)

            # Puck final position
            if len(traj) > 0:
                ax.plot(traj[-1, 0], traj[-1, 1], "x", color="darkred",
                        markersize=10, markeredgewidth=2, label="puck final", zorder=5)

            status = "PASS" if ep["reached"] else "FAIL"
            color = "green" if ep["reached"] else "red"
            ax.set_title(
                f"Episode {i+1}: {status} (d={ep['min_dist']:.3f})",
                fontsize=10, color=color, fontweight="bold",
            )
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

            # Table bounds (rough)
            ax.set_xlim(0.15, 0.95)
            ax.set_ylim(-0.50, 0.50)

        # Hide unused subplots
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        # Shared legend
        handles = [
            mpatches.Patch(color="red", label="Puck start"),
            mpatches.Patch(color="green", label="Goal"),
            mpatches.Patch(color="orange", label="Obstacle"),
            mpatches.Patch(color="steelblue", label="Puck trajectory"),
            mpatches.Patch(color="darkred", label="Puck final"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10)
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Q3 LfD Autograder")
    parser.add_argument("--plot", action="store_true",
                        help="Show a figure with per-episode results after testing")
    # ROS2 may pass extra args; use parse_known_args to ignore them.
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    tester = Q3Tester()
    try:
        success = tester.run_tests(plot=args.plot)
    except KeyboardInterrupt:
        success = False
    finally:
        tester.destroy_node()
        rclpy.try_shutdown()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
