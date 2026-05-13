#!/usr/bin/env python3
"""Q1 Autograder — tests semantic mapping and object-goal navigation.

Test procedure:
  Phase 1 (Exploration): Drives the robot through the student-defined
    waypoints. The student's semantic mapper should accumulate
    detections and build a map.
  Phase 2 (Map Evaluation): Reads the published semantic map and compares
    object positions to ground truth (read from /q1/object_positions).
    Scores position accuracy.
  Phase 3 (Navigation, 3 random queries × 120 s each): For 3 randomly
    chosen objects, the robot must navigate 0.5 m toward the origin along
    the axis with the smaller absolute coordinate.

Grading:
  - Map accuracy: Position error < 0.30 m for at least 4/5 objects. (40 pts)
  - Navigation: Robot reaches within 0.15 m for at least 2/3 queries. (60 pts)

This script is PROVIDED and should NOT be modified.
"""

import json
import math
import random
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose2D, Twist
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from skills_utils import FAST_QoS, wrap_angle

# Import student-defined exploration waypoints
try:
    from semantic_mapper import EXPLORATION_WAYPOINTS
except ImportError:
    # Fallback if student hasn't defined them yet
    EXPLORATION_WAYPOINTS = [
        (0.0, -0.4, 0.0),
    ]


class Q1Tester(Node):

    def __init__(self):
        super().__init__("q1_tester")

        self.robot_pose = None
        self.semantic_map_data = None
        self.nav_status = "idle"
        self.ground_truth: dict[str, list[float]] = {}

        self.create_subscription(Pose2D, "/q1/robot_pose", self._pose_cb, FAST_QoS)
        self.create_subscription(String, "/q1/semantic_map", self._map_cb, FAST_QoS)
        self.create_subscription(String, "/q1/navigation_status", self._nav_cb, FAST_QoS)
        self.create_subscription(String, "/q1/object_positions", self._gt_cb, FAST_QoS)

        self.nav_goal_pub = self.create_publisher(Pose2D, "/q1/navigation_goal", FAST_QoS)
        self.query_pub = self.create_publisher(String, "/q1/query", FAST_QoS)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", FAST_QoS)
        self.wp_marker_pub = self.create_publisher(MarkerArray, "/q1/exploration_waypoints", FAST_QoS)

    def _pose_cb(self, msg):
        self.robot_pose = msg

    def _map_cb(self, msg):
        try:
            self.semantic_map_data = json.loads(msg.data)
        except Exception:
            pass

    def _nav_cb(self, msg):
        self.nav_status = msg.data

    def _gt_cb(self, msg):
        try:
            self.ground_truth = json.loads(msg.data)
        except Exception:
            pass

    def _publish_waypoint_markers(self, waypoints, current_idx):
        """Publish exploration waypoints: green=visited, yellow=current, red=pending."""
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()
        for i, (wx, wy, _) in enumerate(waypoints):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = now
            m.ns = "exploration_waypoints"
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = wx
            m.pose.position.y = wy
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.12
            m.scale.z = 0.10
            if i < current_idx:
                m.color.r, m.color.g, m.color.b = 0.0, 0.8, 0.0  # green = visited
            elif i == current_idx:
                m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0  # yellow = current
            else:
                m.color.r, m.color.g, m.color.b = 1.0, 0.0, 0.0  # red = pending
            m.color.a = 0.85

            # Text label
            label = Marker()
            label.header.frame_id = "map"
            label.header.stamp = now
            label.ns = "wp_labels"
            label.id = i
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = wx
            label.pose.position.y = wy
            label.pose.position.z = 0.20
            label.pose.orientation.w = 1.0
            label.scale.z = 0.08
            label.color.r = label.color.g = label.color.b = 1.0
            label.color.a = 1.0
            label.text = str(i + 1)

            arr.markers.append(m)
            arr.markers.append(label)
        self.wp_marker_pub.publish(arr)

    def _clear_exploration_markers(self):
        """Remove all exploration waypoint markers from RViz."""
        arr = MarkerArray()
        for ns in ("exploration_waypoints", "wp_labels"):
            d = Marker()
            d.ns = ns
            d.action = Marker.DELETEALL
            arr.markers.append(d)
        self.wp_marker_pub.publish(arr)

    def _publish_query_markers(self, query_goals, current_idx):
        """Publish Phase-3 navigation query goals: green=done, yellow=current, magenta=pending."""
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()
        for i, (obj_class, gx, gy) in enumerate(query_goals):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = now
            m.ns = "query_goals"
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = gx
            m.pose.position.y = gy
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.14
            m.scale.z = 0.12
            if i < current_idx:
                m.color.r, m.color.g, m.color.b = 0.0, 0.8, 0.0
            elif i == current_idx:
                m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0
            else:
                m.color.r, m.color.g, m.color.b = 0.85, 0.2, 0.85
            m.color.a = 0.85

            label = Marker()
            label.header.frame_id = "map"
            label.header.stamp = now
            label.ns = "query_labels"
            label.id = i
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = gx
            label.pose.position.y = gy
            label.pose.position.z = 0.22
            label.pose.orientation.w = 1.0
            label.scale.z = 0.09
            label.color.r = label.color.g = label.color.b = 1.0
            label.color.a = 1.0
            label.text = f"Q{i+1}:{obj_class}"

            arr.markers.append(m)
            arr.markers.append(label)
        self.wp_marker_pub.publish(arr)

    def wait_for_pose(self, timeout=10.0):
        t0 = time.time()
        while self.robot_pose is None and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.robot_pose is not None

    def wait_for_ground_truth(self, timeout=10.0):
        """Wait until ground-truth object positions arrive from sensor node."""
        t0 = time.time()
        while len(self.ground_truth) < 5 and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        return len(self.ground_truth) >= 5

    def navigate_to_waypoint(self, wx, wy, wtheta, timeout=120.0):
        """Drive robot to a waypoint via the navigator node."""
        goal = Pose2D()
        goal.x = float(wx);  goal.y = float(wy);  goal.theta = float(wtheta)

        t0 = time.time()

        # Burst-publish to survive DDS best-effort drops
        for _ in range(5):
            self.nav_goal_pub.publish(goal)
            rclpy.spin_once(self, timeout_sec=0.05)

        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.robot_pose is not None:
                dx = wx - self.robot_pose.x
                dy = wy - self.robot_pose.y
                pos_ok = math.sqrt(dx * dx + dy * dy) < 0.15
                yaw_err = (wtheta - self.robot_pose.theta + math.pi) % (2 * math.pi) - math.pi
                yaw_ok = abs(yaw_err) < 0.15
                if pos_ok and yaw_ok:
                    return

        self.get_logger().warn(
            f"navigate_to_waypoint ({wx:.1f}, {wy:.1f}) timed out after {timeout:.0f} s")

    def _generate_nav_queries(self):
        """Pick 3 random objects.  For each, move 0.5 m toward the origin
        along the axis with the smaller absolute coordinate value."""
        rng = random.Random()
        objects = list(self.ground_truth.keys())
        rng.shuffle(objects)
        chosen = objects[:3]

        queries = []
        for obj in chosen:
            gt = self.ground_truth[obj]
            ox, oy = gt[0], gt[1]
            if abs(ox) <= abs(oy):
                # offset on x-axis toward origin
                off_x = -0.5 if ox > 0 else 0.5
                off_y = 0.0
            else:
                # offset on y-axis toward origin
                off_x = 0.0
                off_y = -0.5 if oy > 0 else 0.5
            queries.append((obj, off_x, off_y))
        return queries

    def run_tests(self):
        self.get_logger().info("=" * 50)
        self.get_logger().info("Q1 AUTOGRADE: Semantic Mapping & Navigation")
        self.get_logger().info("=" * 50)

        if not self.wait_for_pose():
            self.get_logger().error("No robot pose received. Is the simulation running?")
            return False

        if not self.wait_for_ground_truth():
            self.get_logger().error("No ground-truth object positions received!")
            return False

        self.get_logger().info("Ground truth received:")
        for obj, pos in self.ground_truth.items():
            self.get_logger().info(f"  {obj}: ({pos[0]:.3f}, {pos[1]:.3f})")

        waypoints = EXPLORATION_WAYPOINTS

        # ── Phase 1: Exploration ──
        self.get_logger().info(f"Phase 1: Exploration ({len(waypoints)} waypoints)...")
        for i, (wx, wy, wt) in enumerate(waypoints):
            self._publish_waypoint_markers(waypoints, i)
            self.get_logger().info(f"  Waypoint {i+1}/{len(waypoints)}: ({wx:.1f}, {wy:.1f})")
            self.navigate_to_waypoint(wx, wy, wt)
            # Pause briefly at each waypoint to let detector accumulate
            t0 = time.time()
            while time.time() - t0 < 2.0:
                rclpy.spin_once(self, timeout_sec=0.1)

        self._publish_waypoint_markers(waypoints, len(waypoints))
        self.get_logger().info("Exploration complete. Waiting for final map updates...")
        t0 = time.time()
        while time.time() - t0 < 3.0:
            rclpy.spin_once(self, timeout_sec=0.1)

        # ── Phase 2: Map Evaluation ──
        self.get_logger().info("Phase 2: Evaluating semantic map...")
        map_score = 0
        if self.semantic_map_data is None:
            self.get_logger().error("  No semantic map received!")
        else:
            for obj_name, gt_pos in self.ground_truth.items():
                if obj_name not in self.semantic_map_data:
                    self.get_logger().warn(f"  {obj_name}: NOT FOUND in map")
                    continue
                est = self.semantic_map_data[obj_name]
                if isinstance(est, dict):
                    est = est.get("position", est.get("pos", [0, 0, 0]))
                err = math.sqrt(sum((a - b)**2 for a, b in zip(est[:2], gt_pos[:2])))
                passed = err < 0.30
                map_score += int(passed)
                status = "PASS" if passed else "FAIL"
                self.get_logger().info(f"  {obj_name}: error={err:.3f} m [{status}]")

        map_pass = map_score >= 4
        self.get_logger().info(f"  Map score: {map_score}/5 objects within tolerance ({'PASS' if map_pass else 'FAIL'})")

        # ── Phase 3: Navigation Queries ──
        self.get_logger().info("Phase 3: Navigation queries...")
        nav_queries = self._generate_nav_queries()
        self._clear_exploration_markers()
        query_goals = [
            (obj, self.ground_truth[obj][0] + ox, self.ground_truth[obj][1] + oy)
            for (obj, ox, oy) in nav_queries
        ]
        nav_score = 0
        for qi, (obj_class, off_x, off_y) in enumerate(nav_queries):
            self._publish_query_markers(query_goals, qi)
            self.get_logger().info(f"  Query {qi+1}: Navigate to ({off_x}, {off_y}) relative to {obj_class}")

            query = {"object": obj_class, "offset": [off_x, off_y]}
            query_msg = String()
            query_msg.data = json.dumps(query)
            self.query_pub.publish(query_msg)

            # Compute expected goal from ground truth
            gt = self.ground_truth[obj_class]
            expected_x = gt[0] + off_x
            expected_y = gt[1] + off_y

            # Wait for navigation
            t0 = time.time()
            reached = False
            while time.time() - t0 < 120.0:
                rclpy.spin_once(self, timeout_sec=0.1)
                if self.robot_pose is not None:
                    dx = expected_x - self.robot_pose.x
                    dy = expected_y - self.robot_pose.y
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 0.15:  # generous tolerance (map error + nav error)
                        reached = True
                        break

            nav_score += int(reached)
            status = "PASS" if reached else "FAIL"
            if self.robot_pose is not None:
                dx = expected_x - self.robot_pose.x
                dy = expected_y - self.robot_pose.y
                final_dist = math.sqrt(dx*dx + dy*dy)
                self.get_logger().info(f"    Final distance to expected goal: {final_dist:.3f} m [{status}]")
            else:
                self.get_logger().info(f"    No pose available [{status}]")

        self._publish_query_markers(query_goals, len(query_goals))
        nav_pass = nav_score >= 2
        self.get_logger().info(f"  Navigation score: {nav_score}/3 queries reached ({'PASS' if nav_pass else 'FAIL'})")

        # ── Summary ──
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"MAP:        {map_score}/5  ({'PASS' if map_pass else 'FAIL'})")
        self.get_logger().info(f"NAVIGATION: {nav_score}/3  ({'PASS' if nav_pass else 'FAIL'})")
        overall = map_pass and nav_pass
        self.get_logger().info(f"OVERALL:    {'PASS' if overall else 'FAIL'}")
        self.get_logger().info("=" * 50)

        return overall


def main():
    rclpy.init()
    tester = Q1Tester()
    try:
        success = tester.run_tests()
    except KeyboardInterrupt:
        success = False
    finally:
        tester.destroy_node()
        rclpy.try_shutdown()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
