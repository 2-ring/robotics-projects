#!/usr/bin/env python3
"""Pure pursuit path follower for planned paths."""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Pose2D, PoseStamped, Twist
from nav_msgs.msg import Path
from rclpy.node import Node

from slam_utils import FAST_QoS, LATCHED_QoS, yaw_to_quaternion_msg


def _wrap_angle_rad(theta_rad: float) -> float:
    """Wrap an angle within `[-pi, pi]`."""
    return (theta_rad + math.pi) % (2.0 * math.pi) - math.pi


class PurePursuitNode(Node):
    """Follow a path of planned waypoints using a pure-pursuit controller."""

    def __init__(self) -> None:
        """Initialize subscriptions, publishers, and controller params."""
        super().__init__("pure_pursuit_node")

        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("lookahead_m", 0.15)
        self.declare_parameter("goal_tolerance_m", 0.05)
        self.declare_parameter("max_vx_mps", 0.7)
        self.declare_parameter("max_wz_radps", 6.0)

        self._lookahead_m = float(self.get_parameter("lookahead_m").value)
        self._goal_tolerance_m = float(self.get_parameter("goal_tolerance_m").value)
        self._max_vx_mps = float(self.get_parameter("max_vx_mps").value)
        self._max_wz_radps = float(self.get_parameter("max_wz_radps").value)

        self._path_xy: list[tuple[float, float]] = []
        self._target_idx = 0
        self._latest_odometry: Pose2D | None = None

        self.create_subscription(
            Path, "/planned_path", self._planned_path_cb, LATCHED_QoS
        )
        self.create_subscription(Pose2D, "/gt/odometry", self._odometry_cb, FAST_QoS)

        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._target_waypoint_pub = self.create_publisher(
            PoseStamped, "/pure_pursuit/target_waypoint", 10
        )

        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.create_timer(1.0 / control_rate_hz, self._control_cb)

    def _planned_path_cb(self, msg: Path) -> None:
        """Store a new waypoint path for execution."""
        new_path_xy: list[tuple[float, float]] = []
        for pose_stamped in msg.poses:
            new_path_xy.append(
                (
                    float(pose_stamped.pose.position.x),
                    float(pose_stamped.pose.position.y),
                )
            )
        self._path_xy = new_path_xy
        self._target_idx = 0

    def _odometry_cb(self, msg: Pose2D) -> None:
        """Store an updated odometry pose."""
        self._latest_odometry = msg

    def _publish_stop(self) -> None:
        """Publish a zero-velocity command."""
        stop_msg = Twist()
        self._cmd_pub.publish(stop_msg)

    def _publish_target_waypoint(
        self,
        target_x_m: float,
        target_y_m: float,
        robot_x_m: float,
        robot_y_m: float,
    ) -> None:
        """Publish the currently selected pure-pursuit target waypoint."""
        yaw_rad = math.atan2(target_y_m - robot_y_m, target_x_m - robot_x_m)
        target_msg = PoseStamped()
        target_msg.header.stamp = self.get_clock().now().to_msg()
        target_msg.header.frame_id = "map"
        target_msg.pose.position.x = float(target_x_m)
        target_msg.pose.position.y = float(target_y_m)
        target_msg.pose.position.z = 0.0
        target_msg.pose.orientation = yaw_to_quaternion_msg(yaw_rad)
        self._target_waypoint_pub.publish(target_msg)

    def _control_cb(self) -> None:
        """Run pure-pursuit control loop."""
        if self._latest_odometry is None:
            self.get_logger().warning(
                "Pure pursuit has not received odometry...", throttle_duration_sec=5.0
            )
            self._publish_stop()
            return

        if not self._path_xy:
            self.get_logger().warn(
                "Pure pursuit has not received a path to follow...",
                throttle_duration_sec=5.0,
            )
            self._publish_stop()
            return

        robot_x_m = float(self._latest_odometry.x)
        robot_y_m = float(self._latest_odometry.y)
        robot_yaw_rad = float(self._latest_odometry.theta)

        goal_x_m = self._path_xy[-1][0]
        goal_y_m = self._path_xy[-1][1]
        goal_dist_m = math.hypot(goal_x_m - robot_x_m, goal_y_m - robot_y_m)
        if goal_dist_m <= self._goal_tolerance_m:
            self._publish_target_waypoint(goal_x_m, goal_y_m, robot_x_m, robot_y_m)
            self._publish_stop()
            return

        # Coulter-style pure pursuit: intersect the lookahead circle with forward path segments
        target_x_m, target_y_m = self._path_xy[-1]
        found_intersection = False
        for seg_idx in range(self._target_idx, len(self._path_xy) - 1):
            ax_m, ay_m = self._path_xy[seg_idx]
            bx_m, by_m = self._path_xy[seg_idx + 1]
            dx_m = bx_m - ax_m
            dy_m = by_m - ay_m

            a = dx_m * dx_m + dy_m * dy_m
            if a <= 1e-9:
                continue

            fx_m = ax_m - robot_x_m
            fy_m = ay_m - robot_y_m

            b = 2.0 * (fx_m * dx_m + fy_m * dy_m)
            c = fx_m * fx_m + fy_m * fy_m - self._lookahead_m * self._lookahead_m

            disc = b * b - 4.0 * a * c
            if disc < 0.0:
                continue
            sqrt_disc = math.sqrt(disc)

            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)
            valid_t = [t for t in (t1, t2) if 0.0 <= t <= 1.0]
            if valid_t:
                alpha = max(valid_t)
                target_x_m = ax_m + alpha * dx_m
                target_y_m = ay_m + alpha * dy_m
                self._target_idx = seg_idx
                found_intersection = True

        if not found_intersection:  # Closest-point fallback
            best_dist_m = float("inf")
            for seg_idx in range(self._target_idx, len(self._path_xy) - 1):
                ax_m, ay_m = self._path_xy[seg_idx]
                bx_m, by_m = self._path_xy[seg_idx + 1]
                dx_m = bx_m - ax_m
                dy_m = by_m - ay_m
                seg_len_sq_m2 = dx_m * dx_m + dy_m * dy_m
                if seg_len_sq_m2 < 1e-9:
                    continue

                t = (
                    (robot_x_m - ax_m) * dx_m + (robot_y_m - ay_m) * dy_m
                ) / seg_len_sq_m2
                t = max(0.0, min(1.0, t))

                cx_m = ax_m + t * dx_m
                cy_m = ay_m + t * dy_m
                dist_m = math.hypot(cx_m - robot_x_m, cy_m - robot_y_m)
                if dist_m < best_dist_m:
                    best_dist_m = dist_m
                    target_x_m, target_y_m = cx_m, cy_m
                    self._target_idx = seg_idx

        self._publish_target_waypoint(target_x_m, target_y_m, robot_x_m, robot_y_m)

        target_heading_rad = math.atan2(target_y_m - robot_y_m, target_x_m - robot_x_m)
        heading_error_rad = _wrap_angle_rad(target_heading_rad - robot_yaw_rad)

        curvature_radpm = (
            2.0 * math.sin(heading_error_rad) / max(self._lookahead_m, 1e-3)
        )
        vx_mps = self._max_vx_mps
        wz_radps = vx_mps * curvature_radpm
        wz_radps = max(-self._max_wz_radps, min(self._max_wz_radps, wz_radps))

        cmd_msg = Twist()
        cmd_msg.linear.x = vx_mps
        cmd_msg.angular.z = wz_radps
        self._cmd_pub.publish(cmd_msg)


def main() -> None:
    """Initialize ROS and spin the pure-pursuit node."""
    rclpy.init()
    node = PurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
