#!/usr/bin/env python3
"""Q2 autograder for discrete Bayes localization."""

from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from slam_utils.ros2_utils import waypoints_xy_to_path_msg
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Path as PathMsg
from numpy.typing import NDArray
from rclpy.node import Node

from slam_utils import FAST_QoS, LATCHED_QoS


class TestBayesLocalizer(Node):
    """Drive the robot and score estimated localization with position and yaw RMSE."""

    def __init__(self) -> None:
        """Initialize the Q2 autograder node."""
        super().__init__("test_bayes_localizer")

        self.declare_parameter("test_duration_s", 35.0)
        self.declare_parameter("update_hz", 20.0)
        self.declare_parameter("pos_rmse_max_m", 0.2)
        self.declare_parameter("yaw_rmse_max_rad", 0.2)
        self.declare_parameter("warmup_s", 10.0)
        self.declare_parameter("waypoint_side_m", 2.0)
        self.declare_parameter("waypoint_reached_within_m", 0.3)
        self.declare_parameter("progress_interval_s", 5.0)
        self.declare_parameter("min_samples", 20)

        self._test_duration_s = float(self.get_parameter("test_duration_s").value)
        self._update_hz = float(self.get_parameter("update_hz").value)
        self._pos_rmse_max_m = float(self.get_parameter("pos_rmse_max_m").value)
        self._yaw_rmse_max_rad = float(self.get_parameter("yaw_rmse_max_rad").value)
        self._warmup_s = float(self.get_parameter("warmup_s").value)
        self._waypoint_side_m = float(self.get_parameter("waypoint_side_m").value)
        self._waypoint_reached_within_m = float(
            self.get_parameter("waypoint_reached_within_m").value
        )
        self._progress_interval_s = float(
            self.get_parameter("progress_interval_s").value
        )
        self._min_samples = int(self.get_parameter("min_samples").value)

        self._latest_est_pose: Pose2D | None = None
        self._latest_gt_pose: Pose2D | None = None

        self._est_xy: list[tuple[float, float]] = []
        self._gt_xy: list[tuple[float, float]] = []
        self._est_yaw_rad: list[float] = []
        self._gt_yaw_rad: list[float] = []

        self.create_subscription(
            Pose2D, "/estimated_odometry", self._est_odometry_cb, FAST_QoS
        )
        self.create_subscription(Pose2D, "/gt/odometry", self._gt_odometry_cb, FAST_QoS)
        self._path_pub = self.create_publisher(PathMsg, "/planned_path", LATCHED_QoS)

    def _est_odometry_cb(self, msg: Pose2D) -> None:
        """Store the latest estimated pose."""
        self._latest_est_pose = msg

    def _gt_odometry_cb(self, msg: Pose2D) -> None:
        """Store the latest ground-truth pose."""
        self._latest_gt_pose = msg

    def _publish_path(self, waypoints_xy: list[tuple[float, float]]) -> None:
        """Publish the path of waypoints followed by the pure-pursuit controller.

        :param waypoints_xy: Ordered list of (x, y) waypoints in meters
        """
        stamp = self.get_clock().now().to_msg()
        path_msg = waypoints_xy_to_path_msg(waypoints_xy, stamp)
        self._path_pub.publish(path_msg)

    @staticmethod
    def _position_rmse_m(
        est_xy: NDArray[np.float32], gt_xy: NDArray[np.float32]
    ) -> float:
        """Compute trajectory position RMSE over (x, y) coordinates.

        :param est_xy: Estimated trajectory array with shape (N, 2)
        :param gt_xy: Ground-truth trajectory array with shape (N, 2)
        :return: Position RMSE in meters
        """
        if est_xy.shape != gt_xy.shape:
            raise ValueError("Trajectory arrays must share the same shape.")

        squared_error = np.sum((est_xy - gt_xy) ** 2, axis=1)
        return float(np.sqrt(np.mean(squared_error)))

    @staticmethod
    def _yaw_rmse_rad(
        est_yaw_rad: NDArray[np.float32], gt_yaw_rad: NDArray[np.float32]
    ) -> float:
        """Compute wrapped yaw RMSE in radians."""
        if est_yaw_rad.shape != gt_yaw_rad.shape:
            raise ValueError("Yaw arrays must share the same shape.")

        wrapped_error_rad = np.arctan2(
            np.sin(est_yaw_rad - gt_yaw_rad), np.cos(est_yaw_rad - gt_yaw_rad)
        )
        return float(np.sqrt(np.mean(wrapped_error_rad**2)))

    def run_test(self) -> bool:
        """Run the Q2 localization test and return pass/fail.

        :return: True if test passed, False otherwise
        """
        self.get_logger().info("Waiting for estimated and GT odometry...")
        wait_start_t = time.time()
        while rclpy.ok() and (time.time() - wait_start_t) < 15.0:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._latest_est_pose is not None and self._latest_gt_pose is not None:
                break

        if self._latest_est_pose is None or self._latest_gt_pose is None:
            self.get_logger().error("Missing odometry data for Q2 test.")
            return False

        start_t = time.time()
        next_progress_t = start_t + self._progress_interval_s
        loop_sleep_s = 1.0 / self._update_hz

        start_x_m = float(self._latest_gt_pose.x)
        start_y_m = float(self._latest_gt_pose.y)
        square_waypoints_xy = [
            (start_x_m, start_y_m),
            (start_x_m + self._waypoint_side_m, start_y_m),
            (start_x_m + self._waypoint_side_m, start_y_m + self._waypoint_side_m),
            (start_x_m, start_y_m + self._waypoint_side_m),
        ]
        path_offset_idx = 0
        pure_pursuit_path_xy = square_waypoints_xy.copy()
        self._publish_path(pure_pursuit_path_xy)
        goal_x_m = pure_pursuit_path_xy[-1][0]
        goal_y_m = pure_pursuit_path_xy[-1][1]

        while rclpy.ok() and (time.time() - start_t) < self._test_duration_s:
            now_t = time.time()

            if self._latest_gt_pose is not None:
                robot_x_m = float(self._latest_gt_pose.x)
                robot_y_m = float(self._latest_gt_pose.y)
                if (
                    math.hypot(goal_x_m - robot_x_m, goal_y_m - robot_y_m)
                    <= self._waypoint_reached_within_m
                ):
                    path_offset_idx = (path_offset_idx + 1) % len(square_waypoints_xy)
                    pure_pursuit_path_xy = (
                        square_waypoints_xy[path_offset_idx:]
                        + square_waypoints_xy[:path_offset_idx]
                    )
                    self._publish_path(pure_pursuit_path_xy)
                    goal_x_m = pure_pursuit_path_xy[-1][0]
                    goal_y_m = pure_pursuit_path_xy[-1][1]

            rclpy.spin_once(self, timeout_sec=0.0)

            elapsed_s = now_t - start_t
            if (
                elapsed_s >= self._warmup_s
                and self._latest_est_pose is not None
                and self._latest_gt_pose is not None
            ):
                self._est_xy.append(
                    (float(self._latest_est_pose.x), float(self._latest_est_pose.y))
                )
                self._gt_xy.append(
                    (float(self._latest_gt_pose.x), float(self._latest_gt_pose.y))
                )
                self._est_yaw_rad.append(float(self._latest_est_pose.theta))
                self._gt_yaw_rad.append(float(self._latest_gt_pose.theta))

                if now_t >= next_progress_t and len(self._est_xy) >= 5:
                    est_arr = np.asarray(self._est_xy, dtype=np.float32)
                    gt_arr = np.asarray(self._gt_xy, dtype=np.float32)
                    pos_rmse_m = self._position_rmse_m(est_xy=est_arr, gt_xy=gt_arr)

                    est_yaw_arr = np.asarray(self._est_yaw_rad, dtype=np.float32)
                    gt_yaw_arr = np.asarray(self._gt_yaw_rad, dtype=np.float32)
                    yaw_rmse_rad = self._yaw_rmse_rad(
                        est_yaw_rad=est_yaw_arr, gt_yaw_rad=gt_yaw_arr
                    )

                    self.get_logger().info(
                        f"[{elapsed_s:5.1f} s] "
                        f"samples={len(self._est_xy):4d} "
                        f"pos_rmse={pos_rmse_m:.3f} m "
                        f"yaw_rmse={yaw_rmse_rad:.3f} rad"
                    )
                    next_progress_t = now_t + self._progress_interval_s

            time.sleep(loop_sleep_s)

        if len(self._est_xy) < self._min_samples:
            self.get_logger().error("Insufficient trajectory samples for Q2 scoring.")
            return False

        est_arr = np.asarray(self._est_xy, dtype=np.float32)
        gt_arr = np.asarray(self._gt_xy, dtype=np.float32)
        pos_rmse_m = self._position_rmse_m(est_xy=est_arr, gt_xy=gt_arr)

        est_yaw_arr_rad = np.asarray(self._est_yaw_rad, dtype=np.float32)
        gt_yaw_arr_rad = np.asarray(self._gt_yaw_rad, dtype=np.float32)
        yaw_rmse_rad = self._yaw_rmse_rad(
            est_yaw_rad=est_yaw_arr_rad, gt_yaw_rad=gt_yaw_arr_rad
        )

        passed = (pos_rmse_m <= self._pos_rmse_max_m) and (
            yaw_rmse_rad <= self._yaw_rmse_max_rad
        )

        print("\n" + "=" * 60)
        print("Q2: Markov Grid Localization Test")
        print("=" * 60)
        print(f"Warmup window (s):   {self._warmup_s:.1f}")
        print(f"Samples:             {len(self._est_xy)}")
        print(f"Position RMSE (m):   {pos_rmse_m:.3f}")
        print(f"Required RMSE (m):   <= {self._pos_rmse_max_m:.3f}")
        print(f"Yaw RMSE (rad):      {yaw_rmse_rad:.3f}")
        print(f"Required RMSE (rad): <= {self._yaw_rmse_max_rad:.3f}")
        print("PASS" if passed else "FAIL")
        print("=" * 60 + "\n")

        return passed


def main() -> None:
    """Initialize ROS and run the Q2 autograder."""
    rclpy.init()
    node = TestBayesLocalizer()
    try:
        success = node.run_test()
        raise SystemExit(0 if success else 1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
