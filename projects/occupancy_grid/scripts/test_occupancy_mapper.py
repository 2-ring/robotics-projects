#!/usr/bin/env python3
"""Q1 autograder for occupancy grid mapping."""

import math
import time
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose2D, PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node

from slam_utils import (
    FAST_QoS,
    LATCHED_QoS,
    build_gt_map,
    occupancy_f1_score,
    unpack_occupancy_grid_msg,
)


class TestOccupancyMapper(Node):
    """Drive the robot, allow the mapper to build and map, and score its accuracy."""

    def __init__(self) -> None:
        """Initialize the Q1 autograder node."""
        super().__init__("test_occupancy_mapper")

        self.declare_parameter("test_duration_s", 30.0)
        self._test_duration_s = float(self.get_parameter("test_duration_s").value)

        self.declare_parameter("occupied_threshold", 70)
        self._occupied_threshold = int(self.get_parameter("occupied_threshold").value)

        self.declare_parameter("update_hz", 20.0)
        self._update_hz = float(self.get_parameter("update_hz").value)

        self._MIN_F1 = 0.85

        self._latest_grid: OccupancyGrid | None = None
        self._latest_odom: Pose2D | None = None

        self.create_subscription(OccupancyGrid, "/occupancy_map", self._map_cb, 10)
        self.create_subscription(Pose2D, "/gt/odometry", self._odometry_cb, FAST_QoS)

        self._path_pub = self.create_publisher(PathMsg, "/planned_path", LATCHED_QoS)
        self._gt_map_pub = self.create_publisher(
            OccupancyGrid, "/gt/occupancy_map", LATCHED_QoS
        )

        q1_share = Path(get_package_share_directory("occupancy_grid"))
        self._mujoco_scene_xml = q1_share / "models" / "turtlebot_scene.xml"

    def _map_cb(self, msg: OccupancyGrid) -> None:
        """Store the student's latest occupancy grid map."""
        self._latest_grid = msg

    def _odometry_cb(self, msg: Pose2D) -> None:
        """Store the latest robot odometry reading."""
        self._latest_odom = msg

    def _publish_path(self, waypoints_xy: list[tuple[float, float]]) -> None:
        """Publish the given target path for external visualization."""
        path_msg = PathMsg()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        for x, y in waypoints_xy:
            pose_msg = PoseStamped()
            pose_msg.header = path_msg.header
            pose_msg.pose.position.x = x
            pose_msg.pose.position.y = y
            pose_msg.pose.orientation.w = 1.0
            path_msg.poses.append(pose_msg)
        self._path_pub.publish(path_msg)

    def _publish_gt_map(
        self, gt_occ_mask: np.ndarray, source_grid_msg: OccupancyGrid
    ) -> None:
        """Publish a ground-truth occupancy grid message.

        :param gt_occ_mask: Ground-truth occupancy mask, where True means occupied
        :param source_grid_msg: Map message providing grid metadata and frame
        """
        gt_msg = OccupancyGrid()
        gt_msg.header.stamp = self.get_clock().now().to_msg()
        gt_msg.header.frame_id = source_grid_msg.header.frame_id
        gt_msg.info = source_grid_msg.info
        gt_msg.data = np.where(gt_occ_mask, 100, 0).astype(np.int8).flatten().tolist()
        self._gt_map_pub.publish(gt_msg)

    def run_test(self) -> bool:
        """Execute the test for Question 1 of the assignment.

        :return: True if test is passed, False if test is failed
        """
        self.get_logger().info("Waiting for /occupancy_map publication...")

        # Wait up to 15 seconds to receive an initial map and robot odometry
        wait_start_t = time.time()
        while rclpy.ok() and (time.time() - wait_start_t) < 15:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._latest_grid is not None and self._latest_odom is not None:
                break

        if self._latest_grid is None:
            self.get_logger().error("No occupancy map received.")
            return False
        if self._latest_odom is None:
            self.get_logger().error("No odometry received.")
            return False

        start_t = time.time()
        next_update_t = start_t + 5.0
        loop_sleep_s = 1.0 / self._update_hz

        start_x = float(self._latest_odom.x)
        start_y = float(self._latest_odom.y)
        side_m = 2.0
        square_waypoints_xy: list[tuple[float, float]] = [
            (start_x, start_y),
            (start_x + side_m, start_y),
            (start_x + side_m, start_y + side_m),
            (start_x, start_y + side_m),
        ]
        path_offset_idx = 0  # Index of the starting square waypoint
        pure_pursuit_path_xy = square_waypoints_xy.copy()
        self._publish_path(pure_pursuit_path_xy)
        goal_x = pure_pursuit_path_xy[-1][0]  # Final waypoint in the path
        goal_y = pure_pursuit_path_xy[-1][1]

        increment_within_m = 0.3

        while rclpy.ok() and (time.time() - start_t) < self._test_duration_s:
            now_t = time.time()
            if self._latest_odom is not None:
                robot_x = float(self._latest_odom.x)
                robot_y = float(self._latest_odom.y)

                if math.hypot(goal_x - robot_x, goal_y - robot_y) <= increment_within_m:
                    path_offset_idx = (path_offset_idx + 1) % len(square_waypoints_xy)
                    pure_pursuit_path_xy = (
                        square_waypoints_xy[path_offset_idx:]
                        + square_waypoints_xy[:path_offset_idx]
                    )
                    self._publish_path(pure_pursuit_path_xy)
                    goal_x = pure_pursuit_path_xy[-1][0]
                    goal_y = pure_pursuit_path_xy[-1][1]

            rclpy.spin_once(self, timeout_sec=0.0)

            # Update logged metrics for the student's occupancy grid
            if now_t >= next_update_t and self._latest_grid is not None:
                grid_data, grid_info = unpack_occupancy_grid_msg(self._latest_grid)
                gt_map = build_gt_map(grid_info, scene_xml=self._mujoco_scene_xml)
                self._publish_gt_map(gt_map, self._latest_grid)

                metrics = occupancy_f1_score(
                    predicted_grid=grid_data,
                    gt_occ_mask=gt_map,
                    occupied_threshold=self._occupied_threshold,
                )

                elapsed_s = now_t - start_t
                self.get_logger().info(
                    f"[{elapsed_s:5.1f} s] "
                    f"precision={metrics.precision:.3f} "
                    f"recall={metrics.recall:.3f} "
                    f"f1={metrics.f1:.3f}"
                )
                next_update_t = now_t + 5.0

            time.sleep(loop_sleep_s)

        if self._latest_grid is None:
            self.get_logger().error("No occupancy grid was available at scoring time.")
            return False

        grid_data, grid_info = unpack_occupancy_grid_msg(self._latest_grid)
        gt_map = build_gt_map(grid_info, scene_xml=self._mujoco_scene_xml)
        self._publish_gt_map(gt_map, self._latest_grid)

        metrics = occupancy_f1_score(
            predicted_grid=grid_data,
            gt_occ_mask=gt_map,
            occupied_threshold=self._occupied_threshold,
        )

        passed = metrics.f1 >= self._MIN_F1

        print("\n" + "=" * 60)
        print("Q1: Occupancy Grid Mapping Test")
        print("=" * 60)
        print(f"Precision:   {metrics.precision:.3f}")
        print(f"Recall:      {metrics.recall:.3f}")
        print(f"F1 score:    {metrics.f1:.3f}")
        print(f"Required F1: {self._MIN_F1:.3f}")
        print("PASS" if passed else "FAIL")
        print("=" * 60 + "\n")

        return passed


def main() -> None:
    """Initialize ROS and run the Q1 autograder."""
    rclpy.init()
    node = TestOccupancyMapper()
    try:
        success: bool = node.run_test()
        raise SystemExit(0 if success else 1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
