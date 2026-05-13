#!/usr/bin/env python3
"""Q3 planner-only evaluator for service-based RRT planning."""

from __future__ import annotations

import math
import time
import zlib
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose2D as Pose2DMsg
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path as PathMsg
from nav_msgs.srv import GetPlan
from numpy.typing import NDArray
from rclpy.node import Node
from rrt_planner import Pose2D

from slam_utils import (
    FAST_QoS,
    GridInfo,
    LATCHED_QoS,
    build_gt_map,
    inflate_costmap,
    quat_msg_to_yaw_rad,
    yaw_to_quaternion_msg,
)


def _wrap_angle_rad(theta_rad: float) -> float:
    """Wrap an angle to the range [-pi, pi]."""
    return (theta_rad + math.pi) % (2.0 * math.pi) - math.pi


class TestRRTPlanner(Node):
    """Evaluate RRT planner output with offline validity and kinematic checks."""

    def __init__(self) -> None:
        """Initialize map publishers, planner client, and evaluation settings."""
        super().__init__("test_rrt_planner")

        self.declare_parameter("origin_x_m", -2.0)
        self.declare_parameter("origin_y_m", -2.0)
        self.declare_parameter("resolution_m", 0.05)
        self.declare_parameter("height_cells", 80)
        self.declare_parameter("width_cells", 80)
        self.declare_parameter("occupied_threshold", 85)

        self.declare_parameter("inflation_radius_m", 0.078)
        self.declare_parameter("start_goal_tolerance_m", 0.3)
        self.declare_parameter("collision_check_step_m", 0.05)
        self.declare_parameter("min_goal_distance_m", 3.0)
        self.declare_parameter("max_goal_sampling_attempts", 500)
        self.declare_parameter("plan_timeout_s", 20.0)
        self.declare_parameter("min_waypoints", 2)
        self.declare_parameter("max_segment_length_m", 0.3)
        # self.declare_parameter("max_curvature_radpm", 3.0)
        self.declare_parameter("enforce_goal_heading", True)
        self.declare_parameter("goal_heading_tolerance_rad", 0.5)
        self.declare_parameter("path_hold_s", 1.0)
        self.declare_parameter("rng_tag", "q3-test-rrt")
        self.declare_parameter("scene_path", "")

        self.declare_parameter("trial_count", 10)
        self.declare_parameter("required_successes", 7)
        self._trial_count = int(self.get_parameter("trial_count").value)
        self._required_successes = int(self.get_parameter("required_successes").value)

        self._occupied_threshold = int(self.get_parameter("occupied_threshold").value)
        self._inflation_radius_m = float(self.get_parameter("inflation_radius_m").value)
        self._start_goal_tolerance_m = float(
            self.get_parameter("start_goal_tolerance_m").value
        )
        self._collision_check_step_m = float(
            self.get_parameter("collision_check_step_m").value
        )
        self._min_goal_distance_m = float(
            self.get_parameter("min_goal_distance_m").value
        )
        self._max_goal_sampling_attempts = int(
            self.get_parameter("max_goal_sampling_attempts").value
        )
        self._plan_timeout_s = float(self.get_parameter("plan_timeout_s").value)
        self._min_waypoints = int(self.get_parameter("min_waypoints").value)
        self._max_segment_length_m = float(
            self.get_parameter("max_segment_length_m").value
        )
        # self._max_curvature_radpm = float(
        #     self.get_parameter("max_curvature_radpm").value
        # )
        self._enforce_goal_heading = bool(
            self.get_parameter("enforce_goal_heading").value
        )
        self._goal_heading_tolerance_rad = float(
            self.get_parameter("goal_heading_tolerance_rad").value
        )
        self._path_hold_s: float = max(
            1.0, float(self.get_parameter("path_hold_s").value)
        )

        self._grid_info = GridInfo(
            origin_x=float(self.get_parameter("origin_x_m").value),
            origin_y=float(self.get_parameter("origin_y_m").value),
            resolution_m=float(self.get_parameter("resolution_m").value),
            height_cells=int(self.get_parameter("height_cells").value),
            width_cells=int(self.get_parameter("width_cells").value),
            parent_frame_id="map",
        )

        rng_tag = str(self.get_parameter("rng_tag").value)
        self._rng = np.random.default_rng(self._seed_from_tag(rng_tag))

        scene_path_param = str(self.get_parameter("scene_path").value)
        if scene_path_param:
            self._scene_path = Path(scene_path_param)
        else:
            q1_share_dir = Path(get_package_share_directory("occupancy_grid"))
            self._scene_path = q1_share_dir / "models" / "turtlebot_scene.xml"

        self._costmap_0_100: NDArray[np.int16] = self._build_costmap()
        self._latest_pose: Pose2DMsg | None = None

        self.create_subscription(Pose2DMsg, "/gt/odometry", self._odometry_cb, FAST_QoS)
        self._map_pub = self.create_publisher(
            OccupancyGrid, "/gt/occupancy_map", LATCHED_QoS
        )
        self._path_pub = self.create_publisher(PathMsg, "/planned_path", LATCHED_QoS)
        self.create_timer(1.0, self._publish_map_cb)

        self._planner_client = self.create_client(GetPlan, "/plan_path")

    def _build_costmap(self) -> NDArray[np.int16]:
        """Create an inflated occupancy costmap from the configured scene."""
        occupied_bool = build_gt_map(
            grid_info=self._grid_info, scene_xml=self._scene_path
        )
        occupied_0_100 = occupied_bool.astype(np.int16) * 100
        return inflate_costmap(
            occupied_0_100,
            self._grid_info,
            self._occupied_threshold,
            self._inflation_radius_m,
        )

    def _odometry_cb(self, msg: Pose2DMsg) -> None:
        """Store the latest ground-truth odometry reading."""
        self._latest_pose = msg

    def _publish_map_cb(self) -> None:
        """Publish the planner test occupancy map."""
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = self._grid_info.parent_frame_id

        map_msg.info.resolution = float(self._grid_info.resolution_m)
        map_msg.info.width = int(self._grid_info.width_cells)
        map_msg.info.height = int(self._grid_info.height_cells)
        map_msg.info.origin.position.x = float(self._grid_info.origin_x)
        map_msg.info.origin.position.y = float(self._grid_info.origin_y)
        map_msg.info.origin.orientation.w = 1.0

        map_msg.data = self._costmap_0_100.astype(np.int8).flatten(order="C").tolist()
        self._map_pub.publish(map_msg)

    def _wait_for_pose(self, timeout_s: float) -> bool:
        """Wait until at least one odometry message has been received."""
        start_t_s = time.time()
        while rclpy.ok() and (time.time() - start_t_s) < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._latest_pose is not None:
                return True
        return False

    def _sample_goal(self, start_pose: Pose2D) -> Pose2D:
        """Sample a free-space goal pose with sufficient displacement."""
        free_cells = np.argwhere(self._costmap_0_100 < self._occupied_threshold)
        if free_cells.size == 0:
            return start_pose

        for _ in range(self._max_goal_sampling_attempts):
            idx = int(self._rng.integers(0, free_cells.shape[0]))
            row = int(free_cells[idx, 0])
            col = int(free_cells[idx, 1])
            goal_x_m = self._grid_info.col_to_x(col)
            goal_y_m = self._grid_info.row_to_y(row)

            if (
                math.hypot(goal_x_m - start_pose.x, goal_y_m - start_pose.y)
                < self._min_goal_distance_m
            ):
                continue

            goal_heading_rad = math.atan2(
                goal_y_m - start_pose.y, goal_x_m - start_pose.x
            )
            return Pose2D(x=goal_x_m, y=goal_y_m, theta_rad=goal_heading_rad)

        return start_pose

    def _is_blocked(self, x: float, y: float) -> bool:
        """Check whether a world-frame point is blocked or out-of-bounds."""
        cell = self._grid_info.coord_to_cell((x, y))
        if not self._grid_info.is_valid_cell(cell):
            return True
        return int(self._costmap_0_100[cell.row, cell.col]) >= self._occupied_threshold

    def _segment_collision_free(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> bool:
        """Check whether a straight segment is collision-free using sampled points."""
        length_m = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(math.ceil(length_m / self._collision_check_step_m)))
        for step_idx in range(steps + 1):
            alpha = step_idx / steps
            x = x0 + alpha * (x1 - x0)
            y = y0 + alpha * (y1 - y0)
            if self._is_blocked(x, y):
                return False

        return True

    def _offline_validate_path(
        self,
        path_msg: PathMsg,
        start_pose: Pose2D,
        goal_pose: Pose2D,
    ) -> tuple[bool, str]:
        """Run geometric and kinematic checks on a returned path."""
        waypoint_count = len(path_msg.poses)
        if waypoint_count < self._min_waypoints:
            return False, "too_few_waypoints"

        start_pt = path_msg.poses[0].pose.position
        goal_pt = path_msg.poses[-1].pose.position
        if (
            math.hypot(start_pt.x - start_pose.x, start_pt.y - start_pose.y)
            > self._start_goal_tolerance_m
        ):
            return False, "start_mismatch"
        if (
            math.hypot(goal_pt.x - goal_pose.x, goal_pt.y - goal_pose.y)
            > self._start_goal_tolerance_m
        ):
            return False, "goal_mismatch"

        segment_lengths_m: list[float] = []
        segment_headings_rad: list[float] = []
        for idx in range(waypoint_count - 1):
            pose_a = path_msg.poses[idx].pose.position
            pose_b = path_msg.poses[idx + 1].pose.position
            dx_m = float(pose_b.x - pose_a.x)
            dy_m = float(pose_b.y - pose_a.y)
            seg_len_m = math.hypot(dx_m, dy_m)

            if not self._segment_collision_free(pose_a.x, pose_a.y, pose_b.x, pose_b.y):
                return False, "collision_segment"

            if seg_len_m > self._max_segment_length_m:
                return False, "segment_too_long"

            if seg_len_m > 1e-6:
                segment_lengths_m.append(seg_len_m)
                segment_headings_rad.append(math.atan2(dy_m, dx_m))

        # for idx in range(1, len(segment_headings_rad)):
        #     prev_heading_rad = segment_headings_rad[idx - 1]
        #     next_heading_rad = segment_headings_rad[idx]
        #     heading_delta_rad = _wrap_angle_rad(next_heading_rad - prev_heading_rad)
        #     mean_step_m = 0.5 * (segment_lengths_m[idx - 1] + segment_lengths_m[idx])
        #     curvature_radpm = abs(heading_delta_rad) / max(mean_step_m, 1e-6)
        #     if curvature_radpm > self._max_curvature_radpm:
        #         return False, "curvature_violation"

        if self._enforce_goal_heading:
            final_yaw_rad = quat_msg_to_yaw_rad(path_msg.poses[-1].pose.orientation)
            goal_heading_error_rad = abs(
                _wrap_angle_rad(final_yaw_rad - goal_pose.theta_rad)
            )
            if goal_heading_error_rad > self._goal_heading_tolerance_rad:
                return False, "goal_heading_mismatch"

        return True, "ok"

    def _call_plan_service(
        self, start_pose: Pose2D, goal_pose: Pose2D
    ) -> tuple[PathMsg | None, float]:
        """Call the `/plan_path` service and return `(path, latency_s)`."""
        if not self._planner_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("Planner service /plan_path is unavailable.")
            return None, 0.0

        req = GetPlan.Request()
        req.start = PoseStamped()
        req.goal = PoseStamped()
        req.start.header.frame_id = self._grid_info.parent_frame_id
        req.goal.header.frame_id = self._grid_info.parent_frame_id
        req.start.pose.position.x = float(start_pose.x)
        req.start.pose.position.y = float(start_pose.y)
        req.goal.pose.position.x = float(goal_pose.x)
        req.goal.pose.position.y = float(goal_pose.y)

        req.start.pose.orientation = yaw_to_quaternion_msg(start_pose.theta_rad)
        req.goal.pose.orientation = yaw_to_quaternion_msg(goal_pose.theta_rad)
        req.tolerance = float(self._start_goal_tolerance_m)

        call_start_t_s = time.time()
        future = self._planner_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._plan_timeout_s)
        latency_s = time.time() - call_start_t_s

        if not future.done():
            return None, latency_s
        if future.result() is None:
            return None, latency_s
        return future.result().plan, latency_s

    @staticmethod
    def _path_length_m(path_msg: PathMsg) -> float:
        """Compute total polyline path length in meters."""
        path_len_m = 0.0
        for idx in range(len(path_msg.poses) - 1):
            p0 = path_msg.poses[idx].pose.position
            p1 = path_msg.poses[idx + 1].pose.position
            path_len_m += math.hypot(p1.x - p0.x, p1.y - p0.y)
        return path_len_m

    def _sleep_with_spin(self, duration_s: float) -> None:
        """Sleep while spinning so periodic publishers keep running."""
        end_t_s = time.time() + duration_s
        while rclpy.ok() and time.time() < end_t_s:
            rclpy.spin_once(self, timeout_sec=0.05)

    @staticmethod
    def _seed_from_tag(seed_tag: str) -> int:
        """Convert a seed tag string into a deterministic uint32 seed."""
        return zlib.crc32(seed_tag.encode("utf-8")) & 0xFFFFFFFF

    def run_test(self) -> bool:
        """Run planner-only Q3 evaluation trials."""
        self._publish_map_cb()
        self._sleep_with_spin(1.0)

        if not self._wait_for_pose(timeout_s=15.0):
            self.get_logger().error("No /gt/odometry pose received.")
            return False

        success_count = 0
        valid_path_count = 0
        timings_s: list[float] = []
        trial_fail_reasons: dict[str, int] = {}

        for trial_idx in range(self._trial_count):
            if self._latest_pose is None:
                trial_fail_reasons["missing_start_pose"] = (
                    trial_fail_reasons.get("missing_start_pose", 0) + 1
                )
                continue

            start_pose = Pose2D(
                x=float(self._latest_pose.x),
                y=float(self._latest_pose.y),
                theta_rad=float(self._latest_pose.theta),
            )
            goal_pose = self._sample_goal(start_pose=start_pose)

            print(
                f"Trial {trial_idx + 1}/{self._trial_count}: "
                f"start=({start_pose.x:.2f},{start_pose.y:.2f},{start_pose.theta_rad:.2f}) "
                f"goal=({goal_pose.x:.2f},{goal_pose.y:.2f},{goal_pose.theta_rad:.2f})"
            )

            path_msg, latency_s = self._call_plan_service(
                start_pose=start_pose, goal_pose=goal_pose
            )
            timings_s.append(latency_s)
            print(f"  planner update: latency={latency_s:.3f} s")

            if path_msg is None:
                print("  planner update: no path returned")
                trial_fail_reasons["no_path"] = trial_fail_reasons.get("no_path", 0) + 1
                continue

            self._path_pub.publish(path_msg)
            self._sleep_with_spin(self._path_hold_s)

            path_len_m = self._path_length_m(path_msg)
            print(
                f"  planner update: {len(path_msg.poses)} waypoints, path_length={path_len_m:.3f} m"
            )

            trial_ok, reason = self._offline_validate_path(
                path_msg=path_msg,
                start_pose=start_pose,
                goal_pose=goal_pose,
            )
            if trial_ok:
                valid_path_count += 1
                success_count += 1
            else:
                trial_fail_reasons[reason] = trial_fail_reasons.get(reason, 0) + 1
                print(f"  planner update: failed offline checks ({reason})")

            print(
                f"Trial {trial_idx + 1}/{self._trial_count}: {'PASS' if trial_ok else 'FAIL'}"
            )

        passed = success_count >= self._required_successes
        mean_latency_s = float(np.mean(timings_s)) if timings_s else float("nan")
        max_latency_s = float(np.max(timings_s)) if timings_s else float("nan")

        print("\n" + "=" * 60)
        print("Q3: RRT Planner (Planner-Only) Test")
        print("=" * 60)
        print(f"Valid paths:        {valid_path_count}/{self._trial_count}")
        print(f"Successes:          {success_count}/{self._trial_count}")
        print(f"Required success:   {self._required_successes}")
        print(f"Mean plan time (s): {mean_latency_s:.3f}")
        print(f"Max plan time (s):  {max_latency_s:.3f}")

        if trial_fail_reasons:
            print(f"Failure reasons:    {trial_fail_reasons}")

        print("PASS" if passed else "FAIL")
        print("=" * 60 + "\n")
        return passed


def main() -> None:
    """Initialize ROS and run the Q3 planner test node."""
    rclpy.init()
    node = TestRRTPlanner()
    try:
        success = node.run_test()
        raise SystemExit(0 if success else 1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
