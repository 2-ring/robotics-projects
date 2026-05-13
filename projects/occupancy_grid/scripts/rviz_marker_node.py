#!/usr/bin/env python3
"""RViz bridge to used visualize obstacles, robot odometry, etc."""

from pathlib import Path

import rclpy
from slam_utils.colors import (
    OBSTACLE_RGBA_255,
    RGBA01,
    ROBOT_EST_RGBA_255,
    ROBOT_GT_RGBA_255,
    rgba_255_to_unit,
)
from slam_utils.ros2_utils import (
    FAST_QoS,
    LATCHED_QoS,
    rgba_01_to_msg,
    yaw_to_quaternion_msg,
)
from slam_utils.world_generation import Environment2D, parse_scene_environment
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose2D
from rclpy.node import Node
from rclpy.time import Time
from visualization_msgs.msg import Marker, MarkerArray

ROBOT_RADIUS_M = 0.078
ROBOT_HEIGHT_M = 0.08

ROBOT_GT_RGBA_01 = rgba_255_to_unit(ROBOT_GT_RGBA_255)
ROBOT_EST_RGBA_01 = rgba_255_to_unit(ROBOT_EST_RGBA_255)

OBSTACLE_HEIGHT_M = 0.2
OBSTACLE_RGBA_01 = rgba_255_to_unit(OBSTACLE_RGBA_255)


class RVizMarkerNode(Node):
    """Publish RViz markers to visualize entities not visible in MuJoCo."""

    def __init__(self) -> None:
        """Initialize subscriptions and publishers."""
        super().__init__("rviz_marker_node")

        self._latest_odometry: Pose2D | None = None
        self._latest_est_odometry: Pose2D | None = None

        self.declare_parameter("publish_rate_hz", 10.0)
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self.declare_parameter("frame_id", "map")
        self._default_frame_id = str(self.get_parameter("frame_id").value)

        self.declare_parameter("scene_path", "")
        scene_path_param = str(self.get_parameter("scene_path").value)
        if scene_path_param:
            self._scene_path = Path(scene_path_param)
        else:
            q1_share_dir = Path(get_package_share_directory("occupancy_grid"))
            self._scene_path = q1_share_dir / "models" / "turtlebot_scene.xml"

        self._environment = Environment2D()
        try:
            self._environment = parse_scene_environment(scene_xml=self._scene_path)
            self.get_logger().info(f"Loaded scene markers from: {self._scene_path}")
        except Exception as exc:
            self.get_logger().warn(
                f"Failed to parse obstacles from scene '{self._scene_path}': {exc}"
            )

        self.create_subscription(Pose2D, "/gt/odometry", self._odometry_cb, FAST_QoS)
        self.create_subscription(
            Pose2D, "/estimated_odometry", self._est_odometry_cb, FAST_QoS
        )

        self._marker_pub = self.create_publisher(
            MarkerArray, "/rviz/markers", LATCHED_QoS
        )

        self.create_timer(1.0 / publish_rate_hz, self._publish_markers_cb)

    def _odometry_cb(self, msg: Pose2D) -> None:
        """Store ground-truth odometry to be visualized."""
        self._latest_odometry = msg

    def _est_odometry_cb(self, msg: Pose2D) -> None:
        """Store estimated odometry to be visualized."""
        self._latest_est_odometry = msg

    def _build_robot_marker(
        self, pose: Pose2D | None, frame_id: str, stamp: Time, ns: str, color: RGBA01
    ) -> Marker | None:
        """Build an RViz marker to visualize the robot at the given pose.

        :param pose: Pose used to place the marker (exits if None)
        :param frame_id: Frame ID of the marker
        :param stamp: Timestamp used for the marker
        :param ns: Namespace of the marker
        :param color: RGBA color of the marker
        :return: Robot marker or `None` if the pose was unavailable
        """
        if pose is None:
            return None

        marker_msg = Marker()
        marker_msg.header.frame_id = frame_id
        marker_msg.header.stamp = stamp

        marker_msg.ns = ns
        marker_msg.id = 0
        marker_msg.type = Marker.CYLINDER
        marker_msg.action = Marker.ADD

        marker_msg.pose.position.x = float(pose.x)
        marker_msg.pose.position.y = float(pose.y)
        marker_msg.pose.position.z = 0.5 * ROBOT_HEIGHT_M
        marker_msg.pose.orientation = yaw_to_quaternion_msg(pose.theta)

        marker_msg.scale.x = 2 * ROBOT_RADIUS_M
        marker_msg.scale.y = 2 * ROBOT_RADIUS_M
        marker_msg.scale.z = ROBOT_HEIGHT_M

        marker_msg.color = rgba_01_to_msg(color)

        return marker_msg

    def _build_obstacle_markers(
        self,
        frame_id: str,
        stamp: Time,
    ) -> list[Marker]:
        """Build static obstacle markers from data parsed from the scene XML.

        :param frame_id: Reference frame used by RViz
        :param stamp: Timestamp of the constructed messages
        :return: List of obstacle messages
        """
        markers: list[Marker] = []
        z_center_m = 0.5 * OBSTACLE_HEIGHT_M

        obstacle_rgba_msg = rgba_01_to_msg(OBSTACLE_RGBA_01)

        for idx, rect in enumerate(self._environment.rectangles):
            marker_msg = Marker()
            marker_msg.header.frame_id = frame_id
            marker_msg.header.stamp = stamp
            marker_msg.ns = "obstacles_rect"
            marker_msg.id = idx
            marker_msg.type = Marker.CUBE
            marker_msg.action = Marker.ADD
            marker_msg.pose.position.x = rect.x_m
            marker_msg.pose.position.y = rect.y_m
            marker_msg.pose.position.z = z_center_m
            marker_msg.pose.orientation = yaw_to_quaternion_msg(rect.yaw_rad)
            marker_msg.scale.x = 2.0 * rect.half_width_m
            marker_msg.scale.y = 2.0 * rect.half_height_m
            marker_msg.scale.z = OBSTACLE_HEIGHT_M
            marker_msg.color = obstacle_rgba_msg
            markers.append(marker_msg)

        for idx, circle in enumerate(self._environment.circles):
            marker_msg = Marker()
            marker_msg.header.frame_id = frame_id
            marker_msg.header.stamp = stamp
            marker_msg.ns = "obstacles_circ"
            marker_msg.id = idx
            marker_msg.type = Marker.CYLINDER
            marker_msg.action = Marker.ADD
            marker_msg.pose.position.x = circle.x_m
            marker_msg.pose.position.y = circle.y_m
            marker_msg.pose.position.z = z_center_m
            marker_msg.pose.orientation.w = 1.0
            marker_msg.scale.x = 2.0 * circle.radius_m
            marker_msg.scale.y = 2.0 * circle.radius_m
            marker_msg.scale.z = OBSTACLE_HEIGHT_M
            marker_msg.color = obstacle_rgba_msg
            markers.append(marker_msg)

        return markers

    @staticmethod
    def _delete_marker(frame_id: str, ns: str, marker_id: int = 0) -> Marker:
        """Construct and return a marker deletion command."""
        marker_msg = Marker()
        marker_msg.header.frame_id = frame_id
        marker_msg.ns = ns
        marker_msg.id = int(marker_id)
        marker_msg.action = Marker.DELETE
        return marker_msg

    def _publish_markers_cb(self) -> None:
        """Publish the currently stored data as RViz markers."""
        frame_id = self._default_frame_id
        stamp = self.get_clock().now().to_msg()
        marker_array_msg = MarkerArray()

        robot_gt_marker = self._build_robot_marker(
            pose=self._latest_odometry,
            frame_id=frame_id,
            stamp=stamp,
            ns="robot_gt",
            color=ROBOT_GT_RGBA_01,
        )
        if robot_gt_marker is None:
            delete_gt = self._delete_marker(frame_id=frame_id, ns="robot_gt")
            marker_array_msg.markers.append(delete_gt)
        else:
            marker_array_msg.markers.append(robot_gt_marker)

        robot_est_marker = self._build_robot_marker(
            pose=self._latest_est_odometry,
            frame_id=frame_id,
            stamp=stamp,
            ns="robot_est",
            color=ROBOT_EST_RGBA_01,
        )
        if robot_est_marker is None:
            delete_est = self._delete_marker(frame_id=frame_id, ns="robot_est")
            marker_array_msg.markers.append(delete_est)
        else:
            marker_array_msg.markers.append(robot_est_marker)

        marker_array_msg.markers.extend(
            self._build_obstacle_markers(frame_id=frame_id, stamp=stamp)
        )

        self._marker_pub.publish(marker_array_msg)


def main() -> None:
    """Initialize ROS and spin the RViz marker visualization node."""
    rclpy.init()
    node = RVizMarkerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
