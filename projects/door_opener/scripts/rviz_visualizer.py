#!/usr/bin/env python3
"""Q2 RViz Visualizer — publishes markers for door, handle, EE, planned waypoints.

Markers displayed:
  - Red sphere:    handle position (from /q2/handle_position sensor)
  - Yellow sphere: end-effector position (from /q2/ee_position sensor)
  - Orange outline: door panel at current angle
  - Axis triads:   planned waypoints with orientation (from /q2/planned_waypoints)

For planning visualization, publish to /q2/planned_waypoints as a
PoseArray in the "world" frame. Each pose will be shown as an X/Y/Z triad.

Run:
  ros2 run q2 rviz_visualizer.py
"""

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import PoseArray, Point
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration

from skills_utils import FAST_QoS


HINGE_POSITION = np.array([0.35, 0.35, 0.42])
HANDLE_OFFSET_FROM_HINGE = np.array([0.225, -0.085, 0.35])

# TCP offset from hand body origin, along hand local +Z (between fingertips)
TCP_OFFSET_LOCAL = np.array([0.0, 0.0, 0.1034])


def quat_to_rot(qw, qx, qy, qz):
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    n = qw * qw + qx * qx + qy * qy + qz * qz
    if n < 1e-8:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * qw * qx, s * qw * qy, s * qw * qz
    xx, xy, xz = s * qx * qx, s * qx * qy, s * qx * qz
    yy, yz, zz = s * qy * qy, s * qy * qz, s * qz * qz
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)]
    ])


class RVizVisualizer(Node):

    def __init__(self):
        super().__init__("q2_rviz_visualizer")

        self.ee_position = None
        self.ee_orientation = None
        self.handle_position = None
        self.door_angle = 0.0
        self.planned_poses = None  # PoseArray

        self.create_subscription(
            Float64MultiArray, "/q2/ee_position", self._ee_cb, FAST_QoS)
        self.create_subscription(
            Float64MultiArray, "/q2/ee_orientation", self._ee_quat_cb, FAST_QoS)
        self.create_subscription(
            Float64MultiArray, "/q2/handle_position", self._handle_cb, FAST_QoS)
        self.create_subscription(
            Float64MultiArray, "/q2/door_angle", self._door_cb, FAST_QoS)
        self.create_subscription(
            PoseArray, "/q2/planned_waypoints", self._plan_cb, 10)

        self.marker_pub = self.create_publisher(
            MarkerArray, "/q2/visualization_markers", 10)
        self.create_timer(0.1, self._publish_markers)
        self.get_logger().info("RViz visualizer started.")

    def _ee_cb(self, msg):
        self.ee_position = np.array(msg.data)

    def _ee_quat_cb(self, msg):
        self.ee_orientation = np.array(msg.data)

    def _handle_cb(self, msg):
        self.handle_position = np.array(msg.data)

    def _door_cb(self, msg):
        if msg.data:
            self.door_angle = msg.data[0]

    def _plan_cb(self, msg):
        self.planned_poses = msg

    def _mk(self, id, mtype, pos, scale, r, g, b, a=1.0):
        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "q2"
        m.id = id
        m.type = mtype
        m.action = Marker.ADD
        m.pose.position.x = float(pos[0])
        m.pose.position.y = float(pos[1])
        m.pose.position.z = float(pos[2])
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
        m.lifetime = Duration(sec=1, nanosec=0)
        return m

    def _make_arrow(self, id, start, end, r, g, b, a=1.0, shaft=0.004, head=0.012):
        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "q2_plan"
        m.id = id
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.scale.x = shaft
        m.scale.y = head
        m.scale.z = 0.01
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
        m.lifetime = Duration(sec=1, nanosec=0)
        m.pose.orientation.w = 1.0
        m.points = [
            Point(x=float(start[0]), y=float(start[1]), z=float(start[2])),
            Point(x=float(end[0]),   y=float(end[1]),   z=float(end[2])),
        ]
        return m

    def _publish_markers(self):
        ma = MarkerArray()

        # Handle position (red sphere)
        if self.handle_position is not None:
            ma.markers.append(self._mk(
                0, Marker.SPHERE, self.handle_position,
                (0.04, 0.04, 0.04), 1.0, 0.0, 0.0))

        # TCP position (yellow sphere) — tip point between fingertips
        if self.ee_position is not None and self.ee_orientation is not None:
            R_hand = quat_to_rot(
                self.ee_orientation[0], self.ee_orientation[1],
                self.ee_orientation[2], self.ee_orientation[3])
            tcp = self.ee_position + R_hand @ TCP_OFFSET_LOCAL
            ma.markers.append(self._mk(
                1, Marker.SPHERE, tcp,
                (0.03, 0.03, 0.03), 1.0, 1.0, 0.0))

        # World axes at origin (X=red, Y=green, Z=blue)
        axis_len = 0.2
        axis_shaft = 0.02
        axis_head = 0.04
        origin = np.zeros(3)
        ma.markers.append(self._make_arrow(
            1000, origin, origin + np.array([axis_len, 0, 0]),
            1.0, 0.0, 0.0, shaft=axis_shaft, head=axis_head))
        ma.markers.append(self._make_arrow(
            1001, origin, origin + np.array([0, axis_len, 0]),
            0.0, 1.0, 0.0, shaft=axis_shaft, head=axis_head))
        ma.markers.append(self._make_arrow(
            1002, origin, origin + np.array([0, 0, axis_len]),
            0.0, 0.0, 1.0, shaft=axis_shaft, head=axis_head))

        # Door panel outline (orange)
        door_m = Marker()
        door_m.header.frame_id = "world"
        door_m.header.stamp = self.get_clock().now().to_msg()
        door_m.ns = "q2"
        door_m.id = 2
        door_m.type = Marker.LINE_STRIP
        door_m.action = Marker.ADD
        door_m.scale.x = 0.008
        door_m.color.r, door_m.color.g, door_m.color.b, door_m.color.a = 1.0, 0.5, 0.0, 0.8
        door_m.lifetime = Duration(sec=1, nanosec=0)
        door_m.pose.orientation.w = 1.0
        c, s = np.cos(self.door_angle), np.sin(self.door_angle)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        for corner in [[0.01, 0, 0.01], [0.29, 0, 0.01],
                       [0.29, 0, 0.77], [0.01, 0, 0.77], [0.01, 0, 0.01]]:
            wc = HINGE_POSITION + Rz @ np.array(corner)
            door_m.points.append(Point(x=float(wc[0]), y=float(wc[1]), z=float(wc[2])))
        ma.markers.append(door_m)

        # Planned waypoint poses (axis triads)
        if self.planned_poses is not None:
            arrow_id = 0
            axis_len = 0.04
            for pose in self.planned_poses.poses:
                p = np.array([pose.position.x, pose.position.y, pose.position.z])
                R = quat_to_rot(pose.orientation.w, pose.orientation.x,
                                pose.orientation.y, pose.orientation.z)
                # X axis (red)
                ma.markers.append(self._make_arrow(
                    arrow_id, p, p + axis_len * R[:, 0], 1.0, 0.0, 0.0))
                arrow_id += 1
                # Y axis (green)
                ma.markers.append(self._make_arrow(
                    arrow_id, p, p + axis_len * R[:, 1], 0.0, 1.0, 0.0))
                arrow_id += 1
                # Z axis (blue)
                ma.markers.append(self._make_arrow(
                    arrow_id, p, p + axis_len * R[:, 2], 0.0, 0.0, 1.0))
                arrow_id += 1

        self.marker_pub.publish(ma)


def main():
    rclpy.init()
    node = RVizVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
