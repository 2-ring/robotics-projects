#!/usr/bin/env python3
"""Q1 Object Detector Node — simulates an object recognition + pose estimation system.

Uses ground-truth object positions from MuJoCo, applies noise and detection constraints
(field-of-view, range, false negatives) to simulate a realistic detector.

Subscribes to:
  /q1/robot_pose         geometry_msgs/Pose2D
  /q1/object_positions   std_msgs/String (JSON ground-truth positions)

Publishes:
  /q1/detections         std_msgs/String (JSON list of detections)
    Each detection: {"class": str, "position_camera": [x, y, z],
                     "position_world": [x, y, z], "confidence": float}

This node is PROVIDED and should NOT be modified.
"""

import json
import math
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose2D
from std_msgs.msg import String

from skills_utils import FAST_QoS, rotation_matrix_2d


# Static wall segments for line-of-sight occlusion checks.
# Each entry: (x1, y1, x2, y2) — endpoints of a wall segment.
_WALL_SEGMENTS = [
    # Outer walls (room ±1.5)
    (-1.5, -1.5,  1.5, -1.5),   # south
    (-1.5,  1.5,  1.5,  1.5),   # north
    (-1.5, -1.5, -1.5,  1.5),   # west
    ( 1.5, -1.5,  1.5,  1.5),   # east
    # Interior partition
    (-0.5,  0.0,  0.5,  0.0),   # horizontal partition at center
]


def _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    """Return True if segment AB intersects segment CD (proper crossing)."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross((cx, cy), (dx, dy), (ax, ay))
    d2 = cross((cx, cy), (dx, dy), (bx, by))
    d3 = cross((ax, ay), (bx, by), (cx, cy))
    d4 = cross((ax, ay), (bx, by), (dx, dy))

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


class ObjectDetectorNode(Node):

    def __init__(self):
        super().__init__("object_detector_node")

        self.declare_parameter("detection_rate_hz", 5.0)
        self.declare_parameter("detection_range_m", 2.0)
        self.declare_parameter("detection_fov_rad", 1.2)
        self.declare_parameter("position_noise_sigma_m", 0.05)
        self.declare_parameter("false_negative_prob", 0.05)

        self.detection_range = self.get_parameter("detection_range_m").value
        self.detection_fov = self.get_parameter("detection_fov_rad").value
        self.noise_sigma = self.get_parameter("position_noise_sigma_m").value
        self.fn_prob = self.get_parameter("false_negative_prob").value
        rate = self.get_parameter("detection_rate_hz").value

        self.rng = np.random.default_rng(42)

        self.robot_pose = None
        self.object_positions = {}

        self.create_subscription(Pose2D, "/q1/robot_pose", self._pose_cb, FAST_QoS)
        self.create_subscription(String, "/q1/object_positions", self._obj_cb, FAST_QoS)

        self.det_pub = self.create_publisher(String, "/q1/detections", FAST_QoS)

        self.create_timer(1.0 / rate, self._detect)
        self.get_logger().info("Object detector node started.")

    def _pose_cb(self, msg: Pose2D):
        self.robot_pose = msg

    def _obj_cb(self, msg: String):
        try:
            self.object_positions = json.loads(msg.data)
        except Exception:
            pass

    def _detect(self):
        if self.robot_pose is None or not self.object_positions:
            return

        rx, ry, rtheta = self.robot_pose.x, self.robot_pose.y, self.robot_pose.theta
        detections = []

        for obj_class, pos in self.object_positions.items():
            ox, oy, oz = pos

            # Distance check
            dx = ox - rx
            dy = oy - ry
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > self.detection_range:
                continue

            # FOV check (is object in front of robot within camera FOV?)
            angle_to_obj = math.atan2(dy, dx)
            angle_diff = (angle_to_obj - rtheta + math.pi) % (2 * math.pi) - math.pi
            if abs(angle_diff) > self.detection_fov / 2:
                continue

            # Line-of-sight occlusion check (walls block detections)
            occluded = False
            for wx1, wy1, wx2, wy2 in _WALL_SEGMENTS:
                if _segments_intersect(rx, ry, ox, oy, wx1, wy1, wx2, wy2):
                    occluded = True
                    break
            if occluded:
                continue

            # False negative
            if self.rng.random() < self.fn_prob:
                continue

            # Add noise to position
            noise = self.rng.normal(0, self.noise_sigma, size=3)
            # Scale noise with distance (farther = noisier)
            noise *= (1.0 + 0.5 * dist)
            noisy_world = [ox + noise[0], oy + noise[1], oz + noise[2]]

            # Transform to camera frame (relative to robot)
            R_inv = rotation_matrix_2d(-rtheta)
            rel_xy = R_inv @ np.array([noisy_world[0] - rx, noisy_world[1] - ry])
            camera_pos = [float(rel_xy[0]), float(rel_xy[1]), float(noisy_world[2])]

            # Confidence decreases with distance
            confidence = max(0.5, 1.0 - 0.3 * dist)
            confidence += self.rng.normal(0, 0.05)
            confidence = float(np.clip(confidence, 0.3, 1.0))

            detections.append({
                "class": obj_class,
                "position_camera": camera_pos,
                "position_world_noisy": noisy_world,
                "confidence": round(confidence, 3),
            })

        det_msg = String()
        det_msg.data = json.dumps(detections)
        self.det_pub.publish(det_msg)


def main():
    rclpy.init()
    node = ObjectDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
