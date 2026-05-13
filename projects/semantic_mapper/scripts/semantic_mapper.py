#!/usr/bin/env python3
"""Semantic mapper: accumulate object detections, answer relative-pose queries.

Three phases:

  1. EXPLORATION — the robot drives through `EXPLORATION_WAYPOINTS`. The
     detector emits one or more class-labeled detections per frame; each
     observation is fused into a per-class running position estimate.
  2. QUERY — a JSON request `{"object": "mug", "offset": [dx, dy]}` is
     resolved against the current map and converted to a Pose2D goal.
  3. NAVIGATION — the navigator drives to the published goal.

Fusion uses an ICP-style weighted centroid update with adaptive outlier
rejection: per-class observations are weighted by detector confidence; once
the observation pool exceeds 5 samples, observations beyond 2σ (floor 0.3 m)
from the current estimate are dropped. The pool is capped at 100 inliers.

Topics:
    in   /q1/detections, /q1/robot_pose, /q1/navigation_status, /q1/query
    out  /q1/navigation_goal, /q1/semantic_map, /q1/semantic_markers
"""

import json
import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from skills_utils import FAST_QoS


# Waypoints (x, y, theta) chosen to give the detector clear sight lines on the
# three tables in the 3 m x 3 m room. The detector has a 1.2 rad FOV and 1 m
# range, so each waypoint orients the robot at the table from within range.
EXPLORATION_WAYPOINTS = [
    # SW table
    (-0.3, -0.9, 3.14),
    (-0.9, -0.3, -1.57),
    # NW table
    (-0.9, 0.3, 1.57),
    (-0.3, 0.9, 3.14),
    # NE table
    (0.3, 0.9, 0),
    (0.9, 0.3, 1.57),
]

# Object colors matching the MuJoCo materials, used for RViz markers.
_OBJ_COLORS = {
    "mug":    (0.85, 0.15, 0.15),
    "bottle": (0.15, 0.70, 0.20),
    "box":    (0.15, 0.25, 0.85),
    "bowl":   (0.90, 0.75, 0.10),
    "can":    (0.80, 0.40, 0.10),
}


class SemanticMapper(Node):

    def __init__(self):
        super().__init__("semantic_mapper")

        # semantic_map[class] = {"position": [x,y,z], "count": n, "confidence": c}
        # observation_pool[class] = list of [x, y, z, confidence] inliers
        self.semantic_map: dict = {}
        self.observation_pool: dict = {}

        self.robot_pose: Pose2D | None = None
        self.nav_status: str = "idle"

        self.create_subscription(
            String, "/q1/detections", self._detections_cb, FAST_QoS)
        self.create_subscription(
            Pose2D, "/q1/robot_pose", self._pose_cb, FAST_QoS)
        self.create_subscription(
            String, "/q1/navigation_status", self._nav_status_cb, FAST_QoS)
        self.create_subscription(
            String, "/q1/query", self._query_cb, FAST_QoS)

        self.goal_pub   = self.create_publisher(Pose2D,       "/q1/navigation_goal",    FAST_QoS)
        self.map_pub    = self.create_publisher(String,       "/q1/semantic_map",       FAST_QoS)
        self.marker_pub = self.create_publisher(MarkerArray,  "/q1/semantic_markers",   FAST_QoS)

        self.create_timer(1.0, self._publish_map)

        self.get_logger().info("Semantic mapper node started.")

    def _pose_cb(self, msg: Pose2D):
        self.robot_pose = msg

    def _nav_status_cb(self, msg: String):
        self.nav_status = msg.data

    def _detections_cb(self, msg: String):
        """Fuse incoming detections into the per-class position estimate.

        Each detection carries a class label, a noisy world-frame position,
        and a confidence in [0.3, 1.0]. For each detection we (a) append the
        observation to its class pool, (b) drop adaptive-threshold outliers
        once the pool has bootstrapped, (c) update the position estimate to
        the confidence-weighted centroid of inliers, and (d) cap the pool
        at the 100 most recent inliers.
        """
        objects = json.loads(msg.data)

        for obj in objects:
            class_name = obj["class"]
            pos_object  = obj["position_world_noisy"]
            confidence = obj["confidence"]
            x, y, z = pos_object

            if class_name not in self.semantic_map:
                self.observation_pool[class_name] = []
                self.semantic_map[class_name] = {"position": [x, y, z], "count": 0, "confidence": confidence}

            count = self.semantic_map[class_name]["count"] + 1
            observations = self.observation_pool[class_name] + [[x, y, z, confidence]]
            inliers = observations

            if count > 5:
                # Adaptive outlier rejection: drop anything beyond max(2σ, 0.3 m).
                pos_estimate = self.semantic_map[class_name]["position"]
                distances = [((pos_estimate[0]-obs[0])**2 +
                              (pos_estimate[1]-obs[1])**2 +
                              (pos_estimate[2]-obs[2])**2)**0.5
                             for obs in observations]
                mean_dist = sum(distances) / count
                std = (sum([(d - mean_dist)**2 for d in distances]) / count) ** 0.5
                threshold = max(2 * std, 0.3)

                inliers = [obs for obs, d in zip(observations, distances) if d <= threshold]
                count = len(inliers)

            # Confidence-weighted centroid.
            total_weight = sum(obs[3] for obs in inliers)
            new_pos = [0, 0, 0]
            for obs in inliers:
                new_pos = [new_pos[0] + ((obs[0] * obs[3]) / total_weight),
                           new_pos[1] + ((obs[1] * obs[3]) / total_weight),
                           new_pos[2] + ((obs[2] * obs[3]) / total_weight)]
            mean_confidence = total_weight / count

            # Keep memory bounded to the 100 most recent inliers.
            self.observation_pool[class_name] = inliers[-100:]
            self.semantic_map[class_name]["count"] = count
            self.semantic_map[class_name]["position"] = new_pos
            self.semantic_map[class_name]["confidence"] = mean_confidence

    def _query_cb(self, msg: String):
        """Resolve `{"object": str, "offset": [dx, dy]}` into a Pose2D goal.

        Goal position = object_xy + offset_xy. Goal heading points back at
        the object from the goal pose, so the robot ends up facing it.
        """
        message = json.loads(msg.data)
        obj_name = message["object"]
        offset = np.array(message["offset"])
        if obj_name in self.semantic_map:
            object_pos = np.array(self.semantic_map[obj_name]["position"])[:2]
            goal_pos = offset + object_pos
            angle = math.atan2(offset[0], offset[1])
            self.goal_pub.publish(Pose2D(x=goal_pos[0], y=goal_pos[1], theta=angle))
        else:
            self.get_logger().warning(f"Object ({obj_name}) not in map.")

    def _get_position_estimate(self, obj_class: str) -> list[float] | None:
        """Adapter for the RViz marker code — return [x, y, z] or None."""
        entry = self.semantic_map.get(obj_class)
        if entry is None:
            return None
        return entry.get("position")

    def _get_detections_count(self, obj_class: str) -> int:
        """Adapter for the RViz marker code — return observation count."""
        entry = self.semantic_map.get(obj_class)
        if entry is None:
            return 0
        return len(entry.get("observations", []))

    def _publish_map(self):
        """Publish the current semantic map as JSON, then refresh RViz markers."""
        self.map_pub.publish(String(data=json.dumps({
            obj: self.semantic_map[obj]["position"]
            for obj in self.semantic_map
        })))
        self._publish_rviz_markers()

    def _publish_rviz_markers(self):
        """Publish semantic-map markers (sphere + text label per object)."""
        arr = MarkerArray()

        del_m        = Marker()
        del_m.action = Marker.DELETEALL
        arr.markers.append(del_m)

        now = self.get_clock().now().to_msg()
        for i, obj_class in enumerate(self.semantic_map.keys()):
            pos = self._get_position_estimate(obj_class)
            if pos is None:
                continue
            count = self._get_detections_count(obj_class)
            r, g, b = _OBJ_COLORS.get(obj_class, (1.0, 1.0, 1.0))

            sphere = Marker()
            sphere.header.frame_id = "map"
            sphere.header.stamp    = now
            sphere.ns              = "semantic_objects"
            sphere.id              = i
            sphere.type            = Marker.SPHERE
            sphere.action          = Marker.ADD
            sphere.pose.position.x = float(pos[0])
            sphere.pose.position.y = float(pos[1])
            sphere.pose.position.z = float(pos[2])
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.15
            sphere.color.r = r;  sphere.color.g = g
            sphere.color.b = b;  sphere.color.a = 0.85
            arr.markers.append(sphere)

            label = Marker()
            label.header.frame_id = "map"
            label.header.stamp    = now
            label.ns              = "semantic_labels"
            label.id              = i
            label.type            = Marker.TEXT_VIEW_FACING
            label.action          = Marker.ADD
            label.pose.position.x = float(pos[0])
            label.pose.position.y = float(pos[1])
            label.pose.position.z = float(pos[2]) + 0.22
            label.pose.orientation.w = 1.0
            label.scale.z          = 0.10
            label.color.r = label.color.g = label.color.b = 1.0
            label.color.a = 1.0
            label.text             = f"{obj_class}:{count}"
            arr.markers.append(label)

        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
