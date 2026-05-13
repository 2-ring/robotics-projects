"""Define functions to perform common ROS 2 idioms."""

from math import cos, sin

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA

from .colors import RGBA01

FAST_QoS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)
"""A low-latency, best-effort QoS profile."""


LATCHED_QoS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)
"""A transient-local, reliable QoS profile for latched messages."""


def yaw_to_quaternion_msg(yaw_rad: float) -> Quaternion:
    """Convert a yaw value (radians) into a geometry_msgs/Quaternion message."""
    msg = Quaternion()
    msg.w = cos(0.5 * yaw_rad)
    msg.z = sin(0.5 * yaw_rad)
    return msg


def rgba_01_to_msg(rgba: RGBA01) -> ColorRGBA:
    """Convert a unit-scale RGBA color into a std_msgs/ColorRGBA message."""
    r, g, b, a = rgba

    msg = ColorRGBA()
    msg.r = r
    msg.g = g
    msg.b = b
    msg.a = a
    return msg


def waypoints_xy_to_path_msg(
    waypoints_xy: list[tuple[float, float]], stamp: Time, frame_id: str = "map"
) -> Path:
    """Convert a list of (x,y) waypoints into a nav_msgs/Path message."""
    path_msg = Path()
    path_msg.header.stamp = stamp
    path_msg.header.frame_id = frame_id

    for x, y in waypoints_xy:
        pose_msg = PoseStamped()
        pose_msg.header = path_msg.header
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.orientation.w = 1.0
        path_msg.poses.append(pose_msg)

    return path_msg
