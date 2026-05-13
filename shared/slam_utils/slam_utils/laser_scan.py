"""Define a class representing 2D laser scans from a particular sensor pose."""

from dataclasses import dataclass

from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import LaserScan


@dataclass(frozen=True)
class PosedLaserScan:
    """A 2D laser scan with range measurements relative to a known sensor pose."""

    sensor_pose: Pose2D
    """ROS 2 message representing the sensor pose when the scan was captured."""

    scan: LaserScan
    """ROS 2 message containing range measurement data."""
