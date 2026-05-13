#!/usr/bin/env python3
"""Script defining a ROS 2 node that runs occupancy mapping."""

import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from occupancy_grid.occupancy_grid import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from slam_utils import FAST_QoS, GridInfo, LATCHED_QoS, PosedLaserScan


class OccupancyMapperNode(Node):
    """ROS 2 node that updates and publishes a 2D occupancy map."""

    def __init__(self) -> None:
        """Initialize map model, subscriptions, publisher, and timer."""
        super().__init__("occupancy_mapper_node")

        self.declare_parameter("origin_x_m", -2.0)
        self.declare_parameter("origin_y_m", -2.0)
        self.declare_parameter("resolution_m", 0.05)
        self.declare_parameter("height_cells", 80)
        self.declare_parameter("width_cells", 80)
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("min_obstacle_depth_m", 0.05)
        self.declare_parameter("publish_rate_hz", 2.0)

        grid_info = GridInfo(
            origin_x=float(self.get_parameter("origin_x_m").value),
            origin_y=float(self.get_parameter("origin_y_m").value),
            resolution_m=float(self.get_parameter("resolution_m").value),
            height_cells=int(self.get_parameter("height_cells").value),
            width_cells=int(self.get_parameter("width_cells").value),
            parent_frame_id=str(self.get_parameter("frame_id").value),
        )
        min_obstacle_depth_m = float(self.get_parameter("min_obstacle_depth_m").value)
        self._publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self._occupancy_grid = OccupancyGrid(
            grid_info=grid_info, min_obstacle_depth_m=min_obstacle_depth_m
        )
        self._latest_odom: Pose2D | None = None

        self.create_subscription(Pose2D, "/gt/odometry", self._odometry_cb, FAST_QoS)
        self.create_subscription(
            LaserScan, "/laser_scans", self._laser_scan_cb, LATCHED_QoS
        )
        self._map_pub = self.create_publisher(OccupancyGridMsg, "/occupancy_map", 5)

        self.create_timer(1.0 / self._publish_rate_hz, self._publish_map_cb)

    def _odometry_cb(self, msg: Pose2D) -> None:
        """Store the latest robot odometry into a member variable."""
        self._latest_odom = msg

    def _laser_scan_cb(self, msg: LaserScan) -> None:
        """Update the occupancy grid using data from a new laser scan.

        :param msg: New LiDAR scan to integrate into the map
        """
        if self._latest_odom is None:
            return

        posed_scan = PosedLaserScan(sensor_pose=self._latest_odom, scan=msg)
        self._occupancy_grid.update(posed_scan)

    def _publish_map_cb(self) -> None:
        """Publish the current occupancy grid as a nav_msgs/OccupancyGrid message."""
        self._map_pub.publish(self._build_occupancy_map_msg())

    def _build_occupancy_map_msg(self) -> OccupancyGridMsg:
        """Build an occupancy grid ROS message from the internal log-odds map.

        :return: Populated occupancy grid message
        """
        grid = self._occupancy_grid
        info = grid.grid_info
        msg = OccupancyGridMsg()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = info.parent_frame_id

        msg.info.resolution = float(info.resolution_m)
        msg.info.width = int(info.width_cells)
        msg.info.height = int(info.height_cells)
        msg.info.origin.position.x = float(info.origin_x)
        msg.info.origin.position.y = float(info.origin_y)
        msg.info.origin.orientation.w = 1.0

        p_occupied = OccupancyGrid.log_odds_to_prob(grid.log_odds)
        occupancy_0_to_100 = np.rint(p_occupied * 100.0).astype(np.int8)

        unknown_mask = np.isclose(grid.log_odds, 0.0)
        occupancy_0_to_100[unknown_mask] = -1

        msg.data = occupancy_0_to_100.flatten(order="C").tolist()
        return msg


def main() -> None:
    """Initialize ROS and spin the occupancy mapper node."""
    rclpy.init()
    node = OccupancyMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
