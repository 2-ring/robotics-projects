"""2D log-odds occupancy grid built from posed laser scans.

Inverse sensor model: ray-trace each valid beam, deposit `p_free` log-odds on
traversed cells and `p_occupied` log-odds on cells inside the obstacle's
assumed thickness band. Standard formulation from Thrun, Burgard, & Fox
(2006), Chapter 9.2 / Table 9.1.
"""

import math
import numpy as np
from numpy.typing import NDArray

from slam_utils import GridInfo, PosedLaserScan, bresenham_line


class OccupancyGrid:
    """A 2D occupancy grid using log odds to represent the probability of occupancy."""

    def __init__(self, grid_info: GridInfo, min_obstacle_depth_m: float) -> None:
        """Initialize the occupancy grid.

        :param grid_info: Defines the origin, resolution, height, and width of the grid
        :param min_obstacle_depth_m: Minimum depth (m) assumed for obstacles when ray-tracing
        """
        self.grid_info = grid_info
        self.min_obstacle_depth_m = min_obstacle_depth_m

        # L = log(p(occupied) / p(free)); init to log(0.5/0.5) = 0.
        self.log_odds = np.zeros(
            (grid_info.height_cells, grid_info.width_cells), dtype=np.float32
        )

    @staticmethod
    def prob_to_log_odds(
        prob: float | NDArray[np.floating],
    ) -> float | NDArray[np.floating]:
        """Probability → log-odds (Thrun et al. 2006, Eq. 9.5)."""
        return (lambda p: np.log(p / (1 - p)))(prob)

    @staticmethod
    def log_odds_to_prob(
        log_odds: float | NDArray[np.floating],
    ) -> float | NDArray[np.floating]:
        """Log-odds → probability (Thrun et al. 2006, Eq. 9.6)."""
        return (lambda l_o: np.exp(l_o) / (1 + np.exp(l_o)))(log_odds)

    def update(
        self, scan: PosedLaserScan, *, p_free: float = 0.2, p_occupied: float = 0.8
    ) -> None:
        """Integrate one posed laser scan into the log-odds map.

        For each valid beam: Bresenham-trace from the sensor cell to the
        endpoint, depositing free-evidence on traversed cells and
        occupied-evidence on cells inside the obstacle band (endpoint →
        endpoint + min_obstacle_depth). Out-of-bounds cells are skipped;
        updates accumulate across scans and are clipped to ±20 to bound
        the dynamic range.

        :param scan: Laser scan from a known pose, to be incorporated into the grid
        :param p_free: Probability that a cell is occupied given a laser passes through it
        :param p_occupied: Probability that a cell is occupied given a laser hits in it
        """

        current_angle = scan.scan.angle_min + scan.sensor_pose.theta
        robot_cell = self.grid_info.coord_to_cell((scan.sensor_pose.x, scan.sensor_pose.y))

        for range in scan.scan.ranges:
            if math.isfinite(range) and range >= scan.scan.range_min and range <= scan.scan.range_max:
                obstacle_start = self.grid_info.coord_to_cell((
                    scan.sensor_pose.x + range * math.cos(current_angle),
                    scan.sensor_pose.y + range * math.sin(current_angle)
                ))
                obstacle_end = self.grid_info.coord_to_cell((
                    scan.sensor_pose.x + (range + self.min_obstacle_depth_m) * math.cos(current_angle),
                    scan.sensor_pose.y + (range + self.min_obstacle_depth_m) * math.sin(current_angle)
                ))
                empty_cells = bresenham_line(robot_cell, obstacle_start)
                occupied_cells = bresenham_line(empty_cells.pop(), obstacle_end)
                for occupied_cell in occupied_cells:
                    if self.grid_info.is_valid_cell(occupied_cell):
                        self.log_odds[occupied_cell.row][occupied_cell.col] += self.prob_to_log_odds(p_occupied)
                for empty_cell in empty_cells:
                    if self.grid_info.is_valid_cell(empty_cell):
                        self.log_odds[empty_cell.row][empty_cell.col] += self.prob_to_log_odds(p_free)
            current_angle += scan.scan.angle_increment

        np.clip(self.log_odds, -20.0, 20.0, out=self.log_odds)
