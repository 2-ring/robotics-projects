#!/usr/bin/env python3
"""Discrete Markov (histogram) localizer over a (theta, x, y) belief grid.

Prediction step: shift the belief by the commanded velocity command using the
ideal velocity model, with bilinear interpolation for sub-cell remainders and
an `off_by_one` smoothing kernel on theta to model angular noise.

Correction step: for each kept beam and each theta bin, project the beam
endpoint into the map and apply a hit/miss likelihood depending on whether
the projected endpoint matches the ground-truth occupancy.

Formulation follows Thrun, Burgard & Fox (2006): Markov grid localization
(Table 8.1), velocity motion model (Eq. 5.9), beam measurement model
(Table 6.1).
"""

import math
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose2D, Twist
from numpy.typing import NDArray
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from slam_utils import FAST_QoS, GridInfo, GridCell, build_gt_map
from slam_utils.motion_models import simulate_velocity_delta


class BayesLocalizer(Node):
    """Discrete Bayes localizer over a fixed occupancy map.

    Prediction runs on a timer using the latest /cmd_vel. Correction runs on
    every /laser_scans message and publishes the MAP pose estimate.
    """

    def __init__(self) -> None:
        """Initialize map, belief, prediction timer, and correction subscription."""
        super().__init__("bayes_localizer")

        self.declare_parameter("origin_x_m", -2.0)
        self.declare_parameter("origin_y_m", -2.0)
        self.declare_parameter("resolution_m", 0.05)
        self.declare_parameter("height_cells", 80)
        self.declare_parameter("width_cells", 80)
        self.declare_parameter("frame_id", "map")

        self.declare_parameter("theta_bins", 36)
        self.declare_parameter("every_nth_beam", 3)
        self.declare_parameter("off_by_one_prob", 0.08)
        self.declare_parameter("prediction_rate_hz", 20.0)
        self.declare_parameter("scene_path", "")

        self._grid_info = GridInfo(
            origin_x=float(self.get_parameter("origin_x_m").value),
            origin_y=float(self.get_parameter("origin_y_m").value),
            resolution_m=float(self.get_parameter("resolution_m").value),
            height_cells=int(self.get_parameter("height_cells").value),
            width_cells=int(self.get_parameter("width_cells").value),
            parent_frame_id=str(self.get_parameter("frame_id").value),
        )
        self._theta_bins = int(self.get_parameter("theta_bins").value)
        self._every_nth_beam = int(self.get_parameter("every_nth_beam").value)
        self._off_by_one_prob = float(self.get_parameter("off_by_one_prob").value)

        scene_path_param = str(self.get_parameter("scene_path").value)
        if scene_path_param:
            scene_path = Path(scene_path_param)
        else:
            q1_share = Path(get_package_share_directory("occupancy_grid"))
            scene_path = q1_share / "models" / "turtlebot_scene.xml"

        occ_bool = build_gt_map(grid_info=self._grid_info, scene_xml=scene_path)
        self._free_mask = ~occ_bool

        self._theta_vals_rad = np.linspace(
            -np.pi, np.pi, self._theta_bins, endpoint=False, dtype=np.float32
        )
        self._theta_step_rad = math.tau / self._theta_bins

        # Belief tensor indexed by (theta bin, row, col); initialized to a
        # uniform distribution over free cells × all headings.
        self._belief = np.zeros(
            (
                self._theta_bins,
                self._grid_info.height_cells,
                self._grid_info.width_cells,
            ),
            dtype=np.float32,
        )

        free_count = np.sum(self._free_mask)
        if not free_count:
            raise RuntimeError("No free cells found in occupancy map.")
        self._belief[:, self._free_mask] = 1.0 / (self._theta_bins * free_count)

        self._latest_vx_mps = 0.0
        self._latest_wz_radps = 0.0
        self._correction_step_count = 0

        self.create_subscription(
            LaserScan, "/laser_scans", self._laser_scan_cb, FAST_QoS
        )
        self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_cb, FAST_QoS)
        self._pose_pub = self.create_publisher(Pose2D, "/estimated_odometry", 10)

        prediction_rate_hz = float(self.get_parameter("prediction_rate_hz").value)
        self._prediction_dt_s = 1.0 / prediction_rate_hz
        self.create_timer(self._prediction_dt_s, self._prediction_timer_cb)

    def _laser_scan_cb(self, msg: LaserScan) -> None:
        """Apply a correction update and publish the MAP pose."""
        self._belief = self._correct_belief(
            self._belief, msg, every_nth_beam=self._every_nth_beam
        )

        pose_msg = self._map_pose_from_belief(self._belief)
        self._pose_pub.publish(pose_msg)

        self._correction_step_count += 1
        if self._correction_step_count % 500 == 0:
            self.get_logger().info(
                f"correction={self._correction_step_count} "
                f"est_pose=(x={pose_msg.x:.3f}, y={pose_msg.y:.3f}, theta={pose_msg.theta:.3f})"
            )

    def _cmd_vel_cb(self, msg: Twist) -> None:
        self._latest_vx_mps = float(msg.linear.x)
        self._latest_wz_radps = float(msg.angular.z)

    def _prediction_timer_cb(self) -> None:
        self._belief = self._predict_belief(
            belief=self._belief,
            vx_mps=self._latest_vx_mps,
            wz_radps=self._latest_wz_radps,
            dt_s=self._prediction_dt_s,
            off_by_one_prob=self._off_by_one_prob,
        )

    def _map_pose_from_belief(self, belief: NDArray[np.float32]) -> Pose2D:
        """MAP pose: argmax over the (theta, row, col) belief tensor."""
        flat_idx = int(np.argmax(belief))
        theta_idx, y_cell, x_cell = np.unravel_index(flat_idx, belief.shape)

        pose_msg = Pose2D()
        pose_msg.x = self._grid_info.col_to_x(int(x_cell))
        pose_msg.y = self._grid_info.row_to_y(int(y_cell))
        pose_msg.theta = float(self._theta_vals_rad[theta_idx])

        return pose_msg

    def _normalize(self, belief: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normalize to a proper distribution; fall back to uniform-on-free if mass collapses."""
        total_mass = np.sum(belief)
        if total_mass <= 1e-15:
            free_count = int(np.sum(self._free_mask))
            uniform_free = np.zeros_like(belief)
            uniform_free[:, self._free_mask] = 1.0 / (self._theta_bins * free_count)
            return uniform_free
        return belief / total_mass

    @staticmethod
    def _shift_no_wrap(grid: NDArray, dr_cells: int, dc_cells: int) -> NDArray:
        """Translate a 2D array without wrap-around — out-of-bounds entries drop."""
        n_rows, n_cols = grid.shape
        out = np.zeros_like(grid)

        src_r0 = max(0, -dr_cells)
        src_r1 = min(n_rows, n_rows - dr_cells)
        dst_r0 = max(0, dr_cells)
        dst_r1 = dst_r0 + max(0, src_r1 - src_r0)

        src_c0 = max(0, -dc_cells)
        src_c1 = min(n_cols, n_cols - dc_cells)
        dst_c0 = max(0, dc_cells)
        dst_c1 = dst_c0 + max(0, src_c1 - src_c0)

        if dst_r0 >= dst_r1 or dst_c0 >= dst_c1:
            return out

        out[dst_r0:dst_r1, dst_c0:dst_c1] = grid[src_r0:src_r1, src_c0:src_c1]
        return out

    def _predict_belief(
        self,
        belief: NDArray[np.float32],
        vx_mps: float,
        wz_radps: float,
        dt_s: float,
        off_by_one_prob: float,
    ) -> NDArray[np.float32]:
        """Bayes-filter prediction step over the discretized pose grid.

        For each theta bin, integrate the velocity command for `dt_s` and
        shift the bin's spatial belief by the resulting (drow, dcol) with
        bilinear interpolation for fractional remainders. Then shift theta
        by `dtheta` (also with bilinear interpolation) and apply an
        off-by-one smoothing kernel to model angular noise. Mass on
        known-occupied cells is zeroed and the belief is renormalized.
        """

        new_belief = np.zeros_like(belief)
        for bin_index, theta_bin in enumerate(belief):
            theta = self._theta_vals_rad[bin_index]
            (dx, dy, dtheta) = simulate_velocity_delta(theta, vx_mps, wz_radps, dt_s)
            dcol = dx / self._grid_info.resolution_m
            drow = dy / self._grid_info.resolution_m

            dcol_floor = math.floor(dcol)
            drow_floor = math.floor(drow)
            frac_col = dcol - dcol_floor
            frac_row = drow - drow_floor

            new_belief[bin_index] = (
                self._shift_no_wrap(theta_bin, drow_floor, dcol_floor) * (1 - frac_col) * (1 - frac_row) +
                self._shift_no_wrap(theta_bin, drow_floor + 1, dcol_floor) * (1 - frac_col) * frac_row +
                self._shift_no_wrap(theta_bin, drow_floor, dcol_floor + 1) * frac_col * (1 - frac_row) +
                self._shift_no_wrap(theta_bin, drow_floor + 1, dcol_floor + 1) * frac_col * frac_row
            )

        # Shift all theta bins by dtheta with bilinear interpolation.
        dtheta_bins = dtheta / self._theta_step_rad
        dbins_floor = math.floor(dtheta_bins)
        frac_theta = dtheta_bins - dbins_floor
        new_belief = (
            np.roll(new_belief, dbins_floor, axis=0) * (1 - frac_theta) +
            np.roll(new_belief, dbins_floor + 1, axis=0) * frac_theta
        )
        # Off-by-one angular noise.
        new_belief = (
            (1 - off_by_one_prob) * new_belief +
            np.roll(new_belief, -1, axis=0) * (off_by_one_prob / 2) +
            np.roll(new_belief, 1, axis=0) * (off_by_one_prob / 2)
        )

        new_belief[:, ~self._free_mask] = 0
        return self._normalize(new_belief)

    def _correct_belief(
        self,
        belief: NDArray[np.float32],
        scan: LaserScan,
        every_nth_beam: int,
        hit_likelihood: float = 0.87,
        miss_likelihood: float = 0.24,
    ) -> NDArray[np.float32]:
        """Bayes-filter correction step using the latest laser scan.

        For every Nth beam and each theta bin, project the endpoint into the
        map; multiply the belief by `hit_likelihood` where the projected
        endpoint matches the ground-truth occupancy and `miss_likelihood`
        where it doesn't (or falls off the map). Mass on occupied cells is
        zeroed and the belief is renormalized.
        """

        occ_map = (~self._free_mask).astype(np.float32)
        for beam_index in range(0, len(scan.ranges), every_nth_beam):
            beam = scan.ranges[beam_index]
            if math.isfinite(beam) and beam >= scan.range_min and beam <= scan.range_max:
                for theta_index in range(self._theta_bins):
                    theta = self._theta_vals_rad[theta_index]
                    beam_angle = scan.angle_min + (scan.angle_increment * beam_index) + theta
                    dx = beam * math.cos(beam_angle)
                    dy = beam * math.sin(beam_angle)
                    # Endpoint delta — constant for all cells in this theta bin.
                    dc = int(math.floor(0.5 + dx / self._grid_info.resolution_m))
                    dr = int(math.floor(0.5 + dy / self._grid_info.resolution_m))
                    shifted_occ = self._shift_no_wrap(occ_map, -dr, -dc)
                    update_multiplier = np.where(shifted_occ > 0.5, hit_likelihood, miss_likelihood)
                    belief[theta_index] *= update_multiplier

        belief[:, ~self._free_mask] = 0
        return self._normalize(belief)


def main() -> None:
    rclpy.init()
    node = BayesLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
