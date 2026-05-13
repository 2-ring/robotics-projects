#!/usr/bin/env python3
"""Q1 Navigator Node — drives the TurtleBot to a requested goal pose.

Implements a pure-pursuit controller with RRT path planning over a static
occupancy grid built from the known scene geometry (walls + tables).

Pipeline:
  1. OCCUPANCY MAPPING: The grid (5 cm cells, 4 m × 4 m window) is populated
     once at startup from the scene's walls and table footprints.
  2. PATH PLANNING: When a new goal arrives, an RRT planner searches the
     inflated grid (robot radius = 0.12 m) for a collision-free path, then
     shortcuts redundant waypoints via line-of-sight simplification.
  3. PATH FOLLOWING: A pure-pursuit controller (Coulter's algorithm) follows
     the planned path.  On reaching the final waypoint, a proportional
     controller rotates to the goal yaw.

Subscribes:
  /q1/robot_pose         geometry_msgs/Pose2D
  /q1/navigation_goal    geometry_msgs/Pose2D

Publishes:
  /cmd_vel               geometry_msgs/Twist
  /q1/navigation_status  std_msgs/String  ("navigating" | "reached" | "idle")

This node is PROVIDED and should NOT be modified.
"""

import math

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose2D, Twist
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from skills_utils import FAST_QoS, RELIABLE_QoS, wrap_angle

# ── Occupancy grid constants ──────────────────────────────────────────────────
_GRID_RES    = 0.05      # m per cell
_GRID_ORIGIN = -2.0      # world coord of cell index 0 (both axes)
_GRID_SIZE   = 80        # cells per axis  →  4 m × 4 m window
_OCC_THRESH  = 2         # hit count to mark a cell occupied

# ── Robot geometry ────────────────────────────────────────────────────────────
_ROBOT_RADIUS   = 0.12   # m — inflation radius for collision checks (robot ~0.10 m + small margin)


def _world_to_cell(x: float, y: float):
    ci = int((x - _GRID_ORIGIN) / _GRID_RES)
    cj = int((y - _GRID_ORIGIN) / _GRID_RES)
    return ci, cj


def _cell_to_world(ci: int, cj: int):
    x = _GRID_ORIGIN + (ci + 0.5) * _GRID_RES
    y = _GRID_ORIGIN + (cj + 0.5) * _GRID_RES
    return x, y


class NavigatorNode(Node):

    def __init__(self):
        super().__init__("navigator_node")

        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("goal_tolerance_m", 0.15)
        self.declare_parameter("goal_tolerance_rad", 0.3)
        self.declare_parameter("max_linear_vel", 1.5)
        self.declare_parameter("max_angular_vel", 3.0)
        self.declare_parameter("kp_angular", 4.0)
        self.declare_parameter("waypoint_tolerance_m", 0.10)
        self.declare_parameter("lookahead_m", 0.20)

        self.goal_tol_m     = self.get_parameter("goal_tolerance_m").value
        self.goal_tol_rad   = self.get_parameter("goal_tolerance_rad").value
        self.max_lin        = self.get_parameter("max_linear_vel").value
        self.max_ang        = self.get_parameter("max_angular_vel").value
        self.kp_ang         = self.get_parameter("kp_angular").value
        self.wp_tol_m       = self.get_parameter("waypoint_tolerance_m").value
        self._lookahead     = self.get_parameter("lookahead_m").value
        rate                = self.get_parameter("control_rate_hz").value

        self.robot_pose = None
        self.goal       = None
        self.status     = "idle"

        # Path: list of (x, y) tuples
        self._path     = []
        self._path_idx = 0
        self._raw_rrt_path = []   # unsmoothed RRT path for visualization

        self._heading_aligned = False  # True once initial rotate-to-face is done for current node
        self._prev_vx = 0.0   # for velocity smoothing
        self._prev_wz = 0.0   # for angular velocity smoothing

        # Occupancy grid (hit counter per cell)
        self._occ_grid = np.zeros((_GRID_SIZE, _GRID_SIZE), dtype=np.int16)
        # Pre-computed inflation kernel radius in cells
        self._infl_cells = max(1, int(math.ceil(_ROBOT_RADIUS / _GRID_RES)))

        # Pre-populate grid with known static geometry (walls, tables)
        self._init_static_map()

        self.create_subscription(Pose2D, "/q1/robot_pose",      self._pose_cb, FAST_QoS)
        self.create_subscription(Pose2D, "/q1/navigation_goal", self._goal_cb, FAST_QoS)

        self.cmd_pub    = self.create_publisher(Twist,         "/cmd_vel",               FAST_QoS)
        self.status_pub = self.create_publisher(String,        "/q1/navigation_status",  FAST_QoS)
        self.occ_pub    = self.create_publisher(OccupancyGrid, "/q1/occupancy_grid",     RELIABLE_QoS)
        self.rrt_marker_pub = self.create_publisher(MarkerArray, "/q1/rrt_path",         RELIABLE_QoS)

        self.create_timer(1.0 / rate, self._control_loop)
        self.create_timer(2.0,        self._publish_occ_grid)
        self.get_logger().info("Navigator node started (RRT + pure-pursuit).")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _publish_rrt_markers(self):
        """Publish both raw RRT path (red) and smoothed path (blue)."""
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()

        # Delete all previous markers
        for ns in ["rrt_raw", "rrt_raw_wp", "rrt_smooth", "rrt_smooth_wp"]:
            d = Marker()
            d.ns = ns
            d.action = Marker.DELETEALL
            arr.markers.append(d)

        # ── Raw RRT path (red, thinner) ──
        if self._raw_rrt_path and len(self._raw_rrt_path) > 1:
            line = Marker()
            line.header.frame_id = "map"
            line.header.stamp = now
            line.ns = "rrt_raw"
            line.id = 0
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.015
            line.color.r, line.color.g, line.color.b = 1.0, 0.2, 0.2
            line.color.a = 0.7
            line.pose.orientation.w = 1.0
            for wp in self._raw_rrt_path:
                p = Point()
                p.x, p.y, p.z = float(wp[0]), float(wp[1]), 0.04
                line.points.append(p)
            arr.markers.append(line)

            for i, wp in enumerate(self._raw_rrt_path):
                s = Marker()
                s.header.frame_id = "map"
                s.header.stamp = now
                s.ns = "rrt_raw_wp"
                s.id = i
                s.type = Marker.SPHERE
                s.action = Marker.ADD
                s.pose.position.x = float(wp[0])
                s.pose.position.y = float(wp[1])
                s.pose.position.z = 0.04
                s.pose.orientation.w = 1.0
                s.scale.x = s.scale.y = s.scale.z = 0.04
                s.color.r, s.color.g, s.color.b = 1.0, 0.2, 0.2
                s.color.a = 0.7
                arr.markers.append(s)

        # ── Smoothed path (blue, thicker) ──
        if self._path and len(self._path) > 1:
            line = Marker()
            line.header.frame_id = "map"
            line.header.stamp = now
            line.ns = "rrt_smooth"
            line.id = 0
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.03
            line.color.r, line.color.g, line.color.b = 0.0, 0.5, 1.0
            line.color.a = 0.9
            line.pose.orientation.w = 1.0
            for wp in self._path:
                p = Point()
                p.x, p.y, p.z = float(wp[0]), float(wp[1]), 0.06
                line.points.append(p)
            arr.markers.append(line)

            for i, wp in enumerate(self._path):
                s = Marker()
                s.header.frame_id = "map"
                s.header.stamp = now
                s.ns = "rrt_smooth_wp"
                s.id = i
                s.type = Marker.SPHERE
                s.action = Marker.ADD
                s.pose.position.x = float(wp[0])
                s.pose.position.y = float(wp[1])
                s.pose.position.z = 0.06
                s.pose.orientation.w = 1.0
                s.scale.x = s.scale.y = s.scale.z = 0.06
                s.color.r, s.color.g, s.color.b = 0.0, 0.5, 1.0
                s.color.a = 0.9
                arr.markers.append(s)

        self.rrt_marker_pub.publish(arr)

    def _pose_cb(self, msg: Pose2D):
        self.robot_pose = msg

    def _goal_cb(self, msg: Pose2D):
        self.goal   = msg
        self.status = "navigating"
        self._heading_aligned = False
        self.get_logger().info(
            f"New navigation goal: ({msg.x:.2f}, {msg.y:.2f}, "
            f"{math.degrees(msg.theta):.1f} deg)")

        if self.robot_pose is None:
            self._path     = [np.array([msg.x, msg.y])]
            self._path_idx = 0
            return

        start   = np.array([self.robot_pose.x, self.robot_pose.y])
        goal_xy = np.array([msg.x, msg.y])

        path = self._rrt_plan(start, goal_xy)
        if path is not None:
            self._path     = path
            self._path_idx = 0
            self.get_logger().info(f"RRT path found with {len(path)} waypoints.")
        else:
            self._path     = []
            self._path_idx = 0
            self.status    = "idle"
            self.get_logger().error("RRT failed — stopping (no fallback).")
        self._publish_rrt_markers()

    # ── Occupancy helpers ─────────────────────────────────────────────────────

    def _build_inflated_grid(self) -> np.ndarray:
        """Return a boolean array: True where the robot centre would collide."""
        binary = (self._occ_grid >= _OCC_THRESH).astype(np.uint8)
        # Dilate using a square kernel via cumulative sums (fast, no scipy needed)
        k = self._infl_cells
        n = _GRID_SIZE
        padded = np.pad(binary, k, mode="edge")
        # 2-D box-sum via integral image.
        # Prepend a zero row+col so boundary indexing is clean (no off-by-one).
        integral = padded.cumsum(axis=0).cumsum(axis=1)
        integral_ext = np.zeros((n + 2 * k + 1, n + 2 * k + 1), dtype=np.int64)
        integral_ext[1:, 1:] = integral
        s = 2 * k   # half-kernel span
        inflated = (
            integral_ext[s + 1:s + n + 1, s + 1:s + n + 1]
            - integral_ext[s + 1:s + n + 1, :n]
            - integral_ext[:n, s + 1:s + n + 1]
            + integral_ext[:n, :n]
        ) > 0
        return inflated

    def _line_free(self, inflated: np.ndarray,
                   a: np.ndarray, b: np.ndarray) -> bool:
        """Return True if the segment a→b is free in the inflated grid."""
        dist      = float(np.linalg.norm(b - a))
        n_samples = max(2, int(dist / (_GRID_RES * 0.25)))
        ts        = np.linspace(0.0, 1.0, n_samples)
        xs        = a[0] + ts * (b[0] - a[0])
        ys        = a[1] + ts * (b[1] - a[1])
        cis       = np.floor((xs - _GRID_ORIGIN) / _GRID_RES).astype(int)
        cjs       = np.floor((ys - _GRID_ORIGIN) / _GRID_RES).astype(int)
        in_bounds = (
            (cis >= 0) & (cis < _GRID_SIZE) &
            (cjs >= 0) & (cjs < _GRID_SIZE)
        )
        if not in_bounds.all():
            return False
        return not inflated[cis, cjs].any()

    # ── RRT planner ────────────────────────────────────────────────────────────

    _RRT_MAX_ITER  = 5000
    _RRT_STEP_SIZE = 0.10   # m — step size for each RRT extension (finer = safer around obstacles)
    _RRT_GOAL_BIAS = 0.15   # probability of sampling the goal directly

    def _rrt_plan(self, start: np.ndarray,
                  goal: np.ndarray) -> list[np.ndarray] | None:
        """RRT planner: returns a smoothed collision-free path, or None."""
        inflated = self._build_inflated_grid()

        # Check start/goal feasibility
        si, sj = _world_to_cell(start[0], start[1])
        gi, gj = _world_to_cell(goal[0], goal[1])
        self.get_logger().info(
            f"RRT: start=({start[0]:.2f},{start[1]:.2f}) cell=({si},{sj})  "
            f"goal=({goal[0]:.2f},{goal[1]:.2f}) cell=({gi},{gj})")
        if not (0 <= si < _GRID_SIZE and 0 <= sj < _GRID_SIZE):
            self.get_logger().error(f"RRT: start cell ({si},{sj}) out of grid bounds")
            return None
        if inflated[si, sj]:
            self.get_logger().warn(f"RRT: start cell ({si},{sj}) is in collision, searching nearby free cell")
            found = False
            for radius in range(1, 15):
                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        ni, nj = si + di, sj + dj
                        if 0 <= ni < _GRID_SIZE and 0 <= nj < _GRID_SIZE:
                            if not inflated[ni, nj]:
                                start = np.array(_cell_to_world(ni, nj))
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
            if not found:
                self.get_logger().error(f"RRT: no free cell near start")
                return None
        if not (0 <= gi < _GRID_SIZE and 0 <= gj < _GRID_SIZE):
            self.get_logger().error(f"RRT: goal cell ({gi},{gj}) out of grid bounds")
            return None

        # If goal is in collision, find nearest free cell
        if inflated[gi, gj]:
            self.get_logger().warn(f"RRT: goal in collision, searching for nearby free cell")
            found = False
            for radius in range(1, 15):
                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        ni, nj = gi + di, gj + dj
                        if 0 <= ni < _GRID_SIZE and 0 <= nj < _GRID_SIZE:
                            if not inflated[ni, nj]:
                                goal = np.array(_cell_to_world(ni, nj))
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
            if not found:
                return None

        # RRT tree: list of nodes, parent indices
        tree_nodes   = [start.copy()]
        tree_parents = [-1]

        rng = np.random.default_rng(42)

        x_min, x_max = -1.4, 1.4
        y_min, y_max = -1.4, 1.4

        for _ in range(self._RRT_MAX_ITER):
            # Sample random point (with goal bias)
            if rng.random() < self._RRT_GOAL_BIAS:
                sample = goal.copy()
            else:
                sample = np.array([
                    rng.uniform(x_min, x_max),
                    rng.uniform(y_min, y_max),
                ])

            # Find nearest node in tree
            nodes_arr = np.array(tree_nodes)
            dists = np.linalg.norm(nodes_arr - sample, axis=1)
            nearest_idx = int(np.argmin(dists))
            nearest = tree_nodes[nearest_idx]

            # Steer toward sample
            direction = sample - nearest
            dist = float(np.linalg.norm(direction))
            if dist < 1e-6:
                continue
            direction = direction / dist
            step = min(self._RRT_STEP_SIZE, dist)
            new_node = nearest + direction * step

            # Check collision along the segment
            if not self._line_free(inflated, nearest, new_node):
                continue

            tree_nodes.append(new_node)
            tree_parents.append(nearest_idx)

            # Check if we reached the goal
            if np.linalg.norm(new_node - goal) < self._RRT_STEP_SIZE:
                # Connect to exact goal if line is free
                if self._line_free(inflated, new_node, goal):
                    tree_nodes.append(goal.copy())
                    tree_parents.append(len(tree_nodes) - 2)

                    # Extract path
                    path = []
                    idx = len(tree_nodes) - 1
                    while idx != -1:
                        path.append(tree_nodes[idx])
                        idx = tree_parents[idx]
                    path.reverse()

                    # Store raw path for visualization, return smoothed
                    self._raw_rrt_path = [p.copy() for p in path]
                    smoothed = self._smooth_path(path, inflated)
                    return smoothed

        self.get_logger().error(
            f"RRT: exhausted {self._RRT_MAX_ITER} iterations, "
            f"tree has {len(tree_nodes)} nodes")
        return None  # failed to find path

    def _smooth_path(self, path: list[np.ndarray],
                     inflated: np.ndarray) -> list[np.ndarray]:
        """Iteratively smooth an RRT path using shortcutting + line-of-sight."""
        if len(path) <= 2:
            return path

        # Pass 1: Line-of-sight simplification (greedy from start)
        simplified = [path[0]]
        i = 0
        while i < len(path) - 1:
            furthest = i + 1
            for j in range(len(path) - 1, i, -1):
                if self._line_free(inflated, path[i], path[j]):
                    furthest = j
                    break
            simplified.append(path[furthest])
            i = furthest

        # Pass 2: Random shortcutting (50 iterations)
        rng = np.random.default_rng(42)
        for _ in range(50):
            if len(simplified) <= 2:
                break
            i = rng.integers(0, len(simplified) - 2)
            j = rng.integers(i + 2, len(simplified))
            if self._line_free(inflated, simplified[i], simplified[j]):
                simplified = simplified[:i + 1] + simplified[j:]

        return simplified

    # ── Static map initialisation ─────────────────────────────────────────────

    def _init_static_map(self):
        """Pre-populate the occupancy grid with known static obstacles from the
        scene XML (walls, table footprints)."""

        def mark_box(cx: float, cy: float, hx: float, hy: float,
                     pad: float = _GRID_RES):
            """Mark all grid cells inside the axis-aligned box [cx±hx, cy±hy]."""
            x_min = cx - hx - pad;  x_max = cx + hx + pad
            y_min = cy - hy - pad;  y_max = cy + hy + pad
            ci0 = max(0, int((x_min - _GRID_ORIGIN) / _GRID_RES))
            ci1 = min(_GRID_SIZE - 1, int((x_max - _GRID_ORIGIN) / _GRID_RES))
            cj0 = max(0, int((y_min - _GRID_ORIGIN) / _GRID_RES))
            cj1 = min(_GRID_SIZE - 1, int((y_max - _GRID_ORIGIN) / _GRID_RES))
            self._occ_grid[ci0:ci1 + 1, cj0:cj1 + 1] = 100

        # Outer room walls  (center_x, center_y, half_x, half_y)
        mark_box( 0.0,  1.5, 1.54, 0.04)   # north
        mark_box( 0.0, -1.5, 1.54, 0.04)   # south
        mark_box( 1.5,  0.0, 0.04, 1.54)   # east
        mark_box(-1.5,  0.0, 0.04, 1.54)   # west

        # Internal partition (horizontal, center)
        mark_box( 0.0,  0.0, 0.50, 0.04)   # partition_1

        # Table footprints (use tabletop half-sizes)
        mark_box( 0.85,  0.85, 0.22, 0.22)   # table_1 (NE)
        mark_box(-0.85, -0.85, 0.25, 0.20)   # table_2 (SW)
        mark_box(-0.85,  0.85, 0.20, 0.22)   # table_3 (NW)

        self.get_logger().info("Static map initialised from scene geometry.")

    # ── Occupancy grid publisher ───────────────────────────────────────────────

    def _publish_occ_grid(self):
        """Publish the current occupancy grid as nav_msgs/OccupancyGrid for RViz."""
        msg = OccupancyGrid()
        msg.header.stamp     = self.get_clock().now().to_msg()
        msg.header.frame_id  = "map"
        msg.info.resolution  = _GRID_RES
        msg.info.width       = _GRID_SIZE
        msg.info.height      = _GRID_SIZE
        msg.info.origin.position.x    = _GRID_ORIGIN
        msg.info.origin.position.y    = _GRID_ORIGIN
        msg.info.origin.orientation.w = 1.0
        # Convert hit counts → 0 (free) / 100 (occupied)
        # Grid is indexed [ci, cj] = [x_cell, y_cell].
        # OccupancyGrid data is row-major: data[j * W + i]  →  transpose first.
        binary = ((self._occ_grid >= _OCC_THRESH).astype(np.int8) * 100)
        msg.data = binary.T.flatten().tolist()
        self.occ_pub.publish(msg)

    # ── Main control loop ────────────────────────────────────────────────
    #
    # Per-waypoint cycle:
    #   1. Rotate in place to face the next waypoint
    #   2. Drive using pure pursuit (A2 Coulter)
    #   3. Advance when close enough or when the robot crosses the waypoint
    # After the final waypoint, rotate in place to match the goal theta.

    def _control_loop(self):
        # Always publish status
        status_msg      = String()
        status_msg.data = self.status
        self.status_pub.publish(status_msg)

        if self.robot_pose is None or self.goal is None:
            return
        if self.status != "navigating":
            return
        if not self._path:
            return

        rx   = self.robot_pose.x
        ry   = self.robot_pose.y
        ryaw = self.robot_pose.theta

        # Per-node cycle: rotate to face node → pure pursuit to node → rotate to next
        if self._path_idx < len(self._path):
            wp = self._path[self._path_idx]
            wx, wy = float(wp[0]), float(wp[1])
            dx = wx - rx
            dy = wy - ry
            dist = math.sqrt(dx * dx + dy * dy)

            # Reached this node
            if dist < self.wp_tol_m:
                # Rotate toward next node before advancing
                if self._path_idx + 1 < len(self._path):
                    nwp = self._path[self._path_idx + 1]
                    next_heading = math.atan2(float(nwp[1]) - ry, float(nwp[0]) - rx)
                    next_h_err = wrap_angle(next_heading - ryaw)
                    if abs(next_h_err) > 0.05:
                        cmd = Twist()
                        cmd.angular.z = float(np.clip(
                            self.kp_ang * next_h_err, -self.max_ang, self.max_ang))
                        self.cmd_pub.publish(cmd)
                        return
                self._path_idx += 1
                self._heading_aligned = False  # require fresh alignment for next node
                return

            # One-time rotate to face current node before driving
            if not self._heading_aligned and dist > 0.15:
                desired_heading = math.atan2(dy, dx)
                h_err = wrap_angle(desired_heading - ryaw)
                if abs(h_err) > 0.15:
                    cmd = Twist()
                    cmd.angular.z = float(np.clip(
                        self.kp_ang * h_err, -self.max_ang, self.max_ang))
                    self.cmd_pub.publish(cmd)
                    return
                self._heading_aligned = True  # aligned — let pure pursuit take over

            # Passed-waypoint detection: robot has crossed the perpendicular at wp
            # along the segment direction (prev_wp → wp). Robust to lateral offset.
            if self._path_idx > 0:
                prev_wp = self._path[self._path_idx - 1]
                seg_x = wx - float(prev_wp[0])
                seg_y = wy - float(prev_wp[1])
                tor_x = rx - wx
                tor_y = ry - wy
                if seg_x * tor_x + seg_y * tor_y > 0.0:
                    self._path_idx += 1
                    self._heading_aligned = False
                    return

            # Pure pursuit drive toward current node
            self._drive_pure_pursuit(rx, ry, ryaw, wx, wy)
            return

        # All nodes done — rotate to match goal orientation
        yaw_err = wrap_angle(self.goal.theta - ryaw)
        if abs(yaw_err) > self.goal_tol_rad:
            cmd = Twist()
            cmd.angular.z = float(np.clip(
                self.kp_ang * yaw_err, -self.max_ang, self.max_ang))
            self.cmd_pub.publish(cmd)
            return

        # Goal reached
        self.status = "reached"
        self.cmd_pub.publish(Twist())
        self.get_logger().info("Navigation goal reached!")

    # ── Drive: A2-style pure pursuit (Coulter) ────────────────────────

    def _drive_pure_pursuit(self, rx, ry, ryaw, gx, gy):
        """Pure pursuit toward a single target point (gx, gy)."""
        # Curvature command (A2 Coulter formula)
        heading = math.atan2(gy - ry, gx - rx)
        h_err = wrap_angle(heading - ryaw)
        lookahead = max(self._lookahead, 0.35)  # floor to avoid over-sensitive steering
        curvature = 2.0 * math.sin(h_err) / lookahead

        # Decelerate near the target node
        dist = math.sqrt((gx - rx)**2 + (gy - ry)**2)
        decel_gain = 4.0
        vx = min(self.max_lin, decel_gain * dist)
        vx = max(vx, 0.02)
        wz = vx * curvature
        wz = float(np.clip(wz, -self.max_ang, self.max_ang))

        # Exponential smoothing to avoid jerky commands
        alpha = 0.4  # 0 = fully smooth, 1 = no smoothing
        vx = alpha * vx + (1.0 - alpha) * self._prev_vx
        wz = alpha * wz + (1.0 - alpha) * self._prev_wz
        self._prev_vx = vx
        self._prev_wz = wz

        cmd = Twist()
        cmd.linear.x = vx
        cmd.angular.z = wz
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = NavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
