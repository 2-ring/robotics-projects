#!/usr/bin/env python3
"""Learn-from-demonstration puck-pushing policy with mode-clustered ProMPs.

Pipeline:

  1. Load recorded demos; resample tip trajectories to T = num_steps.
  2. Label each demo as a "left" or "right" approach mode using the sign of
     the 2D cross product between (puck → goal) and (puck → midpoint).
     Without this split, averaging across modes produces a mean trajectory
     that crosses through the obstacle on the line.
  3. Per mode, fit each demo's tip trajectory to a 12-RBF basis via ridge
     regression and estimate a Gaussian over weights N(μ_w, Σ_w).
  4. At episode start: pick a mode (puck/goal geometry), then condition the
     ProMP on two via-points — contact (tip at the puck) and end (tip at
     the goal) — using the closed-form ProMP posterior. The Z heights for
     each via-point come from cached per-mode averages (planar task).
  5. Convert tip-space waypoints → hand-space via the known weld transform,
     run damped-LS IK each step, and publish joint targets at 50 Hz.
"""

from __future__ import annotations

import glob
from pathlib import Path

import mujoco
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray

from skills_utils import FAST_QoS


CONTROL_DT = 0.02
DEFAULT_NUM_STEPS = 180

# Recorded ee_position is offset from the Panda hand body origin — empirically
# the recording sat 1 cm forward and 3 cm down from the hand frame.
EMPIRICAL_RECORDED_EE_TO_HAND_LOCAL = np.array(
    [0.0100017566, 0.0, -0.0299996650], dtype=float
)

# Pusher tip in the hand body's local frame, derived from the weld relpose
# (0,0,0.1034) with quat (0.5,0.5,0.5,0.5) and pusher_tip_site at (0,0,0.135).
TIP_OFFSET_IN_HAND = np.array([0.135, 0.0, 0.1034], dtype=float)


def _normalize_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=float)
    norm = float(np.linalg.norm(quat_wxyz))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat_wxyz / norm


def quat_to_rot(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def rot_to_axis_angle(rot: np.ndarray) -> np.ndarray:
    cos_theta = float(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.zeros(3)
    axis = np.array([
        rot[2, 1] - rot[1, 2],
        rot[0, 2] - rot[2, 0],
        rot[1, 0] - rot[0, 1],
    ]) / (2.0 * np.sin(theta))
    return axis * theta


def solve_panda_ik(
    mj_model, mj_data, current_qpos, target_pos,
    target_quat_wxyz=None, body_name="hand",
    max_iters=200, damping=1e-3, pos_tol=5e-4, ori_tol=1e-2, max_step=0.3,
) -> np.ndarray | None:
    """Damped least-squares IK on position + (optional) orientation."""
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mj_data.qpos[:7] = np.asarray(current_qpos, dtype=float).copy()
    mujoco.mj_forward(mj_model, mj_data)

    target_rot = quat_to_rot(*target_quat_wxyz) if target_quat_wxyz else None
    target_pos = np.asarray(target_pos, dtype=float)

    for _ in range(max_iters):
        mujoco.mj_forward(mj_model, mj_data)
        hand_pos = mj_data.xpos[body_id].copy()
        rot_current = mj_data.xmat[body_id].reshape(3, 3)
        pos_err = target_pos - hand_pos

        jacp = np.zeros((3, mj_model.nv))
        jacr = np.zeros((3, mj_model.nv))
        mujoco.mj_jac(mj_model, mj_data, jacp, jacr, hand_pos, body_id)

        if target_rot is None:
            if np.linalg.norm(pos_err) < pos_tol:
                break
            jac = jacp[:, :7]
            jjt = jac @ jac.T + damping ** 2 * np.eye(3)
            dq = jac.T @ np.linalg.solve(jjt, pos_err)
        else:
            ori_err = rot_to_axis_angle(target_rot @ rot_current.T)
            if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(ori_err) < ori_tol:
                break
            err6 = np.concatenate([pos_err, ori_err])
            jac = np.vstack([jacp[:, :7], jacr[:, :7]])
            jjt = jac @ jac.T + damping ** 2 * np.eye(6)
            dq = jac.T @ np.linalg.solve(jjt, err6)

        norm = float(np.linalg.norm(dq))
        if norm > max_step:
            dq *= max_step / norm
        mj_data.qpos[:7] += dq

    return mj_data.qpos[:7].copy()


def find_demo_files() -> list[str]:
    src_demo_dir = Path(__file__).resolve().parents[1] / "demos"
    if src_demo_dir.is_dir():
        files = sorted(glob.glob(str(src_demo_dir / "demo_*.npz")))
        if files:
            return files
    q3_share = Path(get_package_share_directory("puck_pusher"))
    return sorted(glob.glob(str(q3_share / "demos" / "demo_*.npz")))


def resample_array(values: np.ndarray, num_steps: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == num_steps:
        return values.copy()
    if len(values) == 1:
        return np.repeat(values, num_steps, axis=0)
    old_t = np.linspace(0.0, 1.0, len(values))
    new_t = np.linspace(0.0, 1.0, num_steps)
    cols = [np.interp(new_t, old_t, values[:, i]) for i in range(values.shape[1])]
    return np.stack(cols, axis=1)


def resample_quaternions(quats: np.ndarray, num_steps: int) -> np.ndarray:
    quats = np.asarray(quats, dtype=float)
    interp = resample_array(quats, num_steps)
    return np.vstack([_normalize_quat(q) for q in interp])


def make_rbf_basis(num_steps: int, num_basis: int, width_scale: float = 1.5) -> np.ndarray:
    """Row-normalized Gaussian RBF basis (T, B) over phase ∈ [0, 1]."""
    phase = np.linspace(0.0, 1.0, num_steps)
    centers = np.linspace(0.0, 1.0, num_basis)
    width = width_scale / max(num_basis - 1, 1)
    basis = np.exp(-0.5 * ((phase[:, None] - centers[None, :]) / max(width, 1e-6)) ** 2)
    return basis / np.maximum(np.sum(basis, axis=1, keepdims=True), 1e-8)


def fit_linear_weights(basis: np.ndarray, trajectory: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Ridge regression: find w (B, D) so that y ≈ Phi @ w.

    Normal equations: (Phi^T Phi + ridge·I) w = Phi^T y. Solved with
    np.linalg.solve for numerical stability (no explicit inverse).
    """
    A = (basis.T @ basis) + (ridge * np.eye(12))
    b = basis.T @ trajectory
    return np.linalg.solve(A, b)


def load_demo_dict(path: str, num_steps: int) -> dict:
    with np.load(path) as data:
        ee_positions = np.asarray(data["ee_positions"], dtype=float)
        ee_orientations = np.asarray(data["ee_orientations"], dtype=float)

        hand_positions = []
        for ee_pos, ee_quat in zip(ee_positions, ee_orientations):
            rot = quat_to_rot(*_normalize_quat(ee_quat))
            hand_positions.append(ee_pos + rot @ EMPIRICAL_RECORDED_EE_TO_HAND_LOCAL)
        hand_positions = np.asarray(hand_positions, dtype=float)

        pusher_positions = np.asarray(data["pusher_positions"], dtype=float)

        return {
            "file": Path(path).name,
            "ee_orientations": ee_orientations,
            "pusher_positions_rs": resample_array(pusher_positions, num_steps),
            "puck_positions": np.asarray(data["puck_positions"], dtype=float),
            "goal_position": np.asarray(data["goal_position"], dtype=float),
            "puck_start": np.asarray(data["puck_positions"][0], dtype=float).copy(),
            "hand_positions_rs": resample_array(hand_positions, num_steps),
            "ee_orientations_rs": resample_quaternions(ee_orientations, num_steps),
        }


class LfDPolicy(Node):

    def __init__(self):
        super().__init__("lfd_policy")

        self.declare_parameter("num_resampled_steps", DEFAULT_NUM_STEPS)
        self.num_steps = int(self.get_parameter("num_resampled_steps").value)

        self.joint_positions = None
        self.puck_position = None
        self.goal_position = None
        self.episode_ready = True

        self.demos = self._load_demos()
        self.active_index = 0
        self.last_command = None
        self.active_hand_traj = None
        self.active_quat_traj = None

        self.create_subscription(Float64MultiArray, "/q3/joint_positions", self._jp_cb, FAST_QoS)
        self.create_subscription(Float64MultiArray, "/q3/puck_position", self._puck_cb, FAST_QoS)
        self.create_subscription(Float64MultiArray, "/q3/goal_position", self._goal_cb, FAST_QoS)
        self.create_subscription(Bool, "/q3/episode_ready", self._episode_ready_cb, FAST_QoS)

        self.target_pub = self.create_publisher(Float64MultiArray, "/panda/position_targets", FAST_QoS)

        q3_share = Path(get_package_share_directory("puck_pusher"))
        model_path = q3_share / "models" / "puck_pushing_scene.xml"
        self.mj_model = mujoco.MjModel.from_xml_path(str(model_path))
        self.mj_data = mujoco.MjData(self.mj_model)

        self.promp_basis = make_rbf_basis(self.num_steps, num_basis=12)
        self._fit_promp()

        self.create_timer(CONTROL_DT, self._control_loop)
        self.get_logger().info("LfD ProMP policy ready.")

    def _load_demos(self) -> list[dict]:
        demo_files = find_demo_files()
        if not demo_files:
            self.get_logger().error("No demonstration files found.")
            return []
        demos = [load_demo_dict(path, self.num_steps) for path in demo_files]
        self._label_demo_modes(demos)
        self.get_logger().info(f"Loaded {len(demos)} demos.")
        for d in demos:
            self.get_logger().info(f"  {d['file']}: {d.get('approach_side','?')}")
        return demos

    def _label_demo_modes(self, demos: list[dict]):
        """Classify each demo as a left or right approach via 2D cross product.

        With (puck → goal) = v1 and (puck → midpoint) = v2, the sign of
        cross_z = v1.x·v2.y - v1.y·v2.x tells us which side of the puck-goal
        line the trajectory curves around. Mixing both modes in a single
        Gaussian would produce a mean that crosses the obstacle.
        """
        for demo in demos:
            puck_start = demo["puck_start"]
            goal = demo["goal_position"]

            mid_index = len(demo["pusher_positions_rs"]) // 2
            traj_midpoint = demo["pusher_positions_rs"][mid_index]

            v1 = goal - puck_start
            v2 = traj_midpoint - puck_start
            cross_z = v1[0] * v2[1] - v1[1] * v2[0]

            demo["approach_side"] = "right" if cross_z <= 0 else "left"

    def _select_mode(self) -> str:
        """Pick "left" or "right" for the current episode.

        Any deterministic rule that commits to a single cluster is fine —
        averaging across modes is what we must avoid. Here we use the sign
        of the goal-to-puck Y delta.
        """
        if self.goal_position is None:
            return "left"
        return "left" if self.goal_position[1] < self.puck_position[1] else "right"

    def _fit_promp(self):
        """Fit a Gaussian over RBF weights per approach mode.

        For each mode: project every demo's tip trajectory onto the 12-RBF
        basis with ridge regression, then estimate (μ_w, Σ_w) where μ_w is
        the per-dimension mean and Σ_w is one (B, B) sample covariance per
        output dim. Single-demo modes fall back to a small diagonal so the
        posterior solve stays numerically stable.

        Cached: the average contact-step (where the tip is closest to the
        puck across all demos) and the per-mode average Z heights at
        contact and end — used as the via-point Z values during
        conditioning.
        """
        if not self.demos:
            return
        self.promp_modes = {}

        def contact_timestep(demo):
            dists = np.linalg.norm(
                demo["pusher_positions_rs"][:, :2] - demo["puck_start"][:2], axis=1
            )
            return int(np.argmin(dists))

        self.promp_contact_step = int(np.mean([contact_timestep(demo) for demo in self.demos]))

        for mode in ["left", "right"]:
            demos = [demo for demo in self.demos if demo["approach_side"] == mode]
            if not demos:
                continue

            end_zs = []
            contact_zs = []
            end_t = self.num_steps - 1
            for demo in demos:
                contact_t = contact_timestep(demo)
                points = demo["pusher_positions_rs"]
                end_zs.append(points[end_t][2])
                contact_zs.append(points[contact_t][2])

            trajectories = np.stack(
                [fit_linear_weights(self.promp_basis, d["pusher_positions_rs"]) for d in demos]
            )
            mu = np.mean(trajectories, axis=0)
            if len(trajectories) == 1:
                # Single-demo fallback so the posterior solve stays stable.
                sigma = [1e-4 * np.eye(12) for _ in range(3)]
            else:
                sigma = np.stack(
                    [np.cov(trajectories[:, :, d], rowvar=False) for d in range(3)]
                )

            self.promp_modes[mode] = {
                "mu": mu,
                "sigma": sigma,
                "contact_z": np.mean(contact_zs),
                "end_z": np.mean(end_zs),
            }

    def _predict_promp(self) -> np.ndarray:
        """ProMP posterior conditioned on contact + end via-points.

        For each output dim d, with prior w_d ~ N(μ_d, Σ_d) and observations
        y_obs at timesteps {contact, end}:

            S      = Φ_cond Σ_d Φ_condᵀ + σ_obs² I
            K      = Σ_d Φ_condᵀ S⁻¹
            μ_new  = μ_d + K (y_obs - Φ_cond μ_d)

        Returns the conditioned mean trajectory Φ @ μ_new of shape (T, 3).
        """
        traj_cond = np.zeros((self.num_steps, 3))

        prior = self.promp_modes[self._select_mode()]
        puck_x, puck_y, _ = self.puck_position
        goal_x, goal_y, _ = self.goal_position
        contact_xyz = np.array([puck_x, puck_y, prior["contact_z"]])
        end_xyz = np.array([goal_x, goal_y, prior["end_z"]])

        contact_t = self.promp_contact_step
        end_t = self.num_steps - 1
        basis_cond = self.promp_basis[[contact_t, end_t], :]  # (2, B)

        for d in range(3):
            y_obs = np.array([contact_xyz[d], end_xyz[d]])
            sigma_prior = prior["sigma"][d]
            mu_prior = prior["mu"][:, d]
            # Looser observation noise on Z since the task is essentially planar.
            sigma_obs = 1e-3 if d < 2 else 1e-2

            S = basis_cond @ sigma_prior @ basis_cond.T + sigma_obs**2 * np.eye(2)
            S_inv = np.linalg.solve(S, np.eye(2))
            K_gain = sigma_prior @ basis_cond.T @ S_inv
            mu_new = mu_prior + K_gain @ (y_obs - basis_cond @ mu_prior)
            traj_cond[:, d] = self.promp_basis @ mu_new

        return traj_cond

    def _tip_traj_to_hand_traj(self, tip_traj: np.ndarray,
                               quat_traj: np.ndarray) -> np.ndarray:
        """hand = tip - R_hand @ TIP_OFFSET_IN_HAND, per step."""
        hand_traj = np.zeros_like(tip_traj)
        for i in range(len(tip_traj)):
            R = quat_to_rot(*_normalize_quat(quat_traj[i]))
            hand_traj[i] = tip_traj[i] - R @ TIP_OFFSET_IN_HAND
        return hand_traj

    def _mean_orientation_trajectory(self) -> np.ndarray:
        all_quats = np.stack([d["ee_orientations_rs"] for d in self.demos], axis=0)
        mean_quats = np.mean(all_quats, axis=0)
        return np.vstack([_normalize_quat(q) for q in mean_quats])

    def _build_active_trajectory(self) -> bool:
        if not self.demos or self.puck_position is None or self.goal_position is None:
            return False

        self.active_quat_traj = self._mean_orientation_trajectory()
        tip_traj = self._predict_promp()
        self.active_hand_traj = self._tip_traj_to_hand_traj(
            tip_traj, self.active_quat_traj
        )

        self.active_index = 0
        self.get_logger().info(
            f"ProMP trajectory: {len(self.active_hand_traj)} steps, "
            f"mode={self._select_mode()}"
        )
        return True

    def _jp_cb(self, msg):
        self.joint_positions = np.array(msg.data, dtype=float)

    def _puck_cb(self, msg):
        self.puck_position = np.array(msg.data, dtype=float)

    def _goal_cb(self, msg):
        self.goal_position = np.array(msg.data, dtype=float)

    def _episode_ready_cb(self, msg: Bool):
        self.episode_ready = bool(msg.data)
        if not self.episode_ready:
            self.active_hand_traj = None
            self.active_quat_traj = None
            self.active_index = 0

    def _publish_joint_target(self, q_target: np.ndarray):
        msg = Float64MultiArray()
        msg.data = np.asarray(q_target, dtype=float).tolist()
        self.target_pub.publish(msg)
        self.last_command = np.asarray(q_target, dtype=float).copy()

    def _ik_target(self, index: int) -> np.ndarray | None:
        if self.last_command is None:
            if self.joint_positions is None:
                return None
            self.last_command = self.joint_positions.copy()

        target_pos = self.active_hand_traj[index]
        target_quat = _normalize_quat(self.active_quat_traj[index])
        return solve_panda_ik(
            self.mj_model, self.mj_data, self.last_command,
            target_pos, target_quat_wxyz=tuple(target_quat.tolist()),
            body_name="hand", max_iters=200,
        )

    def _control_loop(self):
        if self.joint_positions is None or self.puck_position is None or self.goal_position is None:
            return
        if not self.demos or not self.episode_ready:
            return

        if self.active_hand_traj is None:
            self.last_command = self.joint_positions.copy()
            if not self._build_active_trajectory():
                return

        if self.active_index >= len(self.active_hand_traj):
            if self.last_command is not None:
                self._publish_joint_target(self.last_command)
            return

        q_target = self._ik_target(self.active_index)
        if q_target is None:
            return
        self._publish_joint_target(q_target)
        self.active_index += 1


def main(args=None):
    rclpy.init(args=args)
    node = LfDPolicy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
