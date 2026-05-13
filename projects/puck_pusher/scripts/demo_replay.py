#!/usr/bin/env python3
"""Replay a recorded Q3 demonstration in simulation.

This script resets the scene to a chosen demo's initial state:
  - robot starting joint pose
  - puck starting pose
  - goal pose
  - obstacle at one-third between puck and goal

Then it replays the demo using one of two modes:
  - `joint`: publish recorded joint targets directly
  - `ik`: solve IK for each recorded EE pose/orientation and publish joints
"""

from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

from skills_utils import FAST_QoS


CONTROL_DT = 0.02
OBSTACLE_Z_M = 0.432
DEFAULT_DEMO_INDEX = 1
DEFAULT_REPLAY_MODE = "joint"
EMPIRICAL_RECORDED_EE_TO_HAND_LOCAL = np.array([0.0100017566, 0.0, -0.0299996650], dtype=float)
HAND_TO_PUSHER_BODY_POS = np.array([0.0, 0.0, 0.1034], dtype=float)
HAND_TO_PUSHER_BODY_QUAT = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)
PUSHER_TIP_LOCAL_POS = np.array([0.0, 0.0, 0.135], dtype=float)


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


EXPECTED_HAND_TO_PUSHER_TIP = (
    HAND_TO_PUSHER_BODY_POS
    + quat_to_rot(*HAND_TO_PUSHER_BODY_QUAT) @ PUSHER_TIP_LOCAL_POS
)


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
    mj_model,
    mj_data,
    current_qpos: np.ndarray,
    target_pos: np.ndarray,
    target_quat_wxyz: tuple | None = None,
    body_name: str = "hand",
    site_name: str | None = None,
    max_iters: int = 200,
    damping: float = 1e-3,
    pos_tol: float = 5e-4,
    ori_tol: float = 1e-2,
    max_step: float = 0.3,
) -> np.ndarray | None:
    """IK that can track either a body origin or a fixed site on that body."""
    if mj_model is None or current_qpos is None:
        return None

    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    site_id = -1
    if site_name is not None:
        site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    mj_data.qpos[:7] = np.asarray(current_qpos, dtype=float).copy()
    mujoco.mj_forward(mj_model, mj_data)

    target_rot = None
    if target_quat_wxyz is not None:
        target_rot = quat_to_rot(*target_quat_wxyz)

    target_pos = np.asarray(target_pos, dtype=float)

    for _ in range(max_iters):
        mujoco.mj_forward(mj_model, mj_data)
        hand_pos = mj_data.xpos[body_id].copy()
        rot_current = mj_data.xmat[body_id].reshape(3, 3)
        tracked_pos = hand_pos
        if site_id >= 0:
            tracked_pos = mj_data.site_xpos[site_id].copy()
        pos_err = target_pos - tracked_pos

        jacp = np.zeros((3, mj_model.nv))
        jacr = np.zeros((3, mj_model.nv))
        if site_id >= 0:
            mujoco.mj_jacSite(mj_model, mj_data, jacp, jacr, site_id)
        else:
            mujoco.mj_jac(mj_model, mj_data, jacp, jacr, hand_pos, body_id)

        if target_rot is None:
            if np.linalg.norm(pos_err) < pos_tol:
                break
            jac = jacp[:, :7]
            jjt = jac @ jac.T + damping * damping * np.eye(3)
            dq = jac.T @ np.linalg.solve(jjt, pos_err)
        else:
            rot_err = target_rot @ rot_current.T
            ori_err = rot_to_axis_angle(rot_err)
            if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(ori_err) < ori_tol:
                break
            err6 = np.concatenate([pos_err, ori_err])
            jac = np.vstack([jacp[:, :7], jacr[:, :7]])
            jjt = jac @ jac.T + damping * damping * np.eye(6)
            dq = jac.T @ np.linalg.solve(jjt, err6)

        norm = float(np.linalg.norm(dq))
        if norm > max_step:
            dq = dq * (max_step / norm)
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


class DemoReplay(Node):

    def __init__(self):
        super().__init__("demo_replay")

        self.declare_parameter("demo_index", DEFAULT_DEMO_INDEX)
        self.declare_parameter("replay_mode", DEFAULT_REPLAY_MODE)
        self.demo_index = int(self.get_parameter("demo_index").value)
        self.replay_mode = str(self.get_parameter("replay_mode").value).strip().lower()
        if self.replay_mode not in {"joint", "ik"}:
            raise ValueError("replay_mode must be 'joint' or 'ik'")

        self.joint_positions = None
        self.replay_started = False
        self.replay_index = 0
        self.last_q = None

        demo_files = find_demo_files()
        if not demo_files:
            raise RuntimeError("No Q3 demos found.")
        if not (1 <= self.demo_index <= len(demo_files)):
            raise ValueError(f"demo_index must be in [1, {len(demo_files)}]")

        self.demo_path = demo_files[self.demo_index - 1]
        with np.load(self.demo_path) as data:
            self.demo = {
                "file": Path(self.demo_path).name,
                "joint_positions": data["joint_positions"].copy(),
                "ee_positions": data["ee_positions"].copy(),
                "ee_orientations": data["ee_orientations"].copy(),
                "pusher_positions": data["pusher_positions"].copy(),
                "puck_positions": data["puck_positions"].copy(),
                "goal_position": data["goal_position"].copy(),
                "timestamps": data["timestamps"].copy(),
            }

        q3_share = Path(get_package_share_directory("puck_pusher"))
        model_path = q3_share / "models" / "puck_pushing_scene.xml"
        self.mj_model = mujoco.MjModel.from_xml_path(str(model_path))
        self.mj_data = mujoco.MjData(self.mj_model)

        self.create_subscription(
            Float64MultiArray, "/q3/joint_positions", self._jp_cb, FAST_QoS)
        self.target_pub = self.create_publisher(
            Float64MultiArray, "/panda/position_targets", FAST_QoS)
        self.body_pose_pub = self.create_publisher(
            String, "/mujoco/body_pose", FAST_QoS)
        self.mocap_pub = self.create_publisher(
            Float64MultiArray, "/mujoco/mocap_pos", FAST_QoS)
        self.reset_client = self.create_client(Trigger, "/mujoco/reset")

        self.create_timer(CONTROL_DT, self._control_loop)

        self.get_logger().info(
            f"Loaded {self.demo['file']} in replay_mode={self.replay_mode}.")
        self._report_demo_consistency()
        self._reset_to_demo_start()

    def _report_demo_consistency(self):
        ee_positions = self.demo["ee_positions"]
        ee_orientations = self.demo["ee_orientations"]
        pusher_positions = self.demo["pusher_positions"]

        inferred_offsets = []
        predicted_errors = []
        predicted_hand_errors = []

        for ee_pos, ee_quat, pusher_pos in zip(ee_positions, ee_orientations, pusher_positions):
            rot = quat_to_rot(*_normalize_quat(ee_quat))
            inferred_local = rot.T @ (pusher_pos - ee_pos)
            inferred_offsets.append(inferred_local)

            predicted_pusher = ee_pos + rot @ EXPECTED_HAND_TO_PUSHER_TIP
            predicted_errors.append(np.linalg.norm(predicted_pusher - pusher_pos))

            predicted_hand = pusher_pos - rot @ EXPECTED_HAND_TO_PUSHER_TIP
            predicted_hand_errors.append(np.linalg.norm(predicted_hand - ee_pos))

        inferred_offsets = np.asarray(inferred_offsets, dtype=float)
        predicted_errors = np.asarray(predicted_errors, dtype=float)
        predicted_hand_errors = np.asarray(predicted_hand_errors, dtype=float)

        self.get_logger().info(
            f"Expected hand->pusher_tip local offset from XML: {EXPECTED_HAND_TO_PUSHER_TIP.tolist()}")
        self.get_logger().info(
            "Recorded inferred hand->pusher_tip local offset: "
            f"mean={np.mean(inferred_offsets, axis=0).tolist()}, "
            f"std={np.std(inferred_offsets, axis=0).tolist()}")
        self.get_logger().info(
            "Pusher prediction error from recorded ee pose: "
            f"mean={predicted_errors.mean():.6f} m, max={predicted_errors.max():.6f} m")
        self.get_logger().info(
            "Hand back-computation error from recorded pusher pose: "
            f"mean={predicted_hand_errors.mean():.6f} m, max={predicted_hand_errors.max():.6f} m")

    def _jp_cb(self, msg):
        self.joint_positions = np.array(msg.data, dtype=float)

    def _wait_for_state(self, timeout_sec=3.0):
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.joint_positions is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return False

    def _wait_for_home(self, target_q: np.ndarray, timeout_sec=4.0, pos_tol=0.03):
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.joint_positions is None:
                continue
            if np.linalg.norm(self.joint_positions[:7] - target_q[:7]) <= pos_tol:
                return True
        return False

    def _reset_to_demo_start(self):
        start_q = self.demo["joint_positions"][0]
        start_puck = self.demo["puck_positions"][0]
        goal = self.demo["goal_position"]
        obstacle = np.array([
            start_puck[0] + (goal[0] - start_puck[0]) / 3.0,
            start_puck[1] + (goal[1] - start_puck[1]) / 3.0,
            OBSTACLE_Z_M,
        ], dtype=float)
        if not self.reset_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("Reset service not available.")
        future = self.reset_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is None or not future.result().success:
            raise RuntimeError("Failed to reset simulation.")
        self._wait_for_state()
        home_msg = Float64MultiArray()
        home_msg.data = start_q.tolist()
        self.target_pub.publish(home_msg)
        if not self._wait_for_home(np.asarray(start_q, dtype=float)):
            raise RuntimeError("Robot did not reach demo start pose before placing objects.")

        puck_msg = String()
        puck_msg.data = json.dumps({
            "body": "puck",
            "pos": start_puck.tolist(),
            "quat": [1.0, 0.0, 0.0, 0.0],
        })
        self.body_pose_pub.publish(puck_msg)

        goal_msg = Float64MultiArray()
        goal_msg.data = goal.tolist() + obstacle.tolist()
        self.mocap_pub.publish(goal_msg)
        time.sleep(0.1)

        self.last_q = np.array(start_q, dtype=float)
        self.replay_started = True
        self.replay_index = 0

    def _publish_joint_target(self, q_target: np.ndarray):
        msg = Float64MultiArray()
        msg.data = np.asarray(q_target, dtype=float).tolist()
        self.target_pub.publish(msg)
        self.last_q = np.asarray(q_target, dtype=float).copy()

    def _ik_target(self, index: int) -> np.ndarray | None:
        if self.last_q is None:
            return None
        target_ee = np.asarray(self.demo["ee_positions"][index], dtype=float)
        target_quat_arr = _normalize_quat(np.asarray(self.demo["ee_orientations"][index], dtype=float))
        target_rot = quat_to_rot(*target_quat_arr)
        corrected_hand_target = (
            target_ee + target_rot @ EMPIRICAL_RECORDED_EE_TO_HAND_LOCAL
        )
        return solve_panda_ik(
            self.mj_model,
            self.mj_data,
            self.last_q,
            corrected_hand_target,
            target_quat_wxyz=tuple(target_quat_arr.tolist()),
            body_name="hand",
            max_iters=200,
        )

    def _control_loop(self):
        if not self.replay_started:
            return
        if self.replay_index >= len(self.demo["joint_positions"]):
            if self.last_q is not None:
                self._publish_joint_target(self.last_q)
            return

        if self.replay_mode == "joint":
            q_target = self.demo["joint_positions"][self.replay_index]
        else:
            q_target = self._ik_target(self.replay_index)
            if q_target is None:
                return

        self._publish_joint_target(q_target)
        self.replay_index += 1


def main(args=None):
    rclpy.init(args=args)
    node = DemoReplay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
