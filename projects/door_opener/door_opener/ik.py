"""Shared inverse kinematics helper for Q2.

This module provides a single numerical IK solver (``solve_panda_ik``) used by
both ``grasp_controller`` and ``manipulation_planner`` so that the two nodes
always stay in sync. Students should call this helper rather than re-implement
IK from scratch.
"""

from __future__ import annotations

import numpy as np
import mujoco


# TCP (tool center point) offset from the ``hand`` body origin, expressed in
# the hand's local frame. This is approximately the point between the closed
# fingertips — the location the IK actually tries to position.
TCP_OFFSET_LOCAL = np.array([0.0, 0.0, 0.1034])


def quat_to_rot(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Convert a (w, x, y, z) quaternion to a 3x3 rotation matrix."""
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


def rot_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to an axis-angle vector.

    The returned 3-vector ``v`` satisfies ``|v| = theta`` (the rotation angle)
    and ``v/|v|`` is the rotation axis. This form is convenient for use as a
    6-DoF orientation error in Jacobian-based IK.
    """
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.zeros(3)
    if abs(theta - np.pi) < 1e-6:
        # Near pi — use alternative formula for numerical stability
        diag = np.maximum(np.diag(R), -1.0)
        axis = np.sqrt((diag + 1.0) * 0.5)
        if R[0, 1] < 0: axis[1] = -axis[1]
        if R[0, 2] < 0: axis[2] = -axis[2]
        return axis * theta
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(theta))
    return axis * theta


def solve_panda_ik(
    mj_model,
    mj_data,
    current_qpos: np.ndarray,
    target_pos: np.ndarray,
    target_quat_wxyz: tuple | None = None,
    body_name: str = "hand",
    max_iters: int = 200,
    damping: float = 1e-3,
    pos_tol: float = 5e-4,
    ori_tol: float = 1e-2,
    max_step: float = 0.3,
) -> np.ndarray | None:
    """Damped least-squares Jacobian IK for the 7-DoF Panda arm.

    Targets the TCP (between-fingertips) position in world frame, and
    optionally the hand orientation. When ``target_quat_wxyz`` is None the
    solver uses position-only IK (3-DoF error) — when provided it solves the
    full 6-DoF (position + orientation) problem.

    Args:
        mj_model: loaded ``mujoco.MjModel``.
        mj_data: matching ``mujoco.MjData``.
        current_qpos: initial 7-element arm joint configuration (used as the
            starting seed for the iterative solver).
        target_pos: desired TCP position in world frame, shape (3,).
        target_quat_wxyz: desired hand orientation as (w, x, y, z). If None,
            only position is solved.
        body_name: name of the MuJoCo body whose TCP is being driven.

    Returns:
        The 7-element joint configuration that places the TCP at the target,
        or None if ``mj_model`` / ``current_qpos`` were not supplied.
    """
    if mj_model is None or current_qpos is None:
        return None

    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    # Seed the solver from the current joint configuration.
    mj_data.qpos[:7] = np.asarray(current_qpos, dtype=float).copy()
    mujoco.mj_forward(mj_model, mj_data)

    R_target = None
    if target_quat_wxyz is not None:
        R_target = quat_to_rot(*target_quat_wxyz)

    target_pos = np.asarray(target_pos, dtype=float)

    for _ in range(max_iters):
        mujoco.mj_forward(mj_model, mj_data)

        # Current TCP position and orientation in world frame.
        hand_pos = mj_data.xpos[body_id].copy()
        R_current = mj_data.xmat[body_id].reshape(3, 3)
        tcp_pos = hand_pos + R_current @ TCP_OFFSET_LOCAL

        pos_err = target_pos - tcp_pos

        # Jacobian at the TCP world point (NOT the body origin).
        jacp = np.zeros((3, mj_model.nv))
        jacr = np.zeros((3, mj_model.nv))
        mujoco.mj_jac(mj_model, mj_data, jacp, jacr, tcp_pos, body_id)

        if R_target is None:
            # Position-only IK (3-DoF error).
            if np.linalg.norm(pos_err) < pos_tol:
                break
            J = jacp[:, :7]
            JJT = J @ J.T + damping * damping * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, pos_err)
        else:
            # Position + orientation IK (6-DoF error).
            R_err = R_target @ R_current.T
            ori_err = rot_to_axis_angle(R_err)

            if (np.linalg.norm(pos_err) < pos_tol and
                    np.linalg.norm(ori_err) < ori_tol):
                break

            err6 = np.concatenate([pos_err, ori_err])
            J = np.vstack([jacp[:, :7], jacr[:, :7]])  # 6x7
            JJT = J @ J.T + damping * damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJT, err6)

        # Clamp step size for stability.
        n = float(np.linalg.norm(dq))
        if n > max_step:
            dq = dq * (max_step / n)

        mj_data.qpos[:7] += dq

    return mj_data.qpos[:7].copy()
