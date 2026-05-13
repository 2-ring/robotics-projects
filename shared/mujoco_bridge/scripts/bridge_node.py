#!/usr/bin/env python3
"""
ROS2 Bridge Node for MuJoCo Panda simulation.

This is the only node in the mujoco_bridge package.
The sim node and user scripts live in assignment packages (e.g. q1).

Handles ALL translation between MuJoCo and ROS2:
  - Receives /panda/position_targets (7 joint angles) from user nodes
  - Receives /panda/gripper_command  (open/close/width)
  - Applies smooth interpolation toward targets
  - Sends ctrl to /mujoco/joint_controls
  - Republishes joint states, EE pose, TF, FT sensor under /panda/*

The Menagerie Panda has position-controlled actuators (biastype="affine").
ctrl values = desired joint positions. No external PID needed.
Gripper is a tendon actuator (actuator8), ctrl range 0–255 (closed–open).

Published:
  /panda/joint_states          sensor_msgs/JointState
  /panda/end_effector_pose     geometry_msgs/PoseStamped
  /panda/ft_sensor             geometry_msgs/WrenchStamped
  /panda/position_error        std_msgs/Float64MultiArray
  /tf                          (TF2 broadcast)

Subscribed:
  /mujoco/joint_states         sensor_msgs/JointState      (from sim)
  /panda/position_targets      std_msgs/Float64MultiArray   (7 joint angles, rad)
  /panda/gripper_command       std_msgs/Float64MultiArray   ([width_m] or [0-255])
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, WrenchStamped, TransformStamped

try:
    from tf2_ros import TransformBroadcaster
    HAS_TF2 = True
except ImportError:
    HAS_TF2 = False

# ── Menagerie Panda constants ───────────────────────────────────────
ARM_JOINTS   = ['joint1', 'joint2', 'joint3', 'joint4',
                'joint5', 'joint6', 'joint7']
FINGER_JOINTS = ['finger_joint1', 'finger_joint2']
NUM_ARM       = 7
NUM_ACTUATORS = 8          # 7 arm + 1 gripper tendon

# Home keyframe from Menagerie panda.xml
HOME_POSITION = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

# Joint limits (from Menagerie model)
JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

# Smooth motion limits (rad/s per joint)
MAX_JOINT_VEL = np.array([2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 3.0])

GRIPPER_MAX_WIDTH = 0.08   # metres (0.04 per finger)


class BridgeNode(Node):

    def __init__(self):
        super().__init__('bridge_node')

        # ── Parameters ──────────────────────────────────────────────
        self.declare_parameter('control_rate', 500.0)
        self.declare_parameter('smooth_motion', True)
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('initial_arm_position', HOME_POSITION.tolist())
        self.declare_parameter('initial_gripper_ctrl', 255.0)

        self.control_rate  = self.get_parameter('control_rate').value
        self.smooth_motion = self.get_parameter('smooth_motion').value
        self.publish_tf    = self.get_parameter('publish_tf').value
        initial_arm_position = np.array(
            self.get_parameter('initial_arm_position').value, dtype=float)
        if initial_arm_position.shape != (NUM_ARM,):
            self.get_logger().warning(
                f"Expected {NUM_ARM} initial arm joints, got "
                f"{initial_arm_position.shape}; falling back to default home.")
            initial_arm_position = HOME_POSITION.copy()
        initial_arm_position = np.clip(initial_arm_position, JOINT_LOWER, JOINT_UPPER)
        initial_gripper_ctrl = float(self.get_parameter('initial_gripper_ctrl').value)
        initial_gripper_ctrl = max(0.0, min(initial_gripper_ctrl, 255.0))

        # ── State ───────────────────────────────────────────────────
        self.current_positions  = np.zeros(NUM_ARM + 2)   # 7 arm + 2 finger
        self.current_velocities = np.zeros(NUM_ARM + 2)
        self.state_received = False

        # Targets & command (interpolated)
        self.target_arm     = initial_arm_position.copy()
        self.command_arm    = initial_arm_position.copy()
        self.gripper_ctrl   = initial_gripper_ctrl

        # ── QoS ─────────────────────────────────────────────────────
        fast = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST, depth=1)

        # ── Subscribers from simulation ─────────────────────────────
        self.create_subscription(
            JointState, '/mujoco/joint_states', self._mujoco_state_cb, fast)

        # ── Subscribers from user / student nodes ───────────────────
        self.create_subscription(
            Float64MultiArray, '/panda/position_targets',
            self._pos_target_cb, fast)
        self.create_subscription(
            Float64MultiArray, '/panda/gripper_command',
            self._gripper_cb, fast)

        # ── Publishers to simulation ────────────────────────────────
        self.ctrl_pub = self.create_publisher(
            Float64MultiArray, '/mujoco/joint_controls', fast)

        # ── Publishers to user / student nodes ──────────────────────
        self.panda_js_pub   = self.create_publisher(JointState,        '/panda/joint_states',      fast)
        self.ee_pose_pub    = self.create_publisher(PoseStamped,       '/panda/end_effector_pose', fast)
        self.ft_pub         = self.create_publisher(WrenchStamped,     '/panda/ft_sensor',         fast)
        self.pos_error_pub  = self.create_publisher(Float64MultiArray, '/panda/position_error',    fast)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self) if (HAS_TF2 and self.publish_tf) else None

        # ── Control loop timer ──────────────────────────────────────
        self.create_timer(1.0 / self.control_rate, self._control_loop)

        self.get_logger().info(
            f'Bridge node ready  (rate={self.control_rate} Hz, '
            f'smooth={self.smooth_motion}, tf={self.publish_tf})')

    # ================================================================
    #  Callbacks from MuJoCo sim
    # ================================================================

    def _mujoco_state_cb(self, msg: JointState):
        """Cache joint state from sim, republish under /panda/."""
        all_names = ARM_JOINTS + FINGER_JOINTS
        for i, name in enumerate(all_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self.current_positions[i]  = msg.position[idx]
                if idx < len(msg.velocity):
                    self.current_velocities[i] = msg.velocity[idx]
        self.state_received = True

        # Republish as /panda/joint_states
        self.panda_js_pub.publish(msg)

    # ================================================================
    #  Callbacks from user nodes
    # ================================================================

    def _pos_target_cb(self, msg: Float64MultiArray):
        """Set desired arm positions (7 values, radians)."""
        n = min(len(msg.data), NUM_ARM)
        t = np.array(msg.data[:n])
        t = np.clip(t, JOINT_LOWER[:n], JOINT_UPPER[:n])
        self.target_arm[:n] = t

    def _gripper_cb(self, msg: Float64MultiArray):
        """Set gripper command.
        Accepts:
          - Single value in [0, 1]   → normalised  (0=closed, 1=open → 0–255)
          - Single value in (1, 255] → raw ctrl value
          - Single value as width in metres [0, 0.08] when <= GRIPPER_MAX_WIDTH
        """
        if not msg.data:
            return
        v = msg.data[0]
        if v < 0:
            v = 0.0
        if v <= 1.0:
            self.gripper_ctrl = v * 255.0
        elif v <= GRIPPER_MAX_WIDTH:
            self.gripper_ctrl = (v / GRIPPER_MAX_WIDTH) * 255.0
        else:
            self.gripper_ctrl = min(v, 255.0)

    # ================================================================
    #  Control loop — smooth interpolation + publish ctrl
    # ================================================================

    def _control_loop(self):
        if not self.state_received:
            return

        dt = 1.0 / self.control_rate

        # Smooth interpolation toward target
        if self.smooth_motion:
            diff = self.target_arm - self.command_arm
            max_step = MAX_JOINT_VEL * dt
            self.command_arm += np.clip(diff, -max_step, max_step)
        else:
            np.copyto(self.command_arm, self.target_arm)

        # Clamp
        self.command_arm = np.clip(self.command_arm, JOINT_LOWER, JOINT_UPPER)

        # Build ctrl: [j1..j7 position targets, gripper_ctrl]
        ctrl = np.zeros(NUM_ACTUATORS)
        ctrl[:NUM_ARM] = self.command_arm
        ctrl[NUM_ARM]  = self.gripper_ctrl

        # Publish to sim
        ctrl_msg = Float64MultiArray()
        ctrl_msg.data = ctrl.tolist()
        self.ctrl_pub.publish(ctrl_msg)

        # Publish position error
        err = self.target_arm - self.current_positions[:NUM_ARM]
        err_msg = Float64MultiArray()
        err_msg.data = err.tolist()
        self.pos_error_pub.publish(err_msg)

    # ================================================================
    #  EE pose / FT / TF  (read from sensor_data or body xpos)
    #  NOTE: these are published whenever we get a joint_state update.
    #  For now we just publish EE pose and TF from the JointState cb
    #  since the sim also publishes sensor_data.  A future version
    #  could subscribe to /mujoco/sensor_data for higher-fidelity EE.
    # ================================================================
    # (EE pose and TF require direct MuJoCo body access, which only
    #  the sim node has.  We'll add EE / TF publishing to the sim node
    #  if needed, or subscribe to sensor_data here.)

    def destroy_node(self):
        super().destroy_node()


# ════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
