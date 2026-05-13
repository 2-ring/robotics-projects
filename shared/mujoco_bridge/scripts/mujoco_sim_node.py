#!/usr/bin/env python3
"""
MuJoCo Simulation Node — Core physics engine.

Loads a MuJoCo model, runs physics (with optional viewer), and exposes
raw simulation state via ROS2 topics. All ROS-friendly translation
(TF, EE pose, etc.) is handled by bridge_node in mujoco_bridge.

Published:
  /mujoco/joint_states       sensor_msgs/JointState
  /mujoco/sensor_data        std_msgs/Float64MultiArray
  /mujoco/sensor_metadata    std_msgs/String  (latched JSON)
  /mujoco/applied_controls   std_msgs/Float64MultiArray
  /mujoco/sim_status         std_msgs/String  (2 Hz JSON)

Subscribed:
  /mujoco/joint_controls     std_msgs/Float64MultiArray

Services:
  /mujoco/reset              std_srvs/Trigger
  /mujoco/pause              std_srvs/SetBool
  /mujoco/get_model_info     std_srvs/Trigger
"""

import json
import time
import threading
import numpy as np

import mujoco
try:
    import mujoco.viewer   # optional — not available on headless servers
    _VIEWER_AVAILABLE = True
except Exception:
    _VIEWER_AVAILABLE = False

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger, SetBool


class MujocoSimNode(Node):

    def __init__(self):
        super().__init__('mujoco_sim_node')

        # ── Parameters ──────────────────────────────────────────────
        self.declare_parameter('model_path', '')
        self.declare_parameter('sim_timestep', 0.001)
        self.declare_parameter('publish_rate', 500.0)
        self.declare_parameter('realtime_factor', 1.0)
        self.declare_parameter('use_viewer', True)
        self.declare_parameter('paused', False)
        self.declare_parameter('initial_keyframe', 'home')
        self.declare_parameter('gravity_comp_hold', False)  # hold arm with gravity comp until controller connects

        model_path            = self.get_parameter('model_path').value
        self.sim_timestep     = self.get_parameter('sim_timestep').value
        self.publish_rate     = self.get_parameter('publish_rate').value
        self.realtime_factor  = self.get_parameter('realtime_factor').value
        self.use_viewer       = self.get_parameter('use_viewer').value
        self.paused           = self.get_parameter('paused').value
        self.gravity_comp_hold = self.get_parameter('gravity_comp_hold').value
        self.initial_keyframe = self.get_parameter('initial_keyframe').value

        if not model_path:
            self.get_logger().fatal('model_path parameter is required!')
            raise RuntimeError('model_path not set')

        # ── Load model ──────────────────────────────────────────────
        self.get_logger().info(f'Loading: {model_path}')
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)

        if self.sim_timestep > 0:
            self.model.opt.timestep = self.sim_timestep

        self._extract_model_info()

        if self.initial_keyframe:
            self._set_keyframe(self.initial_keyframe)

        self.get_logger().info(
            f'Model: {self.model.nq} qpos, {self.model.nv} qvel, '
            f'{self.model.nu} ctrl, {self.model.nsensor} sensors')

        # ── Sim state ───────────────────────────────────────────────
        self.step_count   = 0
        self.sim_running  = True
        self._lock        = threading.Lock()
        self._ctrl_buffer = self.data.ctrl.copy()   # preserve keyframe ctrl
        self._ctrl_updated = False
        self._external_ctrl_received = False  # True once first /mujoco/joint_controls arrives
        self._pending_mocap_pos = None
        self._pending_body_pose = None
        self._pending_reset = False

        # Post-reset hold: pin data.ctrl to the keyframe ctrl for this many
        # sim ticks after a reset, ignoring external ctrl. Prevents a race
        # where a stale controller target yanks the arm off the home pose
        # immediately after /mujoco/reset.
        self._reset_hold_ticks = 0
        self._reset_hold_ticks_on_reset = int(self.publish_rate)  # ≈1 s
        self._home_ctrl = self.data.ctrl.copy()

        # ── QoS ─────────────────────────────────────────────────────
        fast = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST, depth=1)
        reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST, depth=10)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # ── Pub / Sub ───────────────────────────────────────────────
        self.joint_state_pub  = self.create_publisher(JointState,        '/mujoco/joint_states',     fast)
        self.sensor_data_pub  = self.create_publisher(Float64MultiArray, '/mujoco/sensor_data',      fast)
        self.sensor_meta_pub  = self.create_publisher(String,            '/mujoco/sensor_metadata',  latched)
        self.ctrl_echo_pub    = self.create_publisher(Float64MultiArray, '/mujoco/applied_controls', fast)
        self.sim_status_pub   = self.create_publisher(String,            '/mujoco/sim_status',       reliable)

        self.create_subscription(Float64MultiArray, '/mujoco/joint_controls', self._ctrl_cb, fast)
        self.create_subscription(String, '/mujoco/body_pose', self._body_pose_cb, fast)

        # Mocap body positioning (for movable objects like light sources)
        if self.model.nmocap > 0:
            self.create_subscription(
                Float64MultiArray, '/mujoco/mocap_pos', self._mocap_pos_cb, fast)
            self.get_logger().info(f'Mocap bodies: {self.model.nmocap} — listening on /mujoco/mocap_pos')

        # ── Services ────────────────────────────────────────────────
        self.create_service(Trigger, '/mujoco/reset',          self._reset_cb)
        self.create_service(SetBool, '/mujoco/pause',          self._pause_cb)
        self.create_service(Trigger, '/mujoco/get_model_info', self._info_cb)

        # ── Timers ──────────────────────────────────────────────────
        self.create_timer(1.0 / self.publish_rate, self._publish_state)
        self.create_timer(0.5, self._publish_status)

        self._publish_sensor_metadata()
        self.get_logger().info('Sim node ready.')

    # ================================================================
    #  Model helpers
    # ================================================================

    def _extract_model_info(self):
        self.joint_names        = []
        self.arm_joint_names    = []
        self.finger_joint_names = []
        self.actuator_names     = []
        self.sensor_names       = []
        self.body_names         = []

        for i in range(self.model.njnt):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if n:
                self.joint_names.append(n)
                if 'finger' in n:
                    self.finger_joint_names.append(n)
                elif n.startswith('joint') and n[5:].isdigit():
                    self.arm_joint_names.append(n)

        for i in range(self.model.nu):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if n: self.actuator_names.append(n)

        for i in range(self.model.nsensor):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if n: self.sensor_names.append(n)

        for i in range(self.model.nbody):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if n: self.body_names.append(n)

        self.get_logger().info(f'Arm joints: {self.arm_joint_names}')
        self.get_logger().info(f'Finger joints: {self.finger_joint_names}')
        self.get_logger().info(f'Actuators: {self.actuator_names}')

        # joint name → qpos / qvel address
        self.jnt_qpos = {}
        self.jnt_qvel = {}
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.jnt_qpos[name] = self.model.jnt_qposadr[jid]
            self.jnt_qvel[name] = self.model.jnt_dofadr[jid]

        # actuator → joint (skip tendon actuators)
        self.act_to_jnt = {}
        for i, aname in enumerate(self.actuator_names):
            if self.model.actuator_trntype[i] == 0:
                jid = self.model.actuator_trnid[i, 0]
                if 0 <= jid < self.model.njnt:
                    self.act_to_jnt[aname] = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)

    def _set_keyframe(self, name):
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
            mujoco.mj_forward(self.model, self.data)
            self.get_logger().info(f'Keyframe: {name}')
        else:
            self.get_logger().warn(f'Keyframe not found: {name}')

    # ================================================================
    #  Simulation loop  (main thread — macOS viewer requirement)
    # ================================================================

    def run_simulation(self):
        if self.use_viewer:
            self._run_viewer()
        else:
            self._run_headless()

    def _run_viewer(self):
        if not _VIEWER_AVAILABLE:
            self.get_logger().warn(
                'mujoco.viewer not available on this system (headless?). '
                'Falling back to headless mode.'
            )
            self._run_headless()
            return
        self.get_logger().info('Launching viewer...')

        with mujoco.viewer.launch_passive(self.model, self.data) as v:
            self.get_logger().info('Viewer running.')
            w0, s0 = time.time(), self.data.time
            while v.is_running() and self.sim_running:
                if not self.paused:
                    with self._lock:
                        self._apply_pending_state_updates()
                        if self._reset_hold_ticks > 0:
                            # Pin to keyframe ctrl after reset; discard any
                            # external ctrl that arrived before the user
                            # node had a chance to republish its target.
                            np.copyto(self.data.ctrl, self._home_ctrl)
                            self._ctrl_updated = False
                            self._reset_hold_ticks -= 1
                        elif self._ctrl_updated:
                            np.copyto(self.data.ctrl, self._ctrl_buffer)
                            self._ctrl_updated = False
                        elif self.gravity_comp_hold and not self._external_ctrl_received:
                            # No controller yet — hold arm with gravity comp
                            self.data.ctrl[:self.model.nu] = self.data.qfrc_bias[:self.model.nu]
                        mujoco.mj_step(self.model, self.data)
                        self.step_count += 1
                    if self.realtime_factor > 0:
                        dt = (self.data.time - s0) / self.realtime_factor - (time.time() - w0)
                        if dt > 0: time.sleep(dt)
                    v.sync()
                else:
                    with self._lock:
                        self._apply_pending_state_updates()
                    time.sleep(0.01); v.sync()
        self.sim_running = False
        self.get_logger().info('Viewer closed.')

    def _run_headless(self):
        self.get_logger().info('Headless mode.')
        w0, s0 = time.time(), self.data.time
        while self.sim_running:
            if not self.paused:
                with self._lock:
                    self._apply_pending_state_updates()
                    if self._reset_hold_ticks > 0:
                        # Pin to keyframe ctrl after reset; discard any
                        # external ctrl that arrived before the user
                        # node had a chance to republish its target.
                        np.copyto(self.data.ctrl, self._home_ctrl)
                        self._ctrl_updated = False
                        self._reset_hold_ticks -= 1
                    elif self._ctrl_updated:
                        np.copyto(self.data.ctrl, self._ctrl_buffer)
                        self._ctrl_updated = False
                    elif self.gravity_comp_hold and not self._external_ctrl_received:
                        # No controller yet — hold arm with gravity comp
                        self.data.ctrl[:self.model.nu] = self.data.qfrc_bias[:self.model.nu]
                    mujoco.mj_step(self.model, self.data)
                    self.step_count += 1
                if self.realtime_factor > 0:
                    dt = (self.data.time - s0) / self.realtime_factor - (time.time() - w0)
                    if dt > 0: time.sleep(dt)
            else:
                with self._lock:
                    self._apply_pending_state_updates()
                time.sleep(0.01)

    # ================================================================
    #  Callbacks
    # ================================================================

    def _ctrl_cb(self, msg):
        with self._lock:
            n = min(len(msg.data), self.model.nu)
            self._ctrl_buffer[:n] = msg.data[:n]
            self._ctrl_updated = True
            self._external_ctrl_received = True

    def _mocap_pos_cb(self, msg):
        """Set mocap body positions. Data = [x0, y0, z0, x1, y1, z1, ...]."""
        with self._lock:
            self._pending_mocap_pos = list(msg.data)

    def _body_pose_cb(self, msg):
        """Set a free-joint body pose from JSON: {"body":"name","pos":[x,y,z],"quat":[w,x,y,z]}."""
        try:
            req = json.loads(msg.data)
            body_name = req.get('body')
            pos = req.get('pos')
            quat = req.get('quat', [1.0, 0.0, 0.0, 0.0])
            if not body_name or pos is None or len(pos) != 3 or len(quat) != 4:
                raise ValueError('Expected body, pos[3], quat[4].')

            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid < 0:
                raise ValueError(f'Unknown body: {body_name}')
            if self.model.body_jntnum[bid] < 1:
                raise ValueError(f'Body has no joint: {body_name}')
            jid = self.model.body_jntadr[bid]
            if self.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(f'Body is not attached by a free joint: {body_name}')
            with self._lock:
                self._pending_body_pose = {
                    'body': body_name,
                    'pos': [float(v) for v in pos],
                    'quat': [float(v) for v in quat],
                }
        except Exception as e:
            self.get_logger().warn(f'Failed to set body pose: {e}')

    def _apply_pending_state_updates(self):
        # Reset runs on the sim thread so mj_resetData / mj_forward never
        # overlap with v.sync()'s mj_copyDataVisual on mjData.stack.
        if self._pending_reset:
            mujoco.mj_resetData(self.model, self.data)
            if self.initial_keyframe:
                self._set_keyframe(self.initial_keyframe)
            self.step_count = 0
            # Snapshot fresh keyframe ctrl, then pin to it for a short window.
            self._home_ctrl = self.data.ctrl.copy()
            self._ctrl_buffer[:] = self._home_ctrl
            self._ctrl_updated = False
            self._external_ctrl_received = False
            self._reset_hold_ticks = self._reset_hold_ticks_on_reset
            self._pending_body_pose = None
            self._pending_mocap_pos = None
            self._pending_reset = False
            return

        applied = False
        if self._pending_mocap_pos is not None:
            n = min(len(self._pending_mocap_pos) // 3, self.model.nmocap)
            for i in range(n):
                self.data.mocap_pos[i] = self._pending_mocap_pos[i*3 : i*3+3]
            self._pending_mocap_pos = None
            applied = True

        if self._pending_body_pose is not None:
            req = self._pending_body_pose
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, req['body'])
            jid = self.model.body_jntadr[bid]
            qadr = self.model.jnt_qposadr[jid]
            vadr = self.model.jnt_dofadr[jid]
            self.data.qpos[qadr:qadr+3] = req['pos']
            self.data.qpos[qadr+3:qadr+7] = req['quat']
            self.data.qvel[vadr:vadr+6] = 0.0
            self._pending_body_pose = None
            applied = True

        if applied:
            mujoco.mj_forward(self.model, self.data)

    # ================================================================
    #  Publishing
    # ================================================================

    def _publish_sensor_metadata(self):
        m = {'sensor_names': self.sensor_names,
             'sensor_types': [int(self.model.sensor_type[i]) for i in range(self.model.nsensor)],
             'sensor_dims':  [int(self.model.sensor_dim[i])  for i in range(self.model.nsensor)]}
        msg = String(); msg.data = json.dumps(m)
        self.sensor_meta_pub.publish(msg)

    def _publish_state(self):
        if not self.sim_running:
            return
        with self._lock:
            stamp = self.get_clock().now().to_msg()

            # JointState
            js = JointState()
            js.header.stamp = stamp; js.header.frame_id = 'world'
            names = self.arm_joint_names + self.finger_joint_names
            js.name = names
            pos, vel, eff = [], [], []
            for jn in names:
                pos.append(float(self.data.qpos[self.jnt_qpos[jn]]))
                vel.append(float(self.data.qvel[self.jnt_qvel[jn]]))
                f = 0.0
                for ai, an in enumerate(self.actuator_names):
                    if self.act_to_jnt.get(an) == jn:
                        f = float(self.data.actuator_force[ai]); break
                eff.append(f)
            js.position = pos; js.velocity = vel; js.effort = eff
            self.joint_state_pub.publish(js)

            # Sensor data flat
            sd = Float64MultiArray(); buf = []
            for i in range(self.model.nsensor):
                a = self.model.sensor_adr[i]
                for d in range(self.model.sensor_dim[i]):
                    buf.append(float(self.data.sensordata[a + d]))
            sd.data = buf; self.sensor_data_pub.publish(sd)

            # Ctrl echo
            ce = Float64MultiArray()
            ce.data = [float(c) for c in self.data.ctrl]
            self.ctrl_echo_pub.publish(ce)

    def _publish_status(self):
        s = {'running': self.sim_running, 'paused': self.paused,
             'sim_time': float(self.data.time), 'steps': self.step_count,
             'dt': float(self.model.opt.timestep),
             'nu': int(self.model.nu), 'nsensor': int(self.model.nsensor)}
        msg = String(); msg.data = json.dumps(s)
        self.sim_status_pub.publish(msg)

    # ================================================================
    #  Services
    # ================================================================

    def _reset_cb(self, req, res):
        with self._lock:
            self._pending_reset = True
        # Wait for the sim thread to apply the reset so callers see a fresh
        # state by the time the service returns.
        deadline = time.time() + 3.0
        while time.time() < deadline:
            with self._lock:
                if not self._pending_reset:
                    res.success = True; res.message = 'Reset.'
                    return res
            time.sleep(0.005)
        res.success = False; res.message = 'Reset timeout.'
        return res

    def _pause_cb(self, req, res):
        self.paused = req.data; res.success = True
        res.message = 'paused' if self.paused else 'resumed'
        return res

    def _info_cb(self, req, res):
        info = {
            'joint_names': self.joint_names,
            'arm_joint_names': self.arm_joint_names,
            'finger_joint_names': self.finger_joint_names,
            'actuator_names': self.actuator_names,
            'sensor_names': self.sensor_names,
            'body_names': self.body_names,
            'timestep': float(self.model.opt.timestep),
        }
        info['joint_limits'] = {}
        for jn in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if self.model.jnt_limited[jid]:
                info['joint_limits'][jn] = [
                    float(self.model.jnt_range[jid, 0]),
                    float(self.model.jnt_range[jid, 1])]
        res.success = True; res.message = json.dumps(info)
        return res

    def destroy_node(self):
        self.sim_running = False
        super().destroy_node()


# ════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimNode()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    try:
        node.run_simulation()
    except KeyboardInterrupt:
        pass
    finally:
        node.sim_running = False
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
