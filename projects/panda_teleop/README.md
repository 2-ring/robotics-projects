# panda_teleop

An interactive sandbox for driving the Franka Panda arm by hand. Two ways
to play: a keyboard teleop that moves the end-effector by W/A/S/D-style
keys, and a joint-pose commander that lets you send arbitrary 7-DoF joint
targets one at a time. Mostly useful as the foundation the other Panda
projects build on — the MuJoCo sim, the bridge node that translates
between MuJoCo and ROS 2, and a clean way to poke the arm and see what it
does.

The interesting bit isn't an algorithm — it's the wiring. A single
launch file brings up three coordinated nodes: a MuJoCo sim node that
owns the physics + viewer, a bridge node that publishes joint
positions/velocities and accepts joint commands, and a teleop or
commander process that the user interacts with. The bridge node is
generic across robots; the sim node loads whichever model XML the launch
file points it at; the teleop is the only piece that's robot-specific.
This separation is what lets every other Panda project plug in by
swapping out only the controller process.

## The MuJoCo ↔ ROS 2 bridge

The bridge (in [shared/mujoco_bridge](../../shared/mujoco_bridge)) runs at
500 Hz inside the same Python process as the MuJoCo physics step. It
exposes joint state to the ROS graph and accepts joint commands back. On
macOS it has to launch under `mjpython` for the viewer to work — the
launch file detects this and shells out accordingly.

## Running it

```bash
ros2 launch panda_teleop panda_sim.launch.py
```

Then in a second terminal:

```bash
ros2 run panda_teleop keyboard_teleop.py        # or joint_pose_commander.py
```
