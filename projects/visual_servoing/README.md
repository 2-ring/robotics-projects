# visual_servoing

The Panda arm hovers over a table and follows a moving red ball, keeping
its gripper a fixed height above the ball and pointed straight down —
ready to grasp. It only ever sees the ball through an overhead camera;
there's no direct knowledge of where the ball is in the world.

This is **6-DoF Jacobian visual servoing**. Each tick, the ball is lifted
from the camera frame into the world frame via a homogeneous transform
from the camera extrinsics. A 6-D twist is then assembled: linear
velocity from a P-controller that drives the gripper toward
`(ball_x, ball_y, fixed_z)`; angular velocity from the orientation error
between the gripper's current rotation and a fixed "pointing down" target
rotation. The orientation error is read off the skew-symmetric part of
`R_desired @ R_currentᵀ`. The 6-D twist is then converted into 7-D
joint velocities by inverting the manipulator Jacobian with damped
least-squares: `dq = Jᵀ (J Jᵀ + λ² I)⁻¹ v_desired`. Joint velocities get
clamped and published at 100 Hz.

## Orientation error from a rotation matrix

The clean way to extract a small-angle rotation error from a 3×3 rotation
matrix `R_err = R_desired @ R_currentᵀ` is to read off the
skew-symmetric part:

```
e_ori = 0.5 * [R_err(2,1) - R_err(1,2),
               R_err(0,2) - R_err(2,0),
               R_err(1,0) - R_err(0,1)]
```

This is the same expression you'd get from a first-order log-map: it
recovers the axis-angle representation directly when the error is small,
and degrades gracefully when it isn't. The 3-vector you get points along
the rotation axis with magnitude equal to the rotation angle, which is
exactly what an angular-velocity controller wants as its input.

## Damped LS on the full 6×7 Jacobian

The same DLS trick used in [arm_pid](../arm_pid) for IK reappears here,
but this time on the full 6-D twist Jacobian rather than just the 3-D
positional one. `J` is 6×7 (3 rows of positional + 3 rows of rotational
partials), `J Jᵀ` is 6×6, and the damping term λ²I keeps things stable
when the arm gets close to a singular configuration mid-track.

## Latched QoS for one-shot publishers

The camera extrinsics are static — a 4×4 matrix that gets published once
on startup. If the visual servoing controller subscribes *after* that
publish, it would miss the message and have no idea where the camera is.
The fix is a `TRANSIENT_LOCAL` (latched) QoS profile on that topic, which
caches the most recent message so late subscribers still get it. Small
detail but it took me a while to figure out why the controller was
silently producing garbage on launch.

## Running it

```bash
ros2 launch visual_servoing visual_servo_sim.launch.py
```
