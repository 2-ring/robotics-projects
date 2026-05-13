# arm_pid

The Franka Panda arm reaches out to a Cartesian point in space and holds
there. Targets come in over a ROS topic; the arm tracks them smoothly,
settles without overshoot, and doesn't drift once it gets there even
under gravity.

Three classical pieces stacked together. **Inverse kinematics** maps the
3-D target position into a joint-space goal by iterating damped
least-squares on the positional Jacobian:
`dq = Jᵀ (J Jᵀ + λ² I)⁻¹ (target - ee_pos)`. **A per-joint PID** with
seven hand-tuned gain triples then drives the joints to that goal, with
the integral term anti-windup-clamped to ±0.5 to keep the arm from
creeping after a large transient. **Gravity + Coriolis bias torques**
from MuJoCo's `mj_forward` get added on top, so the PID is only doing
tracking and not also fighting gravity. The whole thing publishes
7-vector torques at 500 Hz.

## Why per-joint gains

The Panda's base joints (1–4) carry the weight of every link above them;
the wrist joints (5–7) just orient a relatively light end-effector. Using
a single gain across all seven joints means picking between *aggressive
enough to move the base* and *gentle enough not to jitter the wrist*.
Per-joint gains let me have both:

```
Kp = [60, 60, 55, 55, 30, 20, 10]
Kd = [12, 12, 10, 10, 4,  3,  1.5]
```

The decay along the chain is roughly proportional to expected inertia.

## Damped least-squares

The naive way to invert the Jacobian is `dq = J⁺ e`, which blows up near
singularities (where `J` loses rank and `J⁺` has near-infinite entries).
Damped LS regularizes by solving
`(J Jᵀ + λ² I) v = e` and then setting `dq = Jᵀ v` — equivalent to a
trust-region step that gracefully degrades near singularities instead of
exploding. λ = 0.1 is small enough that you don't notice it in
well-conditioned regions and large enough that the singular ones don't
ruin your day.

## Gravity compensation

MuJoCo gives you `qfrc_bias` — the joint torques that exactly balance
gravity, Coriolis, and centrifugal effects at the current state. Adding
it directly to the PID output means the controller doesn't have to learn
to fight a (predictable) ~9.8 N·m bias on the shoulder joint. The PID
gains can stay small and the response feels much more natural.

## Running it

```bash
ros2 launch arm_pid pid_sim.launch.py
```
