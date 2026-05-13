# door_opener

The Panda arm walks up to a closed door, grabs the handle, and pulls
the door open in one continuous motion. The handle traces a smooth arc
around the door's hinge; the gripper rotates along with the handle so
the grip stays seated as the door swings.

The motion is planned in **handle-arc coordinates** rather than in
joint space. Given the (known) hinge position, hinge axis, and handle
offset, the planner generates a sequence of handle waypoints by
linearly interpolating door angles between current and target and
computing each waypoint's world-frame position via a Z-axis rotation
of the handle offset around the hinge. For each waypoint, the desired
grasp orientation is also computed — by composing a Z-axis rotation
(matching the door's angle at that waypoint) with the original
closed-door grasp quaternion. Damped-LS IK on position + orientation
turns each waypoint into a joint target; the controller blends toward
that target (α = 0.3) instead of snapping to it; and the plan
re-generates from the current door angle every ~100 ticks because the
EE drifts off the arc as the door swings.

## Rotating the grasp orientation with the door

This is the part I got wrong on my first try. My instinct was to grasp
the handle at the closed-door pose and hold that exact world-frame
orientation through the swing. That works for the first few degrees
and then the gripper twists off the handle, because the *handle* is
rotating with the door but the *gripper* isn't. The fix is to compose
the closed-door grasp quaternion with a Z-axis rotation by the current
door angle:

```
q_grasp(θ) = q_z(θ) · q_grasp_closed
```

This is just Hamilton multiplication. The result is a per-waypoint
target orientation that has the gripper rotating in lockstep with the
door, which keeps the grip seated and the wrist out of singular
configurations.

## Blending toward the target

The naive thing is to publish each new joint target directly and let
the position controller chase it. The arm reaches for it aggressively
and the resulting torque transient is enough to pry the gripper off
the handle (the door has real inertia in the sim). Instead, the
publisher emits:

```
q_pub = q_current + α · (q_target - q_current)
```

with `α = 0.3`. The arm moves toward the target over several ticks
instead of in one shot. The trajectory is the same in steady state,
but the transient is gentle enough that the door's inertia doesn't
break the grasp.

## Periodic re-planning

The plan is *kinematic* — it assumes the gripper traces the arc
perfectly. The dynamics aren't a perfect tracker, so the EE drifts a
little off the arc as the door swings. After ~100 ticks of drift, the
plan no longer reflects what the gripper should do. So every 100
ticks the planner re-runs from the *current* door angle, refreshing
the trajectory. Tracking stays tight.

## Running it

```bash
ros2 launch door_opener q2_sim.launch.py
```
