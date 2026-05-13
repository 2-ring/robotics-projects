# rrt_planner

Given a start pose, a goal pose, and a known map, plan a collision-free
path for a differential-drive robot to follow. The planner returns a
sequence of waypoints; downstream a pure-pursuit controller actually
drives the robot along them.

This is a **single-tree goal-biased RRT** (Rapidly-exploring Random
Tree) with **motion-model rollouts** — meaning the tree expansion
respects what the robot can actually drive, not just any straight line.
With probability `goal_bias` the goal pose itself is used as the random
sample (so the tree pokes toward the goal often enough to find it);
otherwise a random collision-free pose is drawn from the inflated
costmap. From the nearest existing node, a small fixed set of candidate
`(vx, wz)` velocity commands is tried; each is forward-simulated through
the velocity motion model with sub-step collision checks against the
costmap; the collision-free one whose endpoint is closest to the random
target (under a position + heading metric) becomes the new node. When a
new node satisfies the goal tolerance, the path is reconstructed by
walking parent pointers back to the root.

## Why heading-weighted distance

A differential-drive robot can't sideslip — if you're 5 cm from the
goal but facing the wrong way, you can't just slide over. You have to
rotate, drive forward, and rotate back. So a Euclidean distance metric
ignoring heading would underestimate how far the robot really has to
go. The planner uses

```
ρ(a, b) = ||a_xy - b_xy|| + w · |a_θ - b_θ|
```

with `w = 0.2`. It biases the tree toward picking samples that are
heading-compatible with the nearest node, which produces noticeably
straighter, less zig-zaggy paths.

## Sub-step collision checking

Each velocity command rolls out for `control_dt_s` (0.2 s by default),
which can correspond to ~5 cm of motion at this robot's speed. If we
only checked collision at the start and end of that rollout, a thin
wall sitting in the middle would get jumped over entirely. So each
rollout is sub-divided into segments of `collision_step_m` (0.05 m)
and a collision check runs at every substep. Slower than checking just
the endpoint, but it's what makes the planner actually safe.

## Pre-inflated costmap

Instead of doing a swept-volume check against the robot's footprint at
every collision check (expensive), the costmap gets inflated *once* by
the robot's radius at startup. Then every collision check is a single
lookup against a single point. This is the standard trick from nav2
and friends.

## Vectorized nearest neighbor

For every RRT iteration, finding the nearest tree node to the random
sample is naively O(N) over the tree. Naive Python loops here add up
fast — by 5000 nodes you're spending most of the planner's time in
distance computations. The implementation keeps parallel `tree_x`,
`tree_y`, `tree_θ` NumPy arrays alongside the node list, and the
nearest-neighbor search is a single vectorized expression over those
arrays.

## Running it

```bash
ros2 launch rrt_planner q3_rrt_planning.launch.py
```
