# puck_pusher

The Panda arm pushes a puck across the table, around an obstacle in the
middle, and onto a goal pad. It learns this skill from about 30 recorded
demonstrations of humans doing the same task with different start and
goal positions. Given any new start/goal pair within the demonstrated
range, the policy synthesizes a smooth pushing trajectory that respects
the obstacle and lands the puck on the goal.

This is a **Probabilistic Movement Primitive (ProMP)** policy. Each
demonstration's pusher-tip trajectory is resampled to a common length
(T = 180) and projected onto a 12-dimensional Gaussian RBF basis via
ridge regression: `w = (ΦᵀΦ + λI)⁻¹ Φᵀ y`. Across demos in the same
motion mode, the weight vectors form a Gaussian distribution N(μ_w,
Σ_w) — one (B, B) sample covariance per output dimension. At test
time, the policy commits to one mode, then conditions that distribution
on two via-points: the puck at contact-time, and the goal at end-time.
The conditioned mean trajectory comes from a **closed-form Gaussian
posterior** — no MCMC, no optimization. The result is lifted from
tip-space into hand-space via the known weld transform, run through
damped-LS IK, and published at 50 Hz.

## The two motion modes

The first time I trained the ProMP I noticed the resulting puck went
*straight through* the obstacle. I had averaged my weights across all
30 demos, and it turns out the demos fall into two distinct **motion
families**: about half curve around the obstacle to the *left*, the
rest to the *right*. The mean of a left-curving trajectory and a
right-curving one is a trajectory through the middle, which is exactly
where the obstacle is. The fix is to cluster first, fit per-cluster
ProMPs, and commit to one cluster at test time.

The clustering rule is a 2-D cross product:

```
v1 = goal - puck_start
v2 = trajectory_midpoint - puck_start
cross_z = v1.x · v2.y - v1.y · v2.x
```

`cross_z > 0` means the trajectory midpoint is to the left of the
straight line from puck to goal; `cross_z ≤ 0` means to the right.
Cheap, geometric, and exactly captures the topology.

## Closed-form ProMP posterior

Once the weights are Gaussian-distributed, conditioning on a via-point
is a closed-form Gaussian update. Letting `Φ_cond` be the rows of the
basis matrix at the via-point timesteps and `y_obs` the observed
via-point positions:

```
S      = Φ_cond Σ_w Φ_condᵀ + σ_obs² I
K      = Σ_w Φ_condᵀ S⁻¹
μ_new  = μ_w + K (y_obs - Φ_cond μ_w)
```

The conditioned trajectory is then `Φ @ μ_new`. This is just a Kalman
update applied to the weight vector rather than the trajectory itself
— elegant because all the heavy lifting (covariance shrinkage,
trajectory smoothness) happens in the small (B-dim) weight space, not
in the larger (T×3-dim) trajectory space.

The implementation conditions each output dimension (x, y, z)
independently, with looser observation noise on the Z dimension (the
task is essentially planar) and tighter noise on x/y.

## Mean-orientation handle

The end-effector orientation isn't ProMP'd — it's just averaged across
all demos. The pusher tip is rigidly welded to the hand body, so once
you have the desired tip trajectory + a desired hand orientation, the
hand position falls out of `hand = tip - R_hand @ TIP_OFFSET_IN_HAND`.
I tried fitting a ProMP over the orientation too and the gains weren't
worth the complexity for this task.

## Running it

```bash
ros2 launch puck_pusher q3_sim.launch.py
```
