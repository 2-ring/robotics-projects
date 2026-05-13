# semantic_mapper

A robot drives through a small room of tables with assorted objects on
them. An object detector emits noisy class-labeled detections as the
robot explores. The mapper accumulates those detections into a clean
per-class map of where each object is, and afterwards answers
relative-pose queries like "go 0.5 m in front of the mug" by computing
a goal pose for the navigator.

Fusion is an **ICP-style weighted-centroid update with adaptive outlier
rejection**. Each per-class observation pool starts empty; as
detections arrive, each is weighted by detector confidence and stored.
Once the pool exceeds five samples, an adaptive threshold of
`max(2σ, 0.3 m)` filters out observations that fall too far from the
current estimate; the centroid is recomputed as the
confidence-weighted average of the inliers. The pool is capped at the
100 most recent inliers to keep memory bounded. Query resolution is a
straight vector add: `goal_xy = object_xy + offset_xy`, with the goal
heading pointing back at the object so the robot ends up facing it.

## The adaptive outlier threshold

The standard problem with running averages of noisy detections is that
a single bad observation early on poisons the estimate for the rest of
the run. Hard-coding an outlier threshold (e.g. "reject anything more
than 0.3 m away") doesn't work — early in exploration the estimate
itself is bad, so legitimately good detections look like outliers.

The fix is to scale the threshold with the *spread* of the current
observation pool:

```
threshold = max(2 · std(distances_to_estimate), 0.3 m)
```

When the pool is tight, the threshold is tight. When the pool is
loose (early bootstrap), the threshold is loose. The 0.3 m floor
prevents the threshold from collapsing to zero once the pool has
converged tightly — without it, even tiny normal sensor noise would
start getting rejected.

The first five observations skip rejection entirely so the estimate
can bootstrap without paradoxes.

## Confidence-weighted centroid

The detector tags each detection with a confidence in `[0.3, 1.0]`.
Rather than treat all observations equally, the centroid update is:

```
x* = (Σᵢ wᵢ · pᵢ) / (Σᵢ wᵢ)
```

where `wᵢ` is the detection confidence. High-confidence detections
move the estimate more; low-confidence ones contribute proportionally
less. This is the closed-form minimizer of the weighted least-squares
error `E(x) = Σᵢ wᵢ · ||pᵢ - x||²` — pure ICP machinery, but at the
object-position level instead of the point-cloud level.

## Bounded pool size

Without bounding the inlier pool, memory would grow linearly with
observation count. Capping to the 100 most recent inliers gives
constant memory and also gives the estimate a soft "horizon" — old
detections eventually age out, so if the world genuinely changes,
the map catches up.

## Running it

```bash
ros2 launch semantic_mapper q1_sim.launch.py
```
