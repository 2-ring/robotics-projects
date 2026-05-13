# bayes_localizer

The robot has a map of the world but doesn't know where it is on that
map. By watching its own velocity commands and matching the incoming
laser scans against what each candidate pose *would* see, it builds up
a probability distribution over poses that sharpens, over time, into a
single confident estimate.

This is a **discrete Bayes filter** (also called Markov localization or
a histogram filter), implemented over a 3-D belief tensor indexed by
`(theta_bin, row, col)`. Two phases run concurrently. The **prediction
step** shifts the belief in response to commanded velocity, using the
ideal velocity motion model: each theta bin's spatial slice gets
translated by `(drow, dcol)` with bilinear interpolation for sub-cell
remainders, the theta dimension is rolled by `dtheta` (also bilinearly
interpolated), and an off-by-one kernel smooths out angular noise. The
**correction step** picks every Nth beam from the latest scan and, for
each candidate theta, projects the beam endpoint into the map; cells
where the projected endpoint matches the ground-truth occupancy get
multiplied by a hit-likelihood, the rest by a miss-likelihood. After
every step the belief is renormalized and the MAP pose (argmax over
the tensor) gets published.

## Bilinear interpolation in the prediction step

The naive way to integrate velocity into the belief is to compute the
nearest cell to the predicted displacement and just shift everything
there. This works, but the belief teleports cell-by-cell as the robot
moves, and the localizer never converges to a smooth, tight estimate.
With bilinear interpolation, each predicted position contributes
proportionally to all four neighboring cells:

```
new[r, c] = (1-fr)(1-fc) · old[r₀,   c₀  ]
          + (1-fr)( fc ) · old[r₀,   c₀+1]
          + ( fr )(1-fc) · old[r₀+1, c₀  ]
          + ( fr )( fc ) · old[r₀+1, c₀+1]
```

where `fr, fc` are the fractional remainders. The belief now flows
continuously across the grid, the posterior is smooth, and the MAP
estimate tracks well even at velocities that move much less than one
cell per timestep.

## Subsampling beams

Running the correction over all 360 beams per scan per theta bin per
spatial cell is `360 · 36 · 80 · 80 = 83 M` evaluations per scan in
Python — too slow to keep up with a 10 Hz laser. Using every Nth beam
(N=3 by default) drops this by 3× without measurably hurting accuracy,
because adjacent beams are highly correlated when the scan is dense.

## Off-by-one theta smoothing

After the angular shift, a small amount of probability mass gets bled
to neighboring theta bins (`±1` with probability `off_by_one / 2`
each). This models the angular noise in the velocity estimate without
needing a full Gaussian convolution — cheap and effective.

## Running it

```bash
ros2 launch bayes_localizer q2_bayes_localization.launch.py
```
