# occupancy_grid

A robot with a laser scanner drives through an unknown 2-D environment
and builds up a map of where things are as it moves. Cells the laser
beam passes through get marked as probably empty; cells it stops in get
marked as probably occupied. After enough scans you can look at the grid
and see the rooms, walls, and obstacles emerge cleanly.

The map is stored in **log-odds form**: each cell holds
`L = log( p(occupied) / p(free) )`, which starts at 0 (no information)
and accumulates positive values for occupied evidence and negative for
free. The update is the textbook **inverse sensor model** from Thrun,
Burgard & Fox (Ch. 9.2 / Table 9.1): for each valid laser beam,
Bresenham-trace a line from the sensor cell to the endpoint cell;
deposit free-evidence along the line; deposit occupied-evidence in a
thickness band starting at the endpoint. Updates accumulate over all
scans and the log-odds get clipped to ±20 so heavily-observed cells
don't saturate and become impossible to revise later.

## Why log-odds, not probability

The probabilistic update is multiplicative — `p_new ∝ p_old · likelihood`
— which is fiddly to implement (you have to renormalize, you have
numerical issues with tiny probabilities). Taking the log turns
multiplication into addition: every laser beam contributes a fixed
*additive* increment to each affected cell, with no normalization
needed and no underflow. The conversion back to probability for display
is a single sigmoid:

```
p = exp(L) / (1 + exp(L))
```

It's the same trick used in logistic regression, and for exactly the
same reason.

## The obstacle-thickness band

A laser hit tells you a cell is occupied, but obstacles aren't
infinitely thin — a wall has some depth, and any cell *inside* that
wall is also occupied. So the inverse sensor model deposits
occupied-evidence not just at the endpoint cell but along a short
Bresenham line of length `min_obstacle_depth_m` beyond it. This makes
the resulting map register walls as actual walls, not as one-cell-thick
lines, and the planner downstream can inflate them correctly.

## Why clip log-odds

After hundreds of scans of a definitely-occupied cell, the log-odds
would climb to thousands — fine in itself, but if the environment
changes (a chair moves), it takes hundreds of scans of *free* evidence
to walk that cell back below zero. Clipping to ±20 keeps the dynamic
range bounded so the map can respond to changes within a reasonable
number of new observations.

## Running it

```bash
ros2 launch occupancy_grid turtlebot_bringup.launch.py
```
