# braitenberg_bug

A little two-wheeled robot drives toward a glowing target and stops when it
gets there. The robot has a left and right antenna that each report how far
away the light is; everything else falls out of how those two numbers steer
the wheels.

The control law is a classic Braitenberg vehicle — the kind you read about
in the first chapter of any biologically-inspired robotics book. Each
wheel's speed is a linear combination of the two sensor readings: the
average sets the forward speed, the difference sets the steering. To make
the behavior feel smooth instead of jerky, the published wheel commands go
through a small low-pass filter; the forward speed is clamped so the bug
doesn't crawl or overshoot; and there's a brief boost on the steering term
whenever the bug is accelerating. No PID, no learning, no IK — just a
weighted sum that happens to produce a recognizable "I want to go *there*"
motion.

## The acceleration boost

This was the first thing I added after watching the bug do wide spirals
around the light. The issue is that while it's still ramping up to cruising
speed, the forward velocity is dominating the wheel commands — the
steering term gets washed out, the bug barely turns, and by the time the
heading correction "wins" it has overshot. Multiplying the steering term
by 2x specifically while `drive_speed` is still rising punches through that
window cleanly. After cruising speed the boost turns off and the regular
gain takes over.

## Sensor smoothing

The proximity sensors are noisy enough that the raw wheel commands jitter
visibly. A first-order low-pass filter on the wheel velocities with
α = 0.15 smooths this out without adding meaningful lag — at 50 Hz, that's
a few ticks of mixing, which is invisible to the eye but kills the jitter.

## Running it

```bash
ros2 launch braitenberg_bug braitenberg_sim.launch.py
```
