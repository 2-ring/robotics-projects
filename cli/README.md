# cli

A little launcher for the projects in this repo. Prints a menu, lets you
pick a project by number, brings up its full sim + controller, and
handles cleanup (Ctrl+C → graceful shutdown → straggler sweep).

## Run it

From the repo root:

```bash
pip install rich            # one-time, if you don't have it
python cli/robo.py
```

That's it. The CLI keeps running until you pick `q` (or hit Ctrl+C at
the prompt). After each project exits, it brings you back to the menu so
you can try the next one without re-launching.

## What it does

1. **Splash + boot.** Animated title, brief "spinning up" progress bar.
2. **Menu.** A numbered table of every project, grouped by category
   (control / SLAM / planning / skills / symbolic).
3. **Prompt.** Pick `1`-`12` to launch a project, or `q` to quit.
   Anything else gets a friendly nudge.
4. **Pre-flight checks.** Before launching, the CLI verifies the env is
   set up for that project — `ros2` on PATH and `install/` built for ROS
   projects, `household_core` importable for the household projects. If
   not, it prints the exact command you need to run.
5. **Launch.** Spawns the sim/controller in its own process group so the
   whole tree can be torn down together.
6. **Shutdown.** Ctrl+C escalates `SIGINT → SIGTERM → SIGKILL` against
   the process group, then sweeps for any straggler ROS / MuJoCo
   processes (`mujoco_sim_node`, `rviz_marker_node`, bridge nodes, etc.)
   so the next launch starts clean.

## Project map

| # | project | how it's launched |
| --- | --- | --- |
| 1 | braitenberg_bug | `ros2 launch braitenberg_bug braitenberg_sim.launch.py` |
| 2 | panda_teleop    | `ros2 launch panda_teleop panda_sim.launch.py` |
| 3 | arm_pid         | `ros2 launch arm_pid pid_sim.launch.py` |
| 4 | visual_servoing | `ros2 launch visual_servoing visual_servo_sim.launch.py` |
| 5 | occupancy_grid  | `ros2 launch occupancy_grid turtlebot_bringup.launch.py` |
| 6 | bayes_localizer | `ros2 launch bayes_localizer q2_bayes_localization.launch.py` |
| 7 | rrt_planner     | `ros2 launch rrt_planner q3_rrt_planning.launch.py` |
| 8 | semantic_mapper | `ros2 launch semantic_mapper q1_sim.launch.py` |
| 9 | door_opener     | `ros2 launch door_opener q2_sim.launch.py` |
| 10 | puck_pusher    | `ros2 launch puck_pusher q3_sim.launch.py` |
| 11 | household_fsm  | `python projects/household_fsm/evaluate.py` |
| 12 | household_bt   | `python projects/household_bt/evaluate.py` |

## Troubleshooting

- **`error · ros2 isn't on PATH`** — you haven't activated the conda env
  or sourced the workspace. From the repo root:
  `conda activate ros2_env && source install/setup.bash`.

- **`error · no install/ directory`** — you haven't built the workspace
  yet. `colcon build --symlink-install` in the repo root.

- **`error · household_core isn't importable`** — you're trying to run a
  household project but the env doesn't have `household_core` installed.
  `pip install -e shared/household_core` in the repo root (in your `a4`
  conda env, not `ros2_env`).

- **Something hangs after Ctrl+C.** The CLI escalates to `SIGKILL`
  within ~10 seconds and then sweeps stragglers. If something *still*
  hangs, `ps aux | grep -E 'mujoco|rviz|ros2' | grep -v grep` will
  show you what's left.

## Files

- [robo.py](robo.py) — the CLI itself. ~250 lines, single file, only
  external dep is `rich`.
- [README.md](README.md) — this file.
