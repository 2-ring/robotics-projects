# Robotics Projects

I wanted to prepare myself for working in the robotics lab, making sure I understood everything myself, and how it all pieces together from the ground up. The result of this aspiration was this repository: a semester of fun little robotics experiments.

Each thing under projects/ is one of those: a problem wired up cleanly enough that I could focus on the algorithm instead of the plumbing.

I like first principles when learning something new, so I largely tried to avoid off-the-shelf code here where possible. No nav2, no MoveIt, no SLAM toolbox, etcetera. I ultimately found this to be a rather educational decision.

## Just want to try things?

<div align="center"><pre>
██████╗  ██████╗ ██████╗  ██████╗ ████████╗██╗ ██████╗███████╗
██╔══██╗██╔═══██╗██╔══██╗██╔═══██╗╚══██╔══╝██║██╔════╝██╔════╝
██████╔╝██║   ██║██████╔╝██║   ██║   ██║   ██║██║     ███████╗
██╔══██╗██║   ██║██╔══██╗██║   ██║   ██║   ██║██║     ╚════██║
██║  ██║╚██████╔╝██████╔╝╚██████╔╝   ██║   ██║╚██████╗███████║
╚═╝  ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝    ╚═╝   ╚═╝ ╚═════╝╚══════╝
P  R  O  J  E  C  T  S
a launchpad for 12 little experiments
</pre></div>

There's a little launcher in [cli/](cli) that prints a menu of every
project and brings up the sim + controller for whichever one you pick.
Handles graceful shutdown on Ctrl+C and sweeps for stragglers afterwards.

```bash
pip install rich        # one-time
python cli/robo.py      # from the repo root
```

See [cli/README.md](cli/README.md) for details, troubleshooting, and the
exact launch command behind each menu entry.

## The experiments

- [braitenberg_bug](projects/braitenberg_bug) — a two-wheeled robot that drives toward a light source via the classic Braitenberg crossover.
- [panda_teleop](projects/panda_teleop) — keyboard teleop + joint-pose commander for the Panda. The foundation the other arm projects build on.
- [arm_pid](projects/arm_pid) — joint-space PID for the Franka Panda, with damped-LS IK and gravity compensation.
- [visual_servoing](projects/visual_servoing) — 6-DoF Jacobian visual servoing: the Panda tracks a moving ball through an overhead camera, posed to grasp.
- [occupancy_grid](projects/occupancy_grid) — log-odds occupancy mapping from laser scans, à la Thrun, Burgard & Fox.
- [bayes_localizer](projects/bayes_localizer) — discrete Bayes histogram filter over `(θ, x, y)` on a known map.
- [rrt_planner](projects/rrt_planner) — single-tree goal-biased RRT with motion-model rollouts on an inflated costmap.
- [semantic_mapper](projects/semantic_mapper) — fuses noisy object detections into a per-class map with ICP-style adaptive outlier rejection.
- [door_opener](projects/door_opener) — opens a hinged door by planning along its arc and rotating the grasp orientation with the door.
- [puck_pusher](projects/puck_pusher) — learns puck-pushing from demonstrations using mode-clustered ProMPs and closed-form via-point conditioning.
- [household_fsm](projects/household_fsm) — finite-state controller for a household domain with locked doors, keys, and a finite battery.
- [household_bt](projects/household_bt) — reactive behavior tree for the same household task, but with options that can fail.

## Shared infrastructure

[shared/](shared) holds everything that gets reused across projects:

- [shared/franka_panda](shared/franka_panda) — Franka Panda meshes (vendored from MuJoCo Menagerie), symlinked into each package's `models/.../assets/`.
- [shared/mujoco_bridge](shared/mujoco_bridge) — the colcon package that owns the MuJoCo physics step and translates between MuJoCo and ROS 2 (sim node, generic bridge node, TurtleBot bridge node).
- [shared/slam_utils](shared/slam_utils) — utilities for the SLAM stack: `GridInfo`, `GridCell`, Bresenham, costmap inflation, motion models.
- [shared/skills_utils](shared/skills_utils) — utilities for the higher-level skills: TF helpers, QoS profiles, small math.
- [shared/household_core](shared/household_core) — the household env + abstract FSM/BT controller bases + options + evaluation harness.

## Requirements

- [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/) — the node graph, QoS, services, and launch system tying each controller to its sim. Every project except the two household ones runs on it.
- [MuJoCo 3.4](https://mujoco.org/) — the physics simulator under every ROS project. Used directly for forward kinematics, IK Jacobians, and gravity / Coriolis bias terms in addition to the actual physics step.
- [RoboStack](https://robostack.github.io/) — the conda packaging layer that lets ROS 2 live inside a conda env rather than touch the system Python.
- [colcon](https://colcon.readthedocs.io/) — the build tool that walks the workspace, builds each ROS package, and produces an `install/` tree to source.
- [Python 3.11+](https://www.python.org/) — every controller, planner, and mapper in this repo is pure Python.
- [NumPy](https://numpy.org/) — does basically all the math.
- [rich](https://rich.readthedocs.io/) — used by the [CLI launcher](cli) for styling.

## Running things

Most projects are ROS 2 colcon packages and share a single
[RoboStack](https://robostack.github.io/) conda env. From the repo root:

```bash
conda create -n ros2_env -c conda-forge -c robostack-jazzy -c nodefaults python=3.12 -y
conda activate ros2_env
mamba install -c robostack-jazzy ros-jazzy-desktop colcon-common-extensions rosdep
pip install "mujoco==3.4.0"

colcon build --symlink-install
source install/setup.bash    # or setup.zsh
```

Then `ros2 launch <project> <project>_sim.launch.py` (each project's
README has its specific launch command).

The two household projects are different — they're pure Python, no
simulator. To run them:

```bash
pip install -e shared/household_core
cd projects/household_fsm     # or projects/household_bt
python evaluate.py
```

## A note on the asset layout

The Franka Panda meshes are vendored once at
[`shared/franka_panda/`](shared/franka_panda) and symlinked into each
package's `models/.../assets/`. Keeps the working tree to ~46 MB instead
of ~270 MB, and is transparent to MuJoCo and to colcon's `--symlink-install`.

## License

Apache 2.0 — see [LICENSE](LICENSE).
