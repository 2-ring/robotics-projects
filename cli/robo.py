#!/usr/bin/env python3
"""robotics-projects CLI launcher.

Lists the projects in the repo, lets you pick one by number, and runs the
full sim + controller for it. Handles Ctrl+C cleanly, sweeps for stragglers
afterwards, and returns to the menu so you can try the next one.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from rich.align import Align
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.text import Text
except ImportError:
    sys.stderr.write(
        "this CLI needs the `rich` library.\n"
        "  install it:  pip install rich\n"
    )
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parent.parent
console = Console()


@dataclass(frozen=True)
class Project:
    name: str
    group: str
    blurb: str
    kind: str                 # "ros2_launch" or "python"
    pkg_or_dir: str           # ROS pkg name, or repo-relative dir for python
    launch_file: str = ""     # used only by ros2_launch


PROJECTS: list[Project] = [
    # ── control ──
    Project("braitenberg_bug", "control",
            "Light-seeking two-wheeled bug",
            "ros2_launch", "braitenberg_bug", "braitenberg_sim.launch.py"),
    Project("panda_teleop", "control",
            "Keyboard teleop + joint-pose commander for the Panda arm",
            "ros2_launch", "panda_teleop", "panda_sim.launch.py"),
    Project("arm_pid", "control",
            "Joint-space PID with damped-LS IK and gravity compensation",
            "ros2_launch", "arm_pid", "pid_sim.launch.py"),
    Project("visual_servoing", "control",
            "Jacobian visual servoing: Panda tracks a moving ball",
            "ros2_launch", "visual_servoing", "visual_servo_sim.launch.py"),
    # ── SLAM ──
    Project("occupancy_grid", "SLAM",
            "Log-odds occupancy mapping from laser scans",
            "ros2_launch", "occupancy_grid", "turtlebot_bringup.launch.py"),
    Project("bayes_localizer", "SLAM",
            "Discrete Bayes histogram filter over (θ, x, y)",
            "ros2_launch", "bayes_localizer", "q2_bayes_localization.launch.py"),
    # ── planning ──
    Project("rrt_planner", "planning",
            "Goal-biased RRT with motion-model rollouts",
            "ros2_launch", "rrt_planner", "q3_rrt_planning.launch.py"),
    # ── skills ──
    Project("semantic_mapper", "skills",
            "Object-level mapper with adaptive outlier rejection",
            "ros2_launch", "semantic_mapper", "q1_sim.launch.py"),
    Project("door_opener", "skills",
            "Panda opens a hinged door by planning along the arc",
            "ros2_launch", "door_opener", "q2_sim.launch.py"),
    Project("puck_pusher", "skills",
            "ProMP puck-pushing policy learned from demonstrations",
            "ros2_launch", "puck_pusher", "q3_sim.launch.py"),
    # ── symbolic ──
    Project("household_fsm", "symbolic",
            "Finite-state controller for the deterministic household",
            "python", "projects/household_fsm"),
    Project("household_bt", "symbolic",
            "Behavior tree for the stochastic household",
            "python", "projects/household_bt"),
]

GROUP_COLOR = {
    "control":  "cyan",
    "SLAM":     "green",
    "planning": "yellow",
    "skills":   "magenta",
    "symbolic": "blue",
}


# ─── ui ────────────────────────────────────────────────────────────────

# Block-letter logo (figlet "ansi_shadow" font). Embedded so we don't need
# pyfiglet as a runtime dep.
ROBOTICS_ART = r"""
██████╗  ██████╗ ██████╗  ██████╗ ████████╗██╗ ██████╗███████╗
██╔══██╗██╔═══██╗██╔══██╗██╔═══██╗╚══██╔══╝██║██╔════╝██╔════╝
██████╔╝██║   ██║██████╔╝██║   ██║   ██║   ██║██║     ███████╗
██╔══██╗██║   ██║██╔══██╗██║   ██║   ██║   ██║██║     ╚════██║
██║  ██║╚██████╔╝██████╔╝╚██████╔╝   ██║   ██║╚██████╗███████║
╚═╝  ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝    ╚═╝   ╚═╝ ╚═════╝╚══════╝
""".strip("\n")


def splash() -> None:
    """Animated title: reveal the block-letter logo line by line."""
    lines = ROBOTICS_ART.split("\n")

    with Live(console=console, refresh_per_second=30, transient=False) as live:
        for i in range(1, len(lines) + 1):
            shown = "\n".join(lines[:i])
            live.update(Align.center(Text(shown, style="bold cyan")))
            time.sleep(0.045)

    subtitle = Text("P  R  O  J  E  C  T  S", style="bold cyan")
    console.print(Align.center(subtitle))
    tagline = Text("a launchpad for 12 little experiments", style="dim italic")
    console.print(Align.center(tagline))
    console.print()


def boot_bar() -> None:
    """Brief 'spinning up' progress bar that disappears after."""
    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold dim]spinning up the robots..."),
        BarColumn(bar_width=30, complete_style="cyan", finished_style="green"),
        TextColumn("[dim]{task.percentage:>3.0f}%"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("boot", total=100)
        for _ in range(20):
            progress.update(task, advance=5)
            time.sleep(0.02)


def menu() -> None:
    """Print the project table, grouped by category."""
    table = Table(
        show_header=True,
        header_style="bold",
        border_style="dim",
        padding=(0, 1),
    )
    table.add_column("#", justify="right", width=3)
    table.add_column("project", style="bold", min_width=18)
    table.add_column("group", width=10)
    table.add_column("what it does", overflow="fold")

    prev_group = None
    for i, p in enumerate(PROJECTS, start=1):
        if prev_group is not None and p.group != prev_group:
            table.add_section()
        table.add_row(
            f"[cyan]{i:>2}[/]",
            p.name,
            f"[{GROUP_COLOR[p.group]}]{p.group}[/]",
            p.blurb,
        )
        prev_group = p.group

    console.print(Align.center(table))
    console.print()


def prompt_choice() -> int | None:
    """Get the user's pick — index into PROJECTS, or None to quit."""
    while True:
        raw = Prompt.ask(
            "[bold cyan]pick a project[/] "
            f"[dim]([cyan]1-{len(PROJECTS)}[/cyan], [cyan]q[/cyan] to quit)[/]",
            default="",
            show_default=False,
        ).strip().lower()

        if raw in {"q", "quit", "exit", ":q"}:
            return None
        if not raw:
            continue
        try:
            n = int(raw)
        except ValueError:
            console.print(f"[yellow]hmm, '{raw}' isn't a number.[/]")
            continue
        if not 1 <= n <= len(PROJECTS):
            console.print(f"[yellow]pick a number from 1 to {len(PROJECTS)}.[/]")
            continue
        return n - 1


# ─── env checks ─────────────────────────────────────────────────────────

def env_ok(project: Project) -> bool:
    """Verify the runtime env is set up for this project. Print and return False if not."""
    if project.kind == "ros2_launch":
        if not shutil.which("ros2"):
            console.print("[bold red]error[/] · `ros2` isn't on PATH.")
            console.print("[dim]  did you `conda activate ros2_env` and source install/setup.bash?[/]")
            return False
        if not (REPO_ROOT / "install").exists():
            console.print("[bold red]error[/] · no install/ directory — workspace not built.")
            console.print(f"[dim]  run: cd {REPO_ROOT}  &&  colcon build --symlink-install[/]")
            return False
        return True

    if project.kind == "python":
        # Need household_core importable
        check = subprocess.run(
            [sys.executable, "-c", "import household_core"],
            capture_output=True,
        )
        if check.returncode != 0:
            console.print("[bold red]error[/] · `household_core` isn't importable in this env.")
            console.print(f"[dim]  run: pip install -e {REPO_ROOT}/shared/household_core[/]")
            return False
        return True

    console.print(f"[bold red]error[/] · unknown launcher kind: {project.kind}")
    return False


# ─── launching ─────────────────────────────────────────────────────────

def build_command(project: Project) -> tuple[list[str], Path]:
    if project.kind == "ros2_launch":
        return ["ros2", "launch", project.pkg_or_dir, project.launch_file], REPO_ROOT
    # python kind
    proj_dir = REPO_ROOT / project.pkg_or_dir
    return [sys.executable, "evaluate.py"], proj_dir


def launch(project: Project) -> None:
    if not env_ok(project):
        return

    cmd, cwd = build_command(project)
    pretty_cmd = " ".join(str(c) for c in cmd)
    console.print()
    console.print(Panel.fit(
        f"[bold green]→ launching[/] [bold]{project.name}[/]\n"
        f"[dim]$ {pretty_cmd}[/]\n"
        f"[dim]  cwd: {cwd}[/]\n\n"
        f"[dim]Ctrl+C to stop and return to the menu.[/]",
        border_style="green",
        padding=(0, 2),
    ))
    console.print()

    # New process group so we can kill the whole launch tree on Ctrl+C.
    proc = subprocess.Popen(cmd, cwd=cwd, start_new_session=True)

    try:
        rc = proc.wait()
    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]Ctrl+C → bringing down the launch tree...[/]")
        terminate_process_group(proc)
        sweep_stragglers()
        console.print("[dim]back at the menu.[/]")
        return

    console.print()
    if rc == 0:
        console.print(f"[dim]{project.name} exited cleanly.[/]")
    else:
        console.print(f"[yellow]{project.name} exited with code {rc}.[/]")
    # Some ROS 2 nodes don't reliably die on the launch's exit — be polite.
    if project.kind == "ros2_launch":
        sweep_stragglers(quiet=True)


def terminate_process_group(proc: subprocess.Popen) -> None:
    """Escalate SIGINT → SIGTERM → SIGKILL on the process group until dead."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return

    for sig, timeout, label in [
        (signal.SIGINT,  5, "SIGINT"),
        (signal.SIGTERM, 3, "SIGTERM"),
        (signal.SIGKILL, 1, "SIGKILL"),
    ]:
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            console.print(f"[dim yellow]  hanging on {label} — escalating...[/]")
            continue


STRAGGLER_PATTERNS = (
    "mujoco_sim_node",
    "rviz_marker_node",
    "bridge_node",
    "turtlebot_bridge_node",
    "ros2 launch",
)


def sweep_stragglers(quiet: bool = False) -> None:
    """Kill any leftover ROS / MuJoCo processes from this run.

    Uses pgrep + per-pid SIGTERM rather than blanket pkill so we don't nuke
    anything the user may be running outside this CLI.
    """
    pgrep = shutil.which("pgrep")
    if pgrep is None:
        return

    own_pid = os.getpid()
    killed = 0
    for pattern in STRAGGLER_PATTERNS:
        try:
            result = subprocess.run(
                [pgrep, "-f", pattern],
                capture_output=True, text=True, timeout=2,
            )
        except subprocess.TimeoutExpired:
            continue
        for pid_str in result.stdout.strip().split():
            try:
                pid = int(pid_str)
            except ValueError:
                continue
            if pid == own_pid:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                killed += 1
            except (ProcessLookupError, PermissionError):
                pass

    if killed and not quiet:
        console.print(f"[dim]swept {killed} straggler process(es).[/]")


# ─── main loop ─────────────────────────────────────────────────────────

def main() -> int:
    # Ignore SIGINT in the parent's default handler — we manage it ourselves
    # inside launch(). Ctrl+C while idle at the prompt will still raise
    # KeyboardInterrupt and exit via the outer except.
    try:
        console.clear()
        splash()
        boot_bar()
        menu()

        while True:
            choice = prompt_choice()
            if choice is None:
                console.print()
                console.print("[dim]bye.[/]")
                return 0

            launch(PROJECTS[choice])
            console.print()
            menu()
    except KeyboardInterrupt:
        console.print()
        console.print("[dim]bye.[/]")
        return 0


if __name__ == "__main__":
    sys.exit(main())
