"""Launch Q3 RRT planning with an optional grading node."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Generate launch description for Q3 RRT planning evaluation."""
    q1_share = Path(get_package_share_directory("occupancy_grid"))
    q1_bringup_launch = q1_share / "launch" / "turtlebot_bringup.launch.py"
    default_model = q1_share / "models" / "turtlebot_scene.xml"

    args = [
        DeclareLaunchArgument("model_path", default_value=str(default_model)),
        DeclareLaunchArgument("use_viewer", default_value="true"),
        DeclareLaunchArgument("use_map_viz", default_value="true"),
        DeclareLaunchArgument("paused", default_value="false"),
        DeclareLaunchArgument("realtime_factor", default_value="1.0"),
        DeclareLaunchArgument("run_test", default_value="false"),
        DeclareLaunchArgument("rrt_planner_module", default_value="rrt_planner.py"),
    ]

    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(q1_bringup_launch)),
        launch_arguments={
            "model_path": LaunchConfiguration("model_path"),
            "use_viewer": LaunchConfiguration("use_viewer"),
            "use_map_viz": LaunchConfiguration("use_map_viz"),
            "use_pure_pursuit": "false",
            "paused": LaunchConfiguration("paused"),
            "realtime_factor": LaunchConfiguration("realtime_factor"),
        }.items(),
    )

    planner = Node(
        package="q3",
        executable=LaunchConfiguration("rrt_planner_module"),
        name="rrt_planner",
        output="screen",
        parameters=[{"scene_path": LaunchConfiguration("model_path")}],
    )

    grader = Node(
        package="q3",
        executable="test_rrt_planner.py",
        name="test_rrt_planner",
        output="screen",
        parameters=[{"scene_path": LaunchConfiguration("model_path")}],
        condition=IfCondition(LaunchConfiguration("run_test")),
    )

    return LaunchDescription(args + [bringup, planner, grader])
