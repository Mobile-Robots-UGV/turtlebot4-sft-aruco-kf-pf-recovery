from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    slam_arg = DeclareLaunchArgument(
        'slam', default_value='true',
        description='Launch SLAM (true/false)')

    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Launch RViz2 (true/false)')

    slam_enabled = LaunchConfiguration('slam')
    rviz_enabled = LaunchConfiguration('rviz')

    # ── Paths ─────────────────────────────────────────────────────────────
    tracker_pkg  = get_package_share_directory('sft_hardware_tracker')
    board_pkg    = get_package_share_directory('board_pose_ros')
    nav_pkg      = get_package_share_directory('turtlebot4_navigation')
    config_file  = os.path.join(tracker_pkg, 'config', 'sft_hardware_recovery.yaml')

    rviz_config = os.path.join(
        tracker_pkg, '..', '..', '..', '..',
        'src', 'turtlebot4-sft-aruco-kf-pf-recovery-main',
        'tracker.rviz'
    )

    # ── 1. Board pose node ────────────────────────────────────────────────
    board_pose_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(board_pkg, 'launch', 'board_pose.launch.py')
        ),
    )

    # ── 2. Tracker node ───────────────────────────────────────────────────
    tracker = Node(
        package='sft_hardware_tracker',
        executable='board_tracker_node',
        name='board_tracker_node',
        output='screen',
        parameters=[config_file],
    )

    # ── 3. Follower node (delayed 2s) ─────────────────────────────────────
    follower = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='sft_hardware_tracker',
                executable='recovery_follower_node',
                name='recovery_follower_node',
                output='screen',
                parameters=[config_file],
            )
        ]
    )

    # ── 4. SLAM (delayed 3s) ──────────────────────────────────────────────
    slam_launch = TimerAction(
        period=3.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(nav_pkg, 'launch', 'slam.launch.py')
                ),
                launch_arguments={
                    'sync':      'false',
                    'namespace': '/robot_09',
                }.items(),
                condition=IfCondition(slam_enabled),
            )
        ]
    )

    # ── 5. RViz2 (delayed 4s) ─────────────────────────────────────────────
    rviz_node = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_config],
                condition=IfCondition(rviz_enabled),
                output='screen',
            )
        ]
    )

    return LaunchDescription([
        slam_arg,
        rviz_arg,
        board_pose_launch,
        tracker,
        follower,
        slam_launch,
        rviz_node,
    ])