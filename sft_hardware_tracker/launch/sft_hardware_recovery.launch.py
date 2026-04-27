from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    pkg_share = get_package_share_directory('sft_hardware_tracker')
    config_file = os.path.join(pkg_share, 'config', 'sft_hardware_recovery.yaml')

    tracker = Node(
        package='sft_hardware_tracker',
        executable='board_tracker_node',
        name='board_tracker_node',
        output='screen',
        parameters=[config_file],
    )

    follower = Node(
        package='sft_hardware_tracker',
        executable='recovery_follower_node',
        name='recovery_follower_node',
        output='screen',
        parameters=[config_file],
    )

    return LaunchDescription([
        tracker,
        follower,
    ])

