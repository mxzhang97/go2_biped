from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='uwb_hri',
            executable='uwb_hri_node',
            name='uwb_hri_node',
            output='screen',
            emulate_tty=True
        ),
        Node(
            package='main_controller',
            executable = 'main_policy_node',
            name='main_policy_node',
            output='screen',
            emulate_tty=True
        )
    ])