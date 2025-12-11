import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    time_limit_arg = DeclareLaunchArgument(
        'time_limit',
        default_value='120',
        description='Time limit to run competition round'
    )

    time_limit = LaunchConfiguration('time_limit')
    
    aruco_params = os.path.join(
        get_package_share_directory('ros2_aruco'),
        'config',
        'aruco_parameters.yaml'
        )

    rescue_node = Node(
        package='AGJ_zeta_rescue',
        executable='zeta_rescue',
        parameters=[{
            'time_limit': time_limit
        }],
        output='screen'
    )

    aruco_node = Node(
        package='ros2_aruco',
        executable='aruco_node',
        parameters=[aruco_params]
    )

    return LaunchDescription([
        time_limit_arg,
        aruco_node,
        rescue_node
    ])
