import time
import rclpy
import rclpy.node

from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist

import tf_transformations

def create_nav_goal(x, y, theta):
    goal = NavigateToPose.Goal()
    goal.pose.header.frame_id = 'map'
    goal.pose.pose.position.x = x
    goal.pose.pose.position.y = y

    q = tf_transformations.quaternion_from_euler(0, 0, theta, 'rxyz')
    goal.pose.pose.orientation.x = q[0]
    goal.pose.pose.orientation.y = q[1]
    goal.pose.pose.orientation.z = q[2]
    goal.pose.pose.orientation.w = q[3]

    return goal


WAYPOINTS = [
    (0.0, 0.0, 0.0),  # Starting area
    (-2.0, 2.5, 0.0), # Top-left
    (-4.0, 2.0, 0.0),
    (1.5, 2.5, 0.0),  # Top-middle
    (4.0, 2.0, 0.0),
    (6.0, 2.0, -1.57), # Top-right
    (6.0, 0.5, -1.57),
    (-4.0, 0.0, 0.0), # Middle-left
    (-2.0, 0.0, 0.0),
    (1.5, 0.0, 0.0),  # Center
    (4.0, -0.2, 0.0),
    (6.0, -1.0, 1.57), # Middle-right
    (-4.0, -1.5, 0.0), # Bottom-left
    (-2.0, -2.0, 0.0),
    (1.5, -2.0, 0.0), # Bottom-middle
    (4.0, -2.0, 0.0),
    (0.0, 0.0, 3.14)  # return to start
]


class WaypointNavigator(rclpy.node.Node):

    def __init__(self, timeout_per_goal=20.0):
        super().__init__('waypoint_nav')

        self.ac = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.timeout = timeout_per_goal
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

    def go_to_pose(self, x, y, theta):
        goal_msg = create_nav_goal(x, y, theta)

        self.get_logger().info(f"Sending goal: ({x:.2f}, {y:.2f}, {theta:.2f})")
        self.ac.wait_for_server()

        send_goal_future = self.ac.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Navigation server rejected the goal.")
            return False

        result_future = goal_handle.get_result_async()
        start_time = time.time()

        while rclpy.ok() and not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)

            # Timeout safety
            if time.time() - start_time > self.timeout:
                self.get_logger().warn("Timeout! Canceling goal.")
                goal_handle.cancel_goal_async()
                return False

        result = result_future.result()
        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Goal succeeded! Spinning...") # Spin once reached
            self.spin_in_place(speed=0.25, rotations=1.0)
            return True

        self.get_logger().warn(f"Goal failed with status: {status}")
        return False

    def run_waypoints(self):
        for (x, y, theta) in WAYPOINTS:
            success = self.go_to_pose(x, y, theta)
            if not success:
                self.get_logger().warn("Stopping waypoint execution early.")
                return False

        self.get_logger().info("Completed all waypoints successfully.")
        return True

    def spin_in_place(self, speed=0.3, rotations=1.0):
        twist = Twist()
        twist.angular.z = speed

        spin_time = (2 * np.pi * rotations) / speed
        start = time.time()

        while time.time() - start < spin_time:
            self.cmd_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.05)

        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)


def main():
    rclpy.init()

    node = WaypointNavigator(timeout_per_goal=25.0)

    node.run_waypoints()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
