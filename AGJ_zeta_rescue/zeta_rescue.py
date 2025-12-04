import time
import rclpy
import rclpy.node
import numpy as np

from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import Twist

from geometry_msgs.msg import PoseStamped
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs 
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
    (-2.0, 2.5, 0.0), (-4.0, 2.0, 0.0), (1.5, 2.5, 0.0),
    (4.0, 2.0, 0.0), (6.0, 2.0, -1.57), (6.0, 0.5, -1.57), (-4.0, 0.0, 0.0),
    (-2.0, 0.0, 0.0), (1.5, 0.0, 0.0), (4.0, -0.2, 0.0), (6.0, -1.0, 1.57),
    (-4.0, -1.5, 0.0), (-2.0, -2.0, 0.0), (1.5, -2.0, 0.0), (4.0, -2.0, 0.0),
]

class WaypointNavigator(rclpy.node.Node):

    def __init__(self):
        super().__init__('waypoint_nav')

        self.declare_parameter('time_limit', 120)
        self.declare_parameter('recall_time', 50)
        self.time_limit = self.get_parameter('time_limit').value
        self.recall_time = self.get_parameter('recall_time').value
        self.start_time = time.time()
        
        self.time_expired = False
        self.returning = False
        
        self.home_x = 0.0
        self.home_y = 0.0
        self.home_theta = 0.0

        self.ac = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cancel_client = self.create_client(CancelGoal, '/navigate_to_pose/_action/cancel_goal')

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

        self.timer = self.create_timer(0.5, self.timer_check)
        
    def timer_check(self):
        elapsed = time.time() - self.start_time
        time_left = self.time_limit - elapsed
        
        if time_left < self.recall_time and not self.returning:
            self.get_logger().warn(f"Time Low ({time_left:.1f}s)! Force Return.")
            self.time_expired = True
            self.returning = True
            
            req = CancelGoal.Request()
            self.cancel_client.call_async(req)

    def set_home_pose(self):
        """
        Spins and waits until the transform from base_link to map is valid.
        """
        self.get_logger().info("Waiting for Map Transform...")
        
        p1 = PoseStamped()
        p1.header.frame_id = "base_link"
        p1.pose.orientation.w = 1.0

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

            try:
                if self.buffer.can_transform("map", "base_link", rclpy.time.Time()):
                    p2 = self.buffer.transform(p1, "map")
                    
                    self.home_x = p2.pose.position.x
                    self.home_y = p2.pose.position.y
                    
                    q = p2.pose.orientation
                    _, _, yaw = tf_transformations.euler_from_quaternion(
                        [q.x, q.y, q.z, q.w]
                    )
                    self.home_theta = yaw
                    self.get_logger().info(f"Start Location Set: ({self.home_x:.2f}, {self.home_y:.2f})")
                    return
            
            except Exception as e:
                self.get_logger().warn(f"Transform failed: {e}")

            if time.time() - self.start_time > 5.0:
                 self.get_logger().error("Could not find map transform! Defaulting to (0,0).")
                 return

    def go_to_pose(self, x, y, theta):
        if self.time_expired:
            return False

        goal_msg = create_nav_goal(x, y, theta)
        self.get_logger().info(f"Going to ({x:.2f}, {y:.2f})...")
        
        if not self.ac.wait_for_server(timeout_sec=1.0):
            return False

        send_future = self.ac.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_future)
        
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        status = result_future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.spin_in_place()
            return True
        else:
            return False

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

    def run_waypoints(self):
        self.set_home_pose()

        for (x, y, theta) in WAYPOINTS:
            if self.returning:
                break
            self.go_to_pose(x, y, theta)

        self.get_logger().info(f"RETURNING HOME ({self.home_x:.2f}, {self.home_y:.2f}).")
        self.returning = True
        self.time_expired = False 
        
        self.go_to_pose(self.home_x, self.home_y, self.home_theta)
        
        self.get_logger().info("Mission Complete. Waiting...")

def main(args=None):
    rclpy.init(args=args)
    node = WaypointNavigator()
    try:
        node.run_waypoints()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()