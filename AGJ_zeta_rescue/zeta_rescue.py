import time
import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data
import numpy as np

from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import Twist

from geometry_msgs.msg import PointStamped, PoseStamped
from ros2_aruco_interfaces.msg import ArucoMarkers
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs # needed even tho not used explicitly
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

# TODO: change to use relative points based off of competition grid
WAYPOINTS = [
    (-2.0, 2.5, 0.0), (-4.0, 2.0, 0.0), (1.5, 2.5, 0.0),
    (4.0, 2.0, 0.0), (6.0, 2.0, -1.57), (6.0, 0.5, -1.57), (-4.0, 0.0, 0.0),
    (-2.0, 0.0, 0.0), (1.5, 0.0, 0.0), (4.0, -0.2, 0.0), (6.0, -1.0, 1.57),
    (-4.0, -1.5, 0.0), (-2.0, -2.0, 0.0), (1.5, -2.0, 0.0), (4.0, -2.0, 0.0),
]

class WaypointNavigator(rclpy.node.Node):

    def __init__(self):
        super().__init__('waypoint_nav')

        # declare parameters
        self.declare_parameter('time_limit', 120)
        
        self.declare_parameter('recall_time', 50)
        
        self.declare_parameter(
            name="aruco_topic",
            value="/aruco_markers",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Aruco Marker topic to subscribe to",
            ),
        )

        self.declare_parameter(
            name="victim_x_tolerance",
            value=2,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Tolerance value for detecing victims in the x axis"
            )
        )

        self.declare_parameter(
            name="victim_y_tolerance",
            value=2,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Tolerance value for detecing victims in the y axis"
            )
        )

        # grab variables
        self.time_limit = self.get_parameter('time_limit').value
        
        self.recall_time = self.get_parameter('recall_time').value
        
        aruco_topic = ( # assign the aruco topic as a string value
            self.get_parameter("aruco_topic").get_parameter_value().string_value
        )

        self.victim_y_tolerance = (
            self.get_parameter("victim_y_tolerance").get_parameter_value().double_value
        )
        self.victim_x_tolerance = (
            self.get_parameter("victim_x_tolerance").get_parameter_value().double_value
        )

        # mark start time of competition
        
        self.start_time = time.time()

        # set state indicators
        
        self.time_expired = False
        self.returning = False
        
        # initialize home pose to default: 0,0,0
        self.home_x = 0.0
        self.home_y = 0.0
        self.home_theta = 0.0

        # declare services
        self.ac = ActionClient(self, NavigateToPose, '/navigate_to_pose') # this is used to communicate with the nav service
        self.cancel_client = self.create_client(CancelGoal, '/navigate_to_pose/_action/cancel_goal') # this is used to cancel the nav service

        # declare subscribers
        self.aruco_sub = self.create_subscription(
            ArucoMarkers, aruco_topic, self.aruco_callback, qos_profile_sensor_data
        )

        # declare publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10) # honestly do not know why this is here

        # The TransfromListener will listen for all published transforms, which will be stored in the buffer.
        self.tf_buffer = Buffer() # buffer for transformation calculations
        self.tf_listener = TransformListener(self.tf_buffer, self) # when a transformation is calculated, the transformer will send it to the buffer

        # data structure for tracking marked victims
        # stores tuples of rectangle values (x_min, x_max, y_min, y_max)
        self.marked_victims = [] # start empty of course


        self.timer = self.create_timer(0.5, self.timer_callback) # timer for competition management

        # enter main loop TODO: this does not work
        # run_waypoints()
        

    ### COMPETITION MANAGEMENT ###

    def timer_callback(self):
        elapsed = time.time() - self.start_time # how much time has it been since we started running?
        time_left = self.time_limit - elapsed # how much time is left in the competition?
        
        if time_left < self.recall_time and not self.returning: # true if we are out of searching-time and we are not already returning home
            self.get_logger().warn(f"Time Low ({time_left:.1f}s)! Force Return.") # log action
            self.time_expired = True # set variable to indicate we are out of searching-time
            self.returning = True # set variable to indicate we are returning home
            
            req = CancelGoal.Request() # create a request to cancel the goal
            self.cancel_client.call_async(req) # send an asynchronous call to the cancel_client to cancel the navigation service. TODO: tested? 
        else:
            self.run_waypoints()

    def set_home_pose(self):
        """
        Spins and waits until the transform from base_link to map is valid.
        """
        self.get_logger().info("Waiting for Map Transform...") # indicate ...
        
        p1 = PoseStamped() # create a new pose stamped
        p1.header.frame_id = "base_link" # indicate this poseStamped is for the home position
        p1.pose.orientation.w = 1.0 # set home pose orientation

        while rclpy.ok(): # if the instance is still running 
            rclpy.spin_once(self, timeout_sec=0.1)

            try:
                if self.tf_buffer.can_transform("map", "base_link", rclpy.time.Time()):
                    p2 = self.tf_buffer.transform(p1, "map")
                    
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

    ### EXPLORATION ###

    def go_to_pose(self, x, y, theta):
        if self.time_expired: # can we afford to make this move?
            self.get_logger().info(f"go_to_pose failure: Time expired. time left: {self.time.time()-self.start_time}")
            return False

        goal_msg = create_nav_goal(x, y, theta) # build a msg to send to nav service
        self.get_logger().info(f"Attempting to go to ({x:.2f}, {y:.2f})...") # log it
        
        if not self.ac.wait_for_server(timeout_sec=1.0):
            self.get_logger().info(f"go_to_pose failure: self.ac.wait_for_server evaluated to false. Cancelling pursuit of goal. ")
            return False

        send_future = self.ac.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_future)
        
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().info(f"go_to_pose failure: Goal handle not accepted :( ")
            return False

        result_future = goal_handle.get_result_async()
        self.get_logger().info(f"Going to ({x:.2f}, {y:.2f})...") # log it
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info(f"Attempting to go to ({x:.2f}, {y:.2f})...") # log it

        status = result_future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f"Arrived at ({x:.2f}, {y:.2f})!") # log success
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

    ### ArUco Code detection, navigation, and capture ###

    # function to be called when an aruco code is detected
    # TODO: in the event of a failure, unmark victim
    def aruco_callback(self, aruco_msg):
        # ensure we're not returning home
        if self.returning:
            return # return early
        
        if aruco_msg is None:
            self.get_logger().warn("No Aruco info has been received!")
            return

        # transform aruco locations
        map_poses = []
        for pose in aruco_msg.poses:
            map_pose = self.buffer.transform(pose, "map")
            map_poses.append(map_poses)
        
        # check if victim is already marked

            # mark victim

        # mark tracking_victim

        # cancel navigation service
        
        # position to capture

        # capture victim

        # resume random nav
        pass
    
    def position_to_capture(self):
        pass
    
    def capture_victim(self):
        pass
    
    def mark_victim(self, victim_pose):
        x_min = victim_pose.position.x - self.victim_x_tolerance
        x_max = victim_pose.position.x + self.victim_x_tolerance
        y_min = victim_pose.position.y - self.victim_y_tolerance
        y_max = victim_pose.position.y + self.victim_y_tolerance
        victim_rectangle = (x_min, x_max, y_min, y_max)
        self.marked_victims.append(victim_rectangle)
    
    def victim_marked(self, victim_pose):
        victim_x = victim_pose.position.x
        victim_y = victim_pose.position.y
        return any(x_min <= victim_x <= x_max and
                   y_min <= victim_y <= y_max for
                   x_min, x_max, y_min, y_max in self.marked_victims)

    ### End ArUco Code detection, navigation, and capture ###

def main(args=None):
    rclpy.init(args=args)
    node = WaypointNavigator()
    '''
    try:
        node.run_waypoints()
    except KeyboardInterrupt:
        pass
    '''
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()