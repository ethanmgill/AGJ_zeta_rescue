import time
import math
import rclpy
import rclpy.node
import numpy as np

from enum import Enum

from rclpy.action import ActionClient
from rclpy.task import Future
from rclpy.qos import qos_profile_sensor_data, QoSProfile,  QoSDurabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal

from std_msgs.msg import Empty
from std_msgs.msg import Int32
from sensor_msgs.msg import Image

from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose

from geometry_msgs.msg import Twist
from geometry_msgs.msg import PointStamped, PoseStamped, PoseArray, Pose

from jmu_ros2_util import map_utils

from zeta_competition_interfaces.msg import Victim

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


# navigation states NOTE: potentially redundant
class NavState(Enum):
    EXPLORING = 0   # searching for victims
    APPROACHING = 1 # approaching a victim
    HOMEBOUND = 2   # returning home
    WAITING = 3     # waiting for a new goal
    CANCELING = 4   # cancelling navigation

class WaypointNavigator(rclpy.node.Node):

    def __init__(self):
        super().__init__('waypoint_nav')

        # TODO: change to use relative points based off of competition grid
        self.waypoints = [
            (-2.0, 2.5, 0.0), (-4.0, 2.0, 0.0), (1.5, 2.5, 0.0),
            (4.0, 2.0, 0.0), (6.0, 2.0, -1.57), (6.0, 0.5, -1.57), (-4.0, 0.0, 0.0),
            (-2.0, 0.0, 0.0), (1.5, 0.0, 0.0), (4.0, -0.2, 0.0), (6.0, -1.0, 1.57),
            (-4.0, -1.5, 0.0), (-2.0, -2.0, 0.0), (1.5, -2.0, 0.0), (4.0, -2.0, 0.0),
        ]

        self.map = None

        # This QOS Setting is used for topics where the messages
        # should continue to be available indefinitely once they are
        # published. Maps fall into this category.  They typically
        # don't change, so it makes sense to publish them once.
        latching_qos = QoSProfile(depth=1,
                                  durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


        # declare parameters
        self.declare_parameter('time_limit', 120)
        
        self.declare_parameter('recall_time', 50)
        
        self.declare_parameter(
            name="aruco_topic",
            value="/aruco_poses",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Aruco Poses topic to subscribe to",
            ),
        )

        self.declare_parameter(
            name="victim_x_tolerance",
            value=0.5,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Tolerance value for detecing victims in the x axis"
            )
        )

        self.declare_parameter(
            name="victim_y_tolerance",
            value=0.5,
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

        self.nav_state = NavState.WAITING
        self.nav_future = None # most recent status of nav service
        self.cancel_future = None
        self.goal = None # navigation service goal

        self.map = None
        
        # initialize home pose to default: 0,0,0
        self.home_x = 0.0
        self.home_y = 0.0
        self.home_theta = 0.0

        # declare current pose
        self.cur_pose = PoseStamped()

        # declare current image (updated by camera topic)
        self.cur_image = None

        # declare services
        self.ac = ActionClient(self, NavigateToPose, '/navigate_to_pose') # this is used to communicate with the nav service
        #self.cancel_client = self.create_client(CancelGoal, '/navigate_to_pose/_action/cancel_goal') # this is used to cancel the nav service

        # declare subscribers
        self.aruco_sub = self.create_subscription(
            PoseArray, aruco_topic, self.aruco_callback, qos_profile_sensor_data
        )
        self.cur_pose_sub = self.create_subscription(
            PoseStamped, '/amcl_pose', self.cur_pose_callback, qos_profile_sensor_data
        )
        ''' TODO: implement callback function
        self.report_req_sub = self.create_subscription(
            Empty, '/report_requested', self.report_req_callback, 10
        )
        '''
        self.camera_sub = self.create_subscription(
            Image, '/oakd/rgb/preview/image_raw', self.camera_callback, qos_profile_sensor_data
        )
        
        self.create_subscription(OccupancyGrid, 'map',
                                 self.map_callback,
                                 qos_profile=latching_qos)
        
        # declare publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10) # honestly do not know why this is here
        self.victim_pub = self.create_publisher(Victim, '/victim', 10)

        # The TransfromListener will listen for all published transforms, which will be stored in the buffer.
        self.tf_buffer = Buffer() # buffer for transformation calculations
        self.tf_listener = TransformListener(self.tf_buffer, self) # when a transformation is calculated, the transformer will send it to the buffer

        # data structure for tracking marked victims
        # stores tuples of x, y locations of victims
        self.marked_victims = [] # start empty of course

        # data structure for keeping victim data ready for publishing
        self.captured_victims = []

        self.timer = self.create_timer(0.5, self.timer_callback) # timer for competition management
        

    ### COMPETITION MANAGEMENT ###

    def timer_callback(self):
        elapsed = time.time() - self.start_time # how much time has it been since we started running?
        time_left = self.time_limit - elapsed # how much time is left in the competition?
        self.get_logger().info(f"\n\nTimer clock: [{elapsed:.2f}]\nTime left: [{time_left:.2f}]\nMarked Victim Count: [{len(self.marked_victims)}]\nVictim Count: [{len(self.captured_victims)}]\nCurrent Nav State: [{self.nav_state}]\n\n")

        # Are we waiting for a goal to be canceled?
        if self.cancel_future is not None: # goal has been cancelled. Awaiting ACK from server
            self.get_logger().info("Canceling goal...")
            #self.nav_state = NavState.CANCELING TODO: ensure commenting this out doesn't cause issues
            # have we gotten an ACK from the nav server?
            if self.cancel_future.done(): # if we are done cancelling the navigation
                self.nav_future = None # reset nav_future for a new navigation task
                self.cancel_future = None
                if self.nav_state != NavState.APPROACHING: # if we're approaching, navigation goes to aruco_callback
                    self.nav_state = NavState.WAITING # ready for new goal
                self.get_logger().info("Canceled goal. Ready for new goal.")
        
        # are we out of time?
        if time_left < self.recall_time:
            # are we already going home?
            if self.nav_state == NavState.HOMEBOUND:
                # TODO: add program exit logic upon completion
                pass
            # are we waiting to go home?
            elif self.nav_state == NavState.WAITING:
                self.nav_state = NavState.HOMEBOUND # set variable to indicate we are returning home
                self.goal = create_nav_goal(self.home_x, self.home_y, self.home_theta) # set nav goal to home point
                self.send_goal() # send nav goal to nav service
                self.get_logger().info("Home goal sent.")
            # have we canceled any ongoing navigation?
            elif self.nav_state != NavState.CANCELING:
                self.get_logger().warn(f"Time Low ({time_left:.1f}s)! Force Return.") # log action
                self.nav_state = NavState.CANCELING       
                self.cancel_future = self.nav_future.result().cancel_goal_async()

        # we still have time left; go exploring
        else:
            if self.nav_state == NavState.EXPLORING:
                self.handle_navigation_status()
            elif self.nav_state == NavState.APPROACHING:
                pass # wait for victim logging to finish
            elif self.nav_state == NavState.HOMEBOUND:
                pass # TODO: we still have time left. we should be exploring
            elif self.nav_state == NavState.WAITING:
                self.nav_to_next_waypoint()
            elif self.nav_state == NavState.CANCELING:
                pass # wait for navigation to finish cancelling

    def handle_navigation_status(self, recovery_state=NavState.WAITING):
        if self.nav_future is not None and self.nav_future.done():
            result = self.nav_future.result().status
            if result == GoalStatus.STATUS_EXECUTING:
                self.get_logger().info(f"\nExecuting navigation to point: [{self.goal.pose.pose.position.x}, {self.goal.pose.pose.position.y}]\n")
                pass
            elif result == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info(f"\nSuccesfully navigated to point: [{self.goal.pose.pose.position.x}, {self.goal.pose.pose.position.y}]\n")
                self.nav_state = NavState.CANCELING       
                self.cancel_future = self.nav_future.result().cancel_goal_async()
            elif result == GoalStatus.STATUS_CANCELING:
                self.get_logger().info(f"\nCancelling navigation to point: [{self.goal.pose.pose.position.x}, {self.goal.pose.pose.position.y}]\n")
                pass
            elif result == GoalStatus.STATUS_CANCELED:
                self.get_logger().info(f"\nCancelled navigation to point: [{self.goal.pose.pose.position.x}, {self.goal.pose.pose.position.y}]\n")
                pass
            elif result == GoalStatus.STATUS_ABORTED:
                self.get_logger().warn(f"\nABORTED navigation to point: [{self.goal.pose.pose.position.x}, {self.goal.pose.pose.position.y}]")
                self.get_logger().warn(f"Recovering from abort. Resetting...")
                self.nav_future = None
                self.nav_state = recovery_state
                pass
            elif result == GoalStatus.STATUS_ACCEPTED:
                self.get_logger().info(f"\nAccepted navigation to point: [{self.goal.pose.pose.position.x}, {self.goal.pose.pose.position.y}]\n")
                pass
            elif result == GoalStatus.STATUS_UNKNOWN:
                self.get_logger().warn(f"\nUNKNOWN STATUS of navigation to point: [{self.goal.pose.pose.position.x}, {self.goal.pose.pose.position.y}]\n")
                #self.get_logger().warn(f"Recovering from unknown. Resetting...")
                #self.nav_future = None
                #self.nav_state = recovery_state
                pass
            else:
                self.get_logger().warn(f"Unexpected navigation goal status. GoalStatus=[{result}]")
        elif self.nav_future is None: # error handling
            self.get_logger().warn(f"Unexpected navigation status. Resetting...")
            self.nav_state = recovery_state
        elif not self.nav_future.done():
            self.get_logger().warn(f"Navigation server has not accepted goal yet...")


            
    # TODO: this doesn't work with asynchronous programming
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
            
    def cur_pose_callback(self, cur_pose):
        # NOTE: accuracy unkown
        # PoseWithCovarianceStamped -> PoseWithCovariance -> Pose
        self.cur_pose = cur_pose.pose.pose

    def report_req_callback(self):
        # TODO: iterate over victim list and publish to victim topic one by one
        pass


    ### EXPLORATION ###

    # TODO: this doesn't work with asynchronous programming
    # TODO: refactor to use nav_service
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

    def nav_to_next_waypoint(self):
        # TODO: check if waypoint is free in occupancy grid before mapping to it
        self.nav_state = NavState.EXPLORING
        ''' TODO: map wasn't populating on spin_up...
        free = False
        while(not free and self.waypoints):
            (x, y, theta) = self.waypoints.pop(0)
            free = self.cell_is_free(x, y)
        if not self.waypoints:
            #TODO: return home i guess :/
            pass
        '''
        (x, y, theta) = self.waypoints.pop(0)

        self.get_logger().info(f"# of Waypoints remaining: [{len(self.waypoints)}]")
        self.get_logger().info(f"PICKED POINT: ({x:.2f}, {y:.2f}).")
        self.goal = create_nav_goal(x, y, theta) # set nav goal to next waypoint
        self.send_goal() # send nav goal to nav service
        self.get_logger().info(f"Waypoint goal sent.")

    def send_goal(self):
        self.get_logger().info("WAITING FOR NAVIGATION SERVER...")
        self.ac.wait_for_server()
        self.get_logger().info("NAVIGATION SERVER AVAILABLE...")
        self.get_logger().info("SENDING GOAL TO NAVIGATION SERVER...")

        self.nav_future = self.ac.send_goal_async(self.goal)

    ### HELPFUL FUNCTIONS ###
    
    def map_callback(self, map_msg):
        """Process the map message.

        This doesn't really do anything useful, it is purely intended
        as an illustration of the Map class.

        """
        if self.map is None:  # No need to do this every time map is published.

            self.map = map_utils.Map(map_msg)
            '''
            # Use numpy to calculate some statistics about the map:
            total_cells = self.map.width * self.map.height
            pct_occupied = np.count_nonzero(self.map.grid == 100) / total_cells * 100
            pct_unknown = np.count_nonzero(self.map.grid == -1) / total_cells * 100
            pct_free = np.count_nonzero(self.map.grid == 0) / total_cells * 100
            map_str = "Map Statistics: occupied: {:.1f}% free: {:.1f}% unknown: {:.1f}%"
            self.get_logger().info(map_str.format(pct_occupied, pct_free, pct_unknown))

            # Here is how to access map cells to see if they are free:
            x = self.goal.pose.pose.position.x
            y = self.goal.pose.pose.position.y
            val = self.map.get_cell(x, y)
            if val == 100:
                free = "occupied"
            elif val == 0:
                free = "free"
            else:
                free = "unknown"
            self.get_logger().info(f"HEY! Map position ({x:.2f}, {y:.2f}) is {free}")
            '''

    def cell_is_free(self, x, y):
        val = self.map.get_cell(x, y)
        if val == 0:
            return True
        else:
            return False
    

    ### ArUco Code detection, navigation, and capture ###

    # function to be called when an aruco code is detected
    def aruco_callback(self, aruco_msg):
        # ensure we're not returning home
        if self.nav_state == NavState.HOMEBOUND:
            return # return early
        self.get_logger().info("ARUCO MARKER DETECTED")
        if aruco_msg is None:
            self.get_logger().warn("No Aruco info has been received!")
            return

        if self.nav_state != NavState.APPROACHING: # we've already locked on to a victim. No need to calculate
            # transform aruco locations
            map_poses = []
            for pose in aruco_msg.poses:
                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = "oakd_rgb_camera_optical_frame"
                pose_stamped.pose = pose
                for _ in range(10):
                    try:
                        map_pose_stamped = self.tf_buffer.transform(pose_stamped, "map")
                        map_pose = map_pose_stamped.pose
                        map_poses.append(map_pose)
                        break
                    except Exception as e:
                        self.get_logger().warn(f"ArUco pose transformation error: {str(e)}")

            # pick first seen victim
            target_victim_pose = map_poses[0]

            # TODO: iterate over target_victim_poses to find one that hasn't been marked.
            # check if victim is already marked
            if self.victim_marked(target_victim_pose):
                return # exit early
            
            # mark victim
            self.mark_victim(target_victim_pose)
            self.get_logger().info(f"found new victim at: [{target_victim_pose.position.x}, {target_victim_pose.position.y}]")

            # change state to indicate tracking a victim
            self.nav_state = NavState.APPROACHING
            # cancel navigation service
            self.cancel_future = self.nav_future.result().cancel_goal_async()

            # calculate goal position
            self.goal = self.calculate_goal_from_victim_pose(target_victim_pose)

        else: # we are approaching a victim
            if self.cancel_future is None: # we're done canceling
                if self.nav_future is None: # we need to navigate to the victim
                    self.send_goal() # victim goal is already loaded, send it.
                else: # we're already navigating to the victim
                    if self.nav_future.result().status == GoalStatus.STATUS_SUCCEEDED:
                        # capture victim
                        victim = self.build_victim_datum()
                        # store victim data
                        self.captured_victims.append(victim)
                        # reset for exploration
                        self.cancel_future = self.nav_future.result().cancel_goal_async()
                        self.nav_state = NavState.CANCELING
                    else:
                        self.handle_navigation_status(recovery_state=NavState.APPROACHING)

    def build_victim_datum(self):
        victim = Victim()
        victim.id = np.int32(len(self.marked_victims - 1))

        ps = PointStamped()
        ps.header.stamp = time.time() - self.start_time
        ps.header.frame_id = "map"
        (x, y) = self.marked_victims[-1] # grab the most recent victim published
        ps.point.x = x
        ps.point.y = y
        ps.point.z = 0
        victim.point = ps

        img = Image()
        img = self.cur_img
        victim.image = img

        return victim

    def calculate_goal_from_victim_pose(self, victim_pose, distance=0.75):
        victim_quat = [
            victim_pose.orientation.x,
            victim_pose.orientation.y,
            victim_pose.orientation.z,
            victim_pose.orientation.w
        ]
        _, _, victim_yaw = tf_transformations.euler_from_quaternion(victim_quat)
        
        # position distance away from target relative to victim orientation
        goal_x = victim_pose.position.x + distance * math.cos(victim_yaw)
        goal_y = victim_pose.position.y + distance * math.sin(victim_yaw)

        # orient to face victim (180 deg opposit of victim's orientation)
        goal_yaw = victim_yaw + math.pi
        goal_yaw = math.atan2(math.sin(goal_yaw), math.cos(goal_yaw))
        # convert orientation to quat
        # goal_quat = tf_transformations.quaternion_from_euler(0, 0, goal_yaw)

        return create_nav_goal(goal_x, goal_y, goal_yaw)
    
    def mark_victim(self, victim_pose):
        victim_x = victim_pose.position.x
        victim_y = victim_pose.position.y

        self.marked_victims.append((victim_x, victim_y))
        self.get_logger().info(f"Marked Victim at [{victim_pose.position.x:.4f}, {victim_pose.position.y:.4f}]")
    
    def victim_marked(self, victim_pose):
        x_min = victim_pose.position.x - self.victim_x_tolerance
        x_max = victim_pose.position.x + self.victim_x_tolerance
        y_min = victim_pose.position.y - self.victim_y_tolerance
        y_max = victim_pose.position.y + self.victim_y_tolerance
        return any(x_min <= vic_x <= x_max and y_min <= vic_y <= y_max 
                   for vic_x, vic_y in self.marked_victims)

    def camera_callback(self, img_msg):
        self.cur_image = img_msg # update current image from camera

    ### End ArUco Code detection, navigation, and capture ###

def main(args=None):
    rclpy.init(args=args)
    node = WaypointNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()