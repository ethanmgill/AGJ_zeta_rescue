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
from sensor_msgs.msg import Image

from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose

from geometry_msgs.msg import Twist
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped

from jmu_ros2_util import map_utils

from ros2_aruco_interfaces.msg import ArucoMarkers

from zeta_competition_interfaces.msg import Victim

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf_transformations
import tf2_geometry_msgs # needed even tho not used explicitly
import random #for sampling points


# navigation states NOTE: potentially redundant
class NavState(Enum):
    EXPLORING = 0       # searching for victims
    APPROACHING = 1     # approaching a victim
    HOMEBOUND = 2       # returning home
    WAITING = 3         # waiting for a new goal
    RECALIBRATING = 4   # resetting nav service for next state
    SETUP = 5           # setting up competition


class WaypointNavigator(rclpy.node.Node):

    def __init__(self):
        super().__init__('waypoint_nav')

        '''----------------PARAMETERS----------------'''

        # VICTIM TRACKING #
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

        # COMPETITION MANAGEMENT #
        self.declare_parameter(
            name="time_limit",
            value=120,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Time limit for competition round",
                read_only=False
            )
        )

        '''------------INSTANCE VARIABLES------------'''

        # MACROS #
        latching_qos        = QoSProfile(      # QOS setting for persistent messages
            depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        # STATE INDICATORS #
        self.nav_state      = NavState.SETUP    # initial state
        self.next_state     = NavState.WAITING  # next state to switch to
        self.nav_future     = None              # state of canceling navigation
        self.cancel_future  = None              # state of navigation service

        # NAVIGATION #
        self.map        = None
        self.waypoints  = []
        self.home_goal  = None
        self.nav_goal   = None
        self.cur_pose   = None

        # VICTIM TRACKING #
        self.victim_count       = 0
        self.marked_victims     = []
        self.captured_victims   = []
        self.cur_image = None
        self.victim_y_tolerance = (     # tolerance for detecting a new victim
            self.get_parameter("victim_y_tolerance").get_parameter_value().double_value
        )
        self.victim_x_tolerance = (
            self.get_parameter("victim_x_tolerance").get_parameter_value().double_value
        )

        # COMPETITION MANAGEMENT #
        self.start_time = time.time()
        self.time_limit = self.get_parameter('time_limit').value
        self.recall_time = 50           # time at which the robot must return
                                        # TODO: calculate with cur_pose callback

        '''---------------SUBSCRIPTIONS--------------'''

        # COMPETITION MANAGEMENT #
        self.create_subscription(
            Empty, '/report_requested', self.report_req_callback, 10
        )

        # NAVIGATION / EXPLORATION #
        self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, qos_profile=latching_qos
        )
        self.cur_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.amcl_callback, qos_profile_sensor_data
        )

        # VICTIM TRACKING #
        self.create_subscription(
            Image, '/oakd/rgb/preview/image_raw', self.camera_callback, qos_profile_sensor_data
        )
        self.aruco_sub = self.create_subscription(
            ArucoMarkers, '/aruco_markers', self.aruco_callback, qos_profile_sensor_data
        )

        '''----------------PUBLISHERS----------------'''

        # NAVIGATION / EXPLORATION #
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10) # honestly do not know why this is here

        # VICTIM TRACKING #
        self.victim_pub = self.create_publisher(Victim, '/victim', 10)
        
        '''-----------------SERVICES-----------------'''

        # NAVIGATION #
        self.tf_buffer      = Buffer()                                  # buffer for transformation calculations
        self.tf_listener    = TransformListener(self.tf_buffer, self)   # listens for published transforms
        
        self.ac = ActionClient(                                         # this is used to communicate with the nav service
            self, NavigateToPose, '/navigate_to_pose'
        )

        # COMPETITION MANAGEMENT #
        self.timer = self.create_timer(0.5, self.timer_callback) # timer for competition management

        '''-----------------------------------------------------------------------------------------------------------------'''



    #####################################################
    '''''''''''''''''''''''''''''''''''''''''''''''''''''
    '                                                   '
    '                    CALLBACKS                      '
    '                                                   '
    '''''''''''''''''''''''''''''''''''''''''''''''''''''
    #####################################################


    '''-------------------------------------COMPETITION TIMER CALLBACK------------------------------------------------------'''

    def timer_callback(self):
        if (self.nav_state == NavState.SETUP):          # competition not ready
            self.setup()
            return
        
        elapsed = time.time() - self.start_time         # how much time has it been since we started running?
        time_left = self.time_limit - elapsed           # how much time is left in the competition?
        
        self.get_logger().info(f"""\n\n
            Timer clock: [{elapsed:.2f}]\n
            Time left: [{time_left:.2f}]\n
            Marked Victim Count: [{len(self.marked_victims)}]\n
            Victim Count: [{len(self.captured_victims)}] = [{self.victim_count}]\n
            Current Nav State: [{self.nav_state}]
            Home Set: [{self.home_goal is not None}]
            \n\n""")

        # are we out of time?
        if time_left < self.recall_time:
            # are we already going home?
            if self.nav_state == NavState.HOMEBOUND:
                self.get_logger().warn(f"\nTime low! Heading home...\n")
                self.handle_homebound()
            # are we getting ready to go home?
            elif self.nav_state == NavState.RECALIBRATING:
                self.handle_recalibration()
            # are we waiting to go home?
            else:
                self.get_logger().warn(f"\nTime low! Rerouting to home...\n")
                self.transition_to_state(NavState.HOMEBOUND)

        # we still have time left; continue the mission.
        else:
            if self.nav_state == NavState.EXPLORING:
                self.handle_navigation_status()             # Navigate to selected point
            elif self.nav_state == NavState.APPROACHING:
                self.handle_approaching()                   # Approach and capture victim
            elif self.nav_state == NavState.HOMEBOUND:
                self.transition_to_state(NavState.WAITING)  # This should never happen; explore
            elif self.nav_state == NavState.WAITING:
                self.nav_to_next_waypoint()                 # Find a new waypoint to explore
            elif self.nav_state == NavState.RECALIBRATING:
                self.handle_recalibration()                 # Reset nav service and switch to next state

    '''-------------------------------------OUR POSE IN MAP FRAME CALLBACK--------------------------------------------------'''

    def amcl_callback(self, cur_pose):
        # PoseWithCovarianceStamped -> PoseWithCovariance -> Pose
        self.cur_pose = cur_pose.pose.pose
        if self.home_goal is None:
            self.set_home_goal()

    '''-------------------------------------ARUCO MARKER DETECTION CALLBACK-------------------------------------------------'''

    # function to be called when an aruco code is detected
    def aruco_callback(self, aruco_msg):

        # ensure we're not returning home
        if self.nav_state == NavState.HOMEBOUND or self.nav_state == NavState.SETUP:
            return
        if aruco_msg is None:
            self.get_logger().warn("No Aruco info has been received!")
            return
        
        # DEBUG: self.get_logger().info("ARUCO MARKER DETECTED")

        if self.nav_state != NavState.APPROACHING:  # we've already locked on to a victim. No need to calculate
            
            # get a new victim pose if found
            target_victim_ps = self.get_new_victim_pose(aruco_msg)
            if target_victim_ps is None:
                return                              # ignore this aruco callback. No new victim found

            # mark victim
            self.mark_victim(target_victim_ps)
            self.get_logger().info(f"found new victim at: [{target_victim_ps.pose.position.x:.6f}, {target_victim_ps.pose.position.y:.6f}]")

            # calculate goal position
            self.victim_goal = self.calculate_approach_goal(target_victim_ps)
            self.transition_to_state(NavState.APPROACHING)

    '''-------------------------------------CAMERA RAW_IMAGE CALLBACK-------------------------------------------------------'''

    def camera_callback(self, img_msg):
        self.cur_image = img_msg # update current image from camera
    
    '''-------------------------------------REQUEST REPORT CALLBACK---------------------------------------------------------'''

    def report_req_callback(self, empty_msg):
        for victim in self.captured_victims:
            self.victim_pub.publish(victim)
        self.get_logger().info("\n\n---PUBLISHED VICTIMS---\n\n")

    '''-------------------------------------OCCUPANCY GRID CALLBACK---------------------------------------------------------'''

    def map_callback(self, map_msg):
        """Process the map message.

        """
        if self.map is None:  # No need to do this every time map is published.

            # wrap OccupancyGrid with helper
            self.map = map_utils.Map(map_msg)

            # derive world bounds from OccupancyGrid.info
            info = map_msg.info
            # store map info for other helpers
            self.map_info = info
            resolution = info.resolution
            width = info.width
            height = info.height
            origin_x = info.origin.position.x
            origin_y = info.origin.position.y
            min_x = origin_x
            min_y = origin_y
            max_x = origin_x + (width * resolution)
            max_y = origin_y + (height * resolution)

            self.get_logger().info(f"Map received: bounds x:[{min_x:.2f},{max_x:.2f}] y:[{min_y:.2f},{max_y:.2f}] res={resolution:.3f}")

            # --- 2) Populate random exploration waypoints on free cells ---
            num_random = 20  # tuneable: number of exploration waypoints to generate
            max_attempts_per_point = 200
            added = 0 # how many points have been added
            attempts = 0 # how many attempts per point
            # margin to avoid map edges (meters)
            # margin = max(0.5, resolution * 2)

            while added < num_random and attempts < num_random * max_attempts_per_point:
                attempts += 1
                rx = random.uniform(min_x, max_x)
                ry = random.uniform(min_y, max_y)
                if self.cell_is_free(rx, ry):
                    rtheta = random.uniform(-math.pi, math.pi)
                    self.waypoints.append((rx, ry, rtheta))
                    added += 1
            self.get_logger().info(f"Populated {added} random exploration waypoints (requested {num_random}). Total waypoints: {len(self.waypoints)}")

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

    '''---------------------------------------------------------------------------------------------------------------------'''


    #####################################################
    '''''''''''''''''''''''''''''''''''''''''''''''''''''
    '                                                   '
    '               STATE HANDLERS                      '
    '                                                   '
    '''''''''''''''''''''''''''''''''''''''''''''''''''''
    #####################################################

    
    '''-------------------------------------EXPLORATION STATE---------------------------------------------------------------'''

    def handle_navigation_status(self, recovery_state=NavState.WAITING):
        if self.nav_future is not None and self.nav_future.done():
            result = self.nav_future.result().status
            if result == GoalStatus.STATUS_EXECUTING:
                self.get_logger().info(f"\nExecuting navigation to point: [{self.nav_goal.pose.pose.position.x}, {self.nav_goal.pose.pose.position.y}]\n")
                pass
            elif result == GoalStatus.STATUS_SUCCEEDED and self.nav_state != NavState.APPROACHING and self.nav_state != NavState.HOMEBOUND:
                self.get_logger().info(f"\nSuccesfully navigated to point: [{self.nav_goal.pose.pose.position.x}, {self.nav_goal.pose.pose.position.y}]\n")
                self.transition_to_state(NavState.WAITING)
            elif result == GoalStatus.STATUS_CANCELING:
                self.get_logger().info(f"\nCancelling navigation to point: [{self.nav_goal.pose.pose.position.x}, {self.nav_goal.pose.pose.position.y}]\n")
                pass
            elif result == GoalStatus.STATUS_CANCELED:
                self.get_logger().info(f"\nCancelled navigation to point: [{self.nav_goal.pose.pose.position.x}, {self.nav_goal.pose.pose.position.y}]\n")
                pass
            elif result == GoalStatus.STATUS_ABORTED:
                self.get_logger().warn(f"\nABORTED navigation to point: [{self.nav_goal.pose.pose.position.x}, {self.nav_goal.pose.pose.position.y}]")
                self.get_logger().warn(f"Recovering from abort. Resetting...")
                self.transition_to_state(recovery_state)
                pass
            elif result == GoalStatus.STATUS_ACCEPTED:
                self.get_logger().info(f"\nAccepted navigation to point: [{self.nav_goal.pose.pose.position.x}, {self.nav_goal.pose.pose.position.y}]\n")
                pass
            elif result == GoalStatus.STATUS_UNKNOWN:
                self.get_logger().warn(f"\nUNKNOWN STATUS of navigation to point: [{self.nav_goal.pose.pose.position.x}, {self.nav_goal.pose.pose.position.y}]\n")
                pass
            else:
                self.get_logger().warn(f"Unexpected navigation goal status. GoalStatus=[{result}]")
        elif self.nav_future is None: # error handling
            self.get_logger().warn(f"Navigation service lost. Resetting...")
            self.transition_to_state(recovery_state)
        elif not self.nav_future.done():
            self.get_logger().warn(f"Navigation server has not accepted goal yet...")

    '''-------------------------------------APPROACHING STATE---------------------------------------------------------------'''

    def handle_approaching(self):
        if self.nav_future is None:         # nav service is ready for a goal
            self.nav_goal = self.victim_goal
            self.send_goal()                # victim goal is already loaded, send it.
        else:                               # we're already navigating to the victim
            if self.nav_future.result().status == GoalStatus.STATUS_SUCCEEDED:
                # capture victim
                victim = self.build_victim_datum()
                # store victim data
                self.captured_victims.append(victim)
                # reset for exploration
                self.transition_to_state(NavState.WAITING)
            else:
                self.handle_navigation_status(recovery_state=NavState.APPROACHING)

    '''-------------------------------------RECALIBRATING STATE---------------------------------------------------------------'''

    def handle_recalibration(self):
        '''reset nav service then switch to next state'''
        if self.nav_future is not None:
            if self.cancel_future is None:  # nav service still running
                self.cancel_future = self.nav_future.result().cancel_goal_async()
            elif self.cancel_future.done(): # nav service acknowledged cancel
                self.nav_future = None
                self.cancel_future = None
            else:                           # wait for nav service to acknowledge cancel
                pass
        else:                               # nav service reset; move to next state
            self.nav_state = self.next_state

    '''-------------------------------------HOMEBOUND STATE---------------------------------------------------------------'''

    def handle_homebound(self):
        if self.nav_future is None:         # nav service is ready for a goal
            self.nav_goal = self.home_goal  # update navigation goal
            self.send_goal()                # send navigation goal to nav server
            self.get_logger().info("HOME GOAL SENT")
        else:                               # we're already navigating home :)
            if self.nav_future.result().status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info("\n\n\nARRIVED AT HOME GOAL\n\n\n")
                try:
                    rclpy.shutdown()
                except Exception as e:
                    self.get_logger().warn("Failed to shutdown.")
                pass
            else:
                self.handle_navigation_status(recovery_state=NavState.HOMEBOUND)

    '''---------------------------------------------------------------------------------------------------------------------'''



    #####################################################
    '''''''''''''''''''''''''''''''''''''''''''''''''''''
    '                                                   '
    '             HELPER FUNCTIONS                      '
    '                                                   '
    '''''''''''''''''''''''''''''''''''''''''''''''''''''
    #####################################################


    '''-------------------------------------SET UP FUNCTIONS----------------------------------------------------------------'''

    def setup(self):
        if not self.map: # no map
            self.get_logger().info("Waiting on map to be set...")
            pass
        elif not self.waypoints:
            # If waypoints list is empty generate them from the map
            self.get_logger().info("No waypoints set")
            pass
        elif not self.cur_image: # no image data from camera
            self.get_logger().info("No msg from camera...")
            pass
        else: # we are ready. change state to waiting
            self.get_logger().info("\n\n----STARTING COMPETITION----\n.\n.\n.")
            self.nav_state = NavState.WAITING
    
    def set_home_goal(self):
        '''
        Takes the current pose from /amcl_pose and assigns it to the home goal
        '''
        if not self.cur_pose:
            self.get_logger().warn(f"No msg to grab from /amcl_pose")
            return
        home_x = self.cur_pose.position.x
        home_y = self.cur_pose.position.y
        home_theta = self.cur_pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion(
            [home_theta.x, home_theta.y, home_theta.z, home_theta.w]
        )
        home_theta = yaw

        # if any(val is None for val in [home_x, home_y, home_theta]):
        #     return

        self.home_goal = create_nav_goal(home_x, home_y, home_theta)
        self.get_logger().info(f"Start Goal Set: ({home_x:.2f}, {home_y:.2f}, {home_theta})")

    '''-------------------------------------VICTIM TRACKING HELPERS---------------------------------------------------------'''

    def get_new_victim_pose(self, aruco_markers):
        '''
        Returns a target_victim_pose based off of the aruco marker given
        or None if no valid victims found
        '''
        header = aruco_markers.header   # a single header to represent the message
        poses = aruco_markers.poses     # a list of Poses of victims

        # iterate over each Pose, transform, and continue with the first new pose
        for pose in poses:
            ps = PoseStamped(header=header, pose=pose)

            try:
                map_ps = self.tf_buffer.transform(ps, "map")    # transform PoseStamped
                if self.victim_marked(map_ps):             # victim_marked takes a pose
                    continue                                    # skip victims we already marked
                else:
                    return map_ps                                # return the transformed Pose
            except Exception as e:
                self.get_logger().warn(f"Err: get_new_victim_pose() ::\n{e}\n\n")
                continue
        return None

    def build_victim_datum(self):
        victim = Victim()
        victim.id = self.victim_count       # 1- assign id based on when captured
        self.victim_count += 1

        pose_s = self.marked_victims[-1]    # grab the most recent victim marked
        point_s = PointStamped()
        point_s.header = pose_s.header
        point_s.point.x = pose_s.pose.position.x
        point_s.point.y = pose_s.pose.position.y
        point_s.point.z = pose_s.pose.position.z
        
        victim.point = point_s              # 2- assign point

        victim.image = self.cur_image       # 3- assign image

        victim.description = ""             # 4- assign empty image

        return victim

    def calculate_goal_from_victim_pose(self, victim_ps, distance=0.75):
        victim_quat = [
            victim_ps.pose.orientation.x,
            victim_ps.pose.orientation.y,
            victim_ps.pose.orientation.z,
            victim_ps.pose.orientation.w
        ]
        _, _, victim_yaw = tf_transformations.euler_from_quaternion(victim_quat)
        victim_yaw = victim_yaw

        # position distance away from target relative to victim orientation
        goal_x = victim_ps.pose.position.x + distance * math.cos(victim_yaw)
        goal_y = victim_ps.pose.position.y + distance * math.sin(victim_yaw)

        # orient to face victim (180 deg opposit of victim's orientation)
        goal_yaw = victim_yaw + math.pi
        goal_yaw = math.atan2(math.sin(goal_yaw), math.cos(goal_yaw))
        # convert orientation to quat
        # goal_quat = tf_transformations.quaternion_from_euler(0, 0, goal_yaw)

        return create_nav_goal(goal_x, goal_y, goal_yaw)
    
    def calculate_approach_goal(self, victim_ps, distance=0.75):
        ''' Calculate a goal for approaching the victim '''
        victim_pos = np.array([             # extract victim position
            victim_ps.pose.position.x,
            victim_ps.pose.position.y,
            victim_ps.pose.position.z
        ])
        victim_quat = [                     # extract victim orientation
            victim_ps.pose.orientation.x,
            victim_ps.pose.orientation.y,
            victim_ps.pose.orientation.z,
            victim_ps.pose.orientation.w
        ]
        # convert quat to rotation matrix
        victim_rot_matrix = tf_transformations.quaternion_matrix(victim_quat)
        # calculate victim's normal vector
        victim_normal = victim_rot_matrix[:3, 2]
        # calculate approach position [distance away along marker normal]
        approach_pos = victim_pos + victim_normal * distance

        # Calculate orientation
        direction_to_victim = victim_pos - approach_pos
        direction_to_victim = direction_to_victim / np.linalg.norm(direction_to_victim)
        # calculate yaw
        approach_yaw = np.arctan2(direction_to_victim[1], direction_to_victim[0])

        return create_nav_goal(approach_pos[0], approach_pos[1], approach_yaw)

    
    def mark_victim(self, victim_ps):
        self.marked_victims.append(victim_ps)
        self.get_logger().info(f"Marked Victim at [{victim_ps.pose.position.x:.4f}, {victim_ps.pose.position.y:.4f}]")
    
    def victim_marked(self, victim_ps):
        x_min = victim_ps.pose.position.x - self.victim_x_tolerance
        x_max = victim_ps.pose.position.x + self.victim_x_tolerance
        y_min = victim_ps.pose.position.y - self.victim_y_tolerance
        y_max = victim_ps.pose.position.y + self.victim_y_tolerance
        return any(
            x_min <= victim.pose.position.x <= x_max
            and y_min <= victim.pose.position.y <= y_max 
            for victim in self.marked_victims
        )
    
    '''-------------------------------COMPETITION MANAGEMENT HELPERS--------------------------------------------------------'''

    def transition_to_state(self, state):
        '''this triggers the navigation service to recalibrate
            to prepare to transition to a new state'''
        self.next_state = state
        self.nav_state = NavState.RECALIBRATING

    '''-------------------------------NAVIGATION / EXPLORATION HELPERS------------------------------------------------------'''

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
        self.nav_goal = create_nav_goal(x, y, theta) # set nav goal to next waypoint
        self.send_goal() # send nav goal to nav service
        self.get_logger().info(f"Waypoint goal sent.")

    def send_goal(self):
        self.get_logger().info("WAITING FOR NAVIGATION SERVER...")
        self.ac.wait_for_server()
        self.get_logger().info("NAVIGATION SERVER AVAILABLE...")
        self.get_logger().info("SENDING GOAL TO NAVIGATION SERVER...")

        self.nav_future = self.ac.send_goal_async(self.nav_goal)

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

    def cell_is_free(self, x, y):
        val = self.map.get_cell(x, y)
        if val == 0: # cell is definitely free
            return True
        else: # cell might not be free. default to occupied
            return False

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

'''-------------------------------------------------------------------------------------------------------------------------'''

def main(args=None):
    rclpy.init(args=args)
    node = WaypointNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()