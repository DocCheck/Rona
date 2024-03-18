#! /usr/bin/env python3

# add python path inside ROS2 workspace
import sys
sys.path.insert(0, "src/rona_pkg/rona_pkg")

"""
xArm handler for RoNA
""" 
from .robot_sdk.xarm.wrapper.xarm_api import XArmAPI
from sty import fg 

from rona_pkg import config_rona
import logging

### fg(10) -> light green log ###

logging.getLogger().setLevel(logging.INFO)


# Point class
class Point:

    # Constructor for point object
    def __init__(self, point=[0, 0, 0, 0, 0, 0]):
        """
        Constructor with an input - list of values in mm and degrees
        """
        if not self.is_valid_input(point):
            raise ValueError("Invalid input: The input list must contain valid float values.")
        
        self._x = point[0]
        self._y = point[1]
        self._z = point[2]
        self._roll = point[3]
        self._pitch = point[4]
        self._yaw = point[5]

    def is_valid_input(self, input_list):
        """
        Check if the input list to the constructor is valid
        """
        if len(input_list) != 6:
            return False
        try:
            float_values = [float(value) for value in input_list]
            return len(float_values) == 6  # Ensure there are 6 valid float values
        except (ValueError, TypeError):
            return False
        
    @property
    def point(self):
        """
        Return point
        """
        return [self._x, self._y, self._z, self._roll, self._pitch, self._yaw]

    def set_point(self, point):
        """
        Set point attributes
        """
        if len(point) != 6:
            raise ValueError("Input list must contain exactly 6 elements.")
        
        try:
            self._x = float(point[0])
            self._y = float(point[1])
            self._z = float(point[2])
            self._roll = float(point[3])
            self._pitch = float(point[4])
            self._yaw = float(point[5])
        except ValueError as e:
            raise ValueError("Invalid input values: " + str(e))


# Joint class
class jPoint:

    # Constructor for joint object
    def __init__(self, jpoint=[0, 0, 0, 0, 0]):
        """
        Constructor with an input - list of values in degrees
        """
        if not self.is_valid_jnput(jpoint):
            raise ValueError("Invalid input: The input list must contain valid float values.")
        
        self._j1 = jpoint[0]
        self._j2 = jpoint[1]
        self._j3 = jpoint[2]
        self._j4 = jpoint[3]
        self._j5 = jpoint[4]

    def is_valid_jnput(self, input_list):
        """
        Check if the input list to the constructor is valid
        """
        if len(input_list) != 5:
            return False
        try:
            float_values = [float(value) for value in input_list]
            return len(float_values) == 5  # Ensure there are 5 valid float values
        except (ValueError, TypeError):
            return False
    
    @property
    def jpoint(self):
        """
        Return point
        """
        return [self._j1, self._j2, self._j3, self._j4, self._j5]

    def set_jpoint(self, jpoint):
        """
        Set jpoint attributes in degrees
        """
        if len(jpoint) != 5:
            raise ValueError("Input list must contain exactly 5 elements.")
        try:
            self._j1 = float(jpoint[0])
            self._j2 = float(jpoint[1])
            self._j3 = float(jpoint[2])
            self._j4 = float(jpoint[3])
            self._j5 = float(jpoint[4])
        except ValueError as e:
            raise ValueError("Invalid input values: " + str(e))

     
# Arm class
class Rona:

    def __init__(self):
        """
        Basic constructor
        """
        self._tcp_offset = config_rona.tcp_offset # TCP offset to the gripper tip
        self._home_position_j = config_rona.home_position_j # Rona's home position after each move
        self._initial_position_j = config_rona.initial_position_j # Rona's shutdown position

        self.SPEED = config_rona.speed_linear # [mm/s]
        self.ACC = config_rona.acc_linear # [mm/s²]
        self.ANG_SPEED = config_rona.speed_angular # [°/s]
        self.ANG_ACC = config_rona.acc_angular # [°/s²] 
        self.GRIPPER_SPEED_OPEN = config_rona.gripper_speed_open # [r/min]
        self.GRIPPER_SPEED_CLOSE = config_rona.gripper_speed_close # [r/min]
        self.Z_HEIGHT_DOWN = config_rona.z_height_down # [mm]
        self.Z_HEIGHT_UP = config_rona.z_height_up # [mm]
        self.HOSTNAME = config_rona.hostname

        """
        Dictionary with values of gripper widths for different instrument grabbing
        """
        self._gripper_width = {
            "default": 50,  
            "standard tweezers": config_rona.tweezers_width,
            "slim tweezers": config_rona.tweezers_width,
            "surgical tweezers": config_rona.tweezers_width,
            "splitter tweezers": config_rona.tweezers_width,
            "three handle": config_rona.handle_width,
            "four handle": config_rona.handle_width,
            "clenched scalpel": config_rona.scalpel_width,
            "narrow scalpel": config_rona.scalpel_width,
            "pointed scissors": config_rona.scissors_width,
            "blunt scissors": config_rona.scissors_width,
            "standard scissors": config_rona.scissors_width,
            "needle": config_rona.needle_width
        }

        logging.info(fg(10) + "Initializing robot arm." + fg.rs)
        self.arm = XArmAPI(self.HOSTNAME)
        self.arm.clean_error()
        self.arm.clean_warn()

        self.setup()
        self.setup_gripper()

    
    #######################
    ### SETUP FUNCTIONS ###
    #######################
    def setup(self):
        """
        Robot state and setup on start-up functions 
        """
        logging.info(fg(10) + "Enabling motion." + fg.rs)
        self.arm.motion_enable(enable=True)

        logging.info(fg(10) + "Setting settings." + fg.rs)
        self.arm.set_tcp_offset(self._tcp_offset)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)

    def setup_gripper(self):
        """
        Gripper setup functions
        """
        self.arm.set_gripper_mode(0)
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_speed(self.GRIPPER_SPEED_CLOSE)

    def set_mode(self, mode):
        """
        Robot control modes:
            0: position control mode in cartesian coordinates
            1: joint motion control in degrees
            2: teaching mode via free-hand movement
        """
        self.arm.set_mode(mode)
        self.arm.set_state(0)

    def get_position(self, mode=1):
        """
        Mode 1 - cartesian coordinates in mm/degrees
        Mode 2 - joint coordinates in degrees
        """
        ((code1,point)) = self.arm.get_position()
        ((code2,joint)) = self.arm.get_servo_angle()

        if code1 == 0 and code2 == 0:
            if mode == 1:
                return point
            if mode == 2:
                return joint
            else:
                return None

    def reset(self):
        """
        Reset errors and re-enable motion control
        """
        self.arm.clean_error()
        self.arm.motion_enable(enable=True)
        self.arm.set_state(state=0)

    #########################
    ### GRIPPER FUNCTIONS ###
    #########################
    def open_gripper(self, width=300, wait=False):
        """
        Open gripper with specific width
        """
        self.arm.set_gripper_position(pos=width, wait=wait, speed=self.GRIPPER_SPEED_OPEN)

    def close_gripper(self, width="default", wait=True):
        """
        Close gripper with specific width
        """
        self.arm.set_gripper_position(pos=self._gripper_width[width], wait=wait, speed=self.GRIPPER_SPEED_CLOSE)

    def react_gripper(self):
        """
        Reaction sequence of the gripper, when the wake-word is heard
        """
        self.close_gripper()    
        self.open_gripper()


    ##########################
    ### COMPOUND MOVEMENTS ###
    ##########################

    # Simple movements
    def go_home(self):
        """
        Drive Rona to the home position
        """
        logging.warn(fg(10) + "Robot is moving to home. Watch out!" + fg.rs)
        self.arm.set_servo_angle(angle=self._home_position_j, speed=self.ANG_SPEED, mvacc=self.ANG_ACC, wait=True)

    def move_to(self, point):
        """
        Movement command in cartesian coordinates
        """
        logging.warn(fg(10) + "Robot is moving. Watch out!" + fg.rs)
        self.arm.set_position(x=point._x, y=point._y, z=point._z, roll=point._roll, pitch=point._pitch, yaw=point._yaw, speed=self.SPEED, mvacc=self.ACC, wait=True)
    
    def move_to_j(self, joints, wait=False):
        """
        Movement command in joint coordinates
        """
        logging.warn(fg(10) + "Robot is moving. Watch out!" + fg.rs)
        self.arm.set_servo_angle(angle=joints, speed=self.ANG_SPEED, mvacc=self.ANG_ACC, wait=wait)

    def shutdown(self):
        """
        Drive Rona to the initial position and shutdown
        """
        self.arm.set_servo_angle(angle=self._initial_position_j, speed=self.ANG_SPEED, mvacc=self.ANG_ACC, wait=True)
        self.arm.disconnect()

    def pick_instrument(self, point, width="default"):
        """
        The move for the robot to go and pick up an instrument from given x,y and yaw coordinates
        """
        logging.warn(fg(10) + "Robot is moving to pick up!" + fg.rs)

        # Assume cartesian input
        coords = point.point 
        coords[2] = self.Z_HEIGHT_UP
        ((_, joints)) = self.arm.get_inverse_kinematics(pose=coords, input_is_radian=False, return_is_radian=False)
        self.arm.set_servo_angle(angle=joints, speed=self.ANG_SPEED, mvacc=self.ANG_ACC, wait=True)
        ((code,joint)) = self.arm.get_servo_angle()
        
        self.open_gripper()
        self.arm.set_position(x=point._x, y=point._y, z=self.Z_HEIGHT_DOWN, roll=point._roll, pitch=point._pitch, yaw=point._yaw, speed=self.SPEED, mvacc=self.ACC)
        
        self.close_gripper(width)
        self.arm.set_position(x=point._x, y=point._y, z=self.Z_HEIGHT_UP, roll=point._roll, pitch=point._pitch, yaw=point._yaw, speed=self.SPEED, mvacc=self.ACC)
        
        logging.warn(fg(10) + "Pick was successful!" + fg.rs)
       
    def return_instrument(self, point, width="default"):
        """
        The move for the robot to return an instrument from transfer point to free slot on the table via given x,y coordinates
        """
        logging.warn(fg(10) + "Robot is moving to return!" + fg.rs)

        # Assume cartesian input
        coords = point.point 
        coords[2] = self.Z_HEIGHT_UP
        ((_, joints)) = self.arm.get_inverse_kinematics(pose=coords, input_is_radian=False, return_is_radian=False)
        self.arm.set_servo_angle(angle=joints, speed=self.ANG_SPEED, mvacc=self.ANG_ACC, wait=True)
        ((code,joint)) = self.arm.get_servo_angle()

        self.arm.set_position(x=point._x, y=point._y, z=self.Z_HEIGHT_DOWN, roll=point._roll, pitch=point._pitch, yaw=point._yaw, speed=self.SPEED, mvacc=self.ACC)
        self.open_gripper()
        self.arm.set_position(x=point._x, y=point._y, z=self.Z_HEIGHT_UP, roll=point._roll, pitch=point._pitch, yaw=point._yaw, speed=self.SPEED, mvacc=self.ACC)