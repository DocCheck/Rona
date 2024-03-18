#! /usr/bin/env python3
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from robot_sdk.xarm.wrapper.xarm_api import XArmAPI
import config_rona
import time

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
        self.arm.motion_enable(enable=True)

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

    def open_gripper(self, width=300, wait=False):
        """
        Open gripper with specific width
        """
        self.arm.set_gripper_position(pos=width, wait=wait, speed=self.GRIPPER_SPEED_OPEN)

    def close_gripper(self, width="default", wait=True):
        """
        Close gripper with specific width
        """
        self.arm.set_gripper_position(pos=50, wait=wait, speed=self.GRIPPER_SPEED_CLOSE)

    def go_home(self):
        """
        Drive Rona to the home position
        """
        self.arm.set_servo_angle(angle=self._home_position_j, speed=self.ANG_SPEED, mvacc=self.ANG_ACC, wait=True)

    def shutdown(self):
        """
        Drive Rona to the initial position and shutdown
        """
        self.arm.set_servo_angle(angle=self._initial_position_j, speed=self.ANG_SPEED, mvacc=self.ANG_ACC, wait=True)
        self.arm.disconnect()

def main(args=None):
    # Test robot
    rona = Rona()
    rona.go_home()
    time.sleep(1)

    rona.open_gripper()
    time.sleep(1)

    rona.close_gripper()
    time.sleep(1)

    rona.shutdown()

if __name__ == '__main__':
    main()
