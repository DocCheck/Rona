#! /usr/bin/env python3
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import config_rona

sys.path.insert(0, config_rona.path_detector)
sys.path.insert(0, config_rona.path_inference)
sys.path.insert(0, config_rona.path_orientation)

from instrument_detector.load_config import load_mid_config, load_camera_config, load_slot_est_config, load_ori_est_config
from instrument_detector.object_detector.Rona_mid.Rona_mid_Inference import Rona_mid
from instrument_detector.orientation_estimator.RotNet.RotNet_Inference import RotNet
from instrument_detector.empty_space_detector.Instrument_return import Empty_space
from instrument_detector.object_detector.general_utils.img_util import crop_image_from_center
from instrument_detector.camera.Op_Camera import OP_Cam
from robot_sdk.xarm.wrapper.xarm_api import XArmAPI

import pyrealsense2 as rs
import numpy as np
import time


class DetectionTest():

    def __init__(self):

        ##################################
        ############## RONA ##############
        ##################################
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

        ##################################
        ######## INTEL REALSENSE #########
        ##################################

        # Configure depth and color streams for intel real sense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.colorizer = rs.colorizer()

        # Set camera frame size and configure streams
        self.frame_x = 1280
        self.frame_y = 720
        self.config.enable_stream(rs.stream.depth, self.frame_x, self.frame_y, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.frame_x, self.frame_y, rs.format.bgr8, 30)

        # Start stream
        self.profile = self.pipeline.start(self.config)

        ##################################
        ######## OBJECT DETECTION ########
        ##################################

        # Load config files
        self.model_config = load_mid_config()
        self.cam_config = load_camera_config()        
        self.ooe_config = load_ori_est_config()
        self.se_config = load_slot_est_config()

        # Create objects and load models             
        self.rotation_estimator = RotNet(self.ooe_config)
        self.slots_estimator = Empty_space(self.se_config)
        self.mid_detector = Rona_mid(self.model_config) 
        self.op_cam = OP_Cam(self.cam_config)

        # Load models
        self.model, self.classes_dict = self.mid_detector.load_model()
        self.rotation_estimator.load_model()

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

    def run(self):
        # Take frame(s)
        self.go_home()

        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Initiate clear variables for scan results
        predictions = []
    
        # Get image from frame
        color_image_og = np.asanyarray(color_frame.get_data())
        color_image = crop_image_from_center(color_image_og, self.cam_config["cam_frame_output_sz"])
        
        # The basic object detector
        predictions = self.mid_detector.predict(color_image)[0].tolist()

        # Instrument detection 
        if len(predictions) > 0:
            # Rotation estimator for all of the objects
            predictions = self.rotation_estimator.predict(color_image, predictions)
            predictions, _, _, return_slot_coord, _ = self.slots_estimator.fixed_slots_estimation(color_image, predictions, self.classes_dict, viz=True)

        time.sleep(1)
        self.shutdown()

def main():

    test = DetectionTest()
    test.run()

if __name__ == '__main__':
    main()