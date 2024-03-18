#! /usr/bin/env python3
import os
import sys
import inspect

import sys
sys.path.insert(0, "src/rona_pkg/rona_pkg")

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

'''
General config file for RoNA

All the NaN values need to be adjusted for your specific system
All other values are set to the 'optimal', but can be changed as per the user's wish

'''

#####################
### robot_handler ###
#####################

tcp_offset = [0, 0, 172, 0, 0, 0] # TCP offset to the gripper tip in Cartesian coordinates
home_position_j = [0, -49.5, -49.5, 105, 0] # Rona's home position after each move in Joint coordinates
initial_position_j = [0, -52, -8, 60, 0] # Rona's shutdown position in Joint coordinates

speed_linear = 500 # [mm/s]
speed_angular = 250 # [°/s]

acc_linear = 1000 # [mm/s²]
acc_angular = 300 # [°/s²] ! more than this could be unsafe !

gripper_speed_open = 2000 # [r/min]
gripper_speed_close = 1000 # [r/min]

z_height_down = 2.5 # [mm] - from the robot's base, this is where RoNA reaches down to pick up an instrument - adjust accordingly
z_height_up = 50 # [mm] - from the robot's base, this is where RoNA reaches up before picking and after she has picked up - adjust accordingly

hostname = "192.168.1.198" 

tweezers_width = 55
handle_width = 80
scalpel_width = 90
scissors_width = 80
needle_width = 40

############################
### sensor_communication ###
############################

controller_port = "/dev/ttyACM0" # this is usually the default port, change if needed
time_listening_controller = 5 # [s] the time the system is listening for controller feedback

###########################
### service_calibration ###
###########################

eye_in_hand_calibration = [122.0, 33.0, 200.0, 0.0, 0.0, 0.0]  # Position of the camera origin in Cartesian coordinates
transfer_position = [-22.0, -18.20, -57.5, 51.5, 3.5, 0.0, 0.0] # Position in Cartesian coordinates

######################
### service_camera ###
######################

path_detector = 'src/rona_pkg/rona_pkg/instrument_detector'
path_inference = 'src/rona_pkg/rona_pkg/instrument_detector/object_detector/Rona_mid'
path_orientation = 'src/rona_pkg/rona_pkg/instrument_detector/orientation_estimator/RotNet'

time_hand_detection = 10 # [s] the duration of the hand detection running each cycle
time_sensor_timeout = 7 # [s] the duration after which the sensor times out each cycle
time_hand_seen = 0.5 # [s] the minimum time the hand needs to continuously be seen in frame to be considered "detected"
distance_hand = 30 # [cm] the minimum distance of the hand to the camera to be considered close enough to consider instrument release

###########################
### mid_general_config ###
###########################

path_mid_model = "src/rona_pkg/rona_pkg/instrument_detector/object_detector/Rona_mid/Rona_mid_model/weights/instrument_detector_model.pt"
path_rotnet_model = "src/rona_pkg/rona_pkg/instrument_detector/orientation_estimator/RotNet/data/models/instrument_detector_model.pt"
path_empty_space = "src/rona_pkg/rona_pkg/instrument_detector/output/slot_estimation/"
path_output = "src/rona_pkg/rona_pkg/instrument_detector/output/camera/"

#######################
### service_speaker ###
#######################

path_audio_files = "src/rona_pkg/rona_pkg/sound_speaker/"

#################################
### service_voice_recognition ###
#################################

path_ww_model = "src/rona_pkg/rona_pkg/voice_recognition/model_wakeword/hey_rona.onnx"
whisper_model = "tiny"
time_awake = 5 # [s] the amount of time the system stays awake after wake-word is heard

#################
### main_rona ###
#################

instrument_detection_timeout = 5 # [s] the amount of time the system searches for the instrument, before timing out and canceling search