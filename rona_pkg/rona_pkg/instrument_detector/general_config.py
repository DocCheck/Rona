#! /usr/bin/env python3
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import config_rona

########################
### Inference Config ###
########################

model_path = config_rona.path_mid_model
image_size =  (960, 960) # (480, 480)
confidence_threshold = 0.75
iou_threshold = 0.6
classes = None
agnostic_nms = True
maximum_detection = 1000
device = "cpu"

#########################################
### Obj Orientation Estimation Config ###
#########################################

ooe_gaussian_blur_filter = (9, 9)
ooe_binary_thr_min = 125
ooe_binary_thr_max = 255
ooe_crop_padding = 10

ooe_rotnet_model_path = config_rona.path_rotnet_model
ooe_rotnet_padding = 0
ooe_rotnet_image_size = (128, 128)
ooe_device = "cpu"

#####################################
### Empty Space Estimation Config ###
#####################################

slot_est_grid_size = (6, 2) #(width, height)
slot_est_save_frame = True
slot_est_frame_path = config_rona.path_empty_space

return_same_slot = False
elevated_tray_ratio = 0.0001
obj_elevated_ratio = 0.0001
min_gripper_width = 55 #pixels

######################
### Camera Config  ###
######################

cam_port = 10
cam_frame_width =  1280
cam_frame_height =  720
cam_save_frame = True
cam_frame_path = config_rona.path_output
cam_center_crop_size = (960, 720)

#####################
### Runner Config ###
#####################

runner_on_demand = True
runner_period = 1 #seconds