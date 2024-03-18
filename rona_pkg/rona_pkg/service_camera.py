# add python path inside ROS2 workspace
import sys
sys.path.insert(0, "src/rona_pkg/rona_pkg")
import config_rona
 
sys.path.insert(0, config_rona.path_detector)
sys.path.insert(0, config_rona.path_inference)
sys.path.insert(0, config_rona.path_orientation)

from .instrument_detector.load_config import load_mid_config, load_camera_config, load_slot_est_config, \
    load_ori_est_config
from .instrument_detector.object_detector.Rona_mid.Rona_mid_Inference import Rona_mid
from .instrument_detector.orientation_estimator.RotNet.RotNet_Inference import RotNet
from .instrument_detector.empty_space_detector.Instrument_return import Empty_space
from .instrument_detector.object_detector.general_utils.img_util import crop_image_from_center
from .instrument_detector.camera.Op_Camera import OP_Cam


from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from custom_interfaces.msg import AllInstrumentsMsg, ControllerCommandMsg, ControllerFeedbackMsg, CalibrationDataMsg, \
    ExitMsg, SingleInstrumentMsg
from custom_interfaces.srv import CameraSrv
from rclpy.node import Node
from sty import fg

import pyrealsense2 as rs
import mediapipe as mp
import numpy as np
import signal
import rclpy
import time
import cv2
import os


### fg(X) -> X log ###

class CameraService(Node):

    # Init
    def __init__(self):
        self.PID = os.getpid()
        ##################################
        ############# ROS2 ###############
        ##################################

        # Initialize ROS2 node and service
        super().__init__('camera_service')
        self.group = ReentrantCallbackGroup()
        self.srv = self.create_service(CameraSrv, 'camera_service', self.service_callback, callback_group=self.group)

        self.sub1 = self.create_subscription(CalibrationDataMsg, 'calibration_data', self.calibration_callback, 1,
                                             callback_group=self.group)
        self.sub2 = self.create_subscription(ControllerFeedbackMsg, "controller_feedback", self.controller_callback, 1,
                                             callback_group=self.group)
        self.sub3 = self.create_subscription(ExitMsg, 'exit_message', self.exit_callback, 1, callback_group=self.group)

        custom_qos_profile = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                                        reliability=QoSReliabilityPolicy.RELIABLE,
                                        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(ControllerCommandMsg, "controller_command",
                                               qos_profile=custom_qos_profile)

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
        ########## CALIBRATION ###########
        ##################################

        # placeholder

        ##################################
        ######### HAND DETECTION #########
        ##################################

        # Mediapipe parameters for hand detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

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

    # Calibration info callback
    def calibration_callback(self, calibration_msg):
        self.camera_position = calibration_msg.camera_coords

    # Controller info callback
    def controller_callback(self, controller_msg):
        self.get_logger().warn(fg(7) + "Controller feedback received in subscriber." + fg.rs)
        if controller_msg.controller_feedback == "success":
            self.controller_answer = True
        else:
            self.controller_answer = False

        self.scanning = False

    # Main service callback
    def service_callback(self, request, response):

        # Check which sub-service has been called via the request
        mode = request.request

        ##################################
        ########## CALIBRATION ###########
        ##################################
        if mode == "calibration":
            self.get_logger().info(fg(11) + "Starting calibration." + fg.rs)


        ##################################
        ######## OBJECT DETECTION ########
        ##################################
        elif mode == "object_detection":
            self.get_logger().info(fg(11) + "Starting object detection." + fg.rs)

            # Take frame(s)
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # Initiate clear variables for scan results
            full_detection = AllInstrumentsMsg()
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
                predictions, _, _, return_slot_coord, _ = self.slots_estimator.fixed_slots_estimation(color_image,
                                                                                                      predictions,
                                                                                                      self.classes_dict)

            # Only return needed
            if request.instrument == "":
                xr_cf_px = return_slot_coord[0] - 15 + 160  # 160 is pixel correction for the cropped frame
                yr_cf_px = return_slot_coord[1] - 13

                # Get depth information of the image
                align = rs.align(rs.stream.color)
                frames = align.process(frames)
                aligned_depth_frame = frames.get_depth_frame()

                x1 = int(1280 / 2) - 50
                x2 = int(1280 / 2) + 50
                y1 = int(720 / 2) - 50
                y2 = int(720 / 2) + 50

                depth = np.asanyarray(aligned_depth_frame.get_data())
                depth = depth[y1:y2, x1:x2].astype(float)
                depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
                depth = depth * depth_scale

                depth = cv2.mean(depth)
                depth = depth[0]

                # Get camera intrinsic parameters from frame
                color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
                dxr, dyr, _ = rs.rs2_deproject_pixel_to_point(color_intrin, [xr_cf_px, yr_cf_px], depth)

                # Position of the instrument in the camera's coordinate system in mm
                xr_cf_mm = round(dxr, 3) * 1000
                yr_cf_mm = round(dyr, 3) * 1000

                # Position of the camera's coordinate system based on the TCP's coordinate system (eye-in-hand)
                x_eih = self.camera_position[0]
                y_eih = self.camera_position[1]

                # Position of the instrument in the TCP's coordinate system
                xr_tcp_mm = x_eih - yr_cf_mm
                yr_tcp_mm = y_eih - xr_cf_mm

                # Finalize service response
                response.xr = xr_tcp_mm
                response.yr = yr_tcp_mm
                response.success = True

                return response

            # Give or change
            else:
                # Extract the target IDs via the voice recognition command
                instrument_dict = {
                    "standard tweezers": [0.0],
                    "slim tweezers": [1.0],
                    "surgical tweezers": [2.0],
                    "splitter tweezers": [3.0],
                    "three handle": [4.0],
                    "four handle": [5.0],
                    "clenched scalpel": [6.0],
                    "narrow scalpel": [7.0],
                    "pointed scissors": [8.0],
                    "blunt scissors": [9.0],
                    "standard scissors": [10.0],
                    "needle": [11.0]
                }
                targets = instrument_dict[request.instrument]
                instrument = []

                # Prepare information on all of the detected instruments
                for obj in predictions:
                    # Prepare result
                    instrument_info = SingleInstrumentMsg()
                    ###
                    instrument_info.x1 = obj[0] + 160  # 160 is pixel correction for the cropped frame
                    instrument_info.y1 = obj[1]
                    instrument_info.x2 = obj[2] + 160  # 160 is pixel correction for the cropped frame
                    instrument_info.y2 = obj[3]
                    instrument_info.accuracy = obj[4]
                    instrument_info.id = obj[5]
                    instrument_info.rotation = obj[6]
                    ###
                    full_detection.full_detection_list.append(instrument_info)

                all_instruments = full_detection.full_detection_list
                for single_instrument in all_instruments:
                    if single_instrument.id in targets:  # Target found
                        instrument = single_instrument
                        break

                # If instrument list empty
                if not instrument:
                    self.get_logger().error(fg(14) + "Could not find a match." + fg.rs)
                    response.success = False
                    return response

                else:
                    # The list instrument has all the information from the image in camera frame(cf)
                    x_cf_px = int((instrument.x1 + instrument.x2) / 2 - 15)  # [px]
                    y_cf_px = int((instrument.y1 + instrument.y2) / 2 - 13)  # [px]
                    rot_cf = instrument.rotation  # [°]
                    xr_cf_px = return_slot_coord[0] - 15 + 160  # 160 is pixel correction for the cropped frame
                    yr_cf_px = return_slot_coord[1] - 13

                    # Get depth information of the image
                    align = rs.align(rs.stream.color)
                    frames = align.process(frames)
                    aligned_depth_frame = frames.get_depth_frame()

                    x1 = int(1280 / 2) - 50
                    x2 = int(1280 / 2) + 50
                    y1 = int(720 / 2) - 50
                    y2 = int(720 / 2) + 50

                    depth = np.asanyarray(aligned_depth_frame.get_data())
                    depth = depth[y1:y2, x1:x2].astype(float)
                    depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
                    depth = depth * depth_scale

                    depth = cv2.mean(depth)
                    depth = depth[0]

                    self.get_logger().info(fg(14) + f"Target coords: [{x_cf_px}:{y_cf_px}]  [px]" + fg.rs)
                    self.get_logger().info(fg(14) + f"Return coords: [{xr_cf_px}:{yr_cf_px}]  [px]" + fg.rs)

                    # Get camera intrinsic parameters from frame
                    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
                    dx, dy, _ = rs.rs2_deproject_pixel_to_point(color_intrin, [x_cf_px, y_cf_px], depth)
                    dxr, dyr, _ = rs.rs2_deproject_pixel_to_point(color_intrin, [xr_cf_px, yr_cf_px], depth)

                    # Position of the instrument and/or empty slot in the camera's coordinate system in mm
                    x_cf_mm = round(dx, 3) * 1000
                    y_cf_mm = round(dy, 3) * 1000
                    depth_mm = round(depth, 3) * 1000

                    xr_cf_mm = round(dxr, 3) * 1000
                    yr_cf_mm = round(dyr, 3) * 1000

                    # Position of the camera's coordinate system based on the TCP's coordinate system
                    x_eih = self.camera_position[0]
                    y_eih = self.camera_position[1]

                    # Position of the instrument and/or empty slot in the TCP's coordinate system
                    x_tcp_mm = x_eih - y_cf_mm
                    y_tcp_mm = y_eih - x_cf_mm

                    xr_tcp_mm = x_eih - yr_cf_mm
                    yr_tcp_mm = y_eih - xr_cf_mm

                    # Rotation limiter for gripper
                    if rot_cf >= 0 and rot_cf < 180:
                        yaw_tcp = float(rot_cf) - 90.0

                    else:
                        yaw_tcp = float(rot_cf) - 270.0

                    self.get_logger().info(fg(14) + f"Target coords: [{x_tcp_mm}:{y_tcp_mm}]  [mm]" + fg.rs)
                    self.get_logger().info(fg(14) + f"Return coords: [{xr_tcp_mm}:{yr_tcp_mm}]  [mm]" + fg.rs)

                    # Finalize service response
                    response.xi = x_tcp_mm
                    response.yi = y_tcp_mm
                    response.yaw = yaw_tcp
                    response.xr = xr_tcp_mm
                    response.yr = yr_tcp_mm
                    response.success = True

                    return response


        ##################################
        ######### HAND DETECTION #########
        ##################################
        elif mode == "hand_detection":
            time.sleep(1)
            self.get_logger().info(fg(11) + "Starting hand detection." + fg.rs)

            # Detection timer
            initial_detection_time = None
            self.controller_answer = False
            self.scanning = False
            success = False

            # Scan for hands for a certain amount of time
            t_end = time.time() + config_rona.time_hand_detection

            # Hand detection
            with self.mp_hands.Hands(
                    model_complexity=0,
                    min_detection_confidence=0.8,
                    min_tracking_confidence=0.8) as hands:

                self.get_logger().warn(fg(7) + "Scanning for hands now..." + fg.rs)

                # Main while loop - the full time you have to take the instrument
                while time.time() < t_end:

                    # Get frames via Intel Realsense library
                    frames = self.pipeline.wait_for_frames()
                    depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    if not depth_frame or not color_frame:
                        continue

                    # Convert images to numpy arrays
                    color_image = np.asanyarray(color_frame.get_data())
                    image = color_image

                    # To improve performance, optionally mark the image as not writeable to pass by reference.
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = hands.process(image)

                    # Draw the hand annotations on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # If hands are recognized in the frame
                    if results.multi_hand_landmarks:

                        # Calculate the distance based off the palm joint coordinates
                        x_coord = results.multi_hand_landmarks[0].landmark[0].x
                        y_coord = results.multi_hand_landmarks[0].landmark[0].y

                        x_coord = int(x_coord * self.frame_x)
                        y_coord = int(y_coord * self.frame_y)

                        # Filter out bugs that return negative x or y coordinates when hand leaves frame very fast
                        if x_coord <= 0 or x_coord > self.frame_x or y_coord <= 0 or y_coord > self.frame_y:
                            continue

                        # Get depth information of the image and hand
                        align = rs.align(rs.stream.color)
                        frames = align.process(frames)
                        aligned_depth_frame = frames.get_depth_frame()

                        x1 = x_coord - 50
                        x2 = x_coord + 50
                        y1 = y_coord - 50
                        y2 = y_coord + 50

                        depth = np.asanyarray(aligned_depth_frame.get_data())
                        depth = depth[y1:y2, x1:x2].astype(float)
                        depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
                        depth = depth * depth_scale

                        depth = cv2.mean(depth)
                        dist = depth[0]  # distance in [m]
                        dist = round(dist * 100, 2)  # distance in [cm]

                        # If hands are detected for the first time in close enough range, store the time in a variable
                        if initial_detection_time is None and dist < config_rona.distance_hand:
                            initial_detection_time = time.time()

                        # If the hand goes away, reset detection
                        elif dist >= config_rona.distance_hand:
                            initial_detection_time = None

                        # If the continuos hand detection is longer than "time_hand_seen" seconds and the hand is closer than "distance_hand" to the camera, release instrument
                        elif initial_detection_time is not None and dist < config_rona.distance_hand:

                            if time.time() - initial_detection_time >= config_rona.time_hand_seen:

                                self.get_logger().info(fg(11) + "Sending scanning command to controller." + fg.rs)
                                # Start the shake sensor           
                                pub_msg = ControllerCommandMsg()
                                pub_msg.command = "sensor"
                                self.publisher.publish(pub_msg)
                                self.scanning = True

                                timeout_sensor = time.time() + config_rona.time_sensor_timeout
                                # Wait for sensor scanning to finish
                                log_true = True
                                while self.scanning and time.time() < timeout_sensor:
                                    if log_true:
                                        self.get_logger().warn(fg(11) + "Scanning for impulse..." + fg.rs)
                                        log_true = False
                                    time.sleep(0.1)

                                # If the controller sensed a shake - end loop and finish with hand detection
                                if self.controller_answer:
                                    self.get_logger().info(fg(11) + "Scanning successful!" + fg.rs)
                                    success = True
                                    break

                                # If the controller didn't sense a shake - restart hand scanning if the time of the main loop hasn't expired
                                else:
                                    self.get_logger().error(fg(11) + "Scanning failed!" + fg.rs)
                                    initial_detection_time = None
                                    self.controller_answer = False
                                    self.scanning = False

                    else:
                        # If the camera loses track of the hand, reset timer
                        initial_detection_time = None

                    time.sleep(0.1)
            # Save the success boolean and check it in the main function
            response.success = success

        ##################################
        ############# ERROR ##############
        ##################################
        else:
            self.get_logger().error(fg(11) + "Error in service request." + fg.rs)
            response.success = False

        return response

    # Exit callback
    def exit_callback(self, msg):
        if msg.exit == True:
            self.destroy_node()
            os.kill((self.PID), signal.SIGTERM)


def main():
    # Start camera service and spin node
    rclpy.init(args=None)
    camera_service = CameraService()
    executor = MultiThreadedExecutor()
    executor.add_node(camera_service)
    camera_service.get_logger().info(fg(11) + "Camera service has been started" + fg.rs)
    executor.spin()

    # Clean up node and shut down
    camera_service.destroy_node()
    executor.shutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
