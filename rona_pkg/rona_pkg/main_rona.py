"""
RoNA main node
"""
# add python path inside ROS2 workspace
import sys
sys.path.insert(0, "src/rona_pkg/rona_pkg")

from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.node import Node

from custom_interfaces.srv import CalibrationSrv, CameraSrv, SpeakerSrv, VoiceSrv
from custom_interfaces.msg import ExitMsg
from .robot_handler import Rona, Point
from sty import fg

from rona_pkg import config_rona
import rclpy
import time

### fg(2) -> green log ###

class MainRona(Node):
    # init
    def __init__(self):
        super().__init__('main_rona')

        # Calibration service
        self.cl_client = self.create_client(CalibrationSrv, 'calibration_service')
        self.cl_client.wait_for_service()
        self.get_logger().warn(fg(2) + "Calibration client initiated" + fg.rs)
        self.cl_request = CalibrationSrv.Request()

        # Voice recognition service
        self.vr_client = self.create_client(VoiceSrv, 'voice_recognition_service')
        #self.vr_client = self.create_client(VoiceSrv, 'manual_control_service')
        self.vr_client.wait_for_service()
        self.get_logger().warn(fg(2) + "Voice recognition client initiated" + fg.rs)
        self.vr_request = VoiceSrv.Request()

        # Camera service
        self.camera_client = self.create_client(CameraSrv, 'camera_service')
        self.camera_client.wait_for_service()
        self.get_logger().warn(fg(2) + "Camera client initiated" + fg.rs)
        self.camera_request = CameraSrv.Request()

        # Speaker service
        self.speaker_client = self.create_client(SpeakerSrv, 'speaking_service')
        self.speaker_client.wait_for_service()
        self.get_logger().warn(fg(2) + "Speaker client initiated" + fg.rs)
        self.speaker_request = SpeakerSrv.Request()

        # Exit publisher
        custom_qos_profile = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.exit_publisher = self.create_publisher(ExitMsg, "exit_message", qos_profile=custom_qos_profile)
    
    # Calibration
    def calibration_sequence(self):

        # Confirm calibration start
        self.get_logger().info(fg(2)+ "Commencing calibration sequence. Stand back." + fg.rs)

        # Start calibration sequence
        self.cl_response = self.cl_client.call_async(self.cl_request)
        rclpy.spin_until_future_complete(self, self.cl_response)
        if self.cl_response.result().success == False:
            self.get_logger().info(fg(2)+ "Calibration error! Shutting down."+ fg.rs)
            exit()
        else:
            self.get_logger().info(fg(2)+ "Calibration successful."+ fg.rs)

    # Core function
    def system_start(self):

        # Confirm start mission
        self.get_logger().info(fg(2)+ "System has successfully booted up."+ fg.rs)
        system_ready = True

        # Initialize point and arm and go to home position
        self.point = Point()
        self.arm = Rona()
        self.arm.go_home()

        self.transfer_coordinates = self.cl_response.result().dropoff_coords
        return_coord = []
        self.holding_instrument = False

        # If successful, start main system behavior - infinite loop until "shut down" voice command
        while system_ready:

            # Call voice recognition service
            self.vr_response = self.vr_client.call_async(self.vr_request)
            rclpy.spin_until_future_complete(self, self.vr_response)
            spoken_command = self.vr_response.result()

            self.get_logger().info(fg(2) + str(spoken_command.command) + fg.rs)

            # System shut down command
            if spoken_command.command == "shut down":
                
                # Speaker block  
                self.speaker_request.request = "The system is shutting down"
                self.speaker_response = self.speaker_client.call_async(self.speaker_request)
                rclpy.spin_until_future_complete(self, self.speaker_response)
                                
                system_ready = False
                self.get_logger().warn(fg(2) + "Stopping system."+ fg.rs)
                break

            # Reset motors if the robot gets stuck
            elif spoken_command.command == "reset":
                
                # Speaker block  
                self.speaker_request.request = "The system is resetting"
                self.speaker_response = self.speaker_client.call_async(self.speaker_request)
                rclpy.spin_until_future_complete(self, self.speaker_response)

                # Clear errors 
                self.get_logger().warn(fg(2) + "Resetting robot position."+ fg.rs)
                self.arm.reset()

                # Check if we have an instrument in the gripper
                if self.holding_instrument:
                    self.arm.return_instrument(return_coord.pop())

                # Return to the home position 
                self.arm.go_home()

            # Drop-off position dynamic recalibration
            elif spoken_command.command == "recalibration":
                
                # Speaker block         
                self.speaker_request.request = "Recalibration started"
                self.speaker_response = self.speaker_client.call_async(self.speaker_request)
                rclpy.spin_until_future_complete(self, self.speaker_response)
                
                # Set robot into teaching mode
                self.arm.set_mode(mode=2)
                time.sleep(1)

                # Call voice recognition until the changes are saved
                while True:
                    # Call voice recognition service
                    self.vr_response = self.vr_client.call_async(self.vr_request)
                    rclpy.spin_until_future_complete(self, self.vr_response)
                    spoken_command = self.vr_response.result()

                    if spoken_command.command == "save":
                        
                        # Speaker block                
                        self.speaker_request.request = "The new coordinates have been saved"
                        self.speaker_response = self.speaker_client.call_async(self.speaker_request)
                        rclpy.spin_until_future_complete(self, self.speaker_response)
                        self.transfer_coordinates = self.arm.get_position(mode=2)
                        break

                    if spoken_command.command == "cancel":
                        # Speaker block
                        self.speaker_request.request = "Canceling recalibration"
                        self.speaker_response = self.speaker_client.call_async(self.speaker_request)
                        rclpy.spin_until_future_complete(self, self.speaker_response)
                        break

                    else:
                        # Speaker block
                        self.speaker_request.request = "Please save"
                        self.speaker_response = self.speaker_client.call_async(self.speaker_request)
                        rclpy.spin_until_future_complete(self, self.speaker_response)
                
                self.arm.set_mode(mode=1)
                self.arm.go_home()

            # Return current instrument back to tray
            elif spoken_command.command == "return":
                
                # Go to drop off position
                if not return_coord:
                    
                    # Speaker block
                    self.speaker_request.request = "Return is not possible"
                    self.speaker_response = self.speaker_client.call_async(self.speaker_request)
                    rclpy.spin_until_future_complete(self, self.speaker_response)
                
                else:

                    # Speaker block
                    self.speaker_request.request = "Return has been chosen"
                    self.speaker_response = self.speaker_client.call_async(self.speaker_request)
                    rclpy.spin_until_future_complete(self, self.speaker_response)

                    # Call camera service for object detection 
                    self.camera_request.request = "object_detection"
                    self.camera_request.instrument = ""
                    self.get_logger().warn(fg(2) + "### STARTING EMPTY SLOT DETECTION ###" + fg.rs)
                    self.camera_response = self.camera_client.call_async(self.camera_request)
                    rclpy.spin_until_future_complete(self, self.camera_response)

                    if self.camera_response.result().success:
                        current_position = self.arm.get_position()

                        # Move to transfer coordinates and await return
                        self.arm.move_to_j(self.transfer_coordinates, True)
                        self.arm.open_gripper()
                        return_xy = [self.camera_response.result().xr , self.camera_response.result().yr]

                        # Call return sequence in camera service
                        self.camera_request.request = "hand_detection"
                        self.get_logger().warn(fg(2) + "### STARTING HAND DETECTION ###" + fg.rs)
                        self.camera_response = self.camera_client.call_async(self.camera_request)
                        rclpy.spin_until_future_complete(self, self.camera_response)

                        if self.camera_response.result().success:
                            # Take instrument and return it to its place
                            self.arm.close_gripper()
                            self.holding_instrument = True
                            time.sleep(1) # To get your fingers away

                            # Set the point for the position of the instrument and go get it
                            self.point.set_point([current_position[0] + return_xy[0], current_position[1] + return_xy[1], 100, 180, 0, 0])
                            self.get_logger().info(fg(14) + f"Point position: [{self.point.point}] [mm]" + fg.rs)
                            self.arm.return_instrument(self.point)
                            return_coord.pop()
                            self.holding_instrument = False

                        # Go home
                        self.arm.go_home()
                    else:
                        self.get_logger().error(fg(4) + "Could not find free space to return" + fg.rs)  
                        
                        # Speaker block
                        self.speaker_request.request = "Could not find free space"
                        self.speaker_response = self.speaker_client.call_async(self.speaker_request)
                        rclpy.spin_until_future_complete(self, self.speaker_response)


            # Pick up and bring an instrument
            elif spoken_command.command == "give":
                            
                # Speaker block
                self.speaker_request.request = f'{spoken_command.instrument} chosen'
                self.speaker_response = self.speaker_client.call_async(self.speaker_request)
                rclpy.spin_until_future_complete(self, self.speaker_response)

                # Search until finding instrument
                start_time = time.time()
                current_time = time.time()
                while config_rona.instrument_detection_timeout > current_time - start_time: # search for x seconds, before timing out

                    # Call camera service for object detection 
                    self.camera_request.request = "object_detection"
                    self.camera_request.instrument = spoken_command.instrument
                    self.get_logger().warn(fg(2) + "### STARTING OBJECT DETECTION ###" + fg.rs)
                    self.camera_response = self.camera_client.call_async(self.camera_request)
                    rclpy.spin_until_future_complete(self, self.camera_response)

                    # If successful
                    if self.camera_response.result().success:
                        break
                
                    current_time = time.time()

                # If successful
                if self.camera_response.result().success:
                    # Speaker block
                    self.speaker_request.request = f'Target found'
                    self.speaker_response = self.speaker_client.call_async(self.speaker_request)
                    rclpy.spin_until_future_complete(self, self.speaker_response)

                    # Set the point for the position of the instrument and go get it
                    current_position = self.arm.get_position()
                    self.point.set_point([current_position[0]+self.camera_response.result().xi, current_position[1]+self.camera_response.result().yi, 100, 180, 0, self.camera_response.result().yaw])
                    self.get_logger().info(fg(2) + "MOVING TO POINT"+ fg.rs)

                    # Go pick instrument
                    self.arm.pick_instrument(self.point, width=self.camera_request.instrument)
                    self.holding_instrument = True
                    return_coord.append(self.point)

                    # Bring to drop off point
                    self.arm.move_to_j(self.transfer_coordinates, True)

                    # Call hand detection service
                    self.camera_request.request = "hand_detection"
                    self.get_logger().warn(fg(2) + "### STARTING HAND DETECTION ###" + fg.rs)
                    self.camera_response = self.camera_client.call_async(self.camera_request)
                    rclpy.spin_until_future_complete(self, self.camera_response)

                    # When the check is complete - open gripper
                    if self.camera_response.result().success:
                        self.arm.open_gripper(wait=True)
                        self.holding_instrument = False
                        self.arm.go_home()

                    else:
                        # Return instrument back to the tray if timed out
                        self.arm.return_instrument(return_coord.pop())
                        self.holding_instrument = False
                        self.arm.go_home()

                else:
                    self.get_logger().error(fg(4) + "Could not find instrument" + fg.rs)  

                    # Speaker block
                    self.speaker_request.request = "Could not find instrument"
                    self.speaker_response = self.speaker_client.call_async(self.speaker_request)
                    rclpy.spin_until_future_complete(self, self.speaker_response)
                    
            # Change with a new instrument
            elif spoken_command.command == "change": 

                # Speaker block
                self.speaker_request.request = f'Change with {spoken_command.instrument}'
                self.speaker_response = self.speaker_client.call_async(self.speaker_request)
                rclpy.spin_until_future_complete(self, self.speaker_response)

                # Search until finding instrument
                start_time = time.time()
                current_time = time.time()
                while config_rona.instrument_detection_timeout > current_time - start_time: # search for X seconds, before timing out

                    self.get_logger().warn(fg(2) + "### STARTING OBJECT DETECTION ###" + fg.rs)
                    # Call camera service for object detection for the new instrument
                    self.camera_request.request = "object_detection"
                    self.camera_request.instrument = spoken_command.instrument
                    self.camera_response = self.camera_client.call_async(self.camera_request)
                    rclpy.spin_until_future_complete(self, self.camera_response)

                    # If successful
                    if self.camera_response.result().success:
                        break
                    current_time = time.time()
                    
                # If successful
                if self.camera_response.result().success:
                    
                    # Set the point for the position of the instrument and go get it
                    current_position = self.arm.get_position()
                    new_xyyaw = [self.camera_response.result().xi , self.camera_response.result().yi, self.camera_response.result().yaw]
                    return_xy = [self.camera_response.result().xr , self.camera_response.result().yr]

                    # Reset voice recognition request 
                    self.vr_request.recognition_request = ""
                                           
                    # Move to transfer coordinates and await return
                    self.arm.move_to_j(self.transfer_coordinates, True)
                    self.arm.open_gripper()

                    # Call return sequence in camera service
                    self.camera_request.request = "hand_detection"
                    self.get_logger().warn(fg(2) + "### STARTING HAND DETECTION ###" + fg.rs)
                    self.camera_response = self.camera_client.call_async(self.camera_request)
                    rclpy.spin_until_future_complete(self, self.camera_response)

                    if self.camera_response.result().success:
                        # Take instrument and return it to its place
                        self.arm.close_gripper()
                        self.holding_instrument = True
                        time.sleep(1) # To get your fingers away 
                        
                        # Set the point for the position of the instrument and go get it
                        self.point.set_point([current_position[0] + return_xy[0], current_position[1] + return_xy[1], 100, 180, 0, 0])
                        self.arm.return_instrument(self.point)
                        self.holding_instrument = False
                        return_coord.pop()

                        # Go pick instrument and save new empty position
                        self.point.set_point([current_position[0] + new_xyyaw[0], current_position[1] + new_xyyaw[1], 100, 180, 0, new_xyyaw[2]])
                        self.arm.pick_instrument(self.point, width=self.camera_request.instrument)
                        self.holding_instrument = True
                        return_coord.append(self.point)

                        # Bring to drop off point
                        self.arm.move_to_j(self.transfer_coordinates, True)

                        # Call hand detection service
                        self.camera_request.request = "hand_detection"
                        self.get_logger().info(fg(2) + "### STARTING HAND DETECTION ###" + fg.rs)
                        self.camera_response = self.camera_client.call_async(self.camera_request)
                        rclpy.spin_until_future_complete(self, self.camera_response)

                        # When complete open gripper 
                        if self.camera_response.result().success:
                            self.arm.open_gripper(wait=True)
                            self.holding_instrument = False
                            self.arm.go_home()

                        else:
                            # Return instrument back to the tray if timed out
                            self.arm.return_instrument(return_coord.pop())
                            self.holding_instrument = False
                            self.arm.go_home()

                    # If no instrument gets returned, cancel and go home
                    else:
                        self.arm.go_home()
                        continue
                
                # If the chosen instrument does not get recognized by the camera, announce error
                else:
                    self.get_logger().info(fg(4) + "Could not find instrument" + fg.rs)   
                    # Speaker block
                    self.speaker_request.request = "Could not find instrument"
                    self.speaker_response = self.speaker_client.call_async(self.speaker_request)
                    rclpy.spin_until_future_complete(self, self.speaker_response)


        # Clean exit robot arm
        self.arm.shutdown()

        # Clean exit for all service nodes
        exit_msg = ExitMsg()
        exit_msg.exit = True
        self.exit_publisher.publish(exit_msg)
        

def main(args=None):
    rclpy.init(args=args)
    node = MainRona()
    node.calibration_sequence()
    node.system_start()

    node.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()