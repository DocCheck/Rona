# add python path inside ROS2 workspace
import sys
sys.path.insert(0, "src/rona_pkg/rona_pkg")

from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from custom_interfaces.msg import CalibrationDataMsg, ExitMsg
from custom_interfaces.srv import CalibrationSrv
from .robot_handler import Rona
from rclpy.node import Node
from sty import fg

from rona_pkg import config_rona
import rclpy
import time

### fg(5) -> magenta log ###

# Calibration service class
class CalibrationService(Node):

    # Init
    def __init__(self):
        # ROS2 init
        super().__init__('calibration_service')
        self.srv = self.create_service(CalibrationSrv, 'calibration_service', self.service_callback)
        self.arm = Rona()

        custom_qos_profile = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(CalibrationDataMsg, 'calibration_data', qos_profile=custom_qos_profile)

        self.create_subscription(ExitMsg, 'exit_message', self.exit_callback, 1)

    # Drop-off calibration method
    def dropoff_calibration(self):

        self.get_logger().info(fg(5) + "Please move the arm in the desired position for the drop off point and press Enter to save the settings." + fg.rs)
        
        # Set robot into teaching mode
        self.arm.set_mode(mode=2)
        time.sleep(2)
 
        # Set robot back into auto mode
        self.arm.set_mode(mode=0)

        # Get the current position and save it into a point
        position_j = self.arm.get_position(mode=2)

        # Reset to home and move back to the point for test
        self.arm.go_home()
        self.get_logger().info(fg(5) + "Drop-off calibration successful." + fg.rs)

        return position_j


    # Service callback function
    def service_callback(self, request, response):
        response.success = False

        # Release instrument position calibration
        dropoff_position_j= self.dropoff_calibration()

        # Constant parameters to skip initial calibration by every start up of the system
        camera_position_delta = config_rona.eye_in_hand_calibration # eye-in-hand
        dropoff_position_j = config_rona.transfer_position # position for the instrument transfer

        response.camera_coords = camera_position_delta
        response.dropoff_coords = dropoff_position_j
        response.success = True

        # Publish calibrations to topic for instrument localization service
        pub_msg = CalibrationDataMsg()
        pub_msg.camera_coords = camera_position_delta
        self.publisher.publish(pub_msg)

        return response

    # Exit callback
    def exit_callback(self, msg):
        if msg.exit == True:
            exit()

def main():

    # Initialize node
    rclpy.init(args=None)
    calibration_service = CalibrationService()
    calibration_service.get_logger().warn(fg(5) + "Calibration service has been started" + fg.rs)
    rclpy.spin(calibration_service)

    calibration_service.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()