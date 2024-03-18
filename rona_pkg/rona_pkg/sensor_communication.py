# add python path inside ROS2 workspace
import sys
sys.path.insert(0, "src/rona_pkg/rona_pkg")

from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from custom_interfaces.msg import ControllerCommandMsg, ControllerFeedbackMsg, ExitMsg
from rclpy.node import Node
from sty import fg 

import serial.tools.list_ports
from rona_pkg import config_rona
import serial
import rclpy
import time

### sudo chmod a+rw /dev/ttyACM0 ###
class SensorController(Node):

    # Init
    def __init__(self):
        # ROS2 init
        super().__init__('controller_node')
        self.create_subscription(ControllerCommandMsg, 'controller_command', self.subscriber_callback, 1)
        self.create_subscription(ExitMsg, 'exit_message', self.exit_callback, 1)

        custom_qos_profile = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(ControllerFeedbackMsg, "controller_feedback", qos_profile=custom_qos_profile)

        # Controller initialization
        try:
            self.controller = serial.Serial(port=config_rona.controller_port, parity=serial.PARITY_EVEN, stopbits=serial.STOPBITS_ONE, timeout=1)
            self.no_controller = False
        except:
            self.get_logger().info(fg(7) + "No controller found." + fg.rs)
            self.no_controller = True

    # Subscriber callback
    def subscriber_callback(self, msg):
        # Read command from topic
        command = msg.command

        # If no controller connected, skip writing function
        if self.no_controller:
            self.get_logger().info(fg(7) + "No controller found." + fg.rs)

        else:

            # Prepare communication and send command
            self.controller.flush()
            self.controller.write(bytes(command + "\r\n", 'utf-8'))
            time.sleep(0.05)
            self.get_logger().info(fg(7) + "Sent command to controller" + fg.rs)

            # Read feedback from the controller
            sensor_feedback = "fail"
            if command == "sensor":
                controller_feedback_msg = ControllerFeedbackMsg()
                listening_duration = time.time() + 5 # Listens for 5 seconds
                while time.time() < listening_duration:
                    
                    # If there is a serial command send from the controller
                    if self.controller.in_waiting:
                        # Format feedback
                        sensor_feedback = self.controller.readline()
                        sensor_feedback = sensor_feedback.decode('utf-8')
                        sensor_feedback = sensor_feedback.strip()
                        self.get_logger().info(fg(7) + "THIS IS THE SENSOR FEEDBACK STRING: " + sensor_feedback + fg.rs)
                        
                        # If the sensor is successful 
                        if sensor_feedback == "success":
                            controller_feedback_msg.controller_feedback = "success"
                            self.publisher.publish(controller_feedback_msg)
                            break

    # Exit callback
    def exit_callback(self, msg):
        if msg.exit == True:
            exit()

def main():

    # Initialize node
    rclpy.init(args=None)
    controller_control = SensorController()
    controller_control.get_logger().warn(fg(5) + "Controller control has been started" + fg.rs)
    rclpy.spin(controller_control)

    controller_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()