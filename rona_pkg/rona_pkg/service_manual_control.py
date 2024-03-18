# add python path inside ROS2 workspace
import sys
sys.path.insert(0, "src/rona_pkg/rona_pkg")

from rclpy.node import Node

from custom_interfaces.srv import VoiceSrv
from custom_interfaces.msg import ExitMsg
from .robot_handler import Rona
from playsound import playsound
from sty import fg

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String

from rona_pkg import config_rona
import signal
import rclpy
import time
import os

class ManualControlService(Node):
 
    # Init
    def __init__(self):
        # Initialize node
        self.PID = os.getpid()
        super().__init__('manual_control_service')
        self.group = ReentrantCallbackGroup()
        self.srv = self.create_service(VoiceSrv, 'manual_control_service', self.service_callback, callback_group=self.group)
        # self.arm = Rona()

        # Speaker single call variable
        self.system_started = False   
        self.full_path = config_rona.path_audio_files

        self.special_instruments = {
            "tweezers": ["standard", "slim", "surgical"],
            "handle": ["three", "four"],
            "scalpel": ["clenched", "narrow"],
            "scissors": ["pointed", "blunt"]
        }

        self.command = ["x", "x", "x", "x"] # [command, instrument, error, flag]

        # Exit subscriber
        self.create_subscription(ExitMsg, 'exit_message', self.exit_callback, 1, callback_group=self.group)

        # Command subscriber
        self.create_subscription(String, 'command', self.command_callback, 1, callback_group=self.group)

        # Command dictionary
        self.command_dict = {
            "1": "cancel",
            "2": "change",
            "3": "give",
            "4": "recalibration",
            "5": "reset",
            "6": "return",
            "7": "save",
            "8": "shut down",
            "9": "take",
            "10": "standard tweezers",
            "11": "slim tweezers",
            "12": "surgical tweezers",
            "13": "three handle",
            "14": "four handle",
            "15": "clenched scalpel",
            "16": "narrow scalpel",
            "17": "blunt scissors",
            "18": "pointed scissors",
            "19": "needle",
            "20": "tweezers",
            "21": "handle",
            "22": "scalpel",
            "23": "scissors"
        }

    # Manual control callback function
    def service_callback(self, request, response):
        # Initiate closed gripper
        # self.arm.close_gripper()

        self.get_logger().info(fg(4) + "[LISTENING FOR WAKE-WORD] ... Flag 0 ... " + fg.rs)
        while self.command[3] != "0":
            time.sleep(0.01)
            
        self.final_command = self.command_dict[self.command[0]]
        self.final_instrument = ""
        # self.arm.open_gripper()
        
        self.get_logger().info(fg(4) + "[LISTENING FOR COMMAND] ... Flag 1 ... " + fg.rs)
        while self.command[3] != "1":
            time.sleep(0.01)

        if self.command[1] != "x":  # instrument input given         
            self.final_instrument = self.command_dict[self.command[1]]

            if self.command[2] != "x": # error input given
                self.error_variation = self.command_dict[self.command[2]]
                sound = self.full_path + f'choose_{self.error_variation}_variation.mp3'
                playsound(sound, block=True)

                self.final_instrument = self.command_dict[self.command[1]]
                self.get_logger().info(fg(4) + "[LISTENING FOR CORRECT INSTRUMENT] ... Flag 2 ... " + fg.rs)
                while self.command[3] != "2":
                    time.sleep(0.01)

        # Reset for wake word
        # self.arm.close_gripper(wait=False)  

        # Prepare service response
        response.command = self.final_command
        response.instrument = self.final_instrument
        self.get_logger().info(fg(4) + self.final_command + self.final_instrument + fg.rs)
        return response
    

    # Exit callback function
    def exit_callback(self, msg):
        if msg.exit == True:
            self.destroy_node()
            os.kill((self.PID), signal.SIGTERM)

    # Command callback function
    def command_callback(self, msg):
        self.get_logger().info("Data received")
        uinput = msg.data
        self.get_logger().info(uinput)
        self.command = uinput.split()


def main():
    # Start voice recognition service and spin node
    rclpy.init(args=None)
    manual_control_service = ManualControlService()
    executor = MultiThreadedExecutor()
    executor.add_node(manual_control_service)
    manual_control_service.get_logger().info(fg(4) + "Manual control service has been started" + fg.rs)
    executor.spin()

    manual_control_service.destroy_node()
    executor.shutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()