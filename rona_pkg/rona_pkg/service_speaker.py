# add python path inside ROS2 workspace
import sys
sys.path.insert(0, "src/rona_pkg/rona_pkg")

from custom_interfaces.srv import SpeakerSrv
from custom_interfaces.msg import ExitMsg
from rclpy.node import Node

from playsound import playsound
from sty import fg 

from rona_pkg import config_rona
import rclpy

### fg(7) -> white log ###


class SpeakerService(Node):

    def __init__(self):
        # ROS2 init
        super().__init__('speaking_service')
        self.srv = self.create_service(SpeakerSrv, 'speaking_service', self.service_callback)
        self.create_subscription(ExitMsg, 'exit_message', self.exit_callback, 1)

        # Sound files dictionary
        self.prompt_dict={
            "Change with blunt scissors":"change_blunt_scissors.mp3",
            "Change with clenched scalpel":"change_clenched_scalpel.mp3",
            "Change with four handle":"change_handle_four.mp3",
            "Change with three handle":"change_handle_three.mp3",
            "Change with narrow scalpel":"change_narrow_scalpel.mp3",
            "Change with needle":"change_needle.mp3",
            "Change with pointed scissors":"change_pointed_scissors.mp3",
            "Change with slim tweezers":"change_slim_tweezers.mp3",
            "Change with standard tweezers":"change_standard_tweezers.mp3",
            "Change with surgical tweezers":"change_surgical_tweezers.mp3",
            "blunt scissors chosen":"chosen_blunt_scissors.mp3",
            "clenched scalpel chosen":"chosen_clenched_scalpel.mp3",
            "four handle chosen":"chosen_handle_four.mp3",
            "three handle chosen":"chosen_handle_three.mp3",
            "narrow scalpel chosen":"chosen_narrow_scalpel.mp3",
            "needle chosen":"chosen_needle.mp3",
            "pointed scissors chosen":"chosen_pointed_scissors.mp3",
            "slim tweezers chosen":"chosen_slim_tweezers.mp3",
            "standard tweezers chosen":"chosen_standard_tweezers.mp3",
            "surgical tweezers chosen":"chosen_surgical_tweezers.mp3",
            "Return is not possible":"return_impossible.mp3",
            "Return has been chosen":"return.mp3",
            "The system is ready":"system_ready.mp3",
            "Please save":"system_recalibration_please_save.mp3",
            "The new coordinates have been saved":"system_recalibration_saved.mp3",
            "Recalibration started":"system_recalibration_start.mp3",
            "Canceling recalibration": "system_recalibration_cancel.mp3",
            "The system is resetting":"system_reset.mp3",
            "The system is shutting down":"system_shutdown.mp3",
            "Target found":"target_found.mp3",
            "Could not find instrument":"target_not_found.mp3",
            "Could not find free space":"target_space_not_found.mp3"

        }

    # Service callback
    def service_callback(self, request, response):
        prompt = request.request
        
        name = self.prompt_dict[prompt]
        full_path = config_rona.path_audio_files + name
        if prompt == "The system is stopping":
            playsound(full_path, block=True)
        else:
            playsound(full_path, block=False)

        # Return empty response
        return response

    # Exit callback
    def exit_callback(self, msg):
        if msg.exit == True:
            exit()


def main():
    # Initialize node
    rclpy.init(args=None)
    speaking_service = SpeakerService()
    speaking_service.get_logger().info(fg(7) + "Speaking service has been started" + fg.rs)
    rclpy.spin(speaking_service)

    speaking_service.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()