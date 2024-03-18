#! /usr/bin/env python3
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import config_rona

import serial.tools.list_ports
import serial
import time


### sudo chmod a+rw /dev/ttyACM0 ###
class ControllerControl():

    # Init
    def __init__(self):
        # Init ROS2 stuff
        try:
            self.controller = serial.Serial(port=config_rona.controller_port, parity=serial.PARITY_EVEN, stopbits=serial.STOPBITS_ONE, timeout=1)
            self.no_controller = False
        except:
            print("No controller found.")
            self.no_controller = True

    def run(self):
        command = "sensor"
        # If no controller connected, skip writing function
        if self.no_controller:
            print("No controller found.")

        else:

            # Prepare communication and send command
            self.controller.flush()
            self.controller.write(bytes(command + "\r\n", 'utf-8'))
            time.sleep(0.05)
            print("Sent command to controller")
    
            # Read feedback from the controller
            sensor_feedback = "fail"
            listening_duration = time.time() + config_rona.time_listening_controller # Listen to controller feedback
            while time.time() < listening_duration:
                
                # If there is a serial command send from the controller
                if self.controller.in_waiting:
                    # Format feedback
                    sensor_feedback = self.controller.readline()
                    sensor_feedback = sensor_feedback.decode('utf-8')
                    sensor_feedback = sensor_feedback.strip()
                    print("THIS IS THE SENSOR FEEDBACK STRING: " + sensor_feedback)
                    
                    # If the sensor is successful 
                    if sensor_feedback == "success":
                        print("Test successful!")
                        break


def main():
    controller = ControllerControl()
    controller.run()


if __name__ == '__main__':
    main()