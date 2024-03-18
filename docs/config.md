Here you can find a quick guide to the "one-time" configuration you need to perform in the beginning, before you can use Rona

## RaspberryPi

To setup the RBP we recommend using [Thonny](https://thonny.org/), because the code is written in Pi Python. Download and install Thonny and then you can use it to run the code on the RBP. You can find the code in the */rona_pkg/rona_pkg/sensoric/* folder. You need to upload the main.py file to the controller. We have also provided the code as a text file just in case. To upload the file to the RBP you first connect it to the workstation via the USB cable. Open a terminal and unblock the port using **sudo chmod a+rw /dev/ttyACM0**. In the same terminal directly run *thonny*. If the interpreter doesn't get configured automatically,go to the *Run* menu on the toolbar and choose *configure interpreter*. From the menu there select the RBP board. Then use *open* and choose *This computer*. Navigate to the /sensoric folder and open the *main.py*. Afterwards *"save as..."* to the controller preserving the same name *main.py* and run it from thonny with the run button. Now close thonny and the terminal. 

If something went wrong, you can check a very useful tutorial [here](https://www.youtube.com/watch?v=L03jT5slWnw) and then try again.


## Robot arm

To be able to connect to the xArm we need to set the LAN IP Address of the Ethernet port of our workstation - where we would connect the robot - to static, so that the robot can be found. We use the following configurations:

    Address 192.168.1.12
    Netmask 255.255.255.0
    Gateway 192.168.1.1

Before use and testing, release the red safety button on the controller to enable the motors of the robot and be vigilant that the robot can move now.


## Testing

It is advised to test all the connections before starting the whole system. We have provided a few test functions you can use to check:

- Realsense camera connection : **test_camera_intel.py**
- RBP connection: **test_controller_connection.py**
- Robot connection: **test_robot_connection.py**

You can also use the Makefile commands directly:

make test-camera
make test-controller
make test-robot

When using the RBP test, first unlock the port if needed **sudo chmod a+rw /dev/ttyACM0** and after starting the test - shake the sensor to see the feedback.