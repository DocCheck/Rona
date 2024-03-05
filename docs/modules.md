# ROS2 modules
Rona is a ROS2 project and is developed on 'ROS2 humble'. The reason we chose to use ROS2 as a developer platform for the project is the modularity that it brings. Rona aims at being flexible and easy to customize for different scenarios - and the modules' structure allows just that. Speaking of the modules, the system is split into 5 main modules and 3 extra modules, some of which should be adjusted and (if needed) rewritten for the specific robotic arm that is being used.

List of Rona's modules:
1. [camera module](#camera-module-and-the-intel-realsense-d435)
2. [voice recognition module](#voice-recognition-module)
3. [speaker module](#speaker-module)
4. [sensor communication module](#sensor-module)
5. [main algorithm module](#main-algorithm-module)
6. [(manual control module)](#manual-control-module)
7. [(calibration module)](#calibration-module)
8. [(robot handler module)](#robot-handler-module)
9. [config](#rona-config)

Each module uses one or more open source libraries that take into consideration the other very important trait of the Rona system - the whole project runs completely offline without internet connection. This was the main goal from the very beginning, because being a medical assistant and crashing in the middle of an intervention do to bad internet connectivity would be very ... inconvenient. Thus, all the modules are designed with that in mind. In the next section we will discuss what each module does and which libraries are needed to be installed.

## Camera module and the Intel Realsense D435

Let us start with arguably the most important module of the system - the robot's vision. Rona's vision requires a depth camera with a good depth sensor and generally good resolution - **with a minimum requirements of 1280x720 capability for both frames** . For our research and development we have chosen the **Intel Realsense D435**. Of course there is a possibility to exchange that camera with another of the Realsense series or even another depth camera in general, depending on budget and needs. Our research was done with this camera, so we will base all further explanations based on usage of that model.

---

The camera module is divided internally into different submodules. The first one is the **object detection**. As an example we have trained a custom machine learning model for surgery instruments and have used the Yolo_v5 library for real-time instrument detection. The main model is based on the instruments used in a more simple surgery, namely - scalpels, scissors, tweezers, scalpel handles and needles. It actually contains not only an *instrument detection model*, but also an *orientation estimation model* that calculates tha angle at which the instrument is seen and also *empty space image processing* for the returning of the instruments to the table. We provide the trained models from the project and some samples to go with them for clarification in our *repositories* [1](https://github.com/DocCheck/Instrument-Orientation-Estimator/) [2](https://github.com/DocCheck/Surgical-Instrument-Detector/) and the models from our *huggingface* [1](https://huggingface.co/DocCheck/medical-instrument-orientation-estimation) [2](https://huggingface.co/DocCheck/medical-instrument-detection)

Regarding any parameters you can adjust for the detector, you can check the *general_config.py* in the *instrument_detector* folder. For example, adjusting the size of the grid to your needs, depending on the amount of instruments you are using. A good practice is to setup the robot's workspace range, which is defined by the frame of view of the camera, and use the *test_instrument_detection.py* test function you can find in the test folder to adjust the position of the instruments and play around with the grid size.

---

The next submodule is the **hand detection**. We use hand detection during the instrument transfer where we detect the hand and its distance to the gripper to try and minimize the errors during the transfers. We found Google's [*mediapipe*](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) library working great for our purpose so we use the hand landmark detection from Mediapipe. Unfortunately, during our development Mediapipe got a massive overhaul from Google and the newest version 0.10.* actually performed worse than the old 0.9.* version, because of the algorithm changes to their hand detection. Thus for the hand detection to be working properly with highest efficiency we recommend using the **mediapipe 0.9.3** version. The correct version for the library is linked in the requirements.txt file you should install with the help of the [quick start guide](https://gitlab.intranet.doccheck.ag/rd/Robot-xArm-Development/-/blob/develop-license/docs/quickstart.md)


---

The third submodule is reserved for an automated eye-in-hand calibration of the camera's position in case there is a decision to change the camera and/or change the position and angle of the camera holder. Currently it is only a placeholder, but could be expanded later. The actual camera coordinates are saved in the *config_rona.py* with the geometry when using the UFactory xArm5 and our designed 3D printed camera holder. 

If you followed everything until now, you should have everything set up correctly for the camera module. Time to move to the next one.


## Voice recognition module

The second most important module for the system is the voice recognition module. Because our goal was to develop a hands-free assistant, we had to resort to complete voice control over the system - making the experience as authentic as possible to working with an actual human nurse, where you would simply tell her what instrument you need without any physical inputs. 

Similar to the camera module, the voice recognition module has submodules. As with any good voice assistant Rona needed to have a wake-word which can trigger the voice recognition. We wouldn't want the medical team to casually talk about the instruments and Rona flying around with a scalpel when the team isn't ready for it, now do we. For this reason we decided to use the open-source [openWakeword](https://github.com/dscripka/openWakeWord) library which is working completely offline and functions super reliably. It provides some pre-trained models, but most importantly it allows for training new custom wake-words. That is why we trained our own wake word "hey Rona" with which we wake up the system every time before we want to give a command. You can download the wake-word model from our [huggingface](https://huggingface.co/DocCheck/wakeword-hey-rona) and put it in the */rona_pkg/rona_pkg/voice_recognition/model_wakeword/* folder, if you haven't done it already. 

Also if you wish to train your own custom wake-word, check out the documentation here: [model training](https://github.com/dscripka/openWakeWord/blob/main/notebooks/training_models.ipynb)

---

After waking up the system we start the next submodule which is now the full voice recognition. Because we are opting for a completely offline system, we had limited possibilities when it came to voice recognition. Luckily, [whisper](https://github.com/openai/whisper) from openAI is an excellent free and offline voice recognition library that provides a large variety of models in different languages and sizes. In our r&d we found out that the "tiny" size model is working sufficiently accurate and fast enough for the system to feel smooth and realistic - without too much processing delay. The model you would use needs to be downloaded first. The download of the model happens automatically the first time when you use that specific model and then you can use it anytime and offline. You can change the model if you want to by changing the related variable in *config_rona.py* script. 

Whisper takes the spoken words and transcribes them into strings which then get further processed by an intent recognizer - to see if any of the spoken words trigger a command. This also doubles as a filter in case whisper sometimes mishears your command or the instrument you want with another word. This could happen rarely and it can be influenced by many things like noise or bad microphone. As good as whisper is, we recommend getting a high-end microphone or a headset to ensure as few as possible errors with the command inputs. Rona works with predefined list of commands which could be expanded if needed. What they are and how to use them is explained in the main [README.md](https://gitlab.intranet.doccheck.ag/rd/Robot-xArm-Development/-/blob/develop-license/README.md?ref_type=heads).

For all sound inputs recordings we use the python [speech_recognition](https://pypi.org/project/SpeechRecognition/) library. Pyaudio and torch are also needed. All the necessary libraries for the voice recognition are included in the requirements.txt file you should install with the help of the [quick start guide](https://gitlab.intranet.doccheck.ag/rd/Robot-xArm-Development/-/blob/develop-license/docs/quickstart.md)


## Speaker module

We wanted Rona to feel alive so we gave her a voice to reply to you. Audible feedback is always appreciated, especially in situations when you aren't able to focus on some blinking lights or other visual cues - like during a surgery. We gave Rona some predefined lines, because we want to keep the system offline and the offline TTS sound very underwhelming. So we used the Google TTS to record some lines for her in a clear female voice and packed them inside the system. Of course this leaves the opportunity to expand her repertoire and add more custom lines if needed. We also use the speaker module as a guide while conversing with her. When you try and give a command, Rona can guide you towards the complete command while asking you further questions until she understands you or decides to cancel and start over. To be able to use the module you just need to have the [playsound](https://pypi.org/project/playsound/) library installed - this is again included in the requirements.txt file you should install with the help of the [quick start guide](https://gitlab.intranet.doccheck.ag/rd/Robot-xArm-Development/-/blob/develop-license/docs/quickstart.md)


## Sensor module

On top of the camera we wanted another safety level during the transfer of the instruments. We tried many different combinations and variations of sensors and ultimately stopped on an impulse sensor module controlled by a RaspberryPi Pico. The size of the module and the controller fit quite nicely in custom made 3D printed parts that we have also provided for free [here](https://gitlab.intranet.doccheck.ag/rd/Robot-xArm-Development/-/tree/develop-license). With an assembly guide you can find [here](https://gitlab.intranet.doccheck.ag/rd/Robot-xArm-Development/-/blob/develop-license/docs/AssemblyTutorial.md).The [sensor](https://www.az-delivery.de/en/products/sw420-vibration-schuttel-erschutterung-sensor-modul) is a sw 420 vibration sensor and its idea is to only open or close the gripper when there is a "haptic" feedback, meaning someone or something touched the gripper tip. This way we can ensure that instruments won't get dropped on the floor or the gripper won't close without an instrument in it. 

We use the [python serial](https://pypi.org/project/pyserial/) library to exchange information between the RBP and the workstation, thus they need to be connected with an USB cable during use. The simple yet effective sensor communication module ensures that the signal from the sensor can get used in the ROS2 environment as a trigger for the next actions. We need the python serial library for that, which can be installed via:

Sometimes you might need to unblock the serial port while connecting to the controller. You can do this with the command **sudo chmod a+rw /dev/ttyACM0**

Additionally, you need to have the proper code running on the RaspberryPi. We recommend using [Thonny](https://thonny.org/), because the code is written into Pi Python. Download and install Thonny and then you can use it to run the code on the RBP. You can find the code in the */rona_pkg/rona_pkg/sensoric/* folder. You need to upload the main.py file to the controller. We have also provided the code as a text file just in case. To upload the file to the RBP you open the main.py file from the computer then "save as..." to the controller and run it. You can check a very useful tutorial [here](https://www.youtube.com/watch?v=L03jT5slWnw).


## Main algorithm module

This is technically Rona's brain. This is the main module and the main node for the ROS2 system. Here all the service calls are done based on the commands and the communication between all the nodes happens. A more thorough explanation of the system's action sequences is given in the main [README.md](https://gitlab.intranet.doccheck.ag/rd/Robot-xArm-Development/-/blob/develop-license/README.md?ref_type=heads). 


## Manual control module

In case you want more controlled testing or you don't want to use voice commands for some reason, we have added an additional module which you can substitute the voice control module with. This way you can publish on the manual control command topic a list with information about the command (check the command dictionary in the module) and the system will execute this command as if you have given it a vocal command. This module is intended for debugging and more controlled testing of some edge cases, but we decided to leave it so that it can be used or further developed if needed. To activate it you need to comment and uncomment the lines in the *main_rona.py* where the voice recognition service is initialized in the constructor - to essentially change the service that is being used. Of course the launch file needs to be adjusted properly as well.


## Calibration module

This is the module where the calibration information is saved. Mainly we need the eye-in-hand camera position based in the robot's base coordinate system and also the coordinates of the point of transfer. Currently there is no automated initial camera calibration for the eye-in-hand calibration, but could be added in future updates or can be done by yourselves following any eye-in-hand calibration tutorial with a fiducial or QR code. The drop-off coordinate point can be manually changed as you please and also there is a command for dynamic recalibration of that point while the robot is running. This recalibration will **not** overwrite the values in the script permanently. It is only for the actual use cycle of the robot - after you shut the system down and start anew the old transfer point.


## Robot handler module

Finally, we have written a robot handler for quicker and easier access to the robot functions and movements. Of course this module is specific to the robot hand being used. For our development we used the UFactory xarm5 and their [Python-SDK](https://github.com/xArm-Developer/xArm-Python-SDK) respectively.
To clone and prepare the repository for use with Rona follow the [quick start guide](https://gitlab.intranet.doccheck.ag/rd/Robot-xArm-Development/-/blob/develop-license/docs/quickstart.md)

In the "config_rona.py" you will find some kinematic parameters that have been found to be the most optimal, but you can change them around if you please. In the handler you will find a small dictionary with width values of the instruments, because the gripper of the robot doesn't have any force or torque sensors. So to insure proper holding of each instrument we need to adjust for the instruments' width. The rest of the functions are a combination of simple movements summarized into one command. All of the movements are in joint coordinate system, so that the sharper faster movements are possible. For further information on the xArm commands, you can check out the examples in the SDK provided by UFactory.


## Rona config

Lastly, a quick mention of the config file. In the config file you can find all the parameters you could change around if needed and also all of the path files for the different modules. The values there have been tuned for the most optimal experience after our extensive testing, but you can change them around if needed.

---

This concludes the fast overview of the system and its modules. In the next section we will explain the usage, logic and configuration of the system with examples, pictures, videos and charts.