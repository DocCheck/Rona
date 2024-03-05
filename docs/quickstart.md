# Quickstart 
Follow this guide to get a working software workspace to control Rona.
Ít should download all necessary code repositories and download our pretrained machine learning models
If something isn't understandable from here, feel free to check the thorough explanation in the [modules documentation](modules.md) file.


# Getting Started
## Prerequisites
Make sure you have installed all the following prerequisites installed properly:
- Assembly the hardware as described in our assembly guide
- ROS2 (follow the official [installation guide](https://docs.ros.org/en/humble/Installation.html))
- Camera setup for [**Intel Realsense D435**](https://www.intelrealsense.com/depth-camera-d435/) and corresponding  [**libraries for Intel RealSense**](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md) (latest tested version 2.54.2)

### Python Setup
It is advised to use a virtual environment for the python packages e.g. using this guide: [Python Virtual Environment](https://docs.python.org/3/library/venv.html) \
Then install the required packages using pip:
```bash
pip -r requirements.txt
```


## Installation
The following commands will create a workspace and download all necessary code repositories: 
```bash
# Create folder for Rona workspace
echo 'Creating workspace for Rona...'
mkdir -p rona_ws/
cd rona_ws/

# source ROS2
source /opt/ros/humble/setup.bash

# Download Rona repository and move it to the right place
echo 'Downloading Rona repository and moving it to the right place...'
git clone git@gitlab.intranet.doccheck.ag:rd/Robot-xArm-Development.git
mv Robot-xArm-Development/ src/

# Download xArm SDK and move it to the right place
echo 'Downloading xArm SDK and moving it to the right place...'
git clone https://github.com/xArm-Developer/xArm-Python-SDK
mv xArm-Python-SDK/ src/rona_pkg/rona_pkg/robot_sdk
echo Making xarm_ros a module
touch src/rona_pkg/rona_pkg/robot_sdk/__init__.py

# Move Makefile to workspace
mv src/Makefile ./

colcon build
source install/setup.bash
```

In case of any changes to the setup the ROS2 environment needs to be rebuilt and resourced again
```bash
colcon build
source install/setup.bash
```
For a working Rona setup, several machine learning models are required. 
If you want to test the system without training your own custom models, you can download external models for a functioning testing setup.




## Download models (Optional)
Rona does not come with pre-trained models. If you want to test the system without training your own custom models, you can download theses external models and their inference code from the following repositories.
### Prerequisites
The models are hosted on huggingface. You can also download them by hand and copy them to correct folders.\
Alternatively, you can use the following commands to download the models using [git-lfs](https://git-lfs.com/).

```bash 
# Download models and move them to the right place
echo 'Downloading models...'
mkdir models

#Check if git-lfs is installed
git lfs install

# download models via git lfs (alternatively you can download them by hand) 
git clone https://huggingface.co/DocCheck/medical-instrument-orientation-estimation
git clone https://huggingface.co/DocCheck/medical-instrument-detection
git clone https://huggingface.co/DocCheck/wakeword-hey-rona

# Move models to correct folders
mv models/instrument_detector_model.pt  src/rona_pkg/rona_pkg/instrument_detector/object_detector/Rona_mid/Rona_mid_model/weights/instrument_detector_model.pt 
mv models/orientation_estimation_model.pt src/rona_pkg/rona_pkg/instrument_detector/orientation_estimator/RotNet/data/models/orientation_estimation_model.pt
mv models/hey_rona.onnx src/rona_pkg/rona_pkg/voice_recognition/model_wakeword/hey_rona.onnx
rm -rf models

# download inference code for object detection
git clone https://github.com/DocCheck/Surgical-Instrument-Detector/
mv medical-instrument-detection/object_detector/general_utils Robot-xArm-Development/rona_pkg/rona_pkg/instrument_detector/general_utils
mv medical-instrument-detection/object_detector Robot-xArm-Development/rona_pkg/rona_pkg/instrument_detector/
touch Robot-xArm-Development/rona_pkg/rona_pkg/instrument_detector/object_detector/__init__.py
rm -rf medical-instrument-detection

# Move Makefile to workspace
mv src/Makefile ./

# Rebuild and resource the workspace again
colcon build
source install/setup.bash
```





## Usage
Following command will start all associated modules and Rona will start listening to commands.\
**Beware: The arm will start moving!**  
```bash
# source the workspace (if not done prior)
source install/setup.bash
# start the system
make start-system
```