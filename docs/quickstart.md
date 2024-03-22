# Quickstart 
Follow this guide to get a working software workspace to control Rona.
Ít should download all necessary code repositories and download our pre-trained machine learning models
In the end, if something isn't clear about the system structure, feel free to check the more thorough explanation in the [modules documentation](modules.md) file.


# Getting Started
## Prerequisites
To start off, install the  core necessary components:
- Prepare the linux machine (official [installation guide](https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview))
- Install ROS2 Humble(follow the official [installation guide](https://docs.ros.org/en/humble/Installation.html))
- Install python 3.10
- Camera setup for [**Intel Realsense D435**](https://www.intelrealsense.com/depth-camera-d435/) (follow official [installation guide](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)) (latest tested version 2.54.2)

### Python libraries setup
All the necessary libraries for the project are included in the *requirements.txt* file provided in the repository.

Install the required packages using pip:
```bash
pip -r requirements.txt
```


## Rona workspace preparation
As you already probably figured out, Rona is a ROS 2 project, thus we need to setup the ros workspace and then download all the necessary repositories inside. /
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
git clone https://github.com/DocCheck/Rona.git
mv Rona/ src/

# Download xArm SDK and move it to the right place
echo 'Downloading xArm SDK and moving it to the right place...'
git clone https://github.com/xArm-Developer/xArm-Python-SDK
sudo cp -r xArm-Python-SDK/. src/rona_pkg/rona_pkg/robot_sdk
sudo rm -rf xArm-Python-SDK
echo Making robot_sdk a module
sudo touch src/rona_pkg/rona_pkg/robot_sdk/__init__.py

# Move Makefile to workspace
mv src/Makefile ./

colcon build
source install/setup.bash
```

In case of any changes to the setup the ROS2 environment needs to be rebuilt and resourced again. We can start utilizing the Makefile from our repository for convenience as well.

```bash
make build-workspace
```

For a working Rona setup, several machine learning models are required. 
If you want to test the system without training your own custom models, you can download external models for a functioning testing setup.


## Download models (Optional)
Rona does not come with pre-trained models. If you want to test the system without training your own custom models, you can download theses external models and their inference code from the following repositories.
### Prerequisites
The models are hosted on huggingface. You can also download them by hand and copy them to correct folders.\
Alternatively, you can use the following commands to download the models using [git-lfs](https://git-lfs.com/).

```bash 
# download inference code for object detection
git clone https://github.com/DocCheck/Surgical-Instrument-Detector/
sudo cp -r  Surgical-Instrument-Detector/object_detector src/rona_pkg/rona_pkg/instrument_detector/object_detector
sudo rm -rf  Surgical-Instrument-Detector/
sudo touch src/rona_pkg/rona_pkg/instrument_detector/object_detector/__init__.py


# Download models and move them to the right place
echo 'Downloading models...'

#Check if git-lfs is installed
git lfs install

# download models via git lfs (alternatively you can download them by hand) 
git clone https://huggingface.co/DocCheck/medical-instrument-orientation-estimation
git clone https://huggingface.co/DocCheck/medical-instrument-detection
git clone https://huggingface.co/DocCheck/wakeword-hey-rona

# Move models to correct folders
sudo mv medical-instrument-detection/instrument_detector_model.pt src/rona_pkg/rona_pkg/instrument_detector/object_detector/Rona_mid/Rona_mid_model/weights/instrument_detector_model.pt 
sudo mv medical-instrument-orientation-estimation/orientation_estimation_model.pt src/rona_pkg/rona_pkg/instrument_detector/orientation_estimator/RotNet/data/models/orientation_estimation_model.pt
sudo mv wakeword-hey-rona/wake_word_hey_rona.onnx src/rona_pkg/rona_pkg/voice_recognition/model_wakeword/hey_rona.onnx

#Delete the downloaded repositories
rm -rf  medical-instrument-detection
rm -rf  medical-instrument-orientation-estimation
rm -rf  wakeword-hey-rona

# Rebuild and resource the workspace again
source /opt/ros/humble/setup.bash
make build-workspace
```


## Usage
Following command will start all associated modules and Rona will start listening to commands.\
**Beware: The arm will start moving!**  

```bash
# source ROS2 (if not automatically done)
source /opt/ros/humble/setup.bash
# start the system
make start-system
```