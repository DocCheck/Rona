SHELL := /bin/bash

.ONESHELL:

build-workspace:
	@echo Building and sourcing changes...
	colcon build
	source install/setup.bash

start-system:
	@echo Unlocking controller folder...
	sudo chmod a+rw /dev/ttyACM0
	@echo Sourcing ROS workspace...
	source install/setup.bash
	@echo Starting ROS app...
	ros2 launch rona_pkg system.launch.xml

setup:
	@echo Unlocking controller folder...
	sudo chmod a+rw /dev/ttyACM0 
	@echo Sourcing ROS workspace...
	source install/setup.bash


test-camera:
	cd src/rona_pkg/rona_pkg/tests
	python3 test_camera_intel.py

test-controller:
	@echo Unlocking controller port...
	sudo chmod a+rw /dev/ttyACM0
	cd src/rona_pkg/rona_pkg/tests
	python3 test_controller_connection.py

test-robot:
	cd src/rona_pkg/rona_pkg/
	python3 tests/test_robot_connection.py

test-detection:
	python3 src/rona_pkg/rona_pkg/tests/test_instrument_detection.py
