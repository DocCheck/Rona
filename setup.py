from setuptools import setup
from glob import glob
import os

from setuptools import find_packages

package_name = 'rona_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.xml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alex',
    maintainer_email='alexander.vuchkov@doccheck.com',
    description='Controller package for Rona',
    license='X',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'main_rona.py = rona_pkg.main_rona:main',
            'robot_config.py = rona_pkg.robot_config:main',
            'robot_handler.py = rona_pkg.robot_handler:main',
            'sensor_communication.py = rona_pkg.sensor_communication:main',
            'service_calibration.py = rona_pkg.service_calibration:main',
            'service_camera.py = rona_pkg.service_camera:main',
            'service_manual_control.py = rona_pkg.service_manual_control:main',
            'service_speaker.py = rona_pkg.service_speaker:main',
            'service_voice_recognition.py = rona_pkg.service_voice_recognition:main'
        ],
    },
)
