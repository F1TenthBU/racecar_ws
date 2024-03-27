from setuptools import find_packages, setup

package_name = 'racecar_neo'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['launch/teleop.launch.py']),
    ],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'gamepad = racecar_neo.gamepad:main',
            'mux = racecar_neo.mux:main',
            'throttle = racecar_neo.throttle:main',
            'pwm = racecar_neo.pwm:main',
            'camera = racecar_neo.camera:main',
            'imu = racecar_neo.imu:main',
        ]
    },

    author='Christopher Lai',
    author_email='cclai@mit.edu',
    maintainer='Christopher Lai',
    maintainer_email='cclai@mit.edu',
    url="",
    keywords=['ROS2'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT',
        'Programming Language :: Python',
        'Topic :: Software Development',
        ],
    description='base layer utilities for a Raspberry Pi Autonomous racecar'
)
