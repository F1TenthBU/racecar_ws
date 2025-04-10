########################################################################################
# Imports
########################################################################################

import sys, time
import numpy as np
import matplotlib.pyplot as plt
import math
from nptyping import NDArray
from typing import Any, Tuple

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

IS_SIM = False
rc = racecar_core.create_racecar(IS_SIM)

SHOW_PLOT = False
TEST_WITHOUT_SPEED = False

# >> Constants
WINDOW_SIZE = 8 # Window size to calculate the average distance

# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels

# >> !!! TUNING VARIABLES !!!
if IS_SIM:
    # This Values are for SIM
    MINIMUM_SPEED = 0.2
    # Initialize PID control variables for angle
    KP = 1.0
    KI = 0.0
    KD = 0.0
else:
    # This Values are for REAL
    MINIMUM_SPEED = 0.2
    # Initialize PID control variables for angle
    KP = 0.6
    KI = 0.0
    KD = 0.1

prev_error_angle = 0  # Previous error for angle control
integral_angle = 0  # Integral term for angle control


########################################################################################
# Functions
########################################################################################

def path_find(lidar_data):
    angle_error = 0
    # Do your logic to find proper values
    return angle_error

def update_lidar():
    """
    Receive the lidar samples and get the average samples from it
    """
    global average_scan

    scan = rc.lidar.get_samples()
    if (len(scan) == 0):
        return False
    
    scan = np.clip(scan, None, 3000)
    # print(scan)
    
    if not IS_SIM:
        scan_length = len(scan) # 1081, 1 for 0 angle maybe?
        values_per_angle = (scan_length - 1) / 270
        degree_0 = int((scan_length - 1) / 2)
        first_half = scan[:degree_0] # 135 to 0
        backward = np.full(int(90 * values_per_angle - 1), 30)
        second_half = scan[degree_0:] # 0 to -135
        rotated_scan = np.concatenate([second_half, backward, first_half])
    else:
        rotated_scan = scan

    average_scan = np.array([rc_utils.get_lidar_average_distance(rotated_scan, angle, WINDOW_SIZE) for angle in range(360)])
    return

def start():
    """
    This function is run once every time the start button is pressed
    """
    global speed
    global angle

    # Initialize variables
    speed = 0
    angle = 0

    # Set initial driving speed and angle
    rc.drive.set_speed_angle(speed, angle)
    # Set update_slow to refresh every half second
    #rc.set_update_slow_time(0.5)

    # Print start message
    print(
        ">> Wall Following\n"
        "\n"
        "Controls:\n"
        "    A button = print current speed, angle, and closest values\n"
    )

def update():
    global speed
    global angle
    global prev_error_angle
    global prev_error_speed
    global integral_angle
    global integral_speed
    global average_scan

    if update_lidar() == False:
        return
    
    # Your logic
    angle_error = path_find(average_scan)

    # Update angle integral term
    integral_angle += angle_error

    # Update angle derivative term
    angle_derivative = angle_error - prev_error_angle
    prev_error_angle = angle_error

    # Calculate angle PID output
    angle_pid_output = KP * angle_error + KI * integral_angle + KD * angle_derivative

    # Convert angle PID output to angle
    angle = angle_pid_output

    speed = MINIMUM_SPEED
    if TEST_WITHOUT_SPEED:
        speed = 0.0

    angle = max(-1.0, min(1.0, angle))

    # Set the speed and angle of the car
    rc.drive.set_speed_angle(speed, angle)

    # Print the current speed and angle and closest values when the A button is held down
    if rc.controller.is_down(rc.controller.Button.A):
        print("Speed:", speed, "Angle:", angle)
        # print("Left:", closest_left_angle, ",", closest_left_distance)
        # print("Right:", closest_right_angle, ",", closest_right_distance)
        # print("Front:", closest_front_angle, ",", closest_front_distance)

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()
