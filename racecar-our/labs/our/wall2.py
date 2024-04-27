########################################################################################
# Imports
########################################################################################

import sys
import numpy as np
from nptyping import NDArray
from typing import Any, Tuple

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# >> Constants
WINDOW_SIZE = 8 # Window size to calculate the average distance
CURVE_ANGLE_SIZE = 40 # The angle of a gap to check existence of curve
CURVE_DISTANCE_SIZE = 30 # The distance of a gap to check existence of curve

# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels

# Initialize PID control variables for angle
# KP = 0.05  # Proportional constant for angle
KI = 0.0000  # Integral constant for angle
# KD = 0.1  # Derivative constant for angle
prev_error_angle = 0  # Previous error for angle control
integral_angle = 0  # Integral term for angle control

# Initialize PID control variables for speed
KP_speed = 0.1  # Proportional constant for speed
KI_speed = 0.001  # Integral constant for speed
KD_speed = 0.05  # Derivative constant for speed
prev_error_speed = 0  # Previous error for speed control
integral_speed = 0  # Integral term for speed control

# Initialize desired speed
desired_speed = 0.5  # Set desired speed to 0.5 (you can adjust this value)

flag = 0

########################################################################################
# Functions
########################################################################################


def update_lidar():
    """
    Receive the lidar samples and get the average samples from it
    """
    global closest_left_angle
    global closest_left_distance
    global closest_leftfront_angle
    global closest_leftfront_distance
    global closest_right_angle
    global closest_right_distance
    global closest_rightfront_angle
    global closest_rightfront_distance
    global closest_front_angle
    global closest_front_distance
    global average_scan

    scan = rc.lidar.get_samples()
    average_scan = np.array([rc_utils.get_lidar_average_distance(scan, angle, WINDOW_SIZE) for angle in range(360)])

    closest_left_angle, closest_left_distance = rc_utils.get_lidar_closest_point(average_scan, (-91, -89))
    closest_leftfront_angle, closest_leftfront_distance = rc_utils.get_lidar_closest_point(average_scan, (-41, -39))
    closest_right_angle, closest_right_distance = rc_utils.get_lidar_closest_point(average_scan, (89, 91))
    closest_rightfront_angle, closest_rightfront_distance = rc_utils.get_lidar_closest_point(average_scan, (39, 41))
    closest_front_angle, closest_front_distance = rc_utils.get_lidar_closest_point(average_scan, (-10, 10))
    # TODO
    # 30 to 60, -30 to -60 are blind spots. However, this setting makes the movement smoother. 
    # For blind spots, it would be better to find the closest point separately and handle it.
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
    global closest_left_angle
    global closest_left_distance
    global closest_leftfront_angle
    global closest_leftfront_distance
    global closest_right_angle
    global closest_right_distance
    global closest_rightfront_angle
    global closest_rightfront_distance
    global closest_front_angle
    global closest_front_distance
    global average_scan
    global flag

    update_lidar()

    L = 150

    degree = closest_right_angle - closest_rightfront_angle
    theta_value = np.deg2rad(degree)

    a_value = closest_rightfront_distance
    b_value = closest_right_distance

    alpha1 = np.arctan((a_value * np.cos(theta_value) - b_value) / a_value * np.sin(theta_value))
    Dt = b_value * np.cos(alpha1)
    right_Dt1 = Dt + np.sin(alpha1) * L

    degree = closest_leftfront_angle - closest_left_angle
    theta_value = np.deg2rad(degree)

    a_value = closest_leftfront_distance
    b_value = closest_left_distance

    alpha2 = np.arctan((a_value * np.cos(theta_value) - b_value) / a_value * np.sin(theta_value))
    Dt = b_value * np.cos(alpha2)
    left_Dt1 = Dt + np.sin(alpha2) * L

    max_alpha = max(np.rad2deg(alpha1), np.rad2deg(alpha2))
    min_alpha = min(np.rad2deg(alpha1), np.rad2deg(alpha2))

    # print(max_alpha, min_alpha)
    # print(right_Dt1, left_Dt1)

    left_gap = average_scan[270 + CURVE_ANGLE_SIZE] - average_scan[270]
    right_gap = average_scan[90 - CURVE_ANGLE_SIZE] - average_scan[90]

    if closest_left_distance > 200 and closest_right_distance > 200:
        rc.drive.set_speed_angle(1.0, 0.0)
        return
    else:
        if closest_front_distance < 200:
            KP = 0.045
            KD = 0.085
            angle_error = (right_Dt1 - left_Dt1) / 2
        else:
            KP = 0.005
            KD = 0.012
            angle_error = (right_Dt1 - left_Dt1) / 2

    # Update angle integral term
    integral_angle += angle_error

    # Update angle derivative term
    angle_derivative = angle_error - prev_error_angle
    prev_error_angle = angle_error

    # Calculate angle PID output
    angle_pid_output = KP * angle_error + KI * integral_angle + KD * angle_derivative

    # Convert angle PID output to angle
    angle = angle_pid_output

    # PID control for speed
    # Calculate speed error (difference between desired and actual speed)
    speed_error = desired_speed - speed

    # Update speed integral term
    integral_speed += speed_error

    # Update speed derivative term
    speed_derivative = speed_error - prev_error_speed
    prev_error_speed = speed_error

    # Calculate speed PID output
    speed_pid_output = KP_speed * speed_error + KI_speed * integral_speed + KD_speed * speed_derivative

    # Convert speed PID output to speed
    speed += speed_pid_output

    speed = 0.4
    if abs(angle) > 0.2:
        if (flag % 3 == 0):
            speed = 0.1
        else:
            speed = 0.2
    elif abs(angle) > 0.1:
        if (flag % 3 == 0):
            speed = 0.0
        else:
            speed = 0.1

    flag += 1
    flag %= 12

    # Constrain speed and angle within 0.0 to 1.0
    speed = max(0.0, min(1.0, speed))
    angle = max(-1.0, min(1.0, angle))

    # Set the speed and angle of the car
    rc.drive.set_speed_angle(speed, angle)

    # Print the current speed and angle and closest values when the A button is held down
    if rc.controller.is_down(rc.controller.Button.A):
        print("Speed:", speed, "Angle:", angle)
        print("Left:", closest_left_angle, ",", closest_left_distance)
        print("Right:", closest_right_angle, ",", closest_right_distance)
        print("Front:", closest_front_angle, ",", closest_front_distance)

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()
