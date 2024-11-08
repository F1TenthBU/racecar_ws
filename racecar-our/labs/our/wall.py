
########################################################################################
# Imports
########################################################################################

import sys, time
import numpy as np
from nptyping import NDArray
from typing import Any, Tuple
from datetime import datetime

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar(False)

# >> Constants
WINDOW_SIZE = 8 # Window size to calculate the average distance
CURVE_ANGLE_SIZE = 40 # The angle of a gap to check existence of curve
CURVE_DISTANCE_SIZE = 30 # The distance of a gap to check existence of curve
FAR_DISTANCE_SIZE = 100 # If the closest point is over this value, it would be ignored

FRONT_DISTANCE_FOR_CURVE = 200
SIDETOP_DISTANCE_FOR_CURVE = 180
SIDETOP_ANGLE_FOR_CURVE = 20
flag= 0
stopped = 0
# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels

# Initialize PID control variables for angle
KP = 0.0002  # Proportional constant for angle
KP_BACKUP = KP
KI = 0.000000  # Integral constant for angle
KD = 0.3  # Derivative constant for angle
KD_BACKUP = KD
#KD = 7.5  # Derivative constant for angle
prev_error_angle = 0  # Previous error for angle control
integral_angle = 0  # Integral term for angle control
# Initialize PID control variables for speed
KP_speed = 0.01 # Proportional constant for speed
KI_speed = 0.000  # Integral constant for speed
KD_speed = 0.03  # Derivative constant for speed
#KD_speed= 0.8 #temp KD
prev_error_speed = 0  # Previous error for speed control
integral_speed = 0  # Integral term for speed control

# Initialize desired speed
desired_speed = 0.2  # Set desired speed to 0.1 (you can adjust this value)

########################################################################################
# Functions
########################################################################################

current_num = 1
current_time = datetime.now().strftime("%y%m%d%H%M")
logfile = f"log{current_time}.log"

def logging_to_file(array):
    with open(logfile, 'a') as f:
        np.savetxt(f, array, fmt='%s')

def test_lidar(scan):
    print(len(scan))

    forward_distance = scan[0]
    print(f"Forward distance: {forward_distance:.2f} cm")

    rear_distance = scan[180]
    print(f"Rear distance: {rear_distance:.2f} cm")

    left_distance = scan[270]
    print(f"Left distance: {left_distance:.2f} cm")

    right_distance = scan[90]
    print(f"Right distance: {right_distance:.2f} cm")

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
    global closest_front_half_angle
    global closest_front_half_distance
    global average_scan

    scan = rc.lidar.get_samples()
    if (len(scan) == 0):
        return False
    
    scan_length = len(scan)
    degree_270 = int(scan_length / 360.0 * 270)
    rotated_scan = np.concatenate((scan[degree_270:], scan[:degree_270]))
    # test_lidar(rotated_scan)

    average_scan = np.array([rc_utils.get_lidar_average_distance(rotated_scan, angle, WINDOW_SIZE) for angle in range(360)])

    logging_to_file(["-----", current_num, "-----"])
    logging_to_file(average_scan)

    closest_left_angle, closest_left_distance = rc_utils.get_lidar_closest_point(average_scan, (-91, -89))
    closest_leftfront_angle, closest_leftfront_distance = rc_utils.get_lidar_closest_point(average_scan, (-41, -39))
    closest_right_angle, closest_right_distance = rc_utils.get_lidar_closest_point(average_scan, (89, 91))
    closest_rightfront_angle, closest_rightfront_distance = rc_utils.get_lidar_closest_point(average_scan, (39, 41))
    closest_front_angle, closest_front_distance = rc_utils.get_lidar_closest_point(average_scan, (-5, 5))
    closest_front_half_angle, closest_front_half_distance = rc_utils.get_lidar_closest_point(average_scan, (-25, 25))
    # TODO
    # 30 to 60, -30 to -60 are blind spots. However, this setting makes the movement smoother. 
    # For blind spots, it would be better to find the closest point separately and handle it.

    # Values for test
    # closest_front_distance = 200
    # closest_left_distance = 200
    # closest_right_distance = 200

    return True

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
    global closest_right_angle
    global closest_right_distance
    global closest_front_angle
    global closest_front_distance
    global closest_front_half_angle
    global closest_front_half_distance
    global average_scan
    global flag
    if update_lidar() == False:
        return

    KP = KP_BACKUP
    KD = KD_BACKUP

    #TODO: If there are no obstacles on either side, just run straight.    
    if False:
        pass
        # For the testing, never get in here.
    else:
        angle_error = 0
        
        L = 200
        FRONT_CURVE_DISTANCE = 200

        degree = closest_right_angle - closest_rightfront_angle
        theta_value = np.deg2rad(degree)

        a_value = closest_rightfront_distance
        b_value = closest_right_distance

        alpha1 = np.arctan((a_value * np.cos(theta_value) - b_value) / a_value * np.sin(theta_value))
        Dt = b_value * np.cos(alpha1)
        right_Dt = Dt
        right_Dt1 = Dt + np.sin(alpha1) * L

        degree = closest_leftfront_angle - closest_left_angle
        theta_value = np.deg2rad(degree)

        a_value = closest_leftfront_distance
        b_value = closest_left_distance

        alpha2 = np.arctan((a_value * np.cos(theta_value) - b_value) / a_value * np.sin(theta_value))
        Dt = b_value * np.cos(alpha2)
        left_Dt = Dt
        left_Dt1 = Dt + np.sin(alpha2) * L

        max_alpha = max(np.rad2deg(alpha1), np.rad2deg(alpha2))
        min_alpha = min(np.rad2deg(alpha1), np.rad2deg(alpha2))

        left_gap = average_scan[270 + CURVE_ANGLE_SIZE] - average_scan[270]
        right_gap = average_scan[90 - CURVE_ANGLE_SIZE] - average_scan[90]

        # if closest_front_distance < 150 and (left_gap > 30 or right_gap > 30) and (min_alpha < -10 or max_alpha > 10):
        if closest_front_distance < 240: #150
           print('Turn mode')
           KP = 0.015 # 008 0.01
           KD = 0.005 # 005 0.03
        else:
           print('Straight mode')
           KP = 0.001  # 0.003
           KD = 0.04  # 0.03i

        # KP = 0.015
        # KD = 0.005

        if (right_Dt - right_Dt1) > 100:
            right_Dt1 += 100
        if (left_Dt - left_Dt1) > 100:
            left_Dt1 += 100

        angle_error = (right_Dt1 - left_Dt1) / 2

        print("left_Dt: ", left_Dt)
        print("left_Dt1: ", left_Dt1)
        print("right_Dt: ", right_Dt)
        print("right_Dt1: ", right_Dt1)
        print("angle_error: ", angle_error)

        # Update angle integral term
        integral_angle += angle_error

        # Update angle derivative term
        angle_derivative = angle_error - prev_error_angle
        prev_error_angle = angle_error

        # Calculate angle PID output
        angle_pid_output = KP * angle_error + KI * integral_angle + (KD * angle_derivative * 60)

        # print("PID terms: P: ", angle_error, "I: ", integral_angle, "D: ", angle_derivative)

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
    if speed > desired_speed:
        speed = desired_speed

    # min_distance = min(min(closest_front_distance, closest_left_distance), closest_right_distance)
    min_distance = closest_front_half_distance
    speed = 0.2
    # if closest_front_distance < 40:
    #     speed = -1.0
    #     angle = -angle
    if min_distance < 50 and flag < 30:
        speed = -0.1
        flag += 1
    elif min_distance < 50 and flag < 60:
        speed = 0.0
        flag += 1
    elif min_distance < 120 and flag < 30:
        speed = 0.1
        flag += 1
    elif flag > 0:
        flag += 1
    else:
        flag = 0
    
    flag %= 60
    
    #speed = 0.0

    # Constrain speed and angle within 0.0 to 1.0
    speed = max(-1.0, min(1.0, speed))
    angle = max(-1.0, min(1.0, angle))

    # Set the speed and angle of the car
    rc.drive.set_speed_angle(speed, angle)
    # rc.drive.set_speed_angle(0.0, 1.0)
    # rc.drive.set_speed_angle(0.0, angle)

    # Print the current speed and angle and closest values when the A button is held down
    # if rc.controller.is_down(rc.controller.Button.A):
    if True:
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
