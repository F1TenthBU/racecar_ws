########################################################################################
# Imports
########################################################################################

import sys, time
import numpy as np
# import matplotlib.pyplot as plt
import math
# from scipy.interpolate import splprep, splev
# from scipy.spatial import distance
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
mid_distance = 50

########################################################################################
# Functions
########################################################################################

def lidar_to_2d_coordinates(lidar_data):
    coordinates = []
    for angle in range(360):
        distance = lidar_data[angle]
        # 0 degrees is up (positive y-axis), adjusting angle accordingly
        adjusted_angle_rad = math.radians(90 - angle)  # Shift 0 degrees to point upward
        x = distance * math.cos(adjusted_angle_rad)
        y = distance * math.sin(adjusted_angle_rad)
        coordinates.append((x, y))
    return coordinates

def find_farthest_point(coordinates):
    filtered_points = [point for point in coordinates if point[1] > 0]
    if not filtered_points:
        return None
    farthest_point = max(filtered_points, key=lambda p: math.sqrt(p[0]**2 + p[1]**2))
    return farthest_point

def point_along_line(origin, target, distance):
    vector_x = target[0] - origin[0]
    vector_y = target[1] - origin[1]
    vector_length = math.sqrt(vector_x**2 + vector_y**2)

    if vector_length <= distance * 2:
        return target

    scale = distance / vector_length
    point_x = origin[0] + vector_x * scale
    point_y = origin[1] + vector_y * scale

    return [point_x, point_y]

def find_side_points(coordinates, origin):
    lefts = []
    rights = []
    adding_to_lefts = True

    for i in range(len(coordinates)):
        last_point = coordinates[(i - 1) % len(coordinates)]
        point = coordinates[i]

        if point == origin or (last_point[1] < 0 and point[1] > 0):
            adding_to_lefts = not adding_to_lefts
            continue

        # Skip segments where either coordinate has a negative y-value
        if point[1] <= 0:
            continue

        # Add to lefts or rights based on the current state
        if adding_to_lefts:
            lefts.append(point)
        else:
            rights.append(point)
    
    if len(lefts) == 0 or len(rights) == 0:
        print(lefts)
        print(rights)
        return lefts, rights

    return lefts, rights

def find_closest_points_on_sides2(origin, midpoint, left_points, right_points):
    y_threshold = origin[1]

    filtered_lefts = [point for point in left_points if not (point[1] > y_threshold)]
    filtered_rights = [point for point in right_points if not (point[1] > y_threshold)]

    closest_left = min(filtered_lefts, key=lambda p: math.hypot(p[0] - midpoint[0], p[1] - midpoint[1]), default=None)
    closest_right = min(filtered_rights, key=lambda p: math.hypot(p[0] - midpoint[0], p[1] - midpoint[1]), default=None)

    # closest_left = min(left_points, key=lambda p: math.hypot(p[0] - midpoint[0], p[1] - midpoint[1]), default=None)
    # closest_right = min(right_points, key=lambda p: math.hypot(p[0] - midpoint[0], p[1] - midpoint[1]), default=None)

    return closest_left, closest_right

def adjust_midpoint(midpoint, closest_left, closest_right, distance=30):
    distance_left = math.hypot(midpoint[0] - closest_left[0], midpoint[1] - closest_left[1])
    distance_right = math.hypot(midpoint[0] - closest_right[0], midpoint[1] - closest_right[1])

    if min(distance_left, distance_right) > distance:
        return midpoint

    elif distance_left + distance_right > distance * 2:
        # Find the direction vector along the line connecting closest_left and closest_right
        direction_vector = [closest_right[0] - closest_left[0], closest_right[1] - closest_left[1]]
        line_length = math.hypot(direction_vector[0], direction_vector[1])
        unit_vector = [direction_vector[0] / line_length, direction_vector[1] / line_length] if line_length != 0 else [0, 0]

        # Calculate new position along the line
        if distance_left < distance_right:
            scale = distance - distance_left
            new_x = midpoint[0] + unit_vector[0] * scale
            new_y = midpoint[1] + unit_vector[1] * scale
        else:
            scale = distance - distance_right
            new_x = midpoint[0] - unit_vector[0] * scale
            new_y = midpoint[1] - unit_vector[1] * scale

        return [new_x, new_y]

    else:
        return [(closest_left[0] + closest_right[0]) / 2, (closest_left[1] + closest_right[1]) / 2]

def find_adjusted_path_with_points(origin, target, distance, coordinates):
    current_point = origin
    path_points = [current_point]  # Store all path points

    lefts, rights = find_side_points(coordinates, origin)

    while True:
        # if len(path_points) > 2:
        #     last_remain_distance = math.hypot(path_points[-2][0] - target[0], path_points[-2][1] - target[1])
        #     remain_distance = math.hypot(path_points[-1][0] - target[0], path_points[-1][1] - target[1])
        #     print (last_remain_distance, remain_distance)
        #     if last_remain_distance <= remain_distance:
        #         print('here')
        #         distance = 30
        #     else:
        #         print('here?')
        #         distance = 10
        # else: 
        #     print('here!')
        #     distance = 10
        
        distance = 20

        next_point = point_along_line(current_point, target, distance)
        # print('next:', next_point)

        if next_point == target:
            path_points.append(target)
            return path_points
        
        closest_left, closest_right = find_closest_points_on_sides2(current_point, next_point, lefts, rights)
        # print(closest_left, closest_right)
        if not closest_left or not closest_right:
            adjusted_point = [0,0]
        else:
            adjusted_point = adjust_midpoint(next_point, closest_left, closest_right)

        # print('adjusted_point:', adjusted_point)

        path_points.append(adjusted_point)

        current_point = adjusted_point

def calculate_angle(origin, midpoint):
    vector_x = midpoint[0] - origin[0]
    vector_y = midpoint[1] - origin[1]
    
    angle_rad = math.atan2(vector_x, vector_y)  # atan2 gives angle relative to y-axis (0, 1) with correct sign
    angle_deg = math.degrees(angle_rad)

    return angle_deg

def convert_angle_to_ratio(angle):
    max_angle = 45.0
    ratio = max(min(angle / max_angle, 1.0), -1.0)
    return ratio

def plot_lines_to_farthest_point(lidar_data, distance=20):
    coordinates = lidar_to_2d_coordinates(lidar_data)
    farthest_point = find_farthest_point(coordinates)
    if not farthest_point:
        return

    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]

    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, s=10, c='blue', alpha=0.6, label='Lidar Points')
    plt.scatter(*farthest_point, c='red', s=50, label='Farthest Point (y > 0)')

    path_points = find_adjusted_path_with_points(farthest_point, [0, 0], 30, coordinates)

    path_x = [point[0] for point in path_points]
    path_y = [point[1] for point in path_points]
    plt.plot(path_x, path_y, 'g-', label='Adjusted Path')

    plt.scatter(path_x, path_y, c='purple', s=30, alpha=0.7, label='Path Points')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

def path_find(lidar_data):
    coordinates = lidar_to_2d_coordinates(lidar_data)
    farthest_point = find_farthest_point(coordinates)
    points = find_adjusted_path_with_points(farthest_point, [0, 0], 20, coordinates)
    print(points)
    points_distance = 2
    if len(points) < points_distance:
        points_distance = len(points)
    adjusted_midpoint = points[-points_distance]
    angle = calculate_angle([0, 0], adjusted_midpoint)
    ratio = convert_angle_to_ratio(angle)
    print(f"Angle: {angle} degrees")
    print(f"Ratio: {ratio}")
    return ratio, farthest_point

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
    if (len(scan) == 0):
        return False
    
    if not IS_SIM:
        scan_length = len(scan) # 1081, 1 for 0 angle maybe?
        values_per_angle = (scan_length - 1) / 270
        # print(scan_length, values_per_angle)
        degree_0 = int((scan_length - 1) / 2)
        first_half = scan[:degree_0] # 135 to 0
        backward = np.full(int(90 * values_per_angle - 1), 30)
        second_half = scan[degree_0:] # 0 to -135
        rotated_scan = np.concatenate([second_half, backward, first_half])
    else:
        rotated_scan = scan
    
    # scan_length = len(scan)
    # degree_270 = int(scan_length / 360.0 * 270)
    # rotated_scan = np.concatenate((scan[degree_270:], scan[:degree_270]))

    average_scan = np.array([rc_utils.get_lidar_average_distance(rotated_scan, angle, WINDOW_SIZE) for angle in range(360)])

    closest_left_angle, closest_left_distance = rc_utils.get_lidar_closest_point(average_scan, (-91, -89))
    closest_leftfront_angle, closest_leftfront_distance = rc_utils.get_lidar_closest_point(average_scan, (-41, -39))
    closest_right_angle, closest_right_distance = rc_utils.get_lidar_closest_point(average_scan, (89, 91))
    closest_rightfront_angle, closest_rightfront_distance = rc_utils.get_lidar_closest_point(average_scan, (39, 41))
    closest_front_angle, closest_front_distance = rc_utils.get_lidar_closest_point(average_scan, (-10, 10))
    # TODO
    # 30 to 60, -30 to -60 are blind spots. However, this setting makes the movement smoother. 
    # For blind spots, it would be better to find the closest point separately and handle it.
    # plot_lines_to_farthest_point(average_scan)
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

    if update_lidar() == False:
        return
    
    start = time.time()
    angle_error, farthest_point = path_find(average_scan)
    print('time: ', time.time() - start)
    if rc.controller.is_down(rc.controller.Button.A):
        plot_lines_to_farthest_point(average_scan)
        return
    # else:
    #     start = time.time()
    #     angle = path_find(average_scan)
    #     print('time: ', time.time() - start)
    #     # speed = 0.5
    #     # if abs(angle) > 0.4:
    #     #     speed = -0.3
    #     # elif abs(angle) > 0.2:
    #     #     speed = 0.0
    #     speed = 0.5 - (abs(angle))
    #     speed = max(-1.0, min(1.0, speed))
    #     rc.drive.set_speed_angle(speed, angle)
    # return

    speed = 0.5 - (abs(angle_error))
    speed = max(-1.0, min(1.0, speed))

    speed = 0.2

    # if abs(angle_error) > 0.2:
    #     speed = 0.0

    # if abs(angle_error) > 0.5:
    #     speed = -0.4
    # elif abs(angle_error) > 0.4:
    #     speed = -0.3
    # elif abs(angle_error) > 0.3:
    #     speed = -0.2
    # elif abs(angle_error) > 0.2:
    #     speed = -0.1
    # elif abs(angle_error) > 0.1:
    #     speed = 0.0

    # if closest_front_distance < 100:
    #     speed = -0.2

    # if abs(angle) > 0.4:
    #     speed = -0.5
    #     # if (flag % 6 == 0):
    #     #     speed = 0.2
    #     # else:
    #     #     speed = -0.5
    # elif abs(angle) > 0.2:
    #     speed = 0.0
    #     # if (flag % 6 == 0):
    #     #     speed = 0.2
    #     # else:
    #     #     speed = 0.0

    # flag += 1
    # flag %= 12

    # rc.drive.set_speed_angle(speed, angle_error)
    # return

    # angle_error = path_find(average_scan)
    KP = 0.2
    KD = 0.0

    # L = 150

    # degree = closest_right_angle - closest_rightfront_angle
    # theta_value = np.deg2rad(degree)

    # a_value = closest_rightfront_distance
    # b_value = closest_right_distance

    # alpha1 = np.arctan((a_value * np.cos(theta_value) - b_value) / a_value * np.sin(theta_value))
    # Dt = b_value * np.cos(alpha1)
    # right_Dt1 = Dt + np.sin(alpha1) * L

    # degree = closest_leftfront_angle - closest_left_angle
    # theta_value = np.deg2rad(degree)

    # a_value = closest_leftfront_distance
    # b_value = closest_left_distance

    # alpha2 = np.arctan((a_value * np.cos(theta_value) - b_value) / a_value * np.sin(theta_value))
    # Dt = b_value * np.cos(alpha2)
    # left_Dt1 = Dt + np.sin(alpha2) * L

    # max_alpha = max(np.rad2deg(alpha1), np.rad2deg(alpha2))
    # min_alpha = min(np.rad2deg(alpha1), np.rad2deg(alpha2))

    # # print(max_alpha, min_alpha)
    # # print(right_Dt1, left_Dt1)

    # left_gap = average_scan[270 + CURVE_ANGLE_SIZE] - average_scan[270]
    # right_gap = average_scan[90 - CURVE_ANGLE_SIZE] - average_scan[90]

    # if closest_left_distance > 200 and closest_right_distance > 200:
    #     rc.drive.set_speed_angle(1.0, 0.0)
    #     return
    # else:
    #     if closest_front_distance < 200:
    #         KP = 0.045
    #         KD = 0.085
    #         angle_error = (right_Dt1 - left_Dt1) / 2
    #     else:
    #         KP = 0.005
    #         KD = 0.012
    #         angle_error = (right_Dt1 - left_Dt1) / 2

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

    speed = 0.2

    farthest_distance = math.hypot(farthest_point[0], farthest_point[1])
    if farthest_distance < 350 and flag < 30:
        speed = 0.0
        flag += 1
    elif flag > 0:
        flag += 1
    else:
        flag = 0
    flag %= 60
    print('speed: ', speed, ' flag: ', flag)
    
    # if closest_front_distance < 40:
    #     speed = -1.0
    #     angle = -angle
    # if min_distance < 80 and flag < 30:
    #     speed = -0.1
    #     flag += 1
    # elif min_distance < 80 and flag < 60:
    #     speed = 0.0
    #     flag += 1
    # elif min_distance < 180 and flag < 30:
    #     speed = 0.1
    #     flag += 1
    # elif flag > 0:
    #     flag += 1
    # else:
    #     flag = 0

    # if abs(angle) > 0.2:
    #     if (flag % 3 == 0):
    #         speed = 0.0 # -0.2

    # if abs(angle) > 0.2:
    #     speed = 0.1
    # if abs(angle) > 0.2:
    #     if (flag % 3 == 0):
    #         speed = 0.1
    #     else:
    #         speed = 0.2
    # elif abs(angle) > 0.1:
    #     if (flag % 3 == 0):
    #         speed = 0.0
    #     else:
    #         speed = 0.1

    # flag += 1
    # flag %= 12

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
