########################################################################################
# Imports
########################################################################################

import sys, time
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import splprep, splev
from scipy.spatial import distance
from nptyping import NDArray
from typing import Any, Tuple

sys.path.insert(1, "../library")
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
# PID Constants for speed control
SPEED_KP = 0.3  # Proportional gain for speed
SPEED_KI = 0.001  # Integral gain for speed
SPEED_KD = 0.05  # Derivative gain for speed
MAX_SPEED = 1.0  # Maximum allowed speed
MIN_SPEED = -0.5  # Minimum allowed speed (for reverse)
DESIRED_DISTANCE_FROM_CENTER = 0.0  # Target distance from center path

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
desired_speed = 1.0  # Set desired speed to 0.5 (you can adjust this value)

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

def lidar_to_2d_coordinates_vectorized(lidar_data):
    adjusted_angle_rad = np.radians(90 - angle)  # Shift 0 degrees to point upward
    x = distance * np.cos(adjusted_angle_rad)
    y = distance * np.sin(adjusted_angle_rad)
    coordinates = np.array([x,y])
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

def adjust_midpoint(midpoint, closest_left, closest_right, distance=25):
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
        
        distance = 20

        next_point = point_along_line(current_point, target, distance)
        print('next:', next_point)

        if next_point == target:
            path_points.append(target)
            return path_points
        
        closest_left, closest_right = find_closest_points_on_sides2(current_point, next_point, lefts, rights)
        print(closest_left, closest_right)
        if not closest_left or not closest_right:
            adjusted_point = [0,0]
        else:
            adjusted_point = adjust_midpoint(next_point, closest_left, closest_right)

        print('adjusted_point:', adjusted_point)

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

def plot_lines_to_farthest_point_in_func(lidar_data, coordinates, farthest_point, path_points, distance=20):
    plt.clf()  # Clear the previous figure

    # x_coords = [coord[0] for coord in coordinates]
    # y_coords = [coord[1] for coord in coordinates]
    coordinates = np.array(coordinates)
    x_coords = coordinates[:, 0]
    y_coords = coordinates[:, 1]

    plt.scatter(x_coords, y_coords, s=10, c='blue', alpha=0.6, label='Lidar Points')
    plt.scatter(*farthest_point, c='red', s=50, label='Farthest Point (y > 0)')

    # path_x = [point[0] for point in path_points]
    # path_y = [point[1] for point in path_points]
    path_points = np.array(path_points)
    path_x = path_points[:, 0]
    path_y = path_points[:, 1]

    plt.plot(path_x, path_y, 'g-', label='Adjusted Path')

    plt.scatter(path_x, path_y, c='purple', s=30, alpha=0.7, label='Path Points')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()

    plt.pause(0.001)

def path_find(lidar_data):
    coordinates = lidar_to_2d_coordinates(lidar_data)
    farthest_point = find_farthest_point(coordinates)
    points = find_adjusted_path_with_points(farthest_point, [0, 0], 30, coordinates)
    # print(points)
    adjusted_midpoint = points[-3]
    angle = calculate_angle([0, 0], adjusted_midpoint)
    ratio = convert_angle_to_ratio(angle)
    print(f"Angle: {angle} degrees")
    print(f"Ratio: {ratio}")

    plot_lines_to_farthest_point_in_func(lidar_data, coordinates, farthest_point, points, distance=20)
    return ratio

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
    print(average_scan)

    linear = rc.physics.get_linear_acceleration()
    angular = rc.physics.get_angular_velocity()
    print('Linear: ', linear, ' Angular: ', angular)

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
    
def calculate_center_error(closest_left, closest_right):
    """
    Calculate how far the car is from the center of the path
    Returns positive value if closer to left wall, negative if closer to right wall
    """
    if not closest_left or not closest_right:
        return 0.0
        
    left_distance = math.hypot(closest_left[0], closest_left[1])
    right_distance = math.hypot(closest_right[0], closest_right[1])
    
    # Calculate the error (difference from center)
    # Positive error means closer to left wall, negative means closer to right wall
    center_error = (right_distance - left_distance) / 2
    
    return center_error

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
    start = time.time()
    angle_error = path_find(average_scan)
    print('time: ', time.time() - start)

    # speed = 0.5 - (abs(angle_error))
    # speed = max(-1.0, min(1.0, speed))

    # speed = 0.2

    # if abs(angle) > 0.4:
    #     speed = -0.5

    # elif abs(angle) > 0.2:
    #     speed = 0.0


    # flag += 1
    # flag %= 12
    
     # Calculate speed error (difference between desired and actual speed)
    speed_error = desired_speed - speed

    # If a crash is expected, slow down
    closest_side_distance = min(closest_left_distance, closest_right_distance)
    if closest_front_distance < closest_side_distance * 2:
        speed_error -= 0.1
        # speed_error -= (1 * (closest_front_distance / (closest_side_distance * 4)))
        # TODO: Find a better formula..

    # Update speed integral term
    integral_speed += speed_error

    # Update speed derivative term
    speed_derivative = speed_error - prev_error_speed
    prev_error_speed = speed_error

    # Calculate speed PID output
    speed_pid_output = KP_speed * speed_error + KI_speed * integral_speed + KD_speed * speed_derivative

    # Convert speed PID output to speed
    speed = speed_pid_output
        
        # Apply controls
    rc.drive.set_speed_angle(speed, angle_error)

    #rc.drive.set_speed_angle(speed, angle_error)
    return


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    plt.ion()
    rc.set_start_update(start, update)
    rc.go()
