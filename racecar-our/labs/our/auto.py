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
SLOW_TEST = False

# >> Constants
WINDOW_SIZE = 8 # Window size to calculate the average distance

# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels

# >> !!! TUNING VARIABLES !!!
if IS_SIM:
    PREDICT_LEVEL = 3
    CAR_WIDTH = 30
    UNIT_PATH_LENGTH = 20
    BRAKE_DISTANCE = 345
    BRAKE_SECOND = 1/3.5
    HARD_BRAKE_SECOND = 1/3.0
    TARGET_SPEED = 0.2
    # Initialize PID control variables for angle
    KP = 1.0
    KI = 0.0
    KD = 0.0
else:
    # This Values are for REAL
    PREDICT_LEVEL = 2
    CAR_WIDTH = 45 #40
    UNIT_PATH_LENGTH = 20
    BRAKE_DISTANCE = 325 #400 #250
    BRAKE_SECOND = 1/3.25
    # HARD_BRAKE_DISTANCE = 200
    HARD_BRAKE_SECOND = 1/3.5
    TARGET_SPEED = 0.4 #0.4 #0.42
    HARD_BREAK_SPEED = -0.6
    BREAK_SPEED = -0.4
    MINIMUM_SPEED = 0.2
    BOOST_SPEED = 0.5 #0.5
    TARGET_PREDICT_DISTANCE = 50
    EMERGENCY_DISTANCE = 150
    MINIMUM_GAP = 40
    if SLOW_TEST:
        TARGET_SPEED = 0.3
        BOOST_SPEED = 0.3
        MINIMUM_SPEED = 0.2
    KP = 0.6 #0.215 #0.275 #0.25 0.2
    KI = 0.0
    KD = 0.1 #0.22 #0.2 # 0.1  0.05   0.01

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

def get_farthest_distance_in_range(scan, start, end):
    scan_size = len(scan)
    if start < 0:
        start += scan_size
    if end < 0:
        end += scan_size

    if start <= end:
        values_in_range = scan[start:end + 1]
    else:
        values_in_range = np.concatenate((scan[start:], scan[:end + 1]))

    return np.max(values_in_range)

def lidar_to_2d_coordinates(lidar_data):
    coordinates = []
    for angle in range(360):
        distance = lidar_data[angle]
        # 0 degrees is up (positive y-axis), adjusting angle accordingly
        adjusted_angle_rad = math.radians(90 - angle)  # Shift 0 degrees to point upward
        x = distance * math.cos(adjusted_angle_rad)
        y = distance * math.sin(adjusted_angle_rad)
        coordinates.append((x, y, distance))
    return coordinates

def find_farthest_point(coordinates):
    filtered_points = [point for point in coordinates if point[1] > 0]
    if not filtered_points:
        return None, None
    
    farthest_point = max(filtered_points, key=lambda p: p[2])
    farthest_idx = filtered_points.index(farthest_point)
    
    if farthest_point[0] == 0 or farthest_idx == 0 or farthest_idx == len(filtered_points) - 1:
        return farthest_point, None
    
    if farthest_point[0] < 0:
        second_farthest_point = filtered_points[farthest_idx + 1]
    else:
        second_farthest_point = filtered_points[farthest_idx - 1]
    
    return farthest_point, second_farthest_point

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
        print('WARN! No left or right side.')
        # print(lefts)
        # print(rights)
        return lefts, rights

    return lefts, rights

def find_closest_points_on_sides(origin, midpoint, left_points, right_points):
    y_threshold = origin[1]

    filtered_lefts = [point for point in left_points if not (point[1] > y_threshold)]
    filtered_rights = [point for point in right_points if not (point[1] > y_threshold)]

    closest_left = min(filtered_lefts, key=lambda p: math.hypot(p[0] - midpoint[0], p[1] - midpoint[1]), default=None)
    closest_right = min(filtered_rights, key=lambda p: math.hypot(p[0] - midpoint[0], p[1] - midpoint[1]), default=None)

    # closest_left = min(left_points, key=lambda p: math.hypot(p[0] - midpoint[0], p[1] - midpoint[1]), default=None)
    # closest_right = min(right_points, key=lambda p: math.hypot(p[0] - midpoint[0], p[1] - midpoint[1]), default=None)

    return closest_left, closest_right

def adjust_midpoint(midpoint, closest_left, closest_right, distance=CAR_WIDTH):
    # distance here is different to one in find_adjusted_path_with_points()
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

def point_to_line_distance(point, line_start, line_end):
    x0, y0 = point[:2]
    x1, y1 = line_start[:2]
    x2, y2 = line_end[:2]
    
    line_vector = (x2-x1, y2-y1)
    point_vector = (x0-x1, y0-y1)
    
    line_length_sq = line_vector[0]**2 + line_vector[1]**2
    
    if line_length_sq == 0:
        return math.hypot(x0-x1, y0-y1)
    
    t = max(0, min(1, (point_vector[0]*line_vector[0] + point_vector[1]*line_vector[1]) / line_length_sq))
    
    projection_x = x1 + t * line_vector[0]
    projection_y = y1 + t * line_vector[1]
    
    return math.hypot(x0-projection_x, y0-projection_y)

def find_closest_distances_to_line(next_point, target, lefts, rights):
    closest_left_fit = float('inf')
    closest_right_fit = float('inf')
    closest_left_point = None
    closest_right_point = None
    
    for left_point in lefts:
        dist = point_to_line_distance(left_point, next_point, target)
        if dist < closest_left_fit:
            closest_left_fit = dist
            closest_left_point = left_point
    
    for right_point in rights:
        dist = point_to_line_distance(right_point, next_point, target)
        if dist < closest_right_fit:
            closest_right_fit = dist
            closest_right_point = right_point
        
    return closest_left_fit, closest_right_fit, closest_left_point, closest_right_point

def find_adjusted_path_with_points(origin, second_origin, target, coordinates, distance=UNIT_PATH_LENGTH, future_position=None):
    lefts, rights = find_side_points(coordinates, origin)
    target = future_position if future_position else target

    current_point = origin[:2]
    path_points = [current_point]
    
    if not second_origin == None:
        vector_x = second_origin[0] - origin[0]
        vector_y = second_origin[1] - origin[1]
        
        if vector_x != 0 or vector_y != 0:
            magnitude = math.sqrt(vector_x**2 + vector_y**2)
            unit_vector = [vector_x/magnitude, vector_y/magnitude]
        else:
            unit_vector = [0, 1]
    
        current_point = [
            origin[0] + unit_vector[0] * TARGET_PREDICT_DISTANCE,
            origin[1] + unit_vector[1] * TARGET_PREDICT_DISTANCE
        ]
        path_points.append(current_point)

    count = 1
    while True:
        if count % 20 == 0:
            distance *= 2

        next_point = point_along_line(current_point, target, distance)
        
        remaining_distance = math.hypot(next_point[0] - target[0], next_point[1] - target[1])        
        if remaining_distance <= EMERGENCY_DISTANCE:
            closest_left_fit, closest_right_fit, closest_left_point, closest_right_point = find_closest_distances_to_line(next_point, target, lefts, rights)
            if closest_left_fit + closest_right_fit < MINIMUM_GAP:
                print('No path found!', closest_left_fit, closest_right_fit)
                print(closest_left_point, closest_right_point)
                print(path_points)
                return []

        if math.hypot(next_point[0] - target[0], next_point[1] - target[1]) < distance * 2:
            path_points.append(target)
            return path_points

        if next_point == target:
            path_points.append(target)
            return path_points
        
        closest_left, closest_right = find_closest_points_on_sides(current_point, next_point, lefts, rights)
        # print(closest_left, closest_right)
        if not closest_left or not closest_right:
            # adjusted_point = [0,0]
            path_points.append(target)
            return path_points
        else:
            adjusted_point = adjust_midpoint(next_point, closest_left, closest_right)

        # print('adjusted_point:', adjusted_point)
        path_points.append(adjusted_point)
        current_point = adjusted_point

        count += 1

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

def plot_lines_to_farthest_point_in_func(lidar_data, coordinates, farthest_point, path_points):
    plt.clf()  # Clear the previous figure

    coordinates = np.array(coordinates)
    x_coords = coordinates[:, 0]
    y_coords = coordinates[:, 1]

    plt.scatter(x_coords, y_coords, s=10, c='blue', alpha=0.6, label='Lidar Points')
    plt.scatter(*farthest_point, c='red', s=50, label='Farthest Point (y > 0)')

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

def calculate_new_origin(angle, speed, distance=50):
    radians = math.radians(angle * 45)

    travel_distance = speed * distance

    new_x = travel_distance * math.sin(radians)
    new_y = travel_distance * math.cos(radians)

    return [new_x, new_y]

def path_find(lidar_data, angle, speed):
    future_position = calculate_new_origin(angle, speed)
    print(future_position)

    coordinates = lidar_to_2d_coordinates(lidar_data)
    farthest_point, second_farthest_point = find_farthest_point(coordinates)
    # points = find_adjusted_path_with_points(farthest_point, second_farthest_point, [0, 0], coordinates)
    # points = find_adjusted_path_with_points(farthest_point, [0, 0], coordinates)
    points = find_adjusted_path_with_points(farthest_point, second_farthest_point, [0, 0], coordinates, future_position=future_position)
    # print('PATH: ', points)

    if len(points) == 0:
        return 0.0, None

    points_distance = PREDICT_LEVEL
    if len(points) < points_distance:
        points_distance = len(points)

    adjusted_midpoint = points[-points_distance]
    # angle = calculate_angle([0, 0], adjusted_midpoint)
    angle = calculate_angle(future_position, adjusted_midpoint)
    ratio = convert_angle_to_ratio(angle)

    print(f"Angle: {angle} degrees")
    print(f"Ratio: {ratio}")

    if SHOW_PLOT:
        plot_lines_to_farthest_point_in_func(lidar_data, coordinates, farthest_point[:-1], points)
    return ratio, farthest_point

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
    global flag

    # rc.drive.set_speed_angle(0.1, 0.0)
    # return

    if update_lidar() == False:
        return
    
    start = time.time()
    angle_error, farthest_point = path_find(average_scan, angle, speed)
    print('time: ', time.time() - start)

    # Update angle integral term
    integral_angle += angle_error

    # Update angle derivative term
    angle_derivative = angle_error - prev_error_angle
    prev_error_angle = angle_error

    # Calculate angle PID output
    angle_pid_output = KP * angle_error + KI * integral_angle + KD * angle_derivative

    # Convert angle PID output to angle
    angle = angle_pid_output

    # # PID control for speed
    # # Calculate speed error (difference between desired and actual speed)
    # speed_error = desired_speed - speed

    # # Update speed integral term
    # integral_speed += speed_error

    # # Update speed derivative term
    # speed_derivative = speed_error - prev_error_speed
    # prev_error_speed = speed_error

    # # Calculate speed PID output
    # speed_pid_output = KP_speed * speed_error + KI_speed * integral_speed + KD_speed * speed_derivative

    # # Convert speed PID output to speed
    # speed += speed_pid_output

    speed = MINIMUM_SPEED
    if abs(angle) < 0.25:
        speed = TARGET_SPEED

    if farthest_point == None:
        speed = 0.0
    else:
        farthest_distance = farthest_point[1] #2
        if farthest_point[1] < 30 and flag < (60 * HARD_BRAKE_SECOND):
            speed = HARD_BREAK_SPEED
            angle *= 1.5
            print('hard breaking')
            flag += 1
        # if farthest_distance < HARD_BRAKE_DISTANCE and flag < (60 * HARD_BRAKE_SECOND):
        #     speed = -0.2
        #     flag += 1
        elif farthest_distance < BRAKE_DISTANCE and flag < (60 * BRAKE_SECOND):
            speed = BREAK_SPEED
            angle *= 1.2
            print('breaking')
            flag += 1
        elif flag > 0:
            speed = MINIMUM_SPEED
            flag += 1
        else:
            if angle < 0.05 and angle > -0.05 and farthest_distance > 500:
                speed = BOOST_SPEED
            flag = 0
        flag %= 60

   # if distance_left or distance_right < 55:
    #    speed = MINIMUM_SPEED


    # print(speed, flag)
    print(speed)
    # angle = -0.1
    if TEST_WITHOUT_SPEED:
        speed = 0.0
    # speed = -0.75
    # emergency_distance = get_farthest_distance_in_range(average_scan, -45, 45)
    # if emergency_distance < 30:
    #     speed = -1.0

    # Constrain speed and angle within 0.0 to 1.0
    # speed = max(0.0, min(1.0, speed))
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
