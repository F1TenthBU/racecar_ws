#!/usr/bin/python3

# node abstraction for using maestro.py driver for servo/motor

# standard library imports
import sys
import time

# ROS imports
import rclpy
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy, QoSProfile

from ackermann_msgs.msg import AckermannDriveStamped


from . import maestro

CAR_SPEED_MAX_DUTY = 7504      #7488 #7216
CAR_SPEED_CENTER_DUTY = 5984       #6100 #5748
CAR_SPEED_MIN_DUTY = 4204     #4616 #4616 #4188

CAR_ANGLE_MAX_DUTY = 8400
CAR_ANGLE_CENTER_DUTY = 6000
CAR_ANGLE_MIN_DUTY = 3600

CAR_MAX_FORWARD = 1.00  #0.25    Parameter('car_max_forward',type_=Parameter.Type.DOUBLE,value=0.25)
CAR_MAX_BACKWARD = 1.00 #0.25   Parameter('car_max_backward',type_=Parameter.Type.DOUBLE,value=0.25)
CAR_MAX_TURN = float(0.4) # Parameter('car_max_turn', type_=Parameter.Type.DOUBLE, value=0.4)
STEERING_CENTER_REL_OFFSET = float(0) # Parameter('steering_center_rel_offset', type_=Parameter.Type.DOUBLE, value=0.0)

# simple function to do a linear mapping
def map_val(value, inMin, inMax, outMin, outMax):
    inSpan = inMax - inMin
    outSpan = outMax - outMin
    valScaled = float(value - inMin) / float(inSpan)
    return outMin + (valScaled * outSpan)


def main(args=None):

    # default config
    controller = maestro.Controller()

    # setup motor config
    controller.setRange(0,CAR_SPEED_MIN_DUTY,CAR_SPEED_MAX_DUTY)
    controller.setSpeed(0,0)
    controller.setAccel(0,0)
    controller.setTarget(0,CAR_SPEED_CENTER_DUTY)

    # setup steering config
    controller.setRange(1,CAR_ANGLE_MIN_DUTY,CAR_ANGLE_MAX_DUTY)
    controller.setSpeed(1,0)
    controller.setAccel(1,0)
    controller.setTarget(1,CAR_ANGLE_CENTER_DUTY)

    # msg.drive.speed:
    # [-1.5, 1.5] -> | -1.5 full reverse | 1.5 full forward |
    # msg.drive.steering_angle:
    # [-0.4, 0.4] -> | -0.4 full left    | 0.4 full right   |
    def motor_callback(msg):

        print("motor_callback:")
        print(f"    >> speed:  {msg.drive.speed}")
        print(f"    >> steer:  {msg.drive.steering_angle}")

        if msg.drive.speed >= 0:
            lin_vel = map_val(msg.drive.speed, 0, CAR_MAX_FORWARD, CAR_SPEED_CENTER_DUTY, CAR_SPEED_MAX_DUTY)
        else:
            lin_vel = map_val(msg.drive.speed, -CAR_MAX_BACKWARD, 0, CAR_SPEED_MIN_DUTY, CAR_SPEED_CENTER_DUTY)

        turn_angle = map_val(msg.drive.steering_angle, -CAR_MAX_TURN, CAR_MAX_TURN, CAR_ANGLE_MAX_DUTY, CAR_ANGLE_MIN_DUTY)

        print(f"    << speed:  {lin_vel}")
        print(f"    << steer:  {turn_angle}")

        # set drive speed
        controller.setTarget(0, int(lin_vel))

        # added Y_OFFSET to center servo
        controller.setTarget(1, int(turn_angle + STEERING_CENTER_REL_OFFSET))

    # init ros
    rclpy.init(args=args)
    node = rclpy.create_node('pwm')

    try:
        qos_profile = QoSProfile(depth=1)
        qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos_profile.durability = QoSDurabilityPolicy.VOLATILE

        node.create_subscription(AckermannDriveStamped, '/motor', motor_callback, qos_profile)

        # wait and shutdown
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    finally:
        controller.close()
        rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv)
