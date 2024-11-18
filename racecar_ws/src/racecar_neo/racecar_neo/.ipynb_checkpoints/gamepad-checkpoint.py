#!/usr/local/bin/python3

# node for turning gamepad inputs into drive commands

import sys

import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDriveStamped

# get param file values
CAR_THROTTLE_FORWARD = float(0.25) # Parameter('car_throttle_forward',type_=Parameter.Type.DOUBLE,value=0.25)
CAR_THROTTLE_BACKWARD = float(0.25) # Parameter('car_throttle_backward',type_=Parameter.Type.DOUBLE,value=0.25)
CAR_THROTTLE_TURN = 1.0 # float(Parameter('car_throttle_turn',type_=Parameter.Type.DOUBLE,value=0.25).get_parameter_value())
GAMEPAD_THROTTLE_SPEED_SCALE = 1.0 # float(Parameter('gamepad_throttle_speed_scale',type_=Parameter.Type.DOUBLE,value=1.0).get_parameter_value())
GAMEPAD_THROTTLE_AXIS = int(1) # int(Parameter('gamepad_x_axis',type_=Parameter.Type.INTEGER,value=1).get_parameter_value())
GAMEPAD_STEER_AXIS = int(3) # Parameter('gamepad_y_axis',type_=Parameter.Type.INTEGER,value=3).get_parameter_value())

# scale x-stick input [-1, 1] to throttled drive speed output
def scale_x(x_in):
    if x_in >= 0:
        return x_in * CAR_THROTTLE_FORWARD * GAMEPAD_THROTTLE_SPEED_SCALE
    else:
        return x_in * CAR_THROTTLE_BACKWARD * GAMEPAD_THROTTLE_SPEED_SCALE

# scale y-stick input [-1, 1] to throttled drive angle output
def scale_y(y_in):
    return y_in * CAR_THROTTLE_TURN


def main(args=None):

    rclpy.init(args=args)
    node = rclpy.create_node('gamepad_node')

    try:
        pub = node.create_publisher(AckermannDriveStamped, '/gamepad_drive', 1)

        # gamepad callback
        def joy_callback(msg):
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = scale_x(msg.axes[GAMEPAD_THROTTLE_AXIS])
            drive_msg.drive.steering_angle = scale_y(msg.axes[GAMEPAD_STEER_AXIS])
            pub.publish(drive_msg)

        node.create_subscription(Joy, '/joy', joy_callback, 1)

        while rclpy.ok():
            rclpy.spin_once(node)

    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
