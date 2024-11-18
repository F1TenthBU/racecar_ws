#!/usr/local/bin/python3

# node for ensuring commanded speed does not exceed throttle

# stdlib imports
import sys

# import ros libraries
import rclpy
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy, QoSProfile

from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDriveStamped

# get param file values
CAR_THROTTLE_FORWARD = float(0.25) # rclpy.parameter.Parameter('car_throttle_forward',type_=DOUBLE, 0.25)
CAR_THROTTLE_BACKWARD = float(0.25) # rclpy.param.Parameter('car_throttle_backward',type_=DOUBLE, 0.25)
CAR_THROTTLE_TURN = float(1.0) # rclpy.param.Parameter('car_throttle_turn',type_=DOUBLE, 1.0)


def main(args=None):

    # init ROS
    rclpy.init(args=args)
    node = rclpy.create_node('throttle_node')

    try:
        qos_profile = QoSProfile(depth=1)
        qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos_profile.durability = QoSDurabilityPolicy.VOLATILE

        pub = node.create_publisher(AckermannDriveStamped, '/motor', qos_profile)

        # callback that throttles max speed and angle
        def drive_callback(msg):
            if msg.drive.speed > CAR_THROTTLE_FORWARD:
                msg.drive.speed = CAR_THROTTLE_FORWARD

            if msg.drive.speed < -CAR_THROTTLE_BACKWARD:
                msg.drive.speed = -CAR_THROTTLE_BACKWARD

            if msg.drive.steering_angle > CAR_THROTTLE_TURN:
                msg.drive.steering_angle = CAR_THROTTLE_TURN

            if msg.drive.steering_angle < -CAR_THROTTLE_TURN:
                msg.drive.steering_angle = -CAR_THROTTLE_TURN

            pub.publish(msg)

        node.create_subscription(AckermannDriveStamped, '/mux_out', drive_callback, qos_profile)

        # wait before shutdown
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv)
