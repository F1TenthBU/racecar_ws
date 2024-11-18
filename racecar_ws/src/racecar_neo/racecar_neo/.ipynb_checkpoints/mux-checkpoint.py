#!/usr/local/bin/python3

# node for arbitrating who has drive control

# stdlib imports
import sys

# python 3
import rclpy
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy, QoSProfile

from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDriveStamped

from enum import Enum, auto

class MuxMode(Enum):
    """ Denotes which messages pass through the MUX """

    IDLE = (0, None)            # nothing passes through
    GAMEPAD = (auto(), 4)       # gamepad commands pass through
    AUTONOMY = (auto(), 5)      # drive commands pass through
    ERROR = (auto(), None)      # ignore this value
    # note: 'auto()' requires python 3.6+

    def __init__(self, value, enable_button):
        self.enable = None
        self._value_ = value
        self.enable = enable_button

# # If we want to configure these from parameters, do so here...
# MuxMode.GAMEPAD.enable=4
# MuxMode.AUTONOMY.enable=5


def main(args=None):

    # init ROS
    rclpy.init(args=args)
    node = rclpy.create_node('mux_node')

    try:
        # see example: https://github.com/ros2/demos/blob/master/topic_monitor/topic_monitor/scripts/data_publisher.py
        qos_profile = QoSProfile(depth=1)
        qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos_profile.durability = QoSDurabilityPolicy.VOLATILE

        pub = node.create_publisher(AckermannDriveStamped, '/mux_out', qos_profile)

        mode = MuxMode.IDLE

        # gamepad callback
        def joy_callback(msg):
            nonlocal mode
            # if LB is pressed, enable teleop
            if msg.buttons[MuxMode.GAMEPAD.enable] == 1:
                mode = MuxMode.GAMEPAD
            # if RB is pressed, enable autonomy
            elif msg.buttons[MuxMode.AUTONOMY.enable] == 1:
                mode = MuxMode.AUTONOMY

            # otherwise default, publish stop
            else:
                mode = MuxMode.IDLE
                pub.publish(AckermannDriveStamped())


        # callback for gamepad_drive topic
        def gamepad_drive_callback(msg):
            if mode == MuxMode.GAMEPAD:
                pub.publish(msg)

        # callback for drive topic
        def autonomy_drive_callback(msg):
            if mode == MuxMode.AUTONOMY:
                pub.publish(msg)

        node.create_subscription(AckermannDriveStamped, '/gamepad_drive', gamepad_drive_callback, qos_profile)
        node.create_subscription(AckermannDriveStamped, '/drive', autonomy_drive_callback, qos_profile)
        node.create_subscription(Joy, '/joy', joy_callback, qos_profile)

        # wait before shutdown
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()



if __name__ == '__main__':
    main(sys.argv)
