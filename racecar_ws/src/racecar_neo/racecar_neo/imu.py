#!/usr/local/bin/python3

# node for processing data from LSM9DS1 IMU through smbus I2C bus and publishing to IMU node

import smbus
import time
import math

import rclpy
from sensor_msgs.msg import Imu

def main(args=None):

    bus = smbus.SMBus(1)

    def twos_complement(val, bits):
        if (val & (1 << (bits - 1))) != 0:
            val = val - (1 << bits)
        return val
    
    SENSITIVITY_ACCELEROMETER_2 = 0.000061
    SENSITIVITY_GYROSCOPE_245 = 0.00875
    SENSITIVITY_MAGNETOMETER_4 = 0.00014
    
    # enable gyroscope
    bus.write_byte_data(0x6B, 0x10, 0b11000011)
    
    # enable accelerometer
    bus.write_byte_data(0x6B, 0x20, 0b11000110)
    
    # enable magnetometer
    bus.write_byte_data(0x1E, 0x20, 0b11111100)
    bus.write_byte_data(0x1E, 0x21, 0b00000000)
    bus.write_byte_data(0x1E, 0x22, 0b00000000)
    bus.write_byte_data(0x1E, 0x23, 0b00001100)
    
    # init ROS
    rclpy.init(args=args)
    node = rclpy.create_node('imu_node')
    pub_imu = node.create_publisher(Imu,'/imu', 1)
    print("Topics set up!")

    # Get frames, send frames through published thread 
    while rclpy.ok():        
        # Access the IMU data (gyro)
        out_x_g_l = bus.read_byte_data(0x6B, 0x18)
        out_x_g_h = bus.read_byte_data(0x6B, 0x19)
        out_x_g = twos_complement((out_x_g_h << 8) | out_x_g_l, 16) * SENSITIVITY_GYROSCOPE_245
        ############################
        out_y_g_l = bus.read_byte_data(0x6B, 0x1A)
        out_y_g_h = bus.read_byte_data(0x6B, 0x1B)
        out_y_g = twos_complement((out_y_g_h << 8) | out_y_g_l, 16) * SENSITIVITY_GYROSCOPE_245
        ############################
        out_z_g_l = bus.read_byte_data(0x6B, 0x1C)
        out_z_g_h = bus.read_byte_data(0x6B, 0x1D)
        out_z_g = twos_complement((out_z_g_h << 8) | out_z_g_l, 16) * SENSITIVITY_GYROSCOPE_245

        # Convert gyro data from dps to rad/s
        out_x_g = round(out_x_g * (math.pi/180), 10)
        out_y_g = round(out_y_g * (math.pi/180), 10)
        out_z_g = round(out_z_g * (math.pi/180), 10)

        # Access the IMU data (accel)
        out_x_xl_l = bus.read_byte_data(0x6B, 0x28)
        out_x_xl_h = bus.read_byte_data(0x6B, 0x29)
        out_x_xl = twos_complement((out_x_xl_h << 8) | out_x_xl_l, 16) * SENSITIVITY_ACCELEROMETER_2
        ############################
        out_y_xl_l = bus.read_byte_data(0x6B, 0x2A)
        out_y_xl_h = bus.read_byte_data(0x6B, 0x2B)
        out_y_xl = twos_complement((out_y_xl_h << 8) | out_y_xl_l, 16) * SENSITIVITY_ACCELEROMETER_2
        ############################
        out_z_xl_l = bus.read_byte_data(0x6B, 0x2C)
        out_z_xl_h = bus.read_byte_data(0x6B, 0x2D)
        out_z_xl = twos_complement((out_z_xl_h << 8) | out_z_xl_l, 16) * SENSITIVITY_ACCELEROMETER_2

        # Convert accel data from g's to m/s^2
        out_x_xl = round(out_x_xl * 9.80665, 10)
        out_y_xl = round(out_y_xl * 9.80665, 10)
        out_z_xl = round(out_z_xl * 9.80665, 10)
        
        # build Imu message
        msg = Imu()
        msg.angular_velocity.x = out_x_g
        msg.angular_velocity.y = out_y_g
        msg.angular_velocity.z = out_z_g
        msg.linear_acceleration.x = out_x_xl
        msg.linear_acceleration.y = out_y_xl
        msg.linear_acceleration.z = out_z_xl
    
        # publish Imu message (pub_accel = accel data)
        pub_imu.publish(msg)

    pipeline.stop()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
