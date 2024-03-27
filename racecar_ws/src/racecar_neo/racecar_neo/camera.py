#!/usr/local/bin/python3

# node for natively publishing RGB color frames from Logitech BRIO 4k camera

import rclpy
from sensor_msgs.msg import Image

import cv2
import numpy as np
import time

def main(args=None):
    
    # init ROS
    rclpy.init(args=args)
    node = rclpy.create_node('camera_node')
    pub_cam = node.create_publisher(Image,'/camera', 1)
    print("Topics set up!")

    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, change if needed
    _, frame = cap.read()
    if frame is None:
        node.get_logger().warn("Couldn't read frame from webcam, attempting secondary method!")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    _, frame = cap.read()
    if frame is None:
        node.get_logger().warn("Couldn't read frame from webcam, attempting final force search method!")
        cap = cv2.VideoCapture(-1)

    # Get frames, send frames thro`ugh published thread 
    while rclpy.ok():        
        _, frame = cap.read()
        
        if frame is not None:
            color_image = frame
            # Resize Frame (50%)
            # color_image = cv2.resize(color_image, (320, 240))
            # BGR TO RGB
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            # Compress image into jpeg format
            _, color_jpeg = cv2.imencode('.jpg', color_image)
    
            # Create Image ROS2 message
            msg = Image()
            msg.height = color_image.shape[0]
            msg.width = color_image.shape[1]
            msg.encoding = 'jpeg'
            msg.is_bigendian = 0
            msg.step = 3 * msg.width
            msg.data = np.array(color_jpeg).tostring()
        
            # Publish color image
            pub_cam.publish(msg)
        else:
            node.get_logger().warn("Couldn't read frame from the webcam.")
    
    pipeline.stop()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
