#!/usr/bin/env python3
"""Publish FFS disparity visualization to ROS2 for rqt_image_view."""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np


class FfsVizPub(Node):
    def __init__(self):
        super().__init__('ffs_viz_pub')
        self.pub_left = self.create_publisher(Image, '/ffs/left', 1)
        self.pub_disp = self.create_publisher(Image, '/ffs/disp_viz', 1)

        self.left = cv2.imread('/tmp/ffs_output/left.png')
        self.disp = cv2.imread('/tmp/ffs_output/trt_disp_vis.png')

        # Resize to half for bandwidth
        self.left = cv2.resize(self.left, (self.left.shape[1]//2, self.left.shape[0]//2))
        self.disp = cv2.resize(self.disp, (self.disp.shape[1]//2, self.disp.shape[0]//2))

        self.timer = self.create_timer(0.5, self.publish)
        self.get_logger().info(f'Publishing left {self.left.shape} and disp {self.disp.shape} at 2Hz')

    def cv2_to_imgmsg(self, img):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height, msg.width = img.shape[:2]
        msg.encoding = 'bgr8'
        msg.step = img.shape[1] * 3
        msg.data = img.tobytes()
        return msg

    def publish(self):
        self.pub_left.publish(self.cv2_to_imgmsg(self.left))
        self.pub_disp.publish(self.cv2_to_imgmsg(self.disp))


def main():
    rclpy.init()
    node = FfsVizPub()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
