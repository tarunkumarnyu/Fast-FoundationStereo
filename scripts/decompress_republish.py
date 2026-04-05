#!/usr/bin/env python3
"""Decompress CompressedImage topics and republish as raw Image.

Usage:
  ros2 run --prefix 'python3' fast_ffs scripts/decompress_republish.py
  # or just:
  python3 scripts/decompress_republish.py --ns /race6/cam1

Subscribes to {ns}/infra1/image_rect_raw/compressed and {ns}/infra2/image_rect_raw/compressed,
publishes raw Image on {ns}/infra1/image_rect_raw and {ns}/infra2/image_rect_raw.
"""
import argparse
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
import cv2
import numpy as np


class DecompressRepublisher(Node):
    def __init__(self, ns):
        super().__init__('decompress_republish')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )

        self.pub_left = self.create_publisher(Image, f'{ns}/infra1/image_rect_raw', qos)
        self.pub_right = self.create_publisher(Image, f'{ns}/infra2/image_rect_raw', qos)

        self.sub_left = self.create_subscription(
            CompressedImage, f'{ns}/infra1/image_rect_raw/compressed',
            lambda msg: self._cb(msg, self.pub_left), qos)
        self.sub_right = self.create_subscription(
            CompressedImage, f'{ns}/infra2/image_rect_raw/compressed',
            lambda msg: self._cb(msg, self.pub_right), qos)

        self.get_logger().info(f'Decompressing {ns}/infra{{1,2}}/image_rect_raw/compressed -> raw')

    def _cb(self, msg: CompressedImage, pub):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return

        out = Image()
        out.header = msg.header
        out.height, out.width = img.shape[:2]
        if img.ndim == 2:
            out.encoding = 'mono8'
            out.step = out.width
        else:
            out.encoding = 'bgr8'
            out.step = out.width * 3
        out.data = img.tobytes()
        pub.publish(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ns', default='/race6/cam1')
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = DecompressRepublisher(args.ns)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
