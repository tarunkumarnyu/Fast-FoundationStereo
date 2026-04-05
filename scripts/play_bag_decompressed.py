#!/usr/bin/env python3
"""Read mcap bag and publish decompressed infra images as raw sensor_msgs/Image.

Bypasses ros2 bag play (yaml-cpp bug) by reading mcap directly with Python.

Usage:
  python3 scripts/play_bag_decompressed.py --bag /path/to/bag.mcap [--rate 1.0] [--ns /race6/cam1]
"""
import argparse
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time
from mcap_ros2.reader import read_ros2_messages


class BagPlayer(Node):
    def __init__(self, bag_path, ns, rate):
        super().__init__('bag_player')
        self.bag_path = bag_path
        self.ns = ns
        self.rate = rate

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )
        self.pub_left = self.create_publisher(Image, f'{ns}/infra1/image_rect_raw', qos)
        self.pub_right = self.create_publisher(Image, f'{ns}/infra2/image_rect_raw', qos)

        self.get_logger().info(f'Playing {bag_path} at {rate}x -> {ns}/infra{{1,2}}/image_rect_raw')

    def play(self):
        topic_left = f'{self.ns}/infra1/image_rect_raw/compressed'
        topic_right = f'{self.ns}/infra2/image_rect_raw/compressed'

        # Read all messages sorted by time, both topics
        msgs = []
        for msg in read_ros2_messages(self.bag_path, topics=[topic_left, topic_right]):
            msgs.append((msg.channel.topic, msg.ros_msg, msg.log_time))

        if not msgs:
            self.get_logger().error('No messages found!')
            return 0

        # Sort by time
        msgs.sort(key=lambda x: x[2])
        self.get_logger().info(f'Loaded {len(msgs)} messages, duration: {(msgs[-1][2] - msgs[0][2]).total_seconds():.1f}s')

        t0_bag = msgs[0][2]
        t0_wall = time.monotonic()
        count = 0

        for topic, ros_msg, log_time in msgs:
            # Rate control
            dt_bag = (log_time - t0_bag).total_seconds()
            dt_wall = time.monotonic() - t0_wall
            sleep_time = (dt_bag / self.rate) - dt_wall
            if sleep_time > 0.001:
                time.sleep(sleep_time)

            # Decompress
            np_arr = np.frombuffer(ros_msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            # Build raw Image msg (copy header fields to avoid type mismatch)
            out = Image()
            hdr = Header()
            hdr.stamp = Time(sec=ros_msg.header.stamp.sec, nanosec=ros_msg.header.stamp.nanosec)
            hdr.frame_id = ros_msg.header.frame_id
            out.header = hdr
            out.height, out.width = img.shape[:2]
            if img.ndim == 2:
                out.encoding = 'mono8'
                out.step = out.width
            else:
                out.encoding = 'bgr8'
                out.step = out.width * 3
            out.data = img.tobytes()

            # Publish to correct topic
            try:
                if topic == topic_left:
                    self.pub_left.publish(out)
                else:
                    self.pub_right.publish(out)
                count += 1
            except Exception:
                break

        elapsed = time.monotonic() - t0_wall
        self.get_logger().info(f'Done: {count} msgs in {elapsed:.1f}s ({count/elapsed:.1f} msg/s)')
        return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', required=True)
    parser.add_argument('--ns', default='/race6/cam1')
    parser.add_argument('--rate', type=float, default=1.0)
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = BagPlayer(args.bag, args.ns, args.rate)
    count = node.play()
    node.destroy_node()
    rclpy.shutdown()
    return 0 if count > 0 else 1


if __name__ == '__main__':
    exit(main())
