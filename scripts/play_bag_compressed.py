#!/usr/bin/env python3
"""Read mcap bag and republish compressed infra images as-is (no decompression).

Bypasses ros2 bag play (yaml-cpp bug). Just reads compressed messages and publishes them.

Usage:
  python3 scripts/play_bag_compressed.py --bag /path/to/bag.mcap [--rate 1.0] [--ns /race6/cam1]
"""
import argparse
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Header
from sensor_msgs.msg import CompressedImage, CameraInfo
from builtin_interfaces.msg import Time
from mcap_ros2.reader import read_ros2_messages


class BagPlayerCompressed(Node):
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
        self.pub_left = self.create_publisher(CompressedImage, f'{ns}/infra1/image_rect_raw/compressed', qos)
        self.pub_right = self.create_publisher(CompressedImage, f'{ns}/infra2/image_rect_raw/compressed', qos)
        self.pub_rs_depth = self.create_publisher(CompressedImage, f'{ns}/depth/image_rect_raw/compressedDepth', qos)
        self.get_logger().info(f'Playing {bag_path} at {rate}x (compressed passthrough + RS depth)')

    def play(self):
        topic_left = f'{self.ns}/infra1/image_rect_raw/compressed'
        topic_right = f'{self.ns}/infra2/image_rect_raw/compressed'
        topic_rs_depth = f'{self.ns}/depth/image_rect_raw/compressedDepth'

        msgs = []
        for msg in read_ros2_messages(self.bag_path, topics=[topic_left, topic_right, topic_rs_depth]):
            msgs.append((msg.channel.topic, msg.ros_msg, msg.log_time))

        if not msgs:
            self.get_logger().error('No messages found!')
            return 0

        msgs.sort(key=lambda x: x[2])
        self.get_logger().info(f'Loaded {len(msgs)} messages, duration: {(msgs[-1][2] - msgs[0][2]).total_seconds():.1f}s')

        t0_bag = msgs[0][2]
        t0_wall = time.monotonic()
        count = 0

        for topic, ros_msg, log_time in msgs:
            dt_bag = (log_time - t0_bag).total_seconds()
            dt_wall = time.monotonic() - t0_wall
            sleep_time = (dt_bag / self.rate) - dt_wall
            if sleep_time > 0.001:
                time.sleep(sleep_time)

            # Rebuild message with correct header types
            out = CompressedImage()
            hdr = Header()
            hdr.stamp = Time(sec=ros_msg.header.stamp.sec, nanosec=ros_msg.header.stamp.nanosec)
            hdr.frame_id = ros_msg.header.frame_id
            out.header = hdr
            out.format = ros_msg.format
            out.data = ros_msg.data

            try:
                if topic == topic_left:
                    self.pub_left.publish(out)
                elif topic == topic_right:
                    self.pub_right.publish(out)
                elif topic == topic_rs_depth:
                    self.pub_rs_depth.publish(out)
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
    node = BagPlayerCompressed(args.bag, args.ns, args.rate)
    count = node.play()
    node.destroy_node()
    rclpy.shutdown()
    return 0 if count > 0 else 1


if __name__ == '__main__':
    exit(main())
