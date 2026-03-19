#!/usr/bin/env python3
"""Live Fast-FoundationStereo TRT depth from RealSense IR stereo.

Architecture: Independent subscribers for left/right IR, inference thread
grabs latest pair. No message_filters sync (avoids frame drops).
"""
import gc
import os, sys, time, threading
import numpy as np
import cv2
import torch
import yaml
from omegaconf import OmegaConf

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f'{code_dir}/../')
from core.pipelined_runner import PipelinedTrtRunner

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image

gc.disable()


class LiveFFS(Node):
    def __init__(self):
        super().__init__('live_ffs')
        self.declare_parameter('engine_dir', f'{code_dir}/../engines/320x224_3iter')
        self.declare_parameter('ns', '/race16/cam1')
        engine_dir = self.get_parameter('engine_dir').value
        ns = self.get_parameter('ns').value

        # Load TRT model
        with open(f'{engine_dir}/onnx.yaml') as f:
            cfg = yaml.safe_load(f)
        cfg['cv_group'] = cfg.get('cv_group', 8)
        self.cfg = OmegaConf.create(cfg)
        self.H, self.W = cfg['image_size']
        iters = cfg.get('valid_iters')
        self.get_logger().info(f'Loading TRT from {engine_dir} ({self.W}x{self.H}, {iters} iters)')
        self.model = PipelinedTrtRunner(self.cfg,
                                         f'{engine_dir}/feature_runner.engine',
                                         f'{engine_dir}/post_runner.engine')

        # Warmup
        dummy_l = torch.randn(1, 3, self.H, self.W).cuda().float() * 255
        dummy_r = torch.randn(1, 3, self.H, self.W).cuda().float() * 255
        for _ in range(5):
            self.model.forward_pipelined(dummy_l, dummy_r)
        self.model.flush()
        torch.cuda.synchronize()
        self.get_logger().info('TRT warmup done')

        # Pre-allocate CUDA buffers
        self.buf_left = torch.empty(1, 3, self.H, self.W, device='cuda', dtype=torch.float32)
        self.buf_right = torch.empty(1, 3, self.H, self.W, device='cuda', dtype=torch.float32)

        # Independent subscribers — no message_filters
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, f'{ns}/infra1/image_rect_raw', self.cb_left, qos)
        self.create_subscription(Image, f'{ns}/infra2/image_rect_raw', self.cb_right, qos)

        # Latest frames
        self.latest_left = None
        self.latest_right = None
        self.new_frame = threading.Event()

        # Publisher
        pub_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_disp = self.create_publisher(CompressedImage, '/ffs/disp_viz/compressed', pub_qos)
        self.pub_gray = self.create_publisher(CompressedImage, '/ffs/disp_gray/compressed', pub_qos)

        self.running = True
        self.frame_count = 0
        self.t0 = time.time()

        # Inference thread
        self.infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self.infer_thread.start()

        self.get_logger().info('Ready — publishing /ffs/disp_viz/compressed')

    def cb_left(self, msg):
        self.latest_left = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width).copy()
        self.new_frame.set()

    def cb_right(self, msg):
        self.latest_right = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width).copy()

    def _infer_loop(self):
        """Tight inference + publish loop."""
        staging = np.empty((1, 3, self.H, self.W), dtype=np.float32)
        # Smoothed colormap bounds
        smooth_min = None
        smooth_max = None
        alpha = 0.08  # EMA weight for colormap bounds (slow = stable colors)
        # Temporal smoothing on disparity — motion-adaptive per-pixel
        prev_disp = None

        while self.running:
            self.new_frame.wait(timeout=0.1)
            self.new_frame.clear()

            left = self.latest_left
            right = self.latest_right
            if left is None or right is None:
                continue

            t0 = time.time()

            # Resize with INTER_LINEAR (smoother input to network)
            left_r = cv2.resize(left, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            right_r = cv2.resize(right, (self.W, self.H), interpolation=cv2.INTER_LINEAR)

            # Grayscale to 3-channel CHW, copy to GPU
            staging[0, 0] = staging[0, 1] = staging[0, 2] = left_r.astype(np.float32)
            self.buf_left.copy_(torch.as_tensor(staging))
            staging[0, 0] = staging[0, 1] = staging[0, 2] = right_r.astype(np.float32)
            self.buf_right.copy_(torch.as_tensor(staging))

            # Inference
            disp = self.model.forward_pipelined(self.buf_left, self.buf_right)

            if disp is None:
                continue

            # Post-processing: no temporal filter (4-iter disparity is clean enough)
            disp_np = disp.numpy().reshape(self.H, self.W).astype(np.float32)

            # 2. Spatial denoise + guided upsampling
            # Guided upsampling: IR edge guide gives sharp object boundaries
            out_h, out_w = left.shape[0] // 2, left.shape[1] // 2
            guide = cv2.resize(left, (out_w, out_h)).astype(np.float32)
            disp_up = cv2.resize(disp_np, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            disp_up = cv2.ximgproc.guidedFilter(guide, disp_up, radius=4, eps=100)

            # 3. Percentile-based colormap bounds
            dmin_cur = float(np.percentile(disp_up, 2))
            dmax_cur = float(np.percentile(disp_up, 98))

            if smooth_min is None:
                smooth_min, smooth_max = dmin_cur, dmax_cur
            else:
                smooth_min = smooth_min + alpha * (dmin_cur - smooth_min)
                smooth_max = smooth_max + alpha * (dmax_cur - smooth_max)

            rng = smooth_max - smooth_min
            if rng > 0.1:
                vis = cv2.applyColorMap(
                    ((disp_up - smooth_min) * (255.0 / rng)).clip(0, 255).astype(np.uint8),
                    cv2.COLORMAP_TURBO)
            else:
                vis = np.zeros((out_h, out_w, 3), dtype=np.uint8)

            stamp = self.get_clock().now().to_msg()

            _, jpeg = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
            cmsg = CompressedImage()
            cmsg.header.stamp = stamp
            cmsg.format = 'jpeg'
            cmsg.data = jpeg.tobytes()
            self.pub_disp.publish(cmsg)

            # Grayscale depth (white=close, black=far)
            if rng > 0.1:
                gray = ((disp_up - smooth_min) * (255.0 / rng)).clip(0, 255).astype(np.uint8)
            else:
                gray = np.zeros((out_h, out_w), dtype=np.uint8)
            _, jpeg_g = cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, 80])
            gmsg = CompressedImage()
            gmsg.header.stamp = stamp
            gmsg.format = 'jpeg'
            gmsg.data = jpeg_g.tobytes()
            self.pub_gray.publish(gmsg)

            dt = (time.time() - t0) * 1000
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                fps = self.frame_count / (time.time() - self.t0)
                self.get_logger().info(f'Frame {self.frame_count}: {dt:.1f}ms iter, {fps:.1f}Hz avg')


def main():
    rclpy.init()
    node = LiveFFS()
    try:
        rclpy.spin(node)
    finally:
        node.running = False

if __name__ == '__main__':
    main()
