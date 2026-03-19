#!/usr/bin/env python3
"""Live Fast-FoundationStereo TRT depth from RealSense IR stereo.

Architecture: Independent subscribers for left/right IR, inference thread
grabs latest pair. No message_filters sync (avoids frame drops).

GPU-accelerated pipeline v2: pinned memory, batched preprocess, fast percentile.
"""
import gc
import os, sys, time, threading
import numpy as np
import cv2
import torch
import torch.nn.functional as F
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


def _build_turbo_lut_gpu():
    """Pre-build TURBO colormap as a GPU tensor (256x3, uint8)."""
    gray = np.arange(256, dtype=np.uint8)
    turbo = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO).reshape(256, 3)
    return torch.from_numpy(turbo).cuda()


class LiveFFS(Node):
    def __init__(self):
        super().__init__('live_ffs')
        self.declare_parameter('engine_dir', f'{code_dir}/../engines/320x224_4iter')
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

        # Pre-allocate ALL buffers (zero per-frame allocation)
        self.buf_left = torch.empty(1, 3, self.H, self.W, device='cuda', dtype=torch.float32)
        self.buf_right = torch.empty(1, 3, self.H, self.W, device='cuda', dtype=torch.float32)

        # Pinned memory for fast H2D — allocated once, reused every frame
        self.pin_left = torch.empty(480, 848, dtype=torch.uint8).pin_memory()
        self.pin_right = torch.empty(480, 848, dtype=torch.uint8).pin_memory()

        # GPU staging buffer for both images (avoids per-frame alloc)
        self.gpu_raw = torch.empty(2, 1, 480, 848, device='cuda', dtype=torch.float32)

        # GPU colormap LUT
        self.turbo_lut = _build_turbo_lut_gpu()

        # CUDA stream for H2D
        self.h2d_stream = torch.cuda.Stream()

        # Independent subscribers
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, f'{ns}/infra1/image_rect_raw', self.cb_left, qos)
        self.create_subscription(Image, f'{ns}/infra2/image_rect_raw', self.cb_right, qos)

        self.latest_left = None
        self.latest_right = None
        self.input_shape = None
        self.new_frame = threading.Event()

        # Publishers — RELIABLE so rqt/foxglove can subscribe without QoS mismatch
        pub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_gray = self.create_publisher(CompressedImage, '/ffs/disp_gray/compressed', pub_qos)

        self.running = True
        self.frame_count = 0
        self.t0 = time.time()

        self.infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self.infer_thread.start()

        self.get_logger().info('Ready — GPU pipeline v2')

    def cb_left(self, msg):
        if self.input_shape is None:
            self.input_shape = (msg.height, msg.width)
            # Reallocate pinned buffers to actual size if different
            if (msg.height, msg.width) != (480, 848):
                self.pin_left = torch.empty(msg.height, msg.width, dtype=torch.uint8).pin_memory()
                self.pin_right = torch.empty(msg.height, msg.width, dtype=torch.uint8).pin_memory()
                self.gpu_raw = torch.empty(2, 1, msg.height, msg.width, device='cuda', dtype=torch.float32)
        self.pin_left.numpy()[:] = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        self.latest_left = True
        self.new_frame.set()

    def cb_right(self, msg):
        if self.input_shape is None:
            self.input_shape = (msg.height, msg.width)
            if (msg.height, msg.width) != (480, 848):
                self.pin_left = torch.empty(msg.height, msg.width, dtype=torch.uint8).pin_memory()
                self.pin_right = torch.empty(msg.height, msg.width, dtype=torch.uint8).pin_memory()
                self.gpu_raw = torch.empty(2, 1, msg.height, msg.width, device='cuda', dtype=torch.float32)
        self.pin_right.numpy()[:] = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        self.latest_right = True

    def _infer_loop(self):
        """Tight inference + publish loop — temporal smoothing for stable depth."""
        smooth_min = 0.0
        smooth_max = 1.0
        alpha = 0.08
        # Temporal EMA on disparity — stabilizes depth for obstacle avoidance
        prev_disp = None
        temporal_alpha = 0.5  # 0.0=full smoothing, 1.0=no smoothing

        while self.running:
            self.new_frame.wait(timeout=0.1)
            self.new_frame.clear()

            if not self.latest_left or not self.latest_right:
                continue

            t0 = time.time()

            # === Batched H2D via pinned memory (one sync for both images) ===
            with torch.cuda.stream(self.h2d_stream):
                self.gpu_raw[0, 0].copy_(self.pin_left.cuda(non_blocking=True).float())
                self.gpu_raw[1, 0].copy_(self.pin_right.cuda(non_blocking=True).float())
            self.h2d_stream.synchronize()

            # === GPU resize both at once (batch=2) ===
            resized = F.interpolate(self.gpu_raw, size=(self.H, self.W), mode='bilinear', align_corners=False)

            # Expand grayscale to 3-ch via expand (no memory copy)
            self.buf_left[:] = resized[0:1].expand(1, 3, self.H, self.W)
            self.buf_right[:] = resized[1:2].expand(1, 3, self.H, self.W)

            # === TRT inference (pipelined) ===
            disp = self.model.forward_pipelined(self.buf_left, self.buf_right)

            if disp is None:
                continue

            # === Post-processing: Saviolo-style filtering (arXiv:2409.11962) ===
            disp_2d = disp.cuda().reshape(self.H, self.W).float()

            # 1. GPU morphological opening (replaces CPU speckle filter — same effect, no roundtrip)
            #    Erode removes isolated noise pixels, dilate restores object shapes
            disp_4d = disp_2d.unsqueeze(0).unsqueeze(0)
            disp_4d = -F.max_pool2d(-disp_4d, kernel_size=3, stride=1, padding=1)  # erode
            disp_4d = F.max_pool2d(disp_4d, kernel_size=3, stride=1, padding=1)    # dilate
            disp_2d = disp_4d.squeeze()

            # 2. Clamp invalid disparities
            disp_2d = disp_2d.clamp(min=0.5)

            # 3. GPU guided filter — edge-preserving smoothing using IR as guide
            #    (matching Saviolo's guided_filter: smooth depth noise, preserve edges)
            ir_gpu = resized[0, 0]  # left IR at network resolution
            disp_4d = disp_2d.unsqueeze(0).unsqueeze(0)
            ir_4d = ir_gpu.unsqueeze(0).unsqueeze(0)
            r = 1  # radius (small = fast, still edge-preserving)
            ks = 2 * r + 1
            eps_gf = 200.0  # guided filter regularization (higher = more smoothing)

            mean_I = F.avg_pool2d(ir_4d, ks, stride=1, padding=r)
            mean_p = F.avg_pool2d(disp_4d, ks, stride=1, padding=r)
            mean_Ip = F.avg_pool2d(ir_4d * disp_4d, ks, stride=1, padding=r)
            cov_Ip = mean_Ip - mean_I * mean_p
            mean_II = F.avg_pool2d(ir_4d * ir_4d, ks, stride=1, padding=r)
            var_I = mean_II - mean_I * mean_I
            a = cov_Ip / (var_I + eps_gf)
            b = mean_p - a * mean_I
            mean_a = F.avg_pool2d(a, ks, stride=1, padding=r)
            mean_b = F.avg_pool2d(b, ks, stride=1, padding=r)
            disp_2d = (mean_a * ir_4d + mean_b).squeeze()

            # 4. Simple temporal smoothing — always blend, never skip
            #    High alpha = responsive, low alpha = smooth
            if prev_disp is None:
                prev_disp = disp_2d.clone()
            else:
                prev_disp.mul_(0.3).add_(disp_2d, alpha=0.7)
            disp_2d = prev_disp

            # 5. Continuous grayscale: close=dark, far=white (inverse disparity mapping)
            #    Clamp disparity range and map smoothly
            stamp = self.get_clock().now().to_msg()
            d_near = 80.0   # disparity for nearest objects (~0.5m)
            d_far = 3.0     # disparity for farthest visible (~15m)
            normalized = 1.0 - ((disp_2d - d_far) / (d_near - d_far)).clamp(0, 1)
            gray = (normalized * 255.0).to(torch.uint8).cpu().numpy()

            # Single JPEG encode + publish
            _, jpeg_g = cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, 75])
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
