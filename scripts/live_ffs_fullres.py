#!/usr/bin/env python3
"""Live FFS depth at full camera resolution using PyTorch (no TRT).

Captures IR stereo from RealSense, runs FFS at native 848x480, publishes
depth visualization as CompressedImage. Slower than TRT (~0.25Hz) but
maximum quality at full resolution.
"""
import gc, os, sys, time, threading
import numpy as np
import cv2
import torch

code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{code_dir}/../')

from core.utils.utils import InputPadder
from Utils import AMP_DTYPE

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image

gc.disable()

# Pre-build TURBO LUT on GPU
_TURBO_LUT_GPU = None
def _get_turbo_lut():
    global _TURBO_LUT_GPU
    if _TURBO_LUT_GPU is None:
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        lut[:, 0, :] = cv2.applyColorMap(
            np.arange(256, dtype=np.uint8), cv2.COLORMAP_TURBO
        ).reshape(256, 3)
        _TURBO_LUT_GPU = torch.from_numpy(lut).cuda()
    return _TURBO_LUT_GPU


class LiveFFSFullRes(Node):
    def __init__(self):
        super().__init__('live_ffs_fullres')
        self.declare_parameter('model_dir',
            f'{code_dir}/../weights/23-36-37/model_best_bp2_serialize.pth')
        self.declare_parameter('ns', '/race6/cam1')
        self.declare_parameter('valid_iters', 8)
        self.declare_parameter('max_disp', 192)

        model_dir = self.get_parameter('model_dir').value
        ns = self.get_parameter('ns').value
        valid_iters = self.get_parameter('valid_iters').value
        max_disp = self.get_parameter('max_disp').value

        self.get_logger().info(
            f'Loading PyTorch model from {model_dir} '
            f'(iters={valid_iters}, max_disp={max_disp})')

        torch.autograd.set_grad_enabled(False)
        self.model = torch.load(model_dir, map_location='cpu', weights_only=False)
        self.model.args.valid_iters = valid_iters
        self.model.args.max_disp = max_disp
        self.model.cuda().eval()

        # Warmup
        dummy_l = torch.randn(1, 3, 480, 864).cuda().float()
        dummy_r = torch.randn(1, 3, 480, 864).cuda().float()
        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            _ = self.model.forward(dummy_l, dummy_r,
                                    iters=valid_iters, test_mode=True,
                                    optimize_build_volume='pytorch1')
        torch.cuda.synchronize()
        self.get_logger().info('Warmup done')

        # State
        self.left_img = None
        self.right_img = None
        self.lock = threading.Lock()
        self.frame_count = 0
        self.start_time = time.monotonic()
        self.valid_iters = valid_iters

        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        # Subscribers
        self.create_subscription(
            Image, f'{ns}/infra1/image_rect_raw', self.cb_left, qos)
        self.create_subscription(
            Image, f'{ns}/infra2/image_rect_raw', self.cb_right, qos)

        # Publishers
        self.pub_viz = self.create_publisher(
            CompressedImage, '/ffs/disp_viz/compressed', 1)
        self.pub_gray = self.create_publisher(
            CompressedImage, '/ffs/disp_gray/compressed', 1)

        # Inference thread
        self.running = True
        self.infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self.infer_thread.start()

        self.get_logger().info('Ready — full-res PyTorch pipeline')

    def cb_left(self, msg):
        h, w = msg.height, msg.width
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
        with self.lock:
            self.left_img = img

    def cb_right(self, msg):
        h, w = msg.height, msg.width
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
        with self.lock:
            self.right_img = img

    def _infer_loop(self):
        while self.running:
            with self.lock:
                left = self.left_img
                right = self.right_img
            if left is None or right is None:
                time.sleep(0.05)
                continue

            t0 = time.monotonic()

            # Convert mono to 3-channel
            H, W = left.shape[:2]
            left_rgb = np.tile(left[..., None], (1, 1, 3))
            right_rgb = np.tile(right[..., None], (1, 1, 3))

            img0 = torch.as_tensor(left_rgb).cuda().float()[None].permute(0, 3, 1, 2)
            img1 = torch.as_tensor(right_rgb).cuda().float()[None].permute(0, 3, 1, 2)

            padder = InputPadder(img0.shape, divis_by=32, force_square=False)
            img0, img1 = padder.pad(img0, img1)

            with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
                disp = self.model.forward(
                    img0, img1, iters=self.valid_iters,
                    test_mode=True, optimize_build_volume='pytorch1')

            torch.cuda.synchronize()
            disp = padder.unpad(disp.float())
            disp_np = disp.squeeze().cpu().numpy().clip(0, None)

            # Publish turbo colormap visualization
            p2, p98 = np.percentile(disp_np[disp_np > 0.5], [2, 98])
            norm = np.clip((disp_np - p2) / max(p98 - p2, 1e-3), 0, 1)
            norm_u8 = (norm * 255).astype(np.uint8)
            vis = cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)

            msg_viz = CompressedImage()
            msg_viz.header.stamp = self.get_clock().now().to_msg()
            msg_viz.format = 'jpeg'
            msg_viz.data = bytes(cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 85])[1])
            self.pub_viz.publish(msg_viz)

            # Publish grayscale
            gray = (norm * 255).astype(np.uint8)
            msg_gray = CompressedImage()
            msg_gray.header.stamp = msg_viz.header.stamp
            msg_gray.format = 'jpeg'
            msg_gray.data = bytes(cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, 85])[1])
            self.pub_gray.publish(msg_gray)

            self.frame_count += 1
            elapsed = time.monotonic() - t0
            avg_hz = self.frame_count / (time.monotonic() - self.start_time)

            if self.frame_count % 1 == 0:
                self.get_logger().info(
                    f'Frame {self.frame_count}: {elapsed*1000:.0f}ms '
                    f'({1/elapsed:.2f}Hz inst, {avg_hz:.2f}Hz avg) '
                    f'disp=[{disp_np.min():.1f}, {disp_np.max():.1f}]')


def main():
    rclpy.init()
    node = LiveFFSFullRes()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.running = False
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
