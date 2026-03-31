#!/usr/bin/env python3
"""Live FFS with unified TRT engines (GWC baked into post engine).

Just two TRT calls: feature_runner → post_gwc_runner → disparity.
No custom CUDA kernels, no Triton.
Publishes disparity (grayscale) + metric depth (raw float32, lazy).
"""
import gc, os, sys, time, threading
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import yaml
import tensorrt as trt
from omegaconf import OmegaConf

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f'{code_dir}/../')

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image

gc.disable()


class TrtEngine:
    """Simple TRT engine wrapper."""
    def __init__(self, path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, 'rb') as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        self.inputs = {}
        self.outputs = {}
        self.out_bufs = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = list(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = {trt.DataType.FLOAT: torch.float32,
                           trt.DataType.HALF: torch.float16,
                           trt.DataType.INT32: torch.int32}[dtype]
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.inputs[name] = (shape, torch_dtype)
                self.ctx.set_input_shape(name, shape)
            else:
                buf = torch.empty(shape, dtype=torch_dtype, device='cuda')
                self.out_bufs[name] = buf
                self.outputs[name] = (shape, torch_dtype)
                self.ctx.set_tensor_address(name, buf.data_ptr())

        print(f'[TRT] {os.path.basename(path)}: {len(self.inputs)} in, {len(self.outputs)} out')

    def infer(self, input_dict):
        for name, tensor in input_dict.items():
            expected_dtype = self.inputs[name][1]
            if tensor.dtype != expected_dtype:
                tensor = tensor.to(expected_dtype)
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            self.ctx.set_tensor_address(name, tensor.data_ptr())
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return {name: buf.clone() for name, buf in self.out_bufs.items()}


class LiveFFSUnified(Node):
    def __init__(self):
        super().__init__('live_ffs_unified')
        self.declare_parameter('engine_dir', '')
        self.declare_parameter('ns', '/race6/cam1')
        # D455 intrinsics at native 848x480 (from Kalibr calibration)
        self.declare_parameter('fx', 430.83)
        self.declare_parameter('fy', 430.91)
        self.declare_parameter('cx', 427.97)
        self.declare_parameter('cy', 246.90)
        self.declare_parameter('baseline', 0.09508)
        self.declare_parameter('zfar', 20.0)
        self.declare_parameter('native_w', 848)

        engine_dir = self.get_parameter('engine_dir').value
        ns = self.get_parameter('ns').value

        with open(f'{engine_dir}/onnx.yaml') as f:
            cfg = yaml.safe_load(f)
        self.H, self.W = cfg['image_size']
        self.get_logger().info(f'Config: {self.W}x{self.H}, {cfg.get("valid_iters")} iters, unified={cfg.get("unified_post", False)}')

        # Depth conversion: scale fx to engine resolution
        native_w = self.get_parameter('native_w').value
        self.baseline = self.get_parameter('baseline').value
        self.zfar = self.get_parameter('zfar').value
        self.fx_scaled = self.get_parameter('fx').value * (self.W / native_w)
        # fb_product is constant: depth = fb / disp
        self.fb = self.fx_scaled * self.baseline
        self.max_disp = cfg.get('max_disp', 128)
        self.zmin = self.fb / self.max_disp  # min depth at max disparity
        self.get_logger().info(
            f'Depth: fx_scaled={self.fx_scaled:.1f}, baseline={self.baseline:.4f}m, '
            f'zfar={self.zfar}m, zmin={self.zmin:.2f}m, fb={self.fb:.2f}')

        # Load engines
        self.feat = TrtEngine(f'{engine_dir}/feature_runner.engine')
        post_name = 'post_gwc_runner.engine' if cfg.get('unified_post') else 'post_runner.engine'
        self.post = TrtEngine(f'{engine_dir}/{post_name}')

        # Warmup
        dummy = torch.randn(1, 3, self.H, self.W, device='cuda')
        for _ in range(3):
            feat_out = self.feat.infer({'left': dummy, 'right': dummy})
            post_in = {k: v for k, v in feat_out.items() if k in self.post.inputs}
            self.post.infer(post_in)
        torch.cuda.synchronize()
        self.get_logger().info('Warmup done')

        # Buffers
        self.pin_left = torch.empty(480, 848, dtype=torch.uint8).pin_memory()
        self.pin_right = torch.empty(480, 848, dtype=torch.uint8).pin_memory()
        self.gpu_raw = torch.empty(2, 1, 480, 848, device='cuda', dtype=torch.float32)
        self.h2d_stream = torch.cuda.Stream()

        # Subscribers
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, f'{ns}/infra1/image_rect_raw', self.cb_left, qos)
        self.create_subscription(Image, f'{ns}/infra2/image_rect_raw', self.cb_right, qos)

        self.latest_left = None
        self.latest_right = None
        self.input_shape = None
        self.new_frame = threading.Event()

        # Publisher
        pub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_gray = self.create_publisher(CompressedImage, '/ffs/disp_gray/compressed', pub_qos)
        raw_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_depth_raw = self.create_publisher(Image, '/ffs/depth_raw', raw_qos)

        self.running = True
        self.frame_count = 0
        self.t0 = time.time()
        self.prev_disp = None
        self.smooth_min = 0.0
        self.smooth_max = 1.0

        self.infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self.infer_thread.start()
        self.get_logger().info('Ready — unified pipeline')

    def cb_left(self, msg):
        if self.input_shape is None:
            self.input_shape = (msg.height, msg.width)
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
        try:
            self._infer_loop_inner()
        except Exception as e:
            import traceback
            self.get_logger().error(f'CRASH: {e}')
            traceback.print_exc()

    def _infer_loop_inner(self):
        alpha = 0.08
        while self.running:
            self.new_frame.wait(timeout=0.1)
            self.new_frame.clear()
            if not self.latest_left or not self.latest_right:
                continue

            t0 = time.time()

            # H2D + resize
            with torch.cuda.stream(self.h2d_stream):
                self.gpu_raw[0, 0].copy_(self.pin_left.cuda(non_blocking=True).float())
                self.gpu_raw[1, 0].copy_(self.pin_right.cuda(non_blocking=True).float())
            self.h2d_stream.synchronize()
            resized = F.interpolate(self.gpu_raw, size=(self.H, self.W), mode='bilinear', align_corners=False)
            left_3ch = resized[0:1].expand(1, 3, self.H, self.W).contiguous()
            right_3ch = resized[1:2].expand(1, 3, self.H, self.W).contiguous()

            # Feature engine
            feat_out = self.feat.infer({'left': left_3ch, 'right': right_3ch})

            # Post+GWC engine (unified — just pass feature outputs)
            post_in = {}
            for name in self.post.inputs:
                if name in feat_out:
                    post_in[name] = feat_out[name]
            post_out = self.post.infer(post_in)

            # Get disparity
            disp = post_out['disp'].cuda().reshape(self.H, self.W).float()

            # Guard against NaN from fp16 edge cases
            if torch.isnan(disp).any():
                if self.prev_disp is not None:
                    disp = self.prev_disp  # reuse last good frame
                else:
                    continue  # no valid frame yet, skip

            disp = disp.clamp(min=0.5)

            # Box blur
            disp = F.avg_pool2d(disp.unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2).squeeze()

            # Temporal EMA
            if self.prev_disp is None:
                self.prev_disp = disp.clone()
            else:
                self.prev_disp.mul_(0.3).add_(disp, alpha=0.7)
            disp = self.prev_disp

            stamp = self.get_clock().now().to_msg()

            # Disparity grayscale — adaptive normalization for visibility
            d_min = disp.min()
            d_max = disp.max()
            span = max(d_max - d_min, 1.0)
            gray = ((disp - d_min) / span * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()

            _, jpeg = cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, 75])
            msg = CompressedImage()
            msg.header.stamp = stamp
            msg.format = 'jpeg'
            msg.data = jpeg.tobytes()
            self.pub_gray.publish(msg)

            # Raw float32 depth in meters — only if anyone is listening
            if self.pub_depth_raw.get_subscription_count() > 0:
                depth_np = (self.fb / disp).clamp(self.zmin, self.zfar).cpu().numpy()
                depth_msg = Image()
                depth_msg.header.stamp = stamp
                depth_msg.header.frame_id = 'cam1_infra1_optical_frame'
                depth_msg.height = self.H
                depth_msg.width = self.W
                depth_msg.encoding = '32FC1'
                depth_msg.is_bigendian = False
                depth_msg.step = self.W * 4
                depth_msg.data = depth_np.astype(np.float32).tobytes()
                self.pub_depth_raw.publish(depth_msg)

            dt = (time.time() - t0) * 1000
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                fps = self.frame_count / (time.time() - self.t0)
                self.get_logger().info(
                    f'Frame {self.frame_count}: {dt:.1f}ms, {fps:.1f}Hz, '
                    f'disp=[{disp.min():.1f}, {disp.max():.1f}, mean={disp.mean():.1f}], '
                    f'gray=[{gray.min()}, {gray.max()}]')


def main():
    rclpy.init()
    node = LiveFFSUnified()
    try:
        rclpy.spin(node)
    finally:
        node.running = False

if __name__ == '__main__':
    main()
