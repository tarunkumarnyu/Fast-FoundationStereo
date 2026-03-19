#!/usr/bin/env python3
"""Live FFS + DA2 fused depth — FFS primary, DA2 fills holes only.

Architecture:
  - FFS stereo (TRT) = primary depth source, 80% weight
  - DA2 monocular (TRT) = hole filler for textureless regions only
  - One-time polynomial calibration over first N frames, then frozen
  - Both run at FFS resolution (320x224), DA2 resized to match

Based on arXiv:2409.11962 (Saviolo et al.) scale alignment.
"""
import gc
import os, sys, time, threading
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
import tensorrt as trt

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f'{code_dir}/../')
from core.pipelined_runner import PipelinedTrtRunner

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image

gc.disable()

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)


class DA2TRT:
    """DA2-Small via native TensorRT engine."""

    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = engine.create_execution_context()

        # Get IO info
        self.in_name = engine.get_tensor_name(0)
        self.out_name = engine.get_tensor_name(1)
        in_shape = engine.get_tensor_shape(self.in_name)
        out_shape = engine.get_tensor_shape(self.out_name)

        self.in_buf = torch.empty(list(in_shape), dtype=torch.float16, device='cuda')
        self.out_buf = torch.empty(list(out_shape), dtype=torch.float16, device='cuda')

        self.ctx.set_tensor_address(self.in_name, self.in_buf.data_ptr())
        self.ctx.set_tensor_address(self.out_name, self.out_buf.data_ptr())

        self.stream = torch.cuda.Stream()
        self.H = int(in_shape[2])
        self.W = int(in_shape[3])

        # Warmup
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        print(f'[DA2-TRT] Loaded: input {list(in_shape)}, output {list(out_shape)}')

    def predict(self, gray_gpu):
        """Run DA2 on a GPU grayscale tensor (H, W) float32. Returns (H, W) float32."""
        # Resize to DA2 input resolution
        inp = gray_gpu.unsqueeze(0).unsqueeze(0)
        inp = F.interpolate(inp, size=(self.H, self.W), mode='bilinear', align_corners=False)
        # Grayscale to 3ch, normalize ImageNet
        inp = inp.expand(1, 3, self.H, self.W) / 255.0
        inp = (inp - IMAGENET_MEAN) / IMAGENET_STD

        self.in_buf.copy_(inp.half())
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        return self.out_buf.float().squeeze()  # (H, W)


class LiveFFSFused(Node):
    def __init__(self):
        super().__init__('live_ffs_fused')
        self.declare_parameter('engine_dir', f'{code_dir}/../engines/320x224_4iter')
        self.declare_parameter('da2_engine', f'{code_dir}/../engines/da2_224x308/da2_small.engine')
        self.declare_parameter('ns', '/race16/cam1')
        self.declare_parameter('max_depth', 10.0)
        self.declare_parameter('calib_frames', 30)
        engine_dir = self.get_parameter('engine_dir').value
        da2_path = self.get_parameter('da2_engine').value
        ns = self.get_parameter('ns').value
        self.max_depth = self.get_parameter('max_depth').value
        self.calib_frames = self.get_parameter('calib_frames').value

        # Load FFS
        with open(f'{engine_dir}/onnx.yaml') as f:
            cfg = yaml.safe_load(f)
        cfg['cv_group'] = cfg.get('cv_group', 8)
        self.cfg = OmegaConf.create(cfg)
        self.H, self.W = cfg['image_size']
        self.get_logger().info(f'FFS: {self.W}x{self.H}')
        self.ffs = PipelinedTrtRunner(self.cfg,
                                       f'{engine_dir}/feature_runner.engine',
                                       f'{engine_dir}/post_runner.engine')
        d = torch.randn(1, 3, self.H, self.W).cuda().float() * 255
        for _ in range(5):
            self.ffs.forward_pipelined(d, d)
        self.ffs.flush()
        torch.cuda.synchronize()

        # Load DA2 TRT
        self.da2 = DA2TRT(da2_path)
        self.get_logger().info('FFS + DA2-TRT loaded')

        # Buffers
        self.buf_left = torch.empty(1, 3, self.H, self.W, device='cuda', dtype=torch.float32)
        self.buf_right = torch.empty(1, 3, self.H, self.W, device='cuda', dtype=torch.float32)
        self.pin_left = torch.empty(480, 848, dtype=torch.uint8).pin_memory()
        self.pin_right = torch.empty(480, 848, dtype=torch.uint8).pin_memory()
        self.gpu_raw = torch.empty(2, 1, 480, 848, device='cuda', dtype=torch.float32)
        self.h2d_stream = torch.cuda.Stream()

        # Calibration state
        self.frozen_coeffs = None  # [c, b, a] — frozen after calib_frames
        self.calib_count = 0
        self.calib_X_list = []
        self.calib_y_list = []

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

        self.running = True
        self.frame_count = 0
        self.t0 = time.time()
        self.prev_disp = None

        self.infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self.infer_thread.start()
        self.get_logger().info(f'Ready — calibrating over {self.calib_frames} frames...')

    def cb_left(self, msg):
        if self.input_shape is None:
            self.input_shape = (msg.height, msg.width)
            self.get_logger().info(f'First IR frame: {msg.width}x{msg.height} enc={msg.encoding}')
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

    def _calibrate_frame(self, ffs_disp, da2_raw):
        """Accumulate calibration data from one frame.

        DA2 outputs relative inverse depth directly (high=close, like disparity).
        We fit: ffs_disp ≈ a * da2² + b * da2 + c  (direct, no inversion needed)
        """
        ffs_flat = ffs_disp.reshape(-1)
        da2_flat = da2_raw.reshape(-1)

        # Valid: FFS has good stereo AND DA2 is positive (close objects)
        valid = (ffs_flat > 2.0) & (da2_flat > 1.0)
        if valid.sum() < 100:
            return

        # Subsample for speed (2000 points per frame)
        idx = torch.where(valid)[0]
        if len(idx) > 2000:
            step = len(idx) // 2000
            idx = idx[::step]

        da2_v = da2_flat[idx]
        ffs_v = ffs_flat[idx]

        ones = torch.ones_like(da2_v)
        X = torch.stack([ones, da2_v, da2_v ** 2], dim=1)
        self.calib_X_list.append(X.cpu())
        self.calib_y_list.append(ffs_v.cpu())
        self.calib_count += 1

    def _freeze_calibration(self):
        """Fit polynomial from accumulated calibration data and freeze."""
        X = torch.cat(self.calib_X_list, dim=0)
        y = torch.cat(self.calib_y_list, dim=0)

        try:
            coeffs = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze()
            self.frozen_coeffs = coeffs.cuda()
            self.get_logger().info(
                f'Calibration frozen: c={coeffs[0]:.4f} b={coeffs[1]:.4f} a={coeffs[2]:.6f} '
                f'from {len(y)} points over {self.calib_count} frames')
        except Exception as e:
            self.get_logger().error(f'Calibration failed: {e}, using identity')
            self.frozen_coeffs = torch.tensor([0.0, 1.0, 0.0], device='cuda')

        # Free calibration data
        self.calib_X_list = []
        self.calib_y_list = []

    def _infer_loop(self):
        try:
            self._infer_loop_inner()
        except Exception as e:
            import traceback
            self.get_logger().error(f'INFER THREAD CRASH: {e}')
            traceback.print_exc()

    def _infer_loop_inner(self):
        while self.running:
            self.new_frame.wait(timeout=0.1)
            self.new_frame.clear()

            if not self.latest_left or not self.latest_right:
                continue

            t0 = time.time()

            # === H2D + resize ===
            with torch.cuda.stream(self.h2d_stream):
                self.gpu_raw[0, 0].copy_(self.pin_left.cuda(non_blocking=True).float())
                self.gpu_raw[1, 0].copy_(self.pin_right.cuda(non_blocking=True).float())
            self.h2d_stream.synchronize()

            resized = F.interpolate(self.gpu_raw, size=(self.H, self.W), mode='bilinear', align_corners=False)
            self.buf_left[:] = resized[0:1].expand(1, 3, self.H, self.W)
            self.buf_right[:] = resized[1:2].expand(1, 3, self.H, self.W)

            # === FFS stereo ===
            ffs_disp = self.ffs.forward_pipelined(self.buf_left, self.buf_right)

            # === DA2 monocular (runs on left IR) ===
            ir_gpu = resized[0, 0]  # (H, W) float32
            da2_rel = self.da2.predict(ir_gpu)
            # Resize DA2 output to FFS resolution
            da2_rel = F.interpolate(da2_rel.unsqueeze(0).unsqueeze(0),
                                     size=(self.H, self.W), mode='bilinear',
                                     align_corners=False).squeeze()

            if ffs_disp is None:
                continue

            ffs_disp = ffs_disp.cuda().reshape(self.H, self.W).float()

            # === Calibration phase ===
            if self.frozen_coeffs is None:
                self._calibrate_frame(ffs_disp, da2_rel)
                if self.calib_count >= self.calib_frames:
                    self._freeze_calibration()
                # During calibration, just use FFS directly
                disp_2d = ffs_disp.clamp(min=0.5)
            else:
                # === Fused mode: FFS primary, DA2 fills holes ===
                # DA2 output is already disparity-like (high=close) — apply polynomial directly
                da2_clamped = da2_rel.clamp(min=0.0)
                c, b, a = self.frozen_coeffs[0], self.frozen_coeffs[1], self.frozen_coeffs[2]
                da2_aligned = a * da2_clamped ** 2 + b * da2_clamped + c
                da2_aligned = da2_aligned.clamp(min=0.5)

                # FFS is primary — DA2 only fills where FFS has holes (disp < 2)
                ffs_valid = ffs_disp > 2.0
                disp_2d = torch.where(ffs_valid, ffs_disp, da2_aligned)

            # === Guided filter (r=1, IR-guided) ===
            disp_4d = disp_2d.unsqueeze(0).unsqueeze(0)
            ir_4d = ir_gpu.unsqueeze(0).unsqueeze(0)
            r, ks = 1, 3
            eps_gf = 200.0
            mean_I = F.avg_pool2d(ir_4d, ks, stride=1, padding=r)
            mean_p = F.avg_pool2d(disp_4d, ks, stride=1, padding=r)
            mean_Ip = F.avg_pool2d(ir_4d * disp_4d, ks, stride=1, padding=r)
            cov_Ip = mean_Ip - mean_I * mean_p
            mean_II = F.avg_pool2d(ir_4d * ir_4d, ks, stride=1, padding=r)
            var_I = mean_II - mean_I * mean_I
            ag = cov_Ip / (var_I + eps_gf)
            bg = mean_p - ag * mean_I
            mean_a = F.avg_pool2d(ag, ks, stride=1, padding=r)
            mean_b = F.avg_pool2d(bg, ks, stride=1, padding=r)
            disp_2d = (mean_a * ir_4d + mean_b).squeeze()

            # === Temporal smoothing (motion-adaptive) ===
            if self.prev_disp is None:
                self.prev_disp = disp_2d.clone()
            else:
                diff = (disp_2d - self.prev_disp).abs()
                alpha = torch.where(diff > 5.0, torch.ones_like(diff), torch.full_like(diff, 0.4))
                self.prev_disp.mul_(1.0 - alpha).add_(disp_2d * alpha)
            disp_2d = self.prev_disp

            # === Grayscale viz: close=dark, far=white ===
            stamp = self.get_clock().now().to_msg()
            d_near, d_far = 80.0, 3.0
            normalized = 1.0 - ((disp_2d - d_far) / (d_near - d_far)).clamp(0, 1)
            gray = (normalized * 255.0).to(torch.uint8).cpu().numpy()

            _, jpeg = cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, 75])
            gmsg = CompressedImage()
            gmsg.header.stamp = stamp
            gmsg.format = 'jpeg'
            gmsg.data = jpeg.tobytes()
            self.pub_gray.publish(gmsg)

            dt = (time.time() - t0) * 1000
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                fps = self.frame_count / (time.time() - self.t0)
                mode = 'CALIB' if self.frozen_coeffs is None else 'FUSED'
                self.get_logger().info(f'[{mode}] Frame {self.frame_count}: {dt:.1f}ms, {fps:.1f}Hz')


def main():
    rclpy.init()
    node = LiveFFSFused()
    try:
        rclpy.spin(node)
    finally:
        node.running = False

if __name__ == '__main__':
    main()
