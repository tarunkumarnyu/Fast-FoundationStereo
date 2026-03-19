#!/usr/bin/env python3
"""benchmark_realsense.py — Benchmark Fast-FoundationStereo on RealSense D455 IR stereo.

Captures IR left/right pairs from the RealSense and runs FFS at various
scale/iter configurations to find the 30Hz sweet spot on Jetson Orin NX.

Usage:
  # With live RealSense:
  python3 scripts/benchmark_realsense.py --live

  # With saved images:
  python3 scripts/benchmark_realsense.py --left_file left.png --right_file right.png

  # Benchmark sweep:
  python3 scripts/benchmark_realsense.py --sweep
"""

import os, sys, time, argparse, logging
import numpy as np
import cv2
import torch
import yaml
from omegaconf import OmegaConf

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, set_logging_format, set_seed


def load_model(model_dir, valid_iters=4, max_disp=192):
    """Load FFS model from checkpoint."""
    with open(f'{os.path.dirname(model_dir)}/cfg.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['valid_iters'] = valid_iters
    cfg['max_disp'] = max_disp
    args = OmegaConf.create(cfg)

    model = torch.load(model_dir, map_location='cpu', weights_only=False)
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    model.cuda().eval()
    return model, args


def capture_stereo_pair_ros2():
    """Capture IR stereo pair from RealSense via ROS2."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image

    rclpy.init()
    node = rclpy.create_node('ffs_capture')
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST, depth=1)

    frames = {'left': None, 'right': None}

    def cb_left(msg):
        frames['left'] = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width).copy()

    def cb_right(msg):
        frames['right'] = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width).copy()

    # Try different namespace patterns
    for ns in ['/quadrotor/cam1', '/race16/cam1']:
        node.create_subscription(
            Image, f'{ns}/infra1/image_rect_raw', cb_left, qos)
        node.create_subscription(
            Image, f'{ns}/infra2/image_rect_raw', cb_right, qos)

    t0 = time.time()
    while (frames['left'] is None or frames['right'] is None) and time.time() - t0 < 10:
        rclpy.spin_once(node, timeout_sec=0.5)

    node.destroy_node()
    rclpy.shutdown()

    if frames['left'] is None or frames['right'] is None:
        return None, None
    return frames['left'], frames['right']


def capture_stereo_pair_pyrealsense():
    """Capture IR stereo pair directly via pyrealsense2."""
    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
    profile = pipeline.start(config)

    # Get intrinsics
    ir1_profile = profile.get_stream(rs.stream.infrared, 1)
    intrinsics = ir1_profile.as_video_stream_profile().get_intrinsics()
    baseline = profile.get_device().first_depth_sensor().get_depth_scale()
    # Extrinsics between IR1 and IR2
    ir2_profile = profile.get_stream(rs.stream.infrared, 2)
    extrinsics = ir1_profile.get_extrinsics_to(ir2_profile)
    stereo_baseline = abs(extrinsics.translation[0])  # meters

    print(f"IR intrinsics: fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} "
          f"cx={intrinsics.ppx:.1f} cy={intrinsics.ppy:.1f}")
    print(f"Stereo baseline: {stereo_baseline:.4f}m")

    # Write intrinsics file
    K = [intrinsics.fx, 0, intrinsics.ppx, 0, intrinsics.fy, intrinsics.ppy, 0, 0, 1]
    k_str = ' '.join([f'{v:.6f}' for v in K])
    k_file = f'{code_dir}/../realsense_K.txt'
    with open(k_file, 'w') as f:
        f.write(f'{k_str}\n')
        f.write(f'{stereo_baseline}\n')
    print(f"Saved intrinsics to {k_file}")

    # Wait for auto-exposure to stabilize
    for _ in range(30):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    left = np.asanyarray(frames.get_infrared_frame(1).get_data())
    right = np.asanyarray(frames.get_infrared_frame(2).get_data())

    pipeline.stop()
    return left, right


def run_inference(model, img_left, img_right, scale=1.0, valid_iters=4):
    """Run FFS inference and return disparity + timing."""
    # Handle grayscale
    if len(img_left.shape) == 2:
        img_left = np.tile(img_left[..., None], (1, 1, 3))
        img_right = np.tile(img_right[..., None], (1, 1, 3))

    img_left = img_left[..., :3]
    img_right = img_right[..., :3]

    if scale != 1.0:
        img_left = cv2.resize(img_left, fx=scale, fy=scale, dsize=None)
        img_right = cv2.resize(img_right, dsize=(img_left.shape[1], img_left.shape[0]))

    H, W = img_left.shape[:2]

    t_left = torch.as_tensor(img_left).cuda().float()[None].permute(0, 3, 1, 2)
    t_right = torch.as_tensor(img_right).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(t_left.shape, divis_by=32, force_square=False)
    t_left, t_right = padder.pad(t_left, t_right)

    model.args.valid_iters = valid_iters

    # Warm-up (first run includes compilation)
    with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
        _ = model.forward(t_left, t_right, iters=valid_iters, test_mode=True,
                          optimize_build_volume='pytorch1')
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            disp = model.forward(t_left, t_right, iters=valid_iters, test_mode=True,
                                 optimize_build_volume='pytorch1')
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)

    return disp, times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str,
                        default=f'{code_dir}/../weights/20-30-48/model_best_bp2_serialize.pth')
    parser.add_argument('--left_file', type=str, default=None)
    parser.add_argument('--right_file', type=str, default=None)
    parser.add_argument('--live', action='store_true', help='Capture from RealSense')
    parser.add_argument('--pyrs', action='store_true', help='Use pyrealsense2 directly')
    parser.add_argument('--sweep', action='store_true', help='Sweep scale/iters for 30Hz')
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--valid_iters', type=int, default=4)
    parser.add_argument('--max_disp', type=int, default=192)
    parser.add_argument('--out_dir', type=str, default='/tmp/ffs_output')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    # Get stereo pair
    if args.live:
        logging.info("Capturing IR stereo pair from RealSense via ROS2...")
        left, right = capture_stereo_pair_ros2()
        if left is None:
            logging.error("No frames received. Is RealSense running with IR enabled?")
            sys.exit(1)
    elif args.pyrs:
        logging.info("Capturing IR stereo pair via pyrealsense2...")
        left, right = capture_stereo_pair_pyrealsense()
    elif args.left_file and args.right_file:
        left = cv2.imread(args.left_file, cv2.IMREAD_UNCHANGED)
        right = cv2.imread(args.right_file, cv2.IMREAD_UNCHANGED)
    else:
        # Use demo images
        demo_dir = f'{code_dir}/../demo_data'
        if os.path.exists(f'{demo_dir}/left.png'):
            left = cv2.imread(f'{demo_dir}/left.png')
            right = cv2.imread(f'{demo_dir}/right.png')
        else:
            # Generate synthetic test pair
            logging.info("No input images — generating synthetic 640x480 test pair")
            left = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            right = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    logging.info(f"Left: {left.shape}, Right: {right.shape}")
    cv2.imwrite(f'{args.out_dir}/left.png', left)
    cv2.imwrite(f'{args.out_dir}/right.png', right)

    # Load model
    logging.info(f"Loading model from {args.model_dir}...")
    model, cfg = load_model(args.model_dir, args.valid_iters, args.max_disp)

    if args.sweep:
        # Sweep configurations to find 30Hz
        configs = [
            # (scale, iters, max_disp)
            (1.0,   8, 192),
            (1.0,   4, 192),
            (0.75,  8, 192),
            (0.75,  4, 192),
            (0.75,  4, 128),
            (0.5,   8, 192),
            (0.5,   4, 192),
            (0.5,   4, 128),
            (0.5,   4, 96),
            (0.5,   2, 128),
            (0.375, 4, 128),
            (0.375, 4, 96),
            (0.375, 2, 96),
        ]

        print(f"\n{'Scale':>6} {'Iters':>5} {'MaxD':>5} {'Resolution':>12} "
              f"{'Mean(ms)':>9} {'Min(ms)':>8} {'Max(ms)':>8} {'FPS':>6} {'30Hz?':>6}")
        print("-" * 80)

        for scale, iters, max_d in configs:
            model.args.max_disp = max_d
            try:
                disp, times = run_inference(model, left, right, scale, iters)
                mean_ms = np.mean(times) * 1000
                min_ms = np.min(times) * 1000
                max_ms = np.max(times) * 1000
                fps = 1000 / mean_ms
                h, w = disp.shape
                hit = "YES" if fps >= 30 else "no"
                print(f"{scale:>6.3f} {iters:>5d} {max_d:>5d} {w:>5d}x{h:<5d} "
                      f"{mean_ms:>9.1f} {min_ms:>8.1f} {max_ms:>8.1f} {fps:>6.1f} {hit:>6}")
            except Exception as e:
                print(f"{scale:>6.3f} {iters:>5d} {max_d:>5d} {'FAILED':>12} — {e}")
    else:
        # Single run
        disp, times = run_inference(model, left, right, args.scale, args.valid_iters)
        mean_ms = np.mean(times) * 1000
        fps = 1000 / mean_ms
        print(f"\n=== Results (scale={args.scale}, iters={args.valid_iters}, max_disp={args.max_disp}) ===")
        print(f"  Resolution: {disp.shape[1]}x{disp.shape[0]}")
        print(f"  Mean: {mean_ms:.1f}ms ({fps:.1f} Hz)")
        print(f"  Min:  {np.min(times)*1000:.1f}ms")
        print(f"  Max:  {np.max(times)*1000:.1f}ms")
        print(f"  Disp range: [{disp.min():.1f}, {disp.max():.1f}]")

        # Save disparity visualization
        from Utils import vis_disparity
        vis = vis_disparity(disp, color_map=cv2.COLORMAP_TURBO)
        cv2.imwrite(f'{args.out_dir}/disp_vis.png', vis[:, :, ::-1])
        logging.info(f"Saved disparity to {args.out_dir}/disp_vis.png")

        # Convert to depth if intrinsics available
        k_file = f'{code_dir}/../realsense_K.txt'
        if os.path.exists(k_file):
            with open(k_file) as f:
                lines = f.readlines()
                K = np.array(list(map(float, lines[0].split()))).reshape(3, 3)
                baseline = float(lines[1])
            K[:2] *= args.scale
            depth = K[0, 0] * baseline / np.clip(disp, 0.01, None)
            depth = np.clip(depth, 0, 10.0)
            print(f"  Depth range: [{depth.min():.2f}m, {np.median(depth):.2f}m median, {depth.max():.2f}m]")
            np.save(f'{args.out_dir}/depth.npy', depth.astype(np.float32))


if __name__ == '__main__':
    main()
