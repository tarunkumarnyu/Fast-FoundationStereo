#!/usr/bin/env python3
"""Benchmark Fast-FoundationStereo TensorRT engines on Jetson Orin NX."""

import os, sys, time, argparse, logging
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from core.foundation_stereo import TrtRunner
from Utils import set_logging_format, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_dir', type=str, required=True,
                        help='Directory containing .engine files and onnx.yaml')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--runs', type=int, default=20)
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    # Load config
    yaml_path = f'{args.engine_dir}/onnx.yaml'
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    cfg_om = OmegaConf.create(cfg)
    logging.info(f"Config: {cfg}")

    H, W = cfg['image_size']
    logging.info(f"Resolution: {W}x{H}, valid_iters={cfg.get('valid_iters')}, max_disp={cfg.get('max_disp')}")

    # Load TRT model
    model = TrtRunner(cfg_om,
                      f'{args.engine_dir}/feature_runner.engine',
                      f'{args.engine_dir}/post_runner.engine')

    # Create synthetic input
    left = torch.randn(1, 3, H, W).cuda().float() * 255
    right = torch.randn(1, 3, H, W).cuda().float() * 255

    # Warmup
    logging.info(f"Warming up ({args.warmup} runs)...")
    for _ in range(args.warmup):
        _ = model.forward(left, right)
    torch.cuda.synchronize()

    # Benchmark
    logging.info(f"Benchmarking ({args.runs} runs)...")
    times = []
    for _ in range(args.runs):
        torch.cuda.synchronize()
        t0 = time.time()
        disp = model.forward(left, right)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    times = np.array(times) * 1000  # ms
    fps = 1000 / np.mean(times)
    print(f"\n=== TRT Benchmark: {W}x{H} ===")
    print(f"  valid_iters: {cfg.get('valid_iters')}")
    print(f"  max_disp:    {cfg.get('max_disp')}")
    print(f"  Mean:  {np.mean(times):.1f}ms ({fps:.1f} Hz)")
    print(f"  Min:   {np.min(times):.1f}ms")
    print(f"  Max:   {np.max(times):.1f}ms")
    print(f"  Std:   {np.std(times):.1f}ms")
    print(f"  P95:   {np.percentile(times, 95):.1f}ms")
    print(f"  30Hz:  {'YES' if fps >= 30 else 'NO'}")
    print(f"  Disp shape: {disp.shape}")


if __name__ == '__main__':
    main()
