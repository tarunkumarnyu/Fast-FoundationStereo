#!/usr/bin/env python3
"""Benchmark sequential vs pipelined TRT inference."""

import os, sys, time
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.pipelined_runner import PipelinedTrtRunner


def benchmark(runner, left, right, mode, warmup=5, runs=30):
    forward_fn = runner.forward_sequential if mode == 'sequential' else runner.forward_pipelined

    # Warmup
    for _ in range(warmup):
        forward_fn(left, right)
    if mode == 'pipelined':
        runner.flush()
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for i in range(runs):
        torch.cuda.synchronize()
        t0 = time.time()
        disp = forward_fn(left, right)
        if mode == 'sequential':
            torch.cuda.synchronize()
        elif disp is not None:
            # In pipelined mode, we measure time between consecutive outputs
            pass
        times.append(time.time() - t0)

    if mode == 'pipelined':
        runner.flush()
    torch.cuda.synchronize()

    times_ms = np.array(times) * 1000
    fps = 1000 / np.mean(times_ms)
    return times_ms, fps


def main():
    configs = [
        ('320x224_3iter', '/tmp/ffs_trt/320x224_3iter'),
        ('320x224_4iter', '/tmp/ffs_trt/320x224_4iter'),
    ]

    for name, engine_dir in configs:
        with open(f'{engine_dir}/onnx.yaml') as f:
            cfg = OmegaConf.create(yaml.safe_load(f))
        H, W = cfg.image_size

        runner = PipelinedTrtRunner(cfg,
                                     f'{engine_dir}/feature_runner.engine',
                                     f'{engine_dir}/post_runner.engine')

        left = torch.randn(1, 3, H, W).cuda().float() * 255
        right = torch.randn(1, 3, H, W).cuda().float() * 255

        print(f"\n=== {name} ({W}x{H}) ===")

        for mode in ['sequential', 'pipelined']:
            times, fps = benchmark(runner, left, right, mode, warmup=5, runs=40)
            tag = "YES" if fps >= 30 else "NO"
            print(f"  {mode:12s}: {np.mean(times):.1f}ms ({fps:.1f}Hz) "
                  f"min={np.min(times):.1f} max={np.max(times):.1f} std={np.std(times):.1f}  30Hz={tag}")

        del runner
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
