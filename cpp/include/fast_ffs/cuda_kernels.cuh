#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace fast_ffs {

// Preprocess: uint8 grayscale → float32 3-channel NCHW, with bilinear resize
void preprocess_gpu(
    const uint8_t* src,  // device, (src_h, src_w) uint8
    float* dst,          // device, (1, 3, dst_h, dst_w) float32 NCHW
    int src_h, int src_w,
    int dst_h, int dst_w,
    cudaStream_t stream);

// Box blur 5x5 (writes result back into data via temp buffer)
void box_blur_5x5(float* data, float* temp, int H, int W, cudaStream_t stream);

// Convert fp16 → float32
void half_to_float(const __half* src, float* dst, int N, cudaStream_t stream);

// Temporal EMA with NaN guard and clamp.
// prev = (1-alpha)*prev + alpha*clamp(cur, clamp_min)
// NaN in cur is replaced with prev value (or clamp_min on first frame).
// If first_frame, prev = clamp(cur, clamp_min).
void temporal_ema(
    float* prev,        // (N) persistent EMA buffer, updated in-place
    const float* cur,   // (N) current disparity
    int N,
    float alpha,        // 0.7 = trust current 70%
    float clamp_min,    // minimum disparity value (0.5)
    bool first_frame,
    cudaStream_t stream);

}  // namespace fast_ffs
