#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace fast_ffs {

// Preprocess: uint8 grayscale → float32 3-channel, with bilinear resize
void preprocess_gpu(
    const uint8_t* src,  // device, (src_h, src_w) uint8
    float* dst,          // device, (1, 3, dst_h, dst_w) float NCHW
    int src_h, int src_w,
    int dst_h, int dst_w,
    cudaStream_t stream);

// GWC volume: group-wise correlation between left/right features
// Input features: (1, C, H, W) fp16 NCHW
// Output: (1, num_groups, max_disp, H, W) fp16
void build_gwc_volume(
    const __half* feat_left,   // (1, C, H, W)
    const __half* feat_right,  // (1, C, H, W)
    __half* gwc_out,           // (1, G, D, H, W)
    __half* workspace,         // temp: 2 * H*W*C halfs for NHWC permute
    int C, int H, int W,
    int max_disp, int num_groups,
    cudaStream_t stream);

// Box blur 5x5 in-place
void box_blur_5x5(float* data, float* temp, int H, int W, cudaStream_t stream);

// Temporal EMA + normalize to uint8 (fused)
void temporal_normalize(
    float* prev,        // (H, W) persistent EMA buffer
    const float* cur,   // (H, W) current disparity
    uint8_t* out,       // (H, W) output grayscale
    int H, int W,
    float ema_alpha,    // 0.7 = trust current 70%
    float d_max,        // max disparity for normalization
    bool first_frame,   // if true, copy cur→prev instead of blend
    cudaStream_t stream);

// Workspace size needed for GWC
size_t gwc_workspace_bytes(int C, int H, int W);

}  // namespace fast_ffs
