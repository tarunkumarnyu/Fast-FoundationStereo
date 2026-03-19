#include "fast_ffs/cuda_kernels.cuh"
#include <cstdio>
#include <cstdint>

namespace fast_ffs {

// ============================================================
// Preprocess: uint8 gray → float32 3ch NCHW with bilinear resize
// ============================================================
__global__ void preprocess_kernel(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    int src_h, int src_w, int dst_h, int dst_w)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_w || y >= dst_h) return;

  // Bilinear source coordinates
  float sy = (y + 0.5f) * src_h / (float)dst_h - 0.5f;
  float sx = (x + 0.5f) * src_w / (float)dst_w - 0.5f;
  int y0 = max(0, (int)floorf(sy));
  int x0 = max(0, (int)floorf(sx));
  int y1 = min(y0 + 1, src_h - 1);
  int x1 = min(x0 + 1, src_w - 1);
  float fy = sy - y0;
  float fx = sx - x0;

  float val = (1-fy) * ((1-fx) * src[y0*src_w+x0] + fx * src[y0*src_w+x1])
            +    fy  * ((1-fx) * src[y1*src_w+x0] + fx * src[y1*src_w+x1]);

  // Write same value to all 3 channels (NCHW layout)
  int hw = dst_h * dst_w;
  int idx = y * dst_w + x;
  dst[0*hw + idx] = val;  // channel 0
  dst[1*hw + idx] = val;  // channel 1
  dst[2*hw + idx] = val;  // channel 2
}

void preprocess_gpu(const uint8_t* src, float* dst,
                    int src_h, int src_w, int dst_h, int dst_w,
                    cudaStream_t stream)
{
  dim3 block(16, 16);
  dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
  preprocess_kernel<<<grid, block, 0, stream>>>(src, dst, src_h, src_w, dst_h, dst_w);
}

// ============================================================
// NCHW → NHWC permute for GWC input
// ============================================================
__global__ void nchw_to_nhwc_kernel(
    const __half* __restrict__ src,  // (1, C, H, W) = (H, W, C) logically
    __half* __restrict__ dst,        // (H, W, C) contiguous
    int C, int H, int W)
{
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  if (w >= W || h >= H) return;

  for (int c = 0; c < C; ++c) {
    dst[h * W * C + w * C + c] = src[c * H * W + h * W + w];
  }
}

// ============================================================
// GWC kernel: group-wise correlation
// ============================================================
__global__ void gwc_kernel(
    const __half* __restrict__ ref_hwc,  // (H, W, C)
    const __half* __restrict__ tar_hwc,  // (H, W, C)
    __half* __restrict__ out_hgdw,       // (H, G, D, W)
    int H, int W, int C, int D, int G, int K)
{
  // Grid: (H * G, ceil(D/8), ceil(W/64))
  // Block: (64, 8)
  int pid0 = blockIdx.x;
  int h = pid0 / G;
  int g = pid0 % G;
  int d = blockIdx.y * blockDim.y + threadIdx.y;
  int w = blockIdx.z * blockDim.x + threadIdx.x;

  if (h >= H || d >= D || w >= W) return;

  float acc = 0.0f;
  int w_src = w - d;

  if (w_src >= 0) {
    int ref_base = h * W * C + w * C + g * K;
    int tar_base = h * W * C + w_src * C + g * K;
    for (int k = 0; k < K; ++k) {
      acc += __half2float(ref_hwc[ref_base + k]) * __half2float(tar_hwc[tar_base + k]);
    }
  }

  // Output layout: (H, G, D, W)
  out_hgdw[h * G * D * W + g * D * W + d * W + w] = __float2half(acc);
}

// ============================================================
// Permute GWC output: (H, G, D, W) → (1, G, D, H, W) = (G, D, H, W)
// ============================================================
__global__ void hgdw_to_gdhw_kernel(
    const __half* __restrict__ src,  // (H, G, D, W)
    __half* __restrict__ dst,        // (G, D, H, W)
    int H, int G, int D, int W)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = G * D * H * W;
  if (idx >= total) return;

  // Decode dst index: (g, d, h, w)
  int w_ = idx % W; idx /= W;
  int h_ = idx % H; idx /= H;
  int d_ = idx % D; idx /= D;
  int g_ = idx;

  // Read from src: (h, g, d, w)
  __half val = src[h_ * G * D * W + g_ * D * W + d_ * W + w_];
  // Write to dst position (already computed from the idx)
  dst[g_ * D * H * W + d_ * H * W + h_ * W + w_] = val;
}

size_t gwc_workspace_bytes(int C, int H, int W) {
  // Two NHWC permuted buffers + intermediate HGDW output
  return 2 * H * W * C * sizeof(__half);
}

void build_gwc_volume(
    const __half* feat_left, const __half* feat_right,
    __half* gwc_out, __half* workspace,
    int C, int H, int W, int max_disp, int num_groups,
    cudaStream_t stream)
{
  int K = C / num_groups;
  __half* ref_hwc = workspace;
  __half* tar_hwc = workspace + H * W * C;

  // 1. Permute NCHW → NHWC (HWC)
  {
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    nchw_to_nhwc_kernel<<<grid, block, 0, stream>>>(feat_left, ref_hwc, C, H, W);
    nchw_to_nhwc_kernel<<<grid, block, 0, stream>>>(feat_right, tar_hwc, C, H, W);
  }

  // 2. GWC correlation
  // Temp output in (H, G, D, W) layout
  // We'll reuse gwc_out as temp since final permute writes to same buffer...
  // Actually need separate temp. Use gwc_out directly in HGDW, then permute in-place.
  // For simplicity, allocate small temp on stack? No, use gwc_out as GDHW target.
  // We need a temp HGDW buffer. Let's use part of workspace after the two NHWC buffers.
  // Actually workspace only has 2*H*W*C halfs. HGDW needs H*G*D*W halfs.
  // For now, just write directly to gwc_out in GDHW layout by computing the correct index.
  // Rewrite kernel to output GDHW directly:
  // Actually, let's just use a simple approach: output HGDW into gwc_out, then permute in-place.
  // But that requires a temp buffer of same size. Let's just compute GDHW directly in the kernel.

  // Modified: output directly to (G, D, H, W) layout
  {
    dim3 block(64, 8);
    dim3 grid(H * num_groups, (max_disp + 7) / 8, (W + 63) / 64);

    // We need to write directly to GDHW. Modify output index in kernel.
    // For now, use the two-step approach with a small temp alloc.
    __half* temp_hgdw = nullptr;
    size_t temp_bytes = (size_t)H * num_groups * max_disp * W * sizeof(__half);
    cudaMalloc(&temp_hgdw, temp_bytes);

    gwc_kernel<<<grid, block, 0, stream>>>(
        ref_hwc, tar_hwc, temp_hgdw,
        H, W, C, max_disp, num_groups, K);

    // 3. Permute (H, G, D, W) → (G, D, H, W)
    int total = num_groups * max_disp * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    hgdw_to_gdhw_kernel<<<blocks, threads, 0, stream>>>(
        temp_hgdw, gwc_out, H, num_groups, max_disp, W);

    cudaFree(temp_hgdw);
  }
}

// ============================================================
// Box blur 5x5
// ============================================================
__global__ void box_blur_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int H, int W)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;

  float sum = 0.0f;
  int count = 0;
  for (int dy = -2; dy <= 2; ++dy) {
    for (int dx = -2; dx <= 2; ++dx) {
      int ny = y + dy, nx = x + dx;
      if (ny >= 0 && ny < H && nx >= 0 && nx < W) {
        sum += src[ny * W + nx];
        ++count;
      }
    }
  }
  dst[y * W + x] = sum / count;
}

void box_blur_5x5(float* data, float* temp, int H, int W, cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((W + 15) / 16, (H + 15) / 16);
  // data → temp (blurred), then swap
  box_blur_kernel<<<grid, block, 0, stream>>>(data, temp, H, W);
  cudaMemcpyAsync(data, temp, H * W * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

// ============================================================
// Temporal EMA + normalize (fused)
// ============================================================
__global__ void temporal_normalize_kernel(
    float* __restrict__ prev,
    const float* __restrict__ cur,
    uint8_t* __restrict__ out,
    int N, float alpha, float d_max, bool first_frame)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  float c = cur[i];
  float p;
  if (first_frame) {
    p = c;
  } else {
    p = (1.0f - alpha) * prev[i] + alpha * c;
  }
  prev[i] = p;

  // Normalize: close(high disp) = dark(0), far(low disp) = white(255)
  float norm = 1.0f - fminf(fmaxf(p / d_max, 0.0f), 1.0f);
  out[i] = (uint8_t)(norm * 255.0f);
}

void temporal_normalize(float* prev, const float* cur, uint8_t* out,
                        int H, int W, float alpha, float d_max,
                        bool first_frame, cudaStream_t stream)
{
  int N = H * W;
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  temporal_normalize_kernel<<<blocks, threads, 0, stream>>>(
      prev, cur, out, N, alpha, d_max, first_frame);
}

}  // namespace fast_ffs
