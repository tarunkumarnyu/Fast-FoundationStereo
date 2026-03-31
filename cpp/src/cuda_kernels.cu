#include "fast_ffs/cuda_kernels.cuh"

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

  int hw = dst_h * dst_w;
  int idx = y * dst_w + x;
  dst[0*hw + idx] = val;
  dst[1*hw + idx] = val;
  dst[2*hw + idx] = val;
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
  box_blur_kernel<<<grid, block, 0, stream>>>(data, temp, H, W);
  cudaMemcpyAsync(data, temp, H * W * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

// ============================================================
// fp16 → float32
// ============================================================
__global__ void half_to_float_kernel(
    const __half* __restrict__ src,
    float* __restrict__ dst,
    int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) dst[i] = __half2float(src[i]);
}

void half_to_float(const __half* src, float* dst, int N, cudaStream_t stream) {
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  half_to_float_kernel<<<blocks, threads, 0, stream>>>(src, dst, N);
}

// ============================================================
// Temporal EMA with NaN guard and clamp
// ============================================================
__global__ void temporal_ema_kernel(
    float* __restrict__ prev,
    const float* __restrict__ cur,
    int N, float alpha, float clamp_min, bool first_frame)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  float c = cur[i];
  // NaN guard: fall back to previous value (or clamp_min on first frame)
  if (isnan(c)) c = first_frame ? clamp_min : prev[i];
  c = fmaxf(c, clamp_min);

  float p = first_frame ? c : (1.0f - alpha) * prev[i] + alpha * c;
  prev[i] = p;
}

void temporal_ema(float* prev, const float* cur, int N,
                  float alpha, float clamp_min, bool first_frame,
                  cudaStream_t stream)
{
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  temporal_ema_kernel<<<blocks, threads, 0, stream>>>(
      prev, cur, N, alpha, clamp_min, first_frame);
}

}  // namespace fast_ffs
