/**
 * C++ Fast-FoundationStereo ROS2 node — unified pipeline.
 * Supports two modes:
 *   - Two TRT engines: feature_runner → post_gwc_runner → disparity.
 *   - Single TRT engine: foundation_stereo (left, right → disp).
 * Per-stage CUDA event profiling.
 * Publishes: compressed disparity, float32 depth, PointCloud2 (for validation).
 */
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cmath>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <opencv2/imgcodecs.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <yaml-cpp/yaml.h>

#include "fast_ffs/trt_engine.hpp"
#include "fast_ffs/cuda_kernels.cuh"

using namespace std::chrono_literals;

// Number of profiling stages
static constexpr int NUM_EVENTS = 9;
static const char* STAGE_NAMES[] = {
    "H2D", "preprocess", "feature_trt", "post_trt",
    "fp16->fp32", "box_blur", "temporal_ema", "D2H+sync", "total_gpu"
};

class LiveFfsNode : public rclcpp::Node {
public:
  LiveFfsNode() : Node("live_ffs_cpp") {
    // Parameters
    declare_parameter("engine_dir", std::string(""));
    declare_parameter("ns", std::string("/race6/cam1"));
    declare_parameter("fx", 430.83);
    declare_parameter("fy", 430.91);
    declare_parameter("cx", 427.97);
    declare_parameter("cy", 246.90);
    declare_parameter("baseline", 0.09508);
    declare_parameter("zfar", 20.0);
    declare_parameter("native_w", 848);
    declare_parameter("native_h", 480);

    auto engine_dir = get_parameter("engine_dir").as_string();
    auto ns = get_parameter("ns").as_string();

    // Load config
    auto yaml = YAML::LoadFile(engine_dir + "/onnx.yaml");
    H_ = yaml["image_size"][0].as<int>();
    W_ = yaml["image_size"][1].as<int>();
    max_disp_ = yaml["max_disp"].as<int>(128);
    bool unified = yaml["unified_post"].as<bool>(false);
    single_engine_ = yaml["single_onnx"].as<bool>(false);
    int iters = yaml["valid_iters"].as<int>(8);
    RCLCPP_INFO(get_logger(), "Config: %dx%d, %d iters, unified=%s, single=%s",
                W_, H_, iters, unified ? "true" : "false",
                single_engine_ ? "true" : "false");

    // Depth conversion — scale intrinsics to engine resolution
    int native_w = get_parameter("native_w").as_int();
    int native_h = get_parameter("native_h").as_int();
    baseline_ = get_parameter("baseline").as_double();
    zfar_ = get_parameter("zfar").as_double();
    double scale_x = (double)W_ / native_w;
    double scale_y = (double)H_ / native_h;
    fx_ = get_parameter("fx").as_double() * scale_x;
    fy_ = get_parameter("fy").as_double() * scale_y;
    cx_ = get_parameter("cx").as_double() * scale_x;
    cy_ = get_parameter("cy").as_double() * scale_y;
    fb_ = fx_ * baseline_;
    zmin_ = fb_ / max_disp_;
    RCLCPP_INFO(get_logger(), "Depth: fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f, baseline=%.4fm",
                fx_, fy_, cx_, cy_, baseline_);
    RCLCPP_INFO(get_logger(), "Depth: zfar=%.1fm, zmin=%.2fm, fb=%.2f", zfar_, zmin_, fb_);

    // Load TRT engines
    if (single_engine_) {
      single_engine_ptr_ = std::make_unique<fast_ffs::TrtEngine>(
          engine_dir + "/foundation_stereo.engine");
      auto disp_info = single_engine_ptr_->getTensorInfo("disp");
      disp_is_fp16_ = (disp_info.dtype == nvinfer1::DataType::kHALF);
      RCLCPP_INFO(get_logger(), "Loaded single engine (left,right -> disp)");
    } else {
      feat_engine_ = std::make_unique<fast_ffs::TrtEngine>(engine_dir + "/feature_runner.engine");
      std::string post_name = unified ? "post_gwc_runner.engine" : "post_runner.engine";
      post_engine_ = std::make_unique<fast_ffs::TrtEngine>(engine_dir + "/" + post_name);
      auto disp_info = post_engine_->getTensorInfo("disp");
      disp_is_fp16_ = (disp_info.dtype == nvinfer1::DataType::kHALF);
    }

    // Allocate GPU/host buffers
    alloc_buffers();
    cudaStreamCreate(&stream_);

    // Create profiling events
    for (int i = 0; i < NUM_EVENTS; ++i) {
      cudaEventCreate(&events_[i]);
    }

    // Warmup
    warmup();

    // ROS2 subscribers — time-synchronized stereo pair
    declare_parameter("compressed", false);
    compressed_ = get_parameter("compressed").as_bool();

    auto qos = rclcpp::QoS(1).best_effort();
    if (compressed_) {
      RCLCPP_INFO(get_logger(), "Subscribing to COMPRESSED topics");
      sub_left_c_.subscribe(this, ns + "/infra1/image_rect_raw/compressed", qos.get_rmw_qos_profile());
      sub_right_c_.subscribe(this, ns + "/infra2/image_rect_raw/compressed", qos.get_rmw_qos_profile());
      sync_c_ = std::make_shared<message_filters::TimeSynchronizer<
          sensor_msgs::msg::CompressedImage, sensor_msgs::msg::CompressedImage>>(sub_left_c_, sub_right_c_, 2);
      sync_c_->registerCallback(std::bind(&LiveFfsNode::cb_stereo_compressed, this,
          std::placeholders::_1, std::placeholders::_2));
    } else {
      sub_left_.subscribe(this, ns + "/infra1/image_rect_raw", qos.get_rmw_qos_profile());
      sub_right_.subscribe(this, ns + "/infra2/image_rect_raw", qos.get_rmw_qos_profile());
      sync_ = std::make_shared<message_filters::TimeSynchronizer<
          sensor_msgs::msg::Image, sensor_msgs::msg::Image>>(sub_left_, sub_right_, 2);
      sync_->registerCallback(std::bind(&LiveFfsNode::cb_stereo, this,
          std::placeholders::_1, std::placeholders::_2));
    }

    // Publishers
    auto pub_qos = rclcpp::QoS(1).reliable();
    pub_gray_ = create_publisher<sensor_msgs::msg::CompressedImage>(
        "/ffs/disp_gray/compressed", pub_qos);
    auto depth_qos = rclcpp::QoS(1).best_effort();
    pub_depth_ = create_publisher<sensor_msgs::msg::Image>(
        "/ffs/depth_raw", depth_qos);
    pub_pcl_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "/ffs/pointcloud", depth_qos);

    // Inference thread
    running_ = true;
    first_frame_ = true;
    frame_count_ = 0;
    infer_thread_ = std::thread(&LiveFfsNode::infer_loop, this);

    RCLCPP_INFO(get_logger(), "Ready — unified C++ pipeline (profiled)");
  }

  ~LiveFfsNode() {
    running_ = false;
    frame_cv_.notify_all();
    if (infer_thread_.joinable()) infer_thread_.join();
    free_buffers();
    cudaStreamDestroy(stream_);
    for (int i = 0; i < NUM_EVENTS; ++i) {
      cudaEventDestroy(events_[i]);
    }
  }

private:
  void set_input_size(int h, int w) {
    if (input_h_ == h && input_w_ == w) return;
    if (input_h_ != 0) {
      cudaFreeHost(pin_left_); cudaFreeHost(pin_right_);
      cudaFreeHost(snap_left_); cudaFreeHost(snap_right_);
      cudaFree(gpu_left_raw_); cudaFree(gpu_right_raw_);
    }
    input_h_ = h; input_w_ = w;
    cudaMallocHost(&pin_left_, input_h_ * input_w_);
    cudaMallocHost(&pin_right_, input_h_ * input_w_);
    cudaMallocHost(&snap_left_, input_h_ * input_w_);
    cudaMallocHost(&snap_right_, input_h_ * input_w_);
    cudaMalloc(&gpu_left_raw_, input_h_ * input_w_);
    cudaMalloc(&gpu_right_raw_, input_h_ * input_w_);
    RCLCPP_INFO(get_logger(), "Input: %dx%d", input_w_, input_h_);
  }

  void cb_stereo(const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
                 const sensor_msgs::msg::Image::ConstSharedPtr& right_msg) {
    set_input_size(left_msg->height, left_msg->width);
    {
      std::lock_guard<std::mutex> lock(mtx_);
      std::memcpy(pin_left_, left_msg->data.data(), left_msg->data.size());
      std::memcpy(pin_right_, right_msg->data.data(), right_msg->data.size());
      stereo_seq_++;
    }
    frame_cv_.notify_one();
  }

  void cb_stereo_compressed(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& left_msg,
                            const sensor_msgs::msg::CompressedImage::ConstSharedPtr& right_msg) {
    // Decompress JPEG/PNG to grayscale
    cv::Mat left_dec = cv::imdecode(left_msg->data, cv::IMREAD_GRAYSCALE);
    cv::Mat right_dec = cv::imdecode(right_msg->data, cv::IMREAD_GRAYSCALE);
    if (left_dec.empty() || right_dec.empty()) return;

    set_input_size(left_dec.rows, left_dec.cols);
    {
      std::lock_guard<std::mutex> lock(mtx_);
      std::memcpy(pin_left_, left_dec.data, input_h_ * input_w_);
      std::memcpy(pin_right_, right_dec.data, input_h_ * input_w_);
      stereo_seq_++;
    }
    frame_cv_.notify_one();
  }

  void infer_loop() {
    auto t_window_start = std::chrono::steady_clock::now();
    int N = H_ * W_;

    // Accumulate stage timings for averaging
    float stage_accum[NUM_EVENTS] = {};
    int profile_count = 0;

    uint64_t last_seq = 0;

    while (running_) {
      // Wait for a NEW synced stereo pair (frame-drop: always process latest only)
      {
        std::unique_lock<std::mutex> lock(mtx_);
        frame_cv_.wait_for(lock, 100ms, [this, &last_seq] {
          return (stereo_seq_ > last_seq) || !running_;
        });
        if (!running_) break;
        if (stereo_seq_ <= last_seq) continue;
        last_seq = stereo_seq_;
        // Snapshot pinned buffers under lock so callback can't overwrite mid-H2D
        std::memcpy(snap_left_, pin_left_, input_h_ * input_w_);
        std::memcpy(snap_right_, pin_right_, input_h_ * input_w_);
      }

      // === EVENT 0: start ===
      cudaEventRecord(events_[0], stream_);

      // H2D from snapshot (lock-free)
      cudaMemcpyAsync(gpu_left_raw_, snap_left_, input_h_ * input_w_,
                       cudaMemcpyHostToDevice, stream_);
      cudaMemcpyAsync(gpu_right_raw_, snap_right_, input_h_ * input_w_,
                       cudaMemcpyHostToDevice, stream_);
      cudaEventRecord(events_[1], stream_);

      // Preprocess: resize + gray→3ch float32
      fast_ffs::preprocess_gpu(gpu_left_raw_, buf_left_,
                                input_h_, input_w_, H_, W_, stream_);
      fast_ffs::preprocess_gpu(gpu_right_raw_, buf_right_,
                                input_h_, input_w_, H_, W_, stream_);
      cudaEventRecord(events_[2], stream_);

      void* disp_raw_ptr = nullptr;
      if (single_engine_) {
        // Single engine: (left, right) → disp
        cudaEventRecord(events_[3], stream_);  // no separate feature stage
        single_engine_ptr_->infer({{"left", buf_left_}, {"right", buf_right_}}, stream_);
        disp_raw_ptr = single_engine_ptr_->getOutputPtr("disp");
      } else {
        // Two-engine: feature → post+GWC
        feat_engine_->infer({{"left", buf_left_}, {"right", buf_right_}}, stream_);
        cudaEventRecord(events_[3], stream_);

        std::map<std::string, void*> post_inputs;
        for (auto& info : post_engine_->getInputInfos()) {
          void* ptr = feat_engine_->getOutputPtr(info.name);
          if (ptr) post_inputs[info.name] = ptr;
        }
        post_engine_->infer(post_inputs, stream_);
        disp_raw_ptr = post_engine_->getOutputPtr("disp");
      }
      cudaEventRecord(events_[4], stream_);

      // fp16→fp32 if needed
      if (disp_is_fp16_) {
        fast_ffs::half_to_float(
            (const __half*)disp_raw_ptr,
            disp_f32_, N, stream_);
      } else {
        cudaMemcpyAsync(disp_f32_, disp_raw_ptr,
                         N * sizeof(float), cudaMemcpyDeviceToDevice, stream_);
      }
      cudaEventRecord(events_[5], stream_);

      // Box blur disabled — raw disparity
      cudaEventRecord(events_[6], stream_);

      // EMA disabled — pass through
      cudaEventRecord(events_[7], stream_);

      // D2H + sync
      cudaMemcpyAsync(disp_host_, disp_f32_, N * sizeof(float),
                       cudaMemcpyDeviceToHost, stream_);
      cudaStreamSynchronize(stream_);
      cudaEventRecord(events_[8], stream_);
      cudaEventSynchronize(events_[8]);

      // Measure GPU stage times
      float stage_ms[NUM_EVENTS];
      for (int i = 0; i < NUM_EVENTS - 1; ++i) {
        cudaEventElapsedTime(&stage_ms[i], events_[i], events_[i + 1]);
        stage_accum[i] += stage_ms[i];
      }
      // Total GPU time (event 0 to event 8)
      cudaEventElapsedTime(&stage_ms[NUM_EVENTS - 1], events_[0], events_[8]);
      stage_accum[NUM_EVENTS - 1] += stage_ms[NUM_EVENTS - 1];
      profile_count++;

      // === CPU post-processing (timed with wall clock) ===
      auto cpu_t0 = std::chrono::steady_clock::now();

      auto stamp = now();

      // Adaptive normalization → uint8 grayscale
      float d_min = disp_host_[0], d_max = disp_host_[0];
      for (int i = 1; i < N; ++i) {
        if (disp_host_[i] < d_min) d_min = disp_host_[i];
        if (disp_host_[i] > d_max) d_max = disp_host_[i];
      }
      float span = std::max(d_max - d_min, 1.0f);
      for (int i = 0; i < N; ++i) {
        int v = (int)((disp_host_[i] - d_min) / span * 255.0f);
        gray_host_[i] = (uint8_t)std::min(std::max(v, 0), 255);
      }
      auto cpu_t1 = std::chrono::steady_clock::now();

      // JPEG encode
      cv::Mat gray_mat(H_, W_, CV_8UC1, gray_host_);
      jpeg_buf_.clear();
      cv::imencode(".jpg", gray_mat, jpeg_buf_, {cv::IMWRITE_JPEG_QUALITY, 75});
      auto cpu_t2 = std::chrono::steady_clock::now();

      // Publish compressed disparity
      auto gray_msg = sensor_msgs::msg::CompressedImage();
      gray_msg.header.stamp = stamp;
      gray_msg.format = "jpeg";
      gray_msg.data.assign(jpeg_buf_.begin(), jpeg_buf_.end());
      pub_gray_->publish(gray_msg);
      auto cpu_t3 = std::chrono::steady_clock::now();

      // Depth + PointCloud (only if subscribed)
      float cpu_depth_ms = 0.0f, cpu_pcl_ms = 0.0f;
      bool pub_depth = pub_depth_->get_subscription_count() > 0;
      bool pub_pcl = pub_pcl_->get_subscription_count() > 0;

      if (pub_depth || pub_pcl) {
        // Compute depth for all pixels
        for (int i = 0; i < N; ++i) {
          float d = (float)(fb_ / (double)disp_host_[i]);
          depth_host_[i] = std::min(std::max(d, (float)zmin_), (float)zfar_);
        }
        auto cpu_t4 = std::chrono::steady_clock::now();
        cpu_depth_ms = std::chrono::duration<float, std::milli>(cpu_t4 - cpu_t3).count();

        if (pub_depth) {
          auto depth_msg = sensor_msgs::msg::Image();
          depth_msg.header.stamp = stamp;
          depth_msg.header.frame_id = "cam1_infra1_optical_frame";
          depth_msg.height = H_;
          depth_msg.width = W_;
          depth_msg.encoding = "32FC1";
          depth_msg.is_bigendian = false;
          depth_msg.step = W_ * 4;
          depth_msg.data.resize(N * sizeof(float));
          std::memcpy(depth_msg.data.data(), depth_host_, N * sizeof(float));
          pub_depth_->publish(depth_msg);
        }

        if (pub_pcl) {
          publish_pointcloud(stamp, N);
          auto cpu_t5 = std::chrono::steady_clock::now();
          cpu_pcl_ms = std::chrono::duration<float, std::milli>(cpu_t5 - cpu_t4 - std::chrono::duration<float, std::milli>(cpu_depth_ms)).count();
        }
      }

      auto cpu_tend = std::chrono::steady_clock::now();
      float cpu_norm_ms = std::chrono::duration<float, std::milli>(cpu_t1 - cpu_t0).count();
      float cpu_jpeg_ms = std::chrono::duration<float, std::milli>(cpu_t2 - cpu_t1).count();
      float cpu_pub_ms = std::chrono::duration<float, std::milli>(cpu_t3 - cpu_t2).count();
      float cpu_total_ms = std::chrono::duration<float, std::milli>(cpu_tend - cpu_t0).count();
      cpu_norm_accum_ += cpu_norm_ms;
      cpu_jpeg_accum_ += cpu_jpeg_ms;
      cpu_pub_accum_ += cpu_pub_ms;
      cpu_total_accum_ += cpu_total_ms;

      // Stats
      frame_count_++;
      int64_t dropped = (int64_t)last_seq - (int64_t)frame_count_;  // approximate synced pairs skipped
      auto now_tp = std::chrono::steady_clock::now();
      if (frame_count_ % 30 == 0) {
        double window_sec = std::chrono::duration<double>(now_tp - t_window_start).count();
        double fps = 30.0 / window_sec;  // instantaneous over last 30 frames
        t_window_start = now_tp;
        float n = (float)profile_count;

        RCLCPP_INFO(get_logger(),
            "Frame %d: %.1fHz (sync seq %lu, dropped ~%ld), disp=[%.1f, %.1f]",
            frame_count_, fps, last_seq, dropped, d_min, d_max);
        RCLCPP_INFO(get_logger(),
            "  GPU avg: H2D=%.2f  preproc=%.2f  feat_trt=%.2f  post_trt=%.2f  "
            "cvt=%.2f  blur=%.2f  ema=%.2f  d2h=%.2f  TOTAL=%.2f ms",
            stage_accum[0]/n, stage_accum[1]/n, stage_accum[2]/n, stage_accum[3]/n,
            stage_accum[4]/n, stage_accum[5]/n, stage_accum[6]/n, stage_accum[7]/n,
            stage_accum[8]/n);
        RCLCPP_INFO(get_logger(),
            "  CPU avg: normalize=%.2f  jpeg=%.2f  publish=%.2f  TOTAL=%.2f ms",
            cpu_norm_accum_/n, cpu_jpeg_accum_/n, cpu_pub_accum_/n, cpu_total_accum_/n);

        // Reset accumulators
        std::memset(stage_accum, 0, sizeof(stage_accum));
        cpu_norm_accum_ = cpu_jpeg_accum_ = cpu_pub_accum_ = cpu_total_accum_ = 0;
        profile_count = 0;
      }
    }
  }

  void publish_pointcloud(rclcpp::Time stamp, int N) {
    // Build XYZ pointcloud from depth using camera intrinsics
    // Point layout: x, y, z (float32 each, 12 bytes per point)
    auto msg = sensor_msgs::msg::PointCloud2();
    msg.header.stamp = stamp;
    msg.header.frame_id = "cam1_infra1_optical_frame";
    msg.height = H_;
    msg.width = W_;
    msg.is_dense = false;
    msg.is_bigendian = false;
    msg.point_step = 12;  // 3 floats x 4 bytes
    msg.row_step = W_ * 12;

    // Fields: x, y, z
    sensor_msgs::msg::PointField fx, fy, fz;
    fx.name = "x"; fx.offset = 0;  fx.datatype = 7; fx.count = 1;
    fy.name = "y"; fy.offset = 4;  fy.datatype = 7; fy.count = 1;
    fz.name = "z"; fz.offset = 8;  fz.datatype = 7; fz.count = 1;
    msg.fields = {fx, fy, fz};

    msg.data.resize(N * 12);
    float* pts = reinterpret_cast<float*>(msg.data.data());

    for (int v = 0; v < H_; ++v) {
      for (int u = 0; u < W_; ++u) {
        int idx = v * W_ + u;
        float z = depth_host_[idx];
        float x = (u - cx_) * z / fx_;
        float y = (v - cy_) * z / fy_;
        pts[idx * 3 + 0] = x;
        pts[idx * 3 + 1] = y;
        pts[idx * 3 + 2] = z;
      }
    }

    pub_pcl_->publish(msg);
  }

  void alloc_buffers() {
    int default_h = 480, default_w = 848;
    int N = H_ * W_;

    // Pre-allocate input buffers for default camera resolution
    input_h_ = default_h; input_w_ = default_w;
    cudaMallocHost(&pin_left_, default_h * default_w);
    cudaMallocHost(&pin_right_, default_h * default_w);
    cudaMallocHost(&snap_left_, default_h * default_w);
    cudaMallocHost(&snap_right_, default_h * default_w);
    cudaMalloc(&gpu_left_raw_, default_h * default_w);
    cudaMalloc(&gpu_right_raw_, default_h * default_w);
    cudaMalloc(&buf_left_, 3 * N * sizeof(float));
    cudaMalloc(&buf_right_, 3 * N * sizeof(float));
    cudaMalloc(&disp_f32_, N * sizeof(float));
    cudaMalloc(&blur_temp_, N * sizeof(float));
    cudaMalloc(&ema_buf_, N * sizeof(float));
    cudaMemset(ema_buf_, 0, N * sizeof(float));
    cudaMallocHost(&disp_host_, N * sizeof(float));
    cudaMallocHost(&gray_host_, N);
    cudaMallocHost(&depth_host_, N * sizeof(float));
  }

  void free_buffers() {
    cudaFreeHost(pin_left_); cudaFreeHost(pin_right_);
    cudaFreeHost(snap_left_); cudaFreeHost(snap_right_);
    cudaFree(gpu_left_raw_); cudaFree(gpu_right_raw_);
    cudaFree(buf_left_); cudaFree(buf_right_);
    cudaFree(disp_f32_); cudaFree(blur_temp_); cudaFree(ema_buf_);
    cudaFreeHost(disp_host_); cudaFreeHost(gray_host_); cudaFreeHost(depth_host_);
  }

  void warmup() {
    RCLCPP_INFO(get_logger(), "Warming up...");
    cudaMemset(buf_left_, 0, 3 * H_ * W_ * sizeof(float));
    cudaMemset(buf_right_, 0, 3 * H_ * W_ * sizeof(float));
    for (int i = 0; i < 3; ++i) {
      if (single_engine_) {
        single_engine_ptr_->infer({{"left", buf_left_}, {"right", buf_right_}}, stream_);
        cudaStreamSynchronize(stream_);
      } else {
        feat_engine_->infer({{"left", buf_left_}, {"right", buf_right_}}, stream_);
        cudaStreamSynchronize(stream_);

        std::map<std::string, void*> post_inputs;
        for (auto& info : post_engine_->getInputInfos()) {
          void* ptr = feat_engine_->getOutputPtr(info.name);
          if (ptr) post_inputs[info.name] = ptr;
        }
        post_engine_->infer(post_inputs, stream_);
        cudaStreamSynchronize(stream_);
      }
    }
    RCLCPP_INFO(get_logger(), "Warmup done");
  }

  // Config
  int H_, W_, max_disp_;
  int input_h_ = 0, input_w_ = 0;
  bool disp_is_fp16_ = false;
  bool single_engine_ = false;
  bool compressed_ = false;

  // Intrinsics (scaled to engine resolution)
  double fx_, fy_, cx_, cy_;
  double baseline_, zfar_, zmin_, fb_;

  // TRT engines
  std::unique_ptr<fast_ffs::TrtEngine> feat_engine_, post_engine_;
  std::unique_ptr<fast_ffs::TrtEngine> single_engine_ptr_;

  // GPU buffers
  uint8_t* pin_left_ = nullptr;
  uint8_t* pin_right_ = nullptr;
  uint8_t* snap_left_ = nullptr;
  uint8_t* snap_right_ = nullptr;
  uint8_t* gpu_left_raw_ = nullptr;
  uint8_t* gpu_right_raw_ = nullptr;
  float* buf_left_ = nullptr;
  float* buf_right_ = nullptr;
  float* disp_f32_ = nullptr;
  float* blur_temp_ = nullptr;
  float* ema_buf_ = nullptr;

  // Host buffers
  float* disp_host_ = nullptr;
  uint8_t* gray_host_ = nullptr;
  float* depth_host_ = nullptr;

  // Profiling
  cudaEvent_t events_[NUM_EVENTS];
  float cpu_norm_accum_ = 0, cpu_jpeg_accum_ = 0;
  float cpu_pub_accum_ = 0, cpu_total_accum_ = 0;
  std::vector<uchar> jpeg_buf_;

  // CUDA
  cudaStream_t stream_;

  // Threading
  uint64_t stereo_seq_{0};  // guarded by mtx_
  std::atomic<bool> running_{false};
  bool first_frame_;
  int frame_count_;
  std::mutex mtx_;
  std::condition_variable frame_cv_;
  std::thread infer_thread_;

  // ROS2 — raw subscribers
  message_filters::Subscriber<sensor_msgs::msg::Image> sub_left_, sub_right_;
  std::shared_ptr<message_filters::TimeSynchronizer<
      sensor_msgs::msg::Image, sensor_msgs::msg::Image>> sync_;
  // ROS2 — compressed subscribers
  message_filters::Subscriber<sensor_msgs::msg::CompressedImage> sub_left_c_, sub_right_c_;
  std::shared_ptr<message_filters::TimeSynchronizer<
      sensor_msgs::msg::CompressedImage, sensor_msgs::msg::CompressedImage>> sync_c_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr pub_gray_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_depth_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_pcl_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LiveFfsNode>();
  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 2);
  exec.add_node(node);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}
