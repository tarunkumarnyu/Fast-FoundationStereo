/**
 * C++ Fast-FoundationStereo ROS2 node.
 * Pipelined TRT inference with CUDA pre/post processing.
 * Targets 25-30Hz on Jetson Orin NX at 480x320.
 */
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <yaml-cpp/yaml.h>

#include "fast_ffs/trt_engine.hpp"
#include "fast_ffs/cuda_kernels.cuh"

using namespace std::chrono_literals;

class LiveFfsNode : public rclcpp::Node {
public:
  LiveFfsNode() : Node("live_ffs_cpp") {
    // Parameters
    declare_parameter("engine_dir", std::string(""));
    declare_parameter("ns", std::string("/race6/cam1"));
    auto engine_dir = get_parameter("engine_dir").as_string();
    auto ns = get_parameter("ns").as_string();

    // Load config
    auto yaml = YAML::LoadFile(engine_dir + "/onnx.yaml");
    H_ = yaml["image_size"][0].as<int>();
    W_ = yaml["image_size"][1].as<int>();
    max_disp_ = yaml["max_disp"].as<int>();
    cv_group_ = yaml["cv_group"].as<int>();
    RCLCPP_INFO(get_logger(), "Config: %dx%d, max_disp=%d, cv_group=%d", W_, H_, max_disp_, cv_group_);

    // Load TRT engines
    feat_engine_ = std::make_unique<fast_ffs::TrtEngine>(engine_dir + "/feature_runner.engine");
    post_engine_ = std::make_unique<fast_ffs::TrtEngine>(engine_dir + "/post_runner.engine");

    // Get feature output info for GWC
    auto feat04_info = feat_engine_->getTensorInfo("features_left_04");
    feat_C_ = feat04_info.dims.d[1];  // channels
    feat_H_ = feat04_info.dims.d[2];  // H/4
    feat_W_ = feat04_info.dims.d[3];  // W/4
    RCLCPP_INFO(get_logger(), "Feature 04: C=%d, H=%d, W=%d", feat_C_, feat_H_, feat_W_);

    // Allocate GPU buffers
    alloc_buffers();

    // Create CUDA streams and events
    cudaStreamCreate(&stream_main_);
    cudaStreamCreate(&stream_post_);
    cudaEventCreate(&feat_done_);
    cudaEventCreate(&post_done_);

    // Warmup
    warmup();

    // ROS2 subscribers
    auto qos = rclcpp::QoS(1).best_effort();
    sub_left_ = create_subscription<sensor_msgs::msg::Image>(
        ns + "/infra1/image_rect_raw", qos,
        [this](sensor_msgs::msg::Image::SharedPtr msg) { cb_left(msg); });
    sub_right_ = create_subscription<sensor_msgs::msg::Image>(
        ns + "/infra2/image_rect_raw", qos,
        [this](sensor_msgs::msg::Image::SharedPtr msg) { cb_right(msg); });

    // Publisher
    auto pub_qos = rclcpp::QoS(1).reliable();
    pub_ = create_publisher<sensor_msgs::msg::CompressedImage>(
        "/ffs/disp_gray/compressed", pub_qos);

    // Inference thread
    running_ = true;
    first_frame_ = true;
    frame_count_ = 0;
    infer_thread_ = std::thread(&LiveFfsNode::infer_loop, this);

    RCLCPP_INFO(get_logger(), "Ready — C++ pipeline");
  }

  ~LiveFfsNode() {
    running_ = false;
    cv_.notify_all();
    if (infer_thread_.joinable()) infer_thread_.join();
    free_buffers();
    cudaStreamDestroy(stream_main_);
    cudaStreamDestroy(stream_post_);
    cudaEventDestroy(feat_done_);
    cudaEventDestroy(post_done_);
  }

private:
  void cb_left(sensor_msgs::msg::Image::SharedPtr msg) {
    if (input_h_ == 0) {
      input_h_ = msg->height;
      input_w_ = msg->width;
      // Reallocate pinned buffers if size differs
      if (input_h_ * input_w_ != 480 * 848) {
        cudaFreeHost(pin_left_);
        cudaFreeHost(pin_right_);
        cudaMallocHost(&pin_left_, input_h_ * input_w_);
        cudaMallocHost(&pin_right_, input_h_ * input_w_);
        cudaFree(gpu_left_raw_);
        cudaFree(gpu_right_raw_);
        cudaMalloc(&gpu_left_raw_, input_h_ * input_w_);
        cudaMalloc(&gpu_right_raw_, input_h_ * input_w_);
      }
      RCLCPP_INFO(get_logger(), "Input: %dx%d", input_w_, input_h_);
    }
    std::memcpy(pin_left_, msg->data.data(), msg->data.size());
    has_left_ = true;
    cv_.notify_one();
  }

  void cb_right(sensor_msgs::msg::Image::SharedPtr msg) {
    if (input_h_ == 0) return;
    std::memcpy(pin_right_, msg->data.data(), msg->data.size());
    has_right_ = true;
  }

  void infer_loop() {
    auto t_start = std::chrono::steady_clock::now();
    __half* prev_disp_ptr = nullptr;  // pipelined: previous frame's post output
    bool post_pending = false;

    while (running_) {
      // Wait for new frame
      {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait_for(lock, 100ms, [this] { return has_left_.load() && has_right_.load(); });
      }
      if (!has_left_ || !has_right_) continue;

      auto t0 = std::chrono::steady_clock::now();

      // === H2D: pinned → GPU ===
      cudaMemcpyAsync(gpu_left_raw_, pin_left_, input_h_ * input_w_,
                       cudaMemcpyHostToDevice, stream_main_);
      cudaMemcpyAsync(gpu_right_raw_, pin_right_, input_h_ * input_w_,
                       cudaMemcpyHostToDevice, stream_main_);

      // === Preprocess: resize + gray→3ch ===
      fast_ffs::preprocess_gpu(gpu_left_raw_, buf_left_, input_h_, input_w_, H_, W_, stream_main_);
      fast_ffs::preprocess_gpu(gpu_right_raw_, buf_right_, input_h_, input_w_, H_, W_, stream_main_);

      // === Feature engine ===
      feat_engine_->infer({{"left", buf_left_}, {"right", buf_right_}}, stream_main_);
      cudaEventRecord(feat_done_, stream_main_);

      // === GWC volume ===
      cudaStreamWaitEvent(stream_main_, feat_done_);
      auto* fl04 = (float*)feat_engine_->getOutputPtr("features_left_04");
      auto* fr04 = (float*)feat_engine_->getOutputPtr("features_right_04");
      fast_ffs::build_gwc_volume_f32(fl04, fr04, gwc_buf_, gwc_workspace_f32_,
                                      feat_C_, feat_H_, feat_W_,
                                      max_disp_ / 4, cv_group_, stream_main_);

      // === If previous post is pending, wait and process ===
      if (post_pending) {
        cudaEventSynchronize(post_done_);
        // Post-process previous frame's disparity
        float* disp_f32 = (float*)post_engine_->getOutputPtr("disp");
        process_and_publish(disp_f32);
      }

      // === Post engine (pipelined on stream_post) ===
      cudaStreamWaitEvent(stream_post_, feat_done_);
      std::map<std::string, void*> post_inputs;
      // Only pass tensors that the post engine actually expects
      for (auto& info : post_engine_->getInputInfos()) {
        if (info.name == "gwc_volume") {
          post_inputs[info.name] = gwc_buf_;
        } else {
          post_inputs[info.name] = feat_engine_->getOutputPtr(info.name);
        }
      }
      post_engine_->infer(post_inputs, stream_post_);
      cudaEventRecord(post_done_, stream_post_);
      post_pending = true;

      auto t1 = std::chrono::steady_clock::now();
      double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      frame_count_++;
      if (frame_count_ % 30 == 0) {
        double elapsed = std::chrono::duration<double>(t1 - t_start).count();
        double fps = frame_count_ / elapsed;
        RCLCPP_INFO(get_logger(), "Frame %d: %.1fms, %.1fHz", frame_count_, dt_ms, fps);
      }
    }

    // Flush last pending frame
    if (post_pending) {
      cudaEventSynchronize(post_done_);
      float* disp_f32 = (float*)post_engine_->getOutputPtr("disp");
      process_and_publish(disp_f32);
    }
  }

  void process_and_publish(float* disp_gpu) {
    // Box blur
    fast_ffs::box_blur_5x5(disp_gpu, blur_temp_, H_, W_, stream_main_);

    // Temporal EMA + normalize to uint8
    fast_ffs::temporal_normalize(
        ema_buf_, disp_gpu, gray_gpu_, H_, W_,
        0.7f, (float)max_disp_, first_frame_, stream_main_);
    first_frame_ = false;

    // D2H
    cudaMemcpyAsync(gray_host_, gray_gpu_, H_ * W_, cudaMemcpyDeviceToHost, stream_main_);
    cudaStreamSynchronize(stream_main_);

    // JPEG encode
    cv::Mat gray_mat(H_, W_, CV_8UC1, gray_host_);
    std::vector<uchar> jpeg_buf;
    cv::imencode(".jpg", gray_mat, jpeg_buf, {cv::IMWRITE_JPEG_QUALITY, 75});

    // Publish
    auto msg = sensor_msgs::msg::CompressedImage();
    msg.header.stamp = now();
    msg.format = "jpeg";
    msg.data.assign(jpeg_buf.begin(), jpeg_buf.end());
    pub_->publish(msg);
  }

  void alloc_buffers() {
    input_h_ = 480; input_w_ = 848;
    cudaMallocHost(&pin_left_, input_h_ * input_w_);
    cudaMallocHost(&pin_right_, input_h_ * input_w_);
    cudaMalloc(&gpu_left_raw_, input_h_ * input_w_);
    cudaMalloc(&gpu_right_raw_, input_h_ * input_w_);
    cudaMalloc(&buf_left_, 3 * H_ * W_ * sizeof(float));
    cudaMalloc(&buf_right_, 3 * H_ * W_ * sizeof(float));

    // GWC buffers
    int D4 = max_disp_ / 4;
    size_t gwc_bytes = cv_group_ * D4 * feat_H_ * feat_W_ * sizeof(__half);
    cudaMalloc(&gwc_buf_, gwc_bytes);
    size_t ws = fast_ffs::gwc_workspace_bytes(feat_C_, feat_H_, feat_W_);
    cudaMalloc(&gwc_workspace_, ws);

    // Post-process buffers
    cudaMalloc(&blur_temp_, H_ * W_ * sizeof(float));
    cudaMalloc(&ema_buf_, H_ * W_ * sizeof(float));
    cudaMemset(ema_buf_, 0, H_ * W_ * sizeof(float));
    cudaMalloc(&gray_gpu_, H_ * W_);
    cudaMallocHost(&gray_host_, H_ * W_);
  }

  void free_buffers() {
    cudaFreeHost(pin_left_); cudaFreeHost(pin_right_);
    cudaFree(gpu_left_raw_); cudaFree(gpu_right_raw_);
    cudaFree(buf_left_); cudaFree(buf_right_);
    cudaFree(gwc_buf_); cudaFree(gwc_workspace_);
    cudaFree(blur_temp_); cudaFree(ema_buf_);
    cudaFree(gray_gpu_); cudaFreeHost(gray_host_);
  }

  void warmup() {
    RCLCPP_INFO(get_logger(), "Warming up TRT engines...");
    cudaMemset(buf_left_, 0, 3 * H_ * W_ * sizeof(float));
    cudaMemset(buf_right_, 0, 3 * H_ * W_ * sizeof(float));
    for (int i = 0; i < 5; ++i) {
      feat_engine_->infer({{"left", buf_left_}, {"right", buf_right_}}, stream_main_);
      cudaStreamSynchronize(stream_main_);

      // GWC
      auto* fl04 = (__half*)feat_engine_->getOutputPtr("features_left_04");
      auto* fr04 = (__half*)feat_engine_->getOutputPtr("features_right_04");
      fast_ffs::build_gwc_volume(fl04, fr04, gwc_buf_, gwc_workspace_,
                                  feat_C_, feat_H_, feat_W_,
                                  max_disp_ / 4, cv_group_, stream_main_);
      cudaStreamSynchronize(stream_main_);

      // Post
      std::map<std::string, void*> post_inputs;
      for (auto& info : post_engine_->getInputInfos()) {
        if (info.name == "gwc_volume")
          post_inputs[info.name] = gwc_buf_;
        else
          post_inputs[info.name] = feat_engine_->getOutputPtr(info.name);
      }
      post_engine_->infer(post_inputs, stream_main_);
      cudaStreamSynchronize(stream_main_);
    }
    RCLCPP_INFO(get_logger(), "Warmup done");
  }

  // Config
  int H_, W_, max_disp_, cv_group_;
  int feat_C_, feat_H_, feat_W_;
  int input_h_ = 0, input_w_ = 0;

  // TRT engines
  std::unique_ptr<fast_ffs::TrtEngine> feat_engine_, post_engine_;

  // GPU buffers
  uint8_t* pin_left_ = nullptr;
  uint8_t* pin_right_ = nullptr;
  uint8_t* gpu_left_raw_ = nullptr;
  uint8_t* gpu_right_raw_ = nullptr;
  float* buf_left_ = nullptr;
  float* buf_right_ = nullptr;
  __half* gwc_buf_ = nullptr;
  __half* gwc_workspace_ = nullptr;
  float* blur_temp_ = nullptr;
  float* ema_buf_ = nullptr;
  uint8_t* gray_gpu_ = nullptr;
  uint8_t* gray_host_ = nullptr;

  // CUDA
  cudaStream_t stream_main_, stream_post_;
  cudaEvent_t feat_done_, post_done_;

  // Threading
  std::atomic<bool> has_left_{false}, has_right_{false}, running_{false};
  bool first_frame_;
  int frame_count_;
  std::mutex mtx_;
  std::condition_variable cv_;
  std::thread infer_thread_;

  // ROS2
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_left_, sub_right_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LiveFfsNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
