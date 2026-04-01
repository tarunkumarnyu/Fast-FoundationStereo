cmake_minimum_required(VERSION 3.18)
project(fast_ffs LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 87)  # Orin NX SM 8.7

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(message_filters REQUIRED)

# TensorRT
find_library(NVINFER nvinfer PATHS /usr/lib/aarch64-linux-gnu)
find_library(NVINFER_PLUGIN nvinfer_plugin PATHS /usr/lib/aarch64-linux-gnu)
find_path(NVINFER_INCLUDE NvInfer.h PATHS /usr/include/aarch64-linux-gnu)

add_executable(live_ffs_node
  src/live_ffs_node.cpp
  src/trt_engine.cpp
  src/cuda_kernels.cu
)

target_include_directories(live_ffs_node PRIVATE
  include
  ${NVINFER_INCLUDE}
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(live_ffs_node
  ${NVINFER}
  CUDA::cudart
  ${OpenCV_LIBS}
  yaml-cpp
)

ament_target_dependencies(live_ffs_node
  rclcpp sensor_msgs std_msgs message_filters
)

install(TARGETS live_ffs_node DESTINATION lib/${PROJECT_NAME})

ament_package()
