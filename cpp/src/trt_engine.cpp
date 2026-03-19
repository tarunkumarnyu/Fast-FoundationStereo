#include "fast_ffs/trt_engine.hpp"
#include <fstream>
#include <iostream>
#include <cassert>
#include <numeric>

namespace fast_ffs {

void TrtEngine::Logger::log(Severity severity, const char* msg) noexcept {
  if (severity <= Severity::kWARNING)
    std::cerr << "[TRT] " << msg << std::endl;
}

static size_t dtype_size(nvinfer1::DataType dt) {
  switch (dt) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF:  return 2;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kINT8:  return 1;
    case nvinfer1::DataType::kBOOL:  return 1;
    default: return 4;
  }
}

static size_t dims_volume(const nvinfer1::Dims& d) {
  size_t vol = 1;
  for (int i = 0; i < d.nbDims; ++i) vol *= d.d[i];
  return vol;
}

TrtEngine::TrtEngine(const std::string& engine_path) {
  // Read engine file
  std::ifstream file(engine_path, std::ios::binary);
  assert(file.good() && "Failed to open engine file");
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> data(size);
  file.read(data.data(), size);

  // Create runtime and deserialize
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  engine_.reset(runtime_->deserializeCudaEngine(data.data(), size));
  assert(engine_ && "Failed to deserialize engine");
  context_.reset(engine_->createExecutionContext());
  assert(context_ && "Failed to create execution context");

  // Enumerate tensors
  int n = engine_->getNbIOTensors();
  for (int i = 0; i < n; ++i) {
    const char* name = engine_->getIOTensorName(i);
    TensorInfo info;
    info.name = name;
    info.dims = engine_->getTensorShape(name);
    info.dtype = engine_->getTensorDataType(name);
    info.is_input = (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);
    info.bytes = dims_volume(info.dims) * dtype_size(info.dtype);

    tensors_[name] = info;

    // Allocate output buffers
    if (!info.is_input) {
      void* buf = nullptr;
      cudaMalloc(&buf, info.bytes);
      output_bufs_[name] = buf;
    }

    // Set input shapes (fixed)
    if (info.is_input) {
      context_->setInputShape(name, info.dims);
    }
  }

  std::cout << "[TrtEngine] Loaded " << engine_path
            << " (" << n << " tensors, " << size / 1e6 << " MB)" << std::endl;
}

TrtEngine::~TrtEngine() {
  for (auto& [name, buf] : output_bufs_) {
    if (buf) cudaFree(buf);
  }
}

bool TrtEngine::infer(const std::map<std::string, void*>& input_ptrs, cudaStream_t stream) {
  // Set input addresses
  for (auto& [name, ptr] : input_ptrs) {
    context_->setTensorAddress(name.c_str(), ptr);
  }
  // Set output addresses
  for (auto& [name, ptr] : output_bufs_) {
    context_->setTensorAddress(name.c_str(), ptr);
  }
  return context_->enqueueV3(stream);
}

void* TrtEngine::getOutputPtr(const std::string& name) const {
  auto it = output_bufs_.find(name);
  return (it != output_bufs_.end()) ? it->second : nullptr;
}

TensorInfo TrtEngine::getTensorInfo(const std::string& name) const {
  return tensors_.at(name);
}

std::vector<TensorInfo> TrtEngine::getInputInfos() const {
  std::vector<TensorInfo> out;
  for (auto& [name, info] : tensors_)
    if (info.is_input) out.push_back(info);
  return out;
}

std::vector<TensorInfo> TrtEngine::getOutputInfos() const {
  std::vector<TensorInfo> out;
  for (auto& [name, info] : tensors_)
    if (!info.is_input) out.push_back(info);
  return out;
}

}  // namespace fast_ffs
