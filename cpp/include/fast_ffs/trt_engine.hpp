#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace fast_ffs {

struct TensorInfo {
  std::string name;
  nvinfer1::Dims dims;
  nvinfer1::DataType dtype;
  size_t bytes;
  bool is_input;
};

class TrtEngine {
public:
  TrtEngine(const std::string& engine_path);
  ~TrtEngine();

  // Run inference. input_ptrs maps tensor name → device pointer.
  // Output buffers are pre-allocated internally.
  bool infer(const std::map<std::string, void*>& input_ptrs, cudaStream_t stream);

  void* getOutputPtr(const std::string& name) const;
  TensorInfo getTensorInfo(const std::string& name) const;
  std::vector<TensorInfo> getInputInfos() const;
  std::vector<TensorInfo> getOutputInfos() const;

private:
  struct Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override;
  };

  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  std::map<std::string, TensorInfo> tensors_;
  std::map<std::string, void*> output_bufs_;  // GPU buffers for outputs
};

}  // namespace fast_ffs
