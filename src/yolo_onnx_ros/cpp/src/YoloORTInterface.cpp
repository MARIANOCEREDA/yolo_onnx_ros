#include "yolo_onnx/YoloORTInterface.hpp"

#include <fstream>

namespace yolo_onnx
{
namespace yolo_ort_interface
{

YoloORTInterface::YoloORTInterface(std::string model_path, Device device, TaskType task_type, ORTConfig config)
  : model_path_(std::move(model_path)),
    device_(device),
    task_type_(task_type),
    ort_config_(config),
    env_(ORT_LOGGING_LEVEL_WARNING, "YoloORTInterface"),
    session_options_(nullptr),
    session_(nullptr)
{
  if (!ValidateModelFile(model_path_))
  {
    throw std::runtime_error("Invalid model file path: " + model_path_);
  }

  session_options_ = CreateSessionOptions();
  session_ = LoadModel();
  InitializeModelIO();
}

bool YoloORTInterface::ValidateModelFile(std::string model_path)
{
  std::ifstream file(model_path);
  std::string file_extension = model_path.substr(model_path.find_last_of(".") + 1);
  bool extension_ok = (file_extension == "onnx");
  return file.good() && extension_ok;
}

Ort::SessionOptions YoloORTInterface::CreateSessionOptions()
{
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(ort_config_.intra_op_num_threads);
  session_options.SetGraphOptimizationLevel(ort_config_.optimization_level);
  if (device_.type == DeviceType::GPU)
  {
    Ort::GetApi().CreateCUDAProviderOptions(&cuda_provider_);
    /**
     * NOTE:
     * 1. ORT (ONNX Runtime) knows which matrix operations should be performed, but it does not know
     * how to perform them on the GPU.
     * 2. The CUDA provider options tell ORT how to perform these operations on the GPU.
     * 3. For instance, the CUDA execution provider know how to map matrix operations to CUDA
     * kernels, manage GPU memory, and optimize data transfer between CPU and GPU.
     * Check the tutorial: https://youtu.be/Wp5PaRpudlk?t=181
     */
    session_options.AppendExecutionProvider_CUDA_V2(*cuda_provider_);
  }

  return session_options;
}

Ort::Session YoloORTInterface::LoadModel()
{
  return Ort::Session(env_, model_path_.c_str(), session_options_);
}

void YoloORTInterface::InitializeModelIO()
{
  Ort::AllocatorWithDefaultOptions allocator;

  for (size_t i = 0; i < session_.GetInputCount(); ++i)
  {
    kExpectedYoloDetectionTensorShape.push_back(session_.GetInputNameAllocated(i, allocator));
    input_layer_names_.push_back(kExpectedYoloDetectionTensorShape.back().get());
  }

  for (size_t i = 0; i < session_.GetOutputCount(); ++i)
  {
    output_layer_names_ptrs_.push_back(session_.GetOutputNameAllocated(i, allocator));
    output_layer_names_.push_back(output_layer_names_ptrs_.back().get());
  }
}
}  // namespace yolo_ort_interface
}  // namespace yolo_onnx