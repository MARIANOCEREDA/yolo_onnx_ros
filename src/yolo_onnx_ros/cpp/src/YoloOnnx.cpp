#include "yolo_onnx/YoloOnnx.hpp"
#include <fstream>

namespace yolo_onnx
{
    YoloOnnx::YoloOnnx(
        std::string model_path,
        DeviceType device,
        TaskType task_type,
        ORTConfig config)
        : model_path_(std::move(model_path)),
          device_(device),
          task_type_(task_type),
          ort_config_(config),
          env_(ORT_LOGGING_LEVEL_WARNING, "YoloOnnx"),
          session_options_(nullptr),
          session_(nullptr)
    {
        if (!validateModelFile(model_path_))
        {
            throw std::runtime_error("Invalid model file path: " + model_path_);
        }

        session_options_ = CreateSessionOptions();
        session_ = LoadModel();
    }

    bool YoloOnnx::validateModelFile(std::string model_path)
    {
        std::ifstream file(model_path);
        std::string file_extension = model_path.substr(model_path.find_last_of(".") + 1);
        bool extension_ok = (file_extension == "onnx");
        return file.good() && extension_ok;
    }

    Ort::SessionOptions YoloOnnx::CreateSessionOptions()
    {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(ort_config_.intra_op_num_threads);
        session_options.SetGraphOptimizationLevel(ort_config_.optimization_level);
        if (device_ == DeviceType::GPU)
        {
            Ort::GetApi().CreateCUDAProviderOptions(&cudaProvider_);
            /**
             * NOTE:
             * 1. ORT (ONNX Runtime) knows which matrix operations should be performed, but it does not know
             * how to perform them on the GPU.
             * 2. The CUDA provider options tell ORT how to perform these operations on the GPU.
             * 3. For instance, the CUDA execution provider know how to map matrix operations to CUDA
             * kernels, manage GPU memory, and optimize data transfer between CPU and GPU.
             * Check the tutorial: https://youtu.be/Wp5PaRpudlk?t=181
             */
            session_options.AppendExecutionProvider_CUDA_V2(*cudaProvider_);
        }

        return session_options;
    }

    Ort::Session YoloOnnx::LoadModel()
    {
        return Ort::Session(env_, model_path_.c_str(), session_options_);
    }
}