#pragma once

#include <string>
#include <onnxruntime_cxx_api.h>

namespace yolo_onnx
{
    enum class TaskType
    {
        DETECTION,
        SEGMENTATION,
    };

    enum class DeviceType
    {
        CPU,
        GPU,
    };

    struct ORTConfig
    {
        GraphOptimizationLevel optimization_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
        int intra_op_num_threads = 1;
    };

    class YoloOnnx
    {
    public:
        YoloOnnx(
            std::string model_path,
            DeviceType device,
            TaskType task_type,
            ORTConfig config = ORTConfig()
        );
        ~YoloOnnx()
        {
            if (cudaProvider_)
            {
                Ort::GetApi().ReleaseCUDAProviderOptions(cudaProvider_);
            }
        }

        Ort::SessionOptions CreateSessionOptions();
        Ort::Session LoadModel();

        static bool validateModelFile(std::string model_path);

    private:
        // Path to the ONNX model file.
        std::string model_path_;
        // Device type (e.g., CPU or GPU).
        DeviceType device_;
        // Type of task (detection or segmentation).
        TaskType task_type_;
        ORTConfig ort_config_;

        /// ONNX Runtime logging environment.
        Ort::Env env_{nullptr};
        /// Session options used to construct the runtime session.
        Ort::SessionOptions session_options_{nullptr};
        /// Active ONNX Runtime session.
        Ort::Session session_{nullptr};
        /// CUDA provider configuration handle, set only for GPU mode.
        OrtCUDAProviderOptionsV2 *cudaProvider_ = nullptr;
    };
}