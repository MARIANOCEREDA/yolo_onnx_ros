#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "yolo_onnx/YoloUtils.hpp"

namespace yolo_onnx
{
namespace yolo_ort_interface
{

class YoloORTInterface
{
 public:
  /**
   * @brief Construct and initialize a YoloORTInterface instance.
   * @param model_path Filesystem path to the ONNX model file (.onnx).
   * @param device     Target execution device (CPU or GPU).
   * @param task_type  Inference task (DETECTION or SEGMENTATION).
   * @param config     Optional ONNX Runtime session configuration.
   * @throws std::runtime_error if the model file is invalid or cannot be loaded.
   */
  YoloORTInterface(std::string model_path, Device device, TaskType task_type, ORTConfig config = ORTConfig());
  YoloORTInterface(const YoloORTInterface&) = delete;
  YoloORTInterface& operator=(const YoloORTInterface&) = delete;

  /**
   * @brief Destroy the YoloORTInterface instance, releasing any CUDA provider resources.
   */
  ~YoloORTInterface()
  {
    if (cudaProvider_)
    {
      Ort::GetApi().ReleaseCUDAProviderOptions(cudaProvider_);
    }
  }

 protected:
  /**
   * @brief Return a reference to the underlying ORT session.
   * @return Reference to the active @c Ort::Session.
   */
  Ort::Session& GetSession()
  {
    return session_;
  }

  /**
   * @brief Return the cached input node names for the loaded model.
   * @return Reference to the vector of input layer name C-strings.
   */
  std::vector<const char*>& GetInputNames()
  {
    return input_layer_names_;
  }

  /**
   * @brief Return the cached output node names for the loaded model.
   * @return Reference to the vector of output layer name C-strings.
   */
  std::vector<const char*>& GetOutputNames()
  {
    return output_layer_names_;
  }

  /**
   * @brief Return the device configuration for this interface.
   * @return Device struct containing type and ID.
   */
  Device GetDevice() const
  {
    return device_;
  }

  /**
   * @brief Return the task type for this interface.
   * @return TaskType enum value indicating the type of YOLO task (detection or segmentation).
   */
  TaskType GetTaskType()
  {
    return task_type_;
  }

 private:
  /**
   * @brief Build ORT session options from the stored config and device type.
   * @return Configured @c Ort::SessionOptions ready to create a session.
   */
  Ort::SessionOptions CreateSessionOptions();

  /**
   * @brief Load the ONNX model into a new ORT session.
   * @return Initialized @c Ort::Session bound to the model file.
   */
  Ort::Session LoadModel();

  /**
   * @brief Query the session for input/output node names and cache them.
   *
   * Populates @c input_layer_names_ and @c output_layer_names_ from the
   * loaded session metadata.
   */
  void InitializeModelIO();

  /**
   * @brief Check that a model file path is accessible and has a .onnx extension.
   * @param model_path Path to the candidate model file.
   * @return @c true if the file exists and has the correct extension.
   */
  static bool ValidateModelFile(std::string model_path);

 private:
  std::string model_path_;  // Path to the ONNX model file.
  Device device_;           // Device (type and ID).
  TaskType task_type_;      // Task type (e.g., detection or segmentation).
  ORTConfig ort_config_;    // Configuration for ONNX Runtime.

  Ort::Env env_{nullptr};  /// ONNX Runtime logging environment.

  Ort::SessionOptions session_options_{nullptr};      /// Session options used to construct the runtime session.
  Ort::Session session_{nullptr};                     /// Active ONNX Runtime session.
  OrtCUDAProviderOptionsV2* cudaProvider_ = nullptr;  /// CUDA provider configuration handle, set only for GPU mode.

  /// Owns the allocated name strings — must outlive the const char* vectors below.
  std::vector<Ort::AllocatedStringPtr> kExpectedYoloDetectionTensorShape{};
  std::vector<Ort::AllocatedStringPtr> output_layer_names_ptrs_{};

  std::vector<const char*> input_layer_names_{};   /// Input node names passed to ONNX Runtime.
  std::vector<const char*> output_layer_names_{};  /// Output node names passed to ONNX Runtime.
};
}  // namespace yolo_ort_interface
}  // namespace yolo_onnx