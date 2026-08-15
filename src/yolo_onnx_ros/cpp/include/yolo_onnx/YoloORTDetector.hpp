#pragma once

#include <onnxruntime_cxx_api.h>

#include "yolo_onnx/YoloORTInterface.hpp"
#include "yolo_onnx/YoloUtils.hpp"

using namespace yolo_onnx::yolo_ort_interface;

namespace yolo_onnx
{

class YoloORTDetector : public YoloORTInterface
{
 public:
  YoloORTDetector(std::string model_path, Device device, TaskType task_type, ORTConfig config = ORTConfig())
    : YoloORTInterface(model_path, device, task_type, config) {};
  YoloORTDetector(const YoloORTDetector&) = delete;
  YoloORTDetector& operator=(const YoloORTDetector&) = delete;
  ~YoloORTDetector() {};

  void Detect(const cv::Mat& input_image, std::vector<BoundingBox>& bounding_boxes);

 protected:
  void Preprocess(const cv::Mat& input_image, Ort::Value& input_tensor);
  void Postprocess(const cv::Mat& input_image, std::vector<Ort::Value>& output_tensor);

 private:
  Ort::Value input_tensor_{nullptr};                                                        /// Model input tensor.
  Ort::Value output_tensor_{nullptr};                                                       /// Model output tensor.
  cv::Mat letterboxed_image_{};                                                                 /// Letterboxed image for inference.
  cv::Mat input_blob_{};                                                                    /// Buffer backing input_tensor_ — must outlive the tensor.
  static constexpr InputShape input_blob_shape_{1, 3, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE};  /// Model input shape.
};
}  // namespace yolo_onnx