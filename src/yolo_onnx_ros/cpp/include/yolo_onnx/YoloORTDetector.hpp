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
  YoloORTDetector(std::string model_path,
                  Device device,
                  TaskType task_type = TaskType::DETECTION,
                  YoloThresholds yolo_thresholds = YoloThresholds(),
                  ORTConfig config = ORTConfig())
    : YoloORTInterface(model_path, device, task_type, config), yolo_thresholds_(yolo_thresholds) {};
  YoloORTDetector(const YoloORTDetector&) = delete;
  YoloORTDetector& operator=(const YoloORTDetector&) = delete;
  ~YoloORTDetector() {};

  void Detect(const cv::Mat& input_image, std::vector<BoundingBox>& bounding_boxes);

 protected:
  void Preprocess(const cv::Mat& input_image, Ort::Value& input_tensor);
  void Postprocess(const cv::Mat& input_image, std::vector<Ort::Value>& output_tensor);

 private:
  Ort::Value input_tensor_{nullptr};   /// Model input tensor.
  Ort::Value output_tensor_{nullptr};  /// Model output tensor.
  YoloThresholds yolo_thresholds_;     /// YOLO thresholds for confidence and NMS.

  static constexpr InputShape input_blob_shape_{1, 3, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE};  /// Model input shape.
  
  cv::Size raw_input_image_size_{};  /// Original input image size before preprocessing.
  cv::Mat letterboxed_image_{};                 /// Letterboxed image for inference.
  cv::Mat input_blob_{};                        /// Buffer backing input_tensor_
  cv::Mat output_blob_{};                       /// Buffer backing output_tensor_
  std::vector<BoundingBox> detected_bboxes_{};  /// Detected bounding boxes after postprocessing.
  std::vector<BoundingBox> filtered_bboxes_{};  /// Filtered bounding boxes after confidence thresholding.
  std::vector<BoundingBox> final_bboxes_{};     /// Final bounding boxes after NMS.
};
}  // namespace yolo_onnx