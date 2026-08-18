#pragma once

#include <onnxruntime_cxx_api.h>

#include "yolo_onnx/YoloORTInterface.hpp"
#include "yolo_onnx/YoloUtils.hpp"

using namespace yolo_onnx::yolo_ort_interface;

namespace yolo_onnx
{

class YoloORTSegmentor : public YoloORTInterface
{
 public:
  YoloORTSegmentor(std::string model_path,
                   Device device,
                   TaskType task_type = TaskType::SEGMENTATION,
                   YoloThresholds yolo_thresholds = YoloThresholds(),
                   ORTConfig config = ORTConfig())
    : YoloORTInterface(model_path, device, task_type, config), yolo_thresholds_(yolo_thresholds) {};
  YoloORTSegmentor(const YoloORTSegmentor&) = delete;
  YoloORTSegmentor& operator=(const YoloORTSegmentor&) = delete;
  ~YoloORTSegmentor() {};

  void Segment(const cv::Mat& input_image, std::vector<BoundingBox>& bounding_boxes);

 protected:
  void Preprocess(const cv::Mat& input_image, Ort::Value& input_tensor);
  void Postprocess(const cv::Mat& input_image, std::vector<Ort::Value>& output_tensor);

 private:
  Ort::Value input_tensor_{nullptr};   /// Model input tensor.
  Ort::Value output_tensor_{nullptr};  /// Model output tensor.
  YoloThresholds yolo_thresholds_;     /// YOLO thresholds for confidence and NMS.

  static constexpr InputShape input_blob_shape_{1, 3, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE};  /// Model input shape.

  cv::Size raw_input_image_size_{};  /// Original input image size before preprocessing.
  cv::Mat letterboxed_image_{};      /// Letterboxed image for inference.
  cv::Mat masks_{};                  /// Segmentation masks for detected objects.
  cv::Mat raw_proto_masks_{};
  cv::Mat input_blob_{};                        /// Buffer backing input_tensor_
  cv::Mat output_blob_{};                       /// Buffer backing output_tensor_
  std::vector<BoundingBox> detected_bboxes_{};  /// Detected bounding boxes after postprocessing.
  std::vector<BoundingBox> filtered_bboxes_{};  /// Filtered bounding boxes after confidence thresholding.
  std::vector<BoundingBox> final_bboxes_{};     /// Final bounding boxes after NMS.
  int proto_masks_height_{0};                   /// Height of the proto masks.
  int proto_masks_width_{0};                    /// Width of the proto masks.
};
}  // namespace yolo_onnx