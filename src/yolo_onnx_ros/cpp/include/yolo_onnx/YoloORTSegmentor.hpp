#pragma once

#include <onnxruntime_cxx_api.h>

#include <chrono>

#include "yolo_onnx/YoloORTInterface.hpp"
#include "yolo_onnx/YoloUtils.hpp"

using namespace yolo_onnx::yolo_ort_interface;

namespace yolo_onnx
{

/// Wall-clock time [ms] spent in each stage of the last Segment() call (steady_clock, host side).
struct SegmentTimings
{
  double preprocess_ms = 0.0;  ///< Letterbox + alpha drop + blob + tensor wrap.
  double inference_ms = 0.0;   ///< session.Run (includes H2D/D2H copies on GPU).
  double decode_ms = 0.0;      ///< Output tensor -> boxes -> confidence filter -> NMS.
  double masks_ms = 0.0;       ///< Proto masks -> per-instance masks -> letterbox offset removal.
  double copy_out_ms = 0.0;    ///< Copy of final boxes (with masks) into the caller's vector.
};

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

  void Segment(const cv::Mat& input_image,
               std::vector<BoundingBox>& bounding_boxes,
               ChannelOrder input_order = ChannelOrder::BGR);

  /// Per-stage timings of the last Segment() call. Not thread-safe: read from the thread that calls Segment().
  const SegmentTimings& GetLastTimings() const { return timings_; }

 protected:
  void Preprocess(const cv::Mat& input_image, Ort::Value& input_tensor, ChannelOrder input_order);
  void Postprocess(const cv::Mat& input_image, std::vector<Ort::Value>& output_tensor);

 private:
  Ort::Value input_tensor_{nullptr};   /// Model input tensor.
  Ort::Value output_tensor_{nullptr};  /// Model output tensor.
  YoloThresholds yolo_thresholds_;     /// YOLO thresholds for confidence and NMS.
  SegmentTimings timings_{};           /// Per-stage timings of the last Segment() call.

  static constexpr InputShape input_blob_shape_{1, 3, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE};  /// Model input shape.

  cv::Size raw_input_image_size_{};  /// Original input image size before preprocessing.
  cv::Mat letterboxed_image_{};      /// Letterboxed image for inference.
  cv::Mat letterboxed_3ch_{};          /// Alpha-dropped letterbox (4-channel inputs only), reused.
  
  cv::Mat masks_{};                  /// Segmentation masks for detected objects.
  cv::Mat raw_proto_masks_{};

  cv::Mat input_blob_{};                        /// Buffer backing input_tensor_
  cv::Mat output_blob_{};                       /// Buffer backing output_tensor_

  std::vector<BoundingBox> detected_bboxes_{};  /// Detected bounding boxes after postprocessing.
  std::vector<BoundingBox> filtered_bboxes_{};  /// Filtered bounding boxes after confidence thresholding.
  std::vector<BoundingBox> final_bboxes_{};     /// Final bounding boxes after NMS.

  int proto_masks_height_{0};                   /// Height of the proto masks.
  int proto_masks_width_{0};                    /// Width of the proto masks.

  std::vector<cv::Rect> nms_input_boxes_{};  /// Bounding boxes used for NMS.
};
}  // namespace yolo_onnx