#include "yolo_onnx/YoloORTDetector.hpp"

#include <onnxruntime_cxx_api.h>

using namespace yolo_onnx::yolo_ort_interface;

namespace yolo_onnx
{

void YoloORTDetector::Detect(const cv::Mat& input_image,
                             std::vector<BoundingBox>& bounding_boxes,
                             ChannelOrder input_order)
{
  raw_input_image_size_ = cv::Size(input_image.cols, input_image.rows);
  Preprocess(input_image, input_tensor_, input_order);

  auto& session = GetSession();
  auto& input_node_names = GetInputNames();
  auto& output_node_names = GetOutputNames();

  std::vector<Ort::Value> output_tensors =
    session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor_, 1, output_node_names.data(), 1);

  Postprocess(input_image, output_tensors);

  bounding_boxes = final_bboxes_;
}

void YoloORTDetector::Preprocess(const cv::Mat& input_image, Ort::Value& input_tensor, ChannelOrder input_order)
{
  LetterboxImage(input_image, letterboxed_image_, cv::Size(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE));

  // 4-channel input: drop alpha on the 640x640 letterbox, not on the full-res frame. BGRA2BGR keeps the
  // channel order (it only drops the 4th channel), so it is valid for RGBA too. letterboxed_3ch_ is reused.
  const cv::Mat* blob_source = &letterboxed_image_;
  if (letterboxed_image_.channels() == 4)
  {
    cv::cvtColor(letterboxed_image_, letterboxed_3ch_, cv::COLOR_BGRA2BGR);
    blob_source = &letterboxed_3ch_;
  }
  else if (letterboxed_image_.channels() != 3)
  {
    throw std::runtime_error("Input image must have 3 or 4 channels.");
  }

  // Channel reorder to RGB is fused into ImageToBlob (no separate cvtColor pass).
  ImageToBlob(*blob_source, input_blob_, input_blob_shape_, input_order);

  auto device = GetDevice();
  BlobToONNXTensor(input_blob_, input_tensor, input_blob_shape_, device);
}

void YoloORTDetector::Postprocess(const cv::Mat& input_image, std::vector<Ort::Value>& output_tensor)
{
  auto task_type = GetTaskType();
  OutputTensorToBlob(output_tensor, output_blob_, task_type);

  detected_bboxes_.clear();
  GetBoxesFromDetection(detected_bboxes_, output_blob_, input_image.size(), task_type);

  RemoveLetterboxOffset(
    detected_bboxes_, cv::Size(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), raw_input_image_size_, letterboxed_image_.size());

  filtered_bboxes_.clear();
  FilterBoxesByConfidence(detected_bboxes_, filtered_bboxes_, yolo_thresholds_.confidence_threshold);

  final_bboxes_.clear();
  NonMaxSuppression(filtered_bboxes_, final_bboxes_, yolo_thresholds_.nms_threshold);
}

}  // namespace yolo_onnx