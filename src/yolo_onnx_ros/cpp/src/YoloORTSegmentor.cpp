#include "yolo_onnx/YoloORTSegmentor.hpp"

#include <onnxruntime_cxx_api.h>

using namespace yolo_onnx::yolo_ort_interface;

namespace yolo_onnx
{

void YoloORTSegmentor::Segment(const cv::Mat& input_image, std::vector<BoundingBox>& bounding_boxes)
{
  raw_input_image_size_ = cv::Size(input_image.cols, input_image.rows);
  Preprocess(input_image, input_tensor_);

  auto& session = GetSession();
  auto& input_node_names = GetInputNames();
  auto& output_node_names = GetOutputNames();

  std::vector<Ort::Value> output_tensors =
    session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor_, 1, output_node_names.data(), 2);

  Postprocess(input_image, output_tensors);

  bounding_boxes = final_bboxes_;
}

void YoloORTSegmentor::Preprocess(const cv::Mat& input_image, Ort::Value& input_tensor)
{
  LetterboxImage(input_image, letterboxed_image_, cv::Size(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE));

  cv::cvtColor(letterboxed_image_, letterboxed_image_, cv::COLOR_BGR2RGB);

  ImageToBlob(letterboxed_image_, input_blob_, input_blob_shape_);

  auto device = GetDevice();
  BlobToONNXTensor(input_blob_, input_tensor, input_blob_shape_, device);
}

void YoloORTSegmentor::Postprocess(const cv::Mat& input_image, std::vector<Ort::Value>& output_tensor)
{
  auto task_type = GetTaskType();
  OutputTensorToBlob(output_tensor, output_blob_, task_type);

  detected_bboxes_.clear();
  GetBoxesFromDetection(detected_bboxes_, output_blob_, input_image.size(), task_type);

  filtered_bboxes_.clear();
  FilterBoxesByConfidence(detected_bboxes_, filtered_bboxes_, yolo_thresholds_.confidence_threshold);

  final_bboxes_.clear();
  NonMaxSuppression(filtered_bboxes_, final_bboxes_, yolo_thresholds_.nms_threshold);

  raw_proto_masks_.setTo(0);
  GetMasksFromTensor(raw_proto_masks_, output_tensor, proto_masks_height_, proto_masks_width_);

  masks_.setTo(0);
  ProcessYoloMasks(
    masks_, raw_proto_masks_, final_bboxes_, input_image.size(), proto_masks_height_, proto_masks_width_);

  ProcessYoloBoxes(
    final_bboxes_, masks_, proto_masks_height_, proto_masks_width_, cv::Size(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE));

  RemoveLetterboxOffset(
    final_bboxes_, cv::Size(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), raw_input_image_size_, letterboxed_image_.size());
}

}  // namespace yolo_onnx