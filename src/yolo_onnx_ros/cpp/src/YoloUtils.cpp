#include "yolo_onnx/YoloUtils.hpp"

#include <onnxruntime_cxx_api.h>

namespace yolo_onnx
{
void BlobToONNXTensor(const cv::Mat& blob, Ort::Value& tensor, const InputShape& input_shape, const Device& device)
{
  std::vector<int64_t> shape = {input_shape.batch, input_shape.channels, input_shape.height, input_shape.width};

  Ort::MemoryInfo memory_info{nullptr};
  if (device.type == DeviceType::GPU)
  {
    memory_info = Ort::MemoryInfo("Cuda", OrtDeviceAllocator, device.device_id, OrtMemTypeDefault);
  }
  else
  {
    memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  }
  tensor = Ort::Value::CreateTensor<float>(
    memory_info, reinterpret_cast<float*>(blob.data), blob.total(), shape.data(), shape.size());
}

void LetterboxImage(const cv::Mat& input_image, cv::Mat& output_image, cv::Size target_size, cv::Scalar fill_color)
{
  // 1. Calculate the scaling factor to maintain aspect ration.
  float scaling_factor = std::min(static_cast<float>(target_size.width) / input_image.cols,
                                  static_cast<float>(target_size.height) / input_image.rows);

  // 2. Resize the image with the scaling factor.
  cv::Mat resized_input_image;
  cv::resize(input_image, resized_input_image, cv::Size(), scaling_factor, scaling_factor, cv::INTER_LINEAR);

  // 3. Output image creation
  output_image = cv::Mat(target_size, input_image.type(), fill_color);

  // 4. Copy the resized image into the center of the output image.
  int x_offset = (target_size.width - resized_input_image.cols) / 2;
  int y_offset = (target_size.height - resized_input_image.rows) / 2;
  resized_input_image.copyTo(
    output_image(cv::Rect(x_offset, y_offset, resized_input_image.cols, resized_input_image.rows)));
}

void ImageToBlob(const cv::Mat& image, cv::Mat& blob, const InputShape& input_shape)
{
  // 1. Convert the image to float and normalize it to [0, 1].
  cv::Mat float_image;
  image.convertTo(float_image, CV_32F, 1.0 / 255.0);

  // 2. Create a blob with the shape (batch, channels, height, width).
  blob = cv::dnn::blobFromImage(
    float_image, 1.0, cv::Size(input_shape.width, input_shape.height), cv::Scalar(), true, false);
}

std::vector<int64_t> GetTensorShape(const Ort::Value& tensor)
{
  auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
  return tensor_info.GetShape();
}

bool ValidYoloOutputTensor(const Ort::Value& tensor)
{
  auto shape = GetTensorShape(tensor);

  if (shape.size() != 3)
  {
    return false;
  }
  return true;
}

void OutputTensorToBlob(std::vector<Ort::Value>& tensor, cv::Mat& blob)
{
  std::vector<int64_t> shape = GetTensorShape(tensor.front());

  if (shape.size() != kExpectedYoloOutputTensorRank)
  {
    throw std::runtime_error("Output tensor must have 3 dimensions (batch, channels, boxes).");
  }

  int batch = static_cast<int>(shape[0]);
  int box_coords_classes = static_cast<int>(shape[1]);
  int predictions = static_cast<int>(shape[2]);

  if (batch != kExpectedYoloDetectionTensorShape[0])
  {
    throw std::runtime_error("Batch size must be 1 for YOLO output tensor.");
  }

  if (box_coords_classes != kExpectedYoloDetectionTensorShape[1])
  {
    throw std::runtime_error("Number of classes and boxes coordinates must be 84 for YOLO output tensor.");
  }

  if (predictions != kExpectedYoloDetectionTensorShape[2])
  {
    throw std::runtime_error("Number of anchors must be 8400 for YOLO output tensor.");
  }

  float* output_data_ptr = tensor.front().GetTensorMutableData<float>();
  blob = cv::Mat(
    box_coords_classes, predictions, CV_32FC1, output_data_ptr);  // cv::Mat with shape (84, 8400) and type CV_32F
}

void GetBoxesFromDetection(const cv::Mat& detection_blob,
                           std::vector<BoundingBox>& boxes,
                           const cv::Size& original_image_size)
{
  boxes.clear();
  if (detection_blob.empty() || detection_blob.rows != kExpectedYoloDetectionTensorShape[1] ||
      detection_blob.cols != kExpectedYoloDetectionTensorShape[2])
  {
    throw std::runtime_error((std::string("Invalid detection blob shape. Expected shape: ") +
                              std::to_string(kExpectedYoloDetectionTensorShape[1]) + " x " +
                              std::to_string(kExpectedYoloDetectionTensorShape[2]) + ", got: " +
                              std::to_string(detection_blob.rows) + " x " + std::to_string(detection_blob.cols)));
  }

  int attrs_count =
    detection_blob.rows;    // Number of attributes per detection (e.g., x, y, width, height, confidence, class scores)
  int class_start_col = 4;  // Assuming the first 4 columns are bbox

  for (int i = 0; i < detection_blob.cols; i++)
  {
    const float center_x = detection_blob.at<float>(0, i);
    const float center_y = detection_blob.at<float>(1, i);
    const float width = detection_blob.at<float>(2, i);
    const float height = detection_blob.at<float>(3, i);

    float max_class_confidence = -1.0f;
    int class_id = 0;
    for (int c = class_start_col; c < attrs_count; ++c)
    {
      const float score = detection_blob.at<float>(c, i);
      if (score > max_class_confidence)
      {
        max_class_confidence = score;
        class_id = c - class_start_col;
      }
    }

    const float confidence = max_class_confidence;

    const float top_left_x = center_x - width / 2.0f;
    const float top_left_y = center_y - height / 2.0f;

    const cv::Rect bbox(
      static_cast<int>(top_left_x), static_cast<int>(top_left_y), static_cast<int>(width), static_cast<int>(height));

    BoundingBox box(top_left_x, top_left_y, width, height, confidence, class_id);
    box.bbox = bbox;
    boxes.emplace_back(std::move(box));
  }
}

void FilterBoxesByConfidence(const std::vector<BoundingBox>& input_boxes,
                             std::vector<BoundingBox>& filtered_boxes,
                             float confidence_threshold)
{
  filtered_boxes.reserve(input_boxes.size());
  for (const auto& box : input_boxes)
  {
    if (box.confidence >= confidence_threshold)
    {
      filtered_boxes.emplace_back(box);
    }
  }
}

void NonMaxSuppression(const std::vector<BoundingBox>& input_boxes,
                       std::vector<BoundingBox>& output_boxes,
                       float iou_threshold)
{
  std::vector<cv::Rect> nms_input_boxes;
  nms_input_boxes.reserve(input_boxes.size());
  std::vector<float> confidences;
  confidences.reserve(input_boxes.size());
  for (const auto& box : input_boxes)
  {
    nms_input_boxes.emplace_back(box.bbox);
    confidences.emplace_back(box.confidence);
  }

  std::vector<int> filtered_idxs;
  cv::dnn::NMSBoxes(nms_input_boxes, confidences, 0.0f, iou_threshold, filtered_idxs);

  output_boxes.clear();
  output_boxes.reserve(filtered_idxs.size());
  for (const int idx : filtered_idxs)
  {
    output_boxes.emplace_back(input_boxes[idx]);
  }
}

void RemoveLetterboxOffset(std::vector<BoundingBox>& boxes,
                           const cv::Size& input_yolo_size,
                           const cv::Size& original_image_size,
                           const cv::Size& letterboxed_image_size)
{
  const float scale_x = static_cast<float>(input_yolo_size.width) / static_cast<float>(original_image_size.width);
  const float scale_y = static_cast<float>(input_yolo_size.height) / static_cast<float>(original_image_size.height);
  const float scale = std::min(scale_x, scale_y);

  const float pad_x = (static_cast<float>(input_yolo_size.width) - original_image_size.width * scale) * 0.5f;
  const float pad_y = (static_cast<float>(input_yolo_size.height) - original_image_size.height * scale) * 0.5f;

  const float max_x = static_cast<float>(original_image_size.width - 1);
  const float max_y = static_cast<float>(original_image_size.height - 1);

  for (auto& box : boxes)
  {
    const float x1 = std::clamp((box.x - pad_x) / scale, 0.0f, max_x);
    const float y1 = std::clamp((box.y - pad_y) / scale, 0.0f, max_y);
    const float x2 = std::clamp((box.x + box.width - pad_x) / scale, 0.0f, max_x);
    const float y2 = std::clamp((box.y + box.height - pad_y) / scale, 0.0f, max_y);

    box.x = x1;
    box.y = y1;
    box.width = std::max(0.0f, x2 - x1);
    box.height = std::max(0.0f, y2 - y1);
    box.bbox = cv::Rect(
      static_cast<int>(box.x), static_cast<int>(box.y), static_cast<int>(box.width), static_cast<int>(box.height));
  }
}

}  // namespace yolo_onnx