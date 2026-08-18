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
    memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPUInput);
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

void OutputTensorToBlob(std::vector<Ort::Value>& tensor, cv::Mat& blob, const TaskType task_type)
{
  std::vector<int64_t> shape = GetTensorShape(tensor.front());

  std::vector<int> expected_shape;
  if (task_type == TaskType::DETECTION)
  {
    expected_shape = kExpectedYoloDetectionTensorShape;
  }
  else if (task_type == TaskType::SEGMENTATION)
  {
    expected_shape = kExpectedYoloSegmentTensorShape;
  }
  else
  {
    throw std::runtime_error("Unsupported task type for YOLO output tensor.");
  }

  if (shape.size() != kExpectedYoloOutputTensorRank)
  {
    throw std::runtime_error("Output tensor must have 3 dimensions (batch, channels, boxes).");
  }

  int batch = static_cast<int>(shape[0]);
  int box_coords_classes = static_cast<int>(shape[1]);
  int predictions = static_cast<int>(shape[2]);

  if (batch != expected_shape[0])
  {
    throw std::runtime_error("Batch size must be 1 for YOLO output tensor.");
  }

  if (box_coords_classes != expected_shape[1])
  {
    throw std::runtime_error("Number of classes and boxes coordinates must be " + std::to_string(expected_shape[1]) +
                             " for YOLO output tensor.");
  }

  if (predictions != expected_shape[2])
  {
    throw std::runtime_error("Number of anchors must be " + std::to_string(expected_shape[2]) +
                             " for YOLO output tensor.");
  }

  float* output_data_ptr = tensor.front().GetTensorMutableData<float>();
  blob = cv::Mat(
    box_coords_classes, predictions, CV_32FC1, output_data_ptr);  // cv::Mat with shape (84, 8400) and type CV_32F
}

bool ValidateBlobSizeDetection(const cv::Mat& blob)
{
  if (blob.empty())
  {
    return false;
  }
  if (blob.rows != kExpectedYoloDetectionTensorShape[1] || blob.cols != kExpectedYoloDetectionTensorShape[2])
  {
    return false;
  }
  return true;
}

bool ValidateBlobSizeSegmentation(const cv::Mat& blob)
{
  if (blob.empty())
  {
    return false;
  }
  if (blob.rows != kExpectedYoloSegmentTensorShape[1] || blob.cols != kExpectedYoloSegmentTensorShape[2])
  {
    return false;
  }
  return true;
}

void GetBoxesFromDetection(std::vector<BoundingBox>& boxes,
                           const cv::Mat& detection_blob,
                           const cv::Size& original_image_size,
                           TaskType task_type)
{
  boxes.clear();
  if (task_type == TaskType::DETECTION && !ValidateBlobSizeDetection(detection_blob))
  {
    throw std::runtime_error("Invalid detection blob size for YOLO detection.");
  }
  else if (task_type == TaskType::SEGMENTATION && !ValidateBlobSizeSegmentation(detection_blob))
  {
    throw std::runtime_error("Invalid detection blob size for YOLO segmentation.");
  }

  int attrs_count = detection_blob.rows;
  int class_start_col = CLASS_START_INDEX - 1;

  for (int i = 0; i < detection_blob.cols; i++)
  {
    const float center_x = detection_blob.at<float>(0, i);
    const float center_y = detection_blob.at<float>(1, i);
    const float width = detection_blob.at<float>(2, i);
    const float height = detection_blob.at<float>(3, i);

    float max_class_confidence = -1.0f;
    int class_id = 0;

    for (int c = class_start_col; c < N_SEG_MASK_COEFFS_INDEX_START; ++c)
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

    std::vector<float> mask_coeffs;
    mask_coeffs.reserve(N_SEG_MASK_COEFFS);

    if (task_type == TaskType::SEGMENTATION)
    {
      for (size_t mask_idx = 0; mask_idx < N_SEG_MASK_COEFFS; ++mask_idx)
      {
        mask_coeffs.emplace_back(detection_blob.at<float>(N_SEG_MASK_COEFFS_INDEX_START + mask_idx, i));
      }
    }
    box.mask_coefficients = std::move(mask_coeffs);

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
};

void GetMasksFromTensor(cv::Mat& mask,
                        std::vector<Ort::Value>& inference_output_tensor,
                        int& proto_height,
                        int& proto_width)
{
  if (inference_output_tensor.size() < 2)
  {
    throw std::runtime_error("Expected at least 2 output tensors for YOLO segmentation");
  }

  std::vector<int64_t> mask_shape = inference_output_tensor[1].GetTensorTypeAndShapeInfo().GetShape();

  if (mask_shape.size() != kProtoMasksYoloOutputTensorRank)
  {
    throw std::runtime_error("Unexpected YOLO output tensor shape (expected 4 dims)");
  }

  proto_height = static_cast<int>(mask_shape[2]);
  proto_width = static_cast<int>(mask_shape[3]);

  float* output_data = inference_output_tensor[1].GetTensorMutableData<float>();
  mask = cv::Mat(static_cast<int>(mask_shape[1]), static_cast<int>(mask_shape[2] * mask_shape[3]), CV_32F, output_data);
};

void ProcessYoloMasks(cv::Mat& masks,
                      const cv::Mat& raw_proto_masks,
                      std::vector<BoundingBox>& boxes,
                      const cv::Size& original_image_size,
                      int proto_height,
                      int proto_width)
{
  const int num_boxes = static_cast<int>(boxes.size());
  if (num_boxes == 0)
  {
    return;
  }

  const int num_proto_channels = raw_proto_masks.rows;

  masks = cv::Mat(num_boxes, proto_height * proto_width, CV_32F);

  for (int i = 0; i < num_boxes; ++i)
  {
    auto& box = boxes[i];
    const std::vector<float>& mask_coeffs = box.mask_coefficients;

    cv::Mat mask_flat = cv::Mat::zeros(proto_height * proto_width, 1, CV_32F);
    for (int j = 0; j < num_proto_channels; ++j)
    {
      cv::Mat proto_channel(
        proto_height, proto_width, CV_32F, raw_proto_masks.data + j * proto_height * proto_width * sizeof(float));
      mask_flat += mask_coeffs[j] * proto_channel.reshape(1, proto_height * proto_width);
    }

    masks.row(i) = mask_flat.t();
    box.mask = mask_flat.reshape(1, proto_height);
  }
};

void ProcessYoloBoxes(std::vector<BoundingBox>& boxes,
                      const cv::Mat& proto_masks,
                      int proto_height,
                      int proto_width,
                      const cv::Size& model_input_size)
{
  if (boxes.empty() || proto_masks.empty())
  {
    return;
  }

  for (size_t i = 0; i < boxes.size(); ++i)
  {
    auto& box = boxes[i];

    // Get the mask row index (stored during parse or NMS tracking)
    if (i >= static_cast<size_t>(proto_masks.rows))
    {
      continue;
    }

    // Extract and reshape mask from flat row to 2D
    cv::Mat mask_flat = proto_masks.row(i);
    cv::Mat mask_proto = mask_flat.reshape(1, proto_height);

    // Apply sigmoid activation
    cv::Mat mask_sigmoid;
    cv::exp(-mask_proto, mask_sigmoid);
    mask_sigmoid = 1.0 / (1.0 + mask_sigmoid);

    // Resize from proto size (e.g., 160x160) to model input size (e.g., 640x640)
    cv::Mat mask_resized;
    cv::resize(mask_sigmoid, mask_resized, model_input_size, 0, 0, cv::INTER_LINEAR);

    // Crop mask to bounding box region (in model input space)
    int x1 = std::max(0, static_cast<int>(box.x));
    int y1 = std::max(0, static_cast<int>(box.y));
    int x2 = std::min(model_input_size.width - 1, static_cast<int>(box.x + box.width));
    int y2 = std::min(model_input_size.height - 1, static_cast<int>(box.y + box.height));

    int crop_width = std::max(1, x2 - x1);
    int crop_height = std::max(1, y2 - y1);

    // Ensure box dimensions are at least 1 pixel when casting
    int box_width_int = std::max(1, static_cast<int>(std::round(box.width)));
    int box_height_int = std::max(1, static_cast<int>(std::round(box.height)));

    if (crop_width > 0 && crop_height > 0 && box_width_int > 0 && box_height_int > 0)
    {
      cv::Rect crop_region(x1, y1, crop_width, crop_height);
      cv::Mat mask_cropped = mask_resized(crop_region);

      // Resize to bounding box size
      cv::Mat mask_box;
      cv::resize(mask_cropped, mask_box, cv::Size(box_width_int, box_height_int), 0, 0, cv::INTER_LINEAR);

      // Apply threshold to get binary mask
      cv::threshold(mask_box, box.mask, 0.5, 1.0, cv::THRESH_BINARY);
    }
  }
}

}  // namespace yolo_onnx