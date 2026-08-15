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
  blob = cv::Mat(predictions,
                 box_coords_classes,
                 CV_32FC(batch),
                 output_data_ptr);
}

}  // namespace yolo_onnx