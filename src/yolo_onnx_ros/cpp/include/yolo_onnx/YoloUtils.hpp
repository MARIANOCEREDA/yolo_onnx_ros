#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>

namespace yolo_onnx
{

const std::vector<int> kExpectedYoloDetectionTensorShape = {
  1, 84, 8400};  ///< Expected shape of the YOLO detection tensor (NCHW format).
const std::vector<int> kExpectedYoloSegmentTensorShape = {
  1, 116, 8400};  ///< Expected shape of the YOLO segmentation tensor (NCHW format).
static constexpr int kExpectedYoloOutputTensorRank = 3;  ///< Expected rank of the YOLO output tensor (NCHW format).

struct BoundingBox
{
  float x;                               /// Top-left x coordinate in image space.
  float y;                               /// Top-left y coordinate in image space.
  float width;                           /// Bounding box width in pixels.
  float height;                          /// Bounding box height in pixels.
  float confidence;                      /// Detection confidence score in [0, 1].
  int class_id;                          /// Predicted class index.
  cv::Rect bbox;                         /// OpenCV rectangle representation of the same box.
  std::vector<float> mask_coefficients;  /// Mask coefficients for instance segmentation.
  cv::Mat mask;                          /// Segmentation mask for instance segmentation.

  /**
   * @brief Construct an empty bounding box.
   */
  BoundingBox() : x(0), y(0), width(0), height(0), confidence(0), class_id(0) {}

  /**
   * @brief Construct a bounding box from scalar values.
   * @param x Top-left x coordinate.
   * @param y Top-left y coordinate.
   * @param width Box width in pixels.
   * @param height Box height in pixels.
   * @param confidence Detection confidence score.
   * @param class_id Predicted class index.
   */
  BoundingBox(float x, float y, float width, float height, float confidence, int class_id)
    : x(x), y(y), width(width), height(height), confidence(confidence), class_id(class_id)
  {
  }
};

enum class TaskType
{
  DETECTION,
  SEGMENTATION,
};

enum class DeviceType
{
  CPU,
  GPU,
};

struct Device
{
  DeviceType type;
  int device_id;

  Device(DeviceType type = DeviceType::CPU, int device_id = 0) : type(type), device_id(device_id) {}
};

/// Configuration parameters forwarded to the ONNX Runtime session.
struct ORTConfig
{
  GraphOptimizationLevel optimization_level =
    GraphOptimizationLevel::ORT_ENABLE_EXTENDED;  ///< Graph optimization level applied by ORT.
  int intra_op_num_threads = 1;                   ///< Number of threads used for intra-op parallelism.
};

struct InputShape
{
  int batch;     /// Batch size (usually 1 for inference).
  int channels;  /// Number of channels (e.g., 3 for RGB).
  int height;    /// Input image height in pixels.
  int width;     /// Input image width in pixels.
};

struct OutputShape
{
  int batch;     /// Batch size (usually 1 for inference).
  int channels;  /// Number of channels (e.g., number of classes).
  int height;    /// Output height in pixels.
  int width;     /// Output width in pixels.
};

void BlobToONNXTensor(const cv::Mat& blob, Ort::Value& tensor, const InputShape& input_shape, const Device& device);

void LetterboxImage(const cv::Mat& input_image,
                    cv::Mat& output_image,
                    cv::Size target_size,
                    cv::Scalar fill_color = cv::Scalar(114, 114, 114));

void ImageToBlob(const cv::Mat& image, cv::Mat& blob, const InputShape& input_shape);
void OutputTensorToBlob(std::vector<Ort::Value>& tensor, cv::Mat& blob);
std::vector<int64_t> GetTensorShape(const Ort::Value& tensor);

constexpr int INPUT_IMAGE_SIZE = 640;  ///< Default input image size for YOLO models.

}  // namespace yolo_onnx