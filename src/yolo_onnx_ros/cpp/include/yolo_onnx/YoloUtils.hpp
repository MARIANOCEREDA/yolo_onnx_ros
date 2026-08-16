#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>

namespace yolo_onnx
{

struct YoloThresholds
{
  float confidence_threshold = 0.70f;  ///< Minimum confidence score to consider a detection valid.
  float nms_threshold = 0.50f;         ///< Non-Maximum Suppression (NMS) threshold for filtering overlapping boxes.
};

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
    : x(x),
      y(y),
      width(width),
      height(height),
      confidence(confidence),
      class_id(class_id),
      bbox(static_cast<int>(x), static_cast<int>(y), static_cast<int>(width), static_cast<int>(height))
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

/**
 * @brief Copy pixel data from a preprocessed OpenCV blob into an ONNX Runtime input tensor.
 * @param blob Preprocessed image blob (typically NCHW, float32, normalized) produced by ImageToBlob.
 * @param tensor ONNX Runtime tensor to be filled with the blob's data. Must already be allocated
 *        with a buffer matching input_shape.
 * @param input_shape Dimensions (batch, channels, height, width) describing the blob/tensor layout.
 * @param device Target device (CPU/GPU) the tensor memory is associated with; determines how the
 *        copy is performed (e.g., host copy vs. device-aware copy).
 */
void BlobToONNXTensor(const cv::Mat& blob, Ort::Value& tensor, const InputShape& input_shape, const Device& device);

/**
 * @brief Resize an image to a target size while preserving aspect ratio, padding the remaining
 *        area with a fill color ("letterboxing"). Commonly used to prepare images for YOLO
 *        inference without distorting object proportions.
 * @param input_image Source image to be letterboxed.
 * @param output_image Destination image containing the resized, padded result.
 * @param target_size Desired output dimensions (width, height).
 * @param fill_color Color used to pad the borders added around the resized image.
 */
void LetterboxImage(const cv::Mat& input_image,
                    cv::Mat& output_image,
                    cv::Size target_size,
                    cv::Scalar fill_color = cv::Scalar(114, 114, 114));

/**
 * @brief Convert an image (e.g., letterboxed and color-converted) into a network-ready blob,
 *        typically performing normalization and NHWC-to-NCHW conversion.
 * @param image Source image to convert.
 * @param blob Destination blob suitable for consumption by BlobToONNXTensor.
 * @param input_shape Dimensions (batch, channels, height, width) the blob should conform to.
 */
void ImageToBlob(const cv::Mat& image, cv::Mat& blob, const InputShape& input_shape);

/**
 * @brief Convert raw ONNX Runtime output tensor(s) into an OpenCV Mat for downstream
 *        post-processing (e.g., decoding boxes, class scores, or mask coefficients).
 * @param tensor Output tensor(s) produced by the ONNX Runtime session run.
 * @param blob Destination Mat holding the tensor data in a format convenient for post-processing.
 */
void OutputTensorToBlob(std::vector<Ort::Value>& tensor, cv::Mat& blob);

/**
 * @brief Retrieve the shape (dimensions) of an ONNX Runtime tensor.
 * @param tensor Tensor whose shape is queried.
 * @return Vector of dimension sizes, e.g., {1, 84, 8400} for a detection output.
 */
std::vector<int64_t> GetTensorShape(const Ort::Value& tensor);

/**
 * @brief Extract bounding boxes from a YOLO detection output blob, converting from the
 *        network's output format (center_x, center_y, width, height, confidence, class scores) to a more convenient
 * BoundingBox representation.
 * @param detection_blob OpenCV Mat containing the raw detection output from the YOLO model (shape: [num_attributes,
 * num_predictions]).
 * @param boxes Vector to be filled with BoundingBox instances representing detected objects.
 * @param original_image_size Size of the original input image (width, height) before preprocessing
 */
void GetBoxesFromDetection(const cv::Mat& detection_blob,
                           std::vector<BoundingBox>& boxes,
                           const cv::Size& original_image_size);

/**
 * @brief Filter bounding boxes based on their confidence scores.
 * @param input_boxes Vector of candidate bounding boxes.
 * @param filtered_boxes Vector to be filled with bounding boxes that pass the confidence threshold.
 * @param confidence_threshold Minimum confidence score required for a bounding box to be kept.
 */
void FilterBoxesByConfidence(const std::vector<BoundingBox>& input_boxes,
                             std::vector<BoundingBox>& filtered_boxes,
                             float confidence_threshold);

/**
 * @brief Apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes based on their
 *        confidence scores and Intersection-over-Union (IoU) thresholds. This is a common
 *        post-processing step in object detection pipelines to reduce duplicate detections.
 * @param input_boxes Vector of candidate bounding boxes (with confidence scores and class IDs).
 * @param output_boxes Vector to be filled with the final bounding boxes after NMS filtering.
 * @param iou_threshold IoU threshold for determining whether boxes overlap too much and should be suppressed. Typical
 * values are in the range [0.3, 0.7].
 */
void NonMaxSuppression(const std::vector<BoundingBox>& input_boxes,
                       std::vector<BoundingBox>& output_boxes,
                       float iou_threshold);

void RemoveLetterboxOffset(std::vector<BoundingBox>& boxes,
                           const cv::Size& input_yolo_size,
                           const cv::Size& original_image_size,
                           const cv::Size& letterboxed_image_size);

  constexpr int INPUT_IMAGE_SIZE = 640;  ///< Default input image size for YOLO models.

}  // namespace yolo_onnx