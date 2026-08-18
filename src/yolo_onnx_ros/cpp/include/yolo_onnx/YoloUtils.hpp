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
  1, 84, 8400};  ///< Expected shape of the YOLO detection tensor.
const std::vector<int> kExpectedYoloSegmentTensorShape = {
  1, 116, 8400};  ///< Expected shape of the YOLO segmentation tensor.
const std::vector<int> kProtoMasksYoloOutputTensorShape = {
  1, 32, 160, 160};  ///< Expected shape of the YOLO segmentation proto masks tensor.
static constexpr int kProtoMasksYoloOutputTensorRank =
  4;  ///< Expected rank of the prototype masks output tensor (batch, channels, H, W).
static constexpr int kExpectedYoloOutputTensorRank =
  3;  ///< Expected rank of the primary YOLO output tensor (batch, attributes, predictions).
static constexpr int N_SEG_MASK_COEFFS = 32;  ///< Number of mask coefficients per detection in the segmentation output.
static constexpr int N_SEG_MASK_COEFFS_INDEX_START =
  84;  ///< 0-based row index where mask coefficients begin in the segmentation output tensor (4 box coords + 80 class
       ///< scores).
static constexpr int CLASS_START_INDEX =
  5;  ///< 1-based column index where class scores start; the corresponding 0-based index is CLASS_START_INDEX - 1.

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

/// Specifies the YOLO task performed by the loaded model.
enum class TaskType
{
  DETECTION,     ///< Object detection: predicts bounding boxes and class labels.
  SEGMENTATION,  ///< Instance segmentation: predicts bounding boxes, class labels, and pixel-level masks.
};

/// Target hardware device for ONNX Runtime inference.
enum class DeviceType
{
  CPU,  ///< Run inference on the CPU.
  GPU,  ///< Run inference on a CUDA-enabled GPU.
};

/// Identifies the target device (CPU or GPU) used during ONNX Runtime inference.
struct Device
{
  DeviceType type;  ///< Device category (CPU or GPU).
  int device_id;    ///< Zero-based index of the device; relevant for multi-GPU systems.

  /**
   * @brief Construct a Device descriptor.
   * @param type      Device type; defaults to CPU.
   * @param device_id Zero-based device index; defaults to 0.
   */
  Device(DeviceType type = DeviceType::CPU, int device_id = 0) : type(type), device_id(device_id) {}
};

/// Configuration parameters forwarded to the ONNX Runtime session.
struct ORTConfig
{
  GraphOptimizationLevel optimization_level =
    GraphOptimizationLevel::ORT_ENABLE_EXTENDED;  ///< Graph optimization level applied by ORT.
  int intra_op_num_threads = 1;                   ///< Number of threads used for intra-op parallelism.
};

/// Dimensions of the network input tensor in NCHW order.
struct InputShape
{
  int batch;     ///< Batch size (usually 1 for inference).
  int channels;  ///< Number of channels (e.g., 3 for RGB).
  int height;    ///< Input image height in pixels.
  int width;     ///< Input image width in pixels.
};

/// Dimensions of the network output tensor.
struct OutputShape
{
  int batch;     ///< Batch size (usually 1 for inference).
  int channels;  ///< Number of output channels (e.g., number of attributes per prediction).
  int height;    ///< Output spatial height in pixels (relevant for mask output tensors).
  int width;     ///< Output spatial width in pixels (relevant for mask output tensors).
};

/**
 * @brief Check whether a blob matches the expected shape for YOLO detection output.
 * @param blob OpenCV Mat to validate.
 * @return true if the blob dimensions match kExpectedYoloDetectionTensorShape, false otherwise.
 */
bool ValidateBlobSizeDetection(const cv::Mat& blob);

/**
 * @brief Check whether a blob matches the expected shape for YOLO segmentation output.
 * @param blob OpenCV Mat to validate.
 * @return true if the blob dimensions match kExpectedYoloSegmentTensorShape, false otherwise.
 */
bool ValidateBlobSizeSegmentation(const cv::Mat& blob);

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
 * @param task_type Type of YOLO task (detection or segmentation) to determine expected tensor shape.
 */
void OutputTensorToBlob(std::vector<Ort::Value>& tensor, cv::Mat& blob, const TaskType task_type);

/**
 * @brief Retrieve the shape (dimensions) of an ONNX Runtime tensor.
 * @param tensor Tensor whose shape is queried.
 * @return Vector of dimension sizes, e.g., {1, 84, 8400} for a detection output.
 */
std::vector<int64_t> GetTensorShape(const Ort::Value& tensor);

/**
 * @brief Extract bounding boxes from a YOLO output blob, converting from the network's output
 *        format (cx, cy, w, h, class_scores...) into BoundingBox instances. For segmentation
 *        tasks, mask coefficients are also extracted per box.
 * @param boxes Vector to be filled with BoundingBox instances representing detected objects.
 * @param detection_blob OpenCV Mat containing the raw YOLO output (shape: [num_attributes, num_predictions]).
 * @param original_image_size Dimensions of the original input image (width, height) before preprocessing.
 * @param task_type Determines whether mask coefficients are extracted (SEGMENTATION) in addition to box data.
 */
void GetBoxesFromDetection(std::vector<BoundingBox>& boxes,
                           const cv::Mat& detection_blob,
                           const cv::Size& original_image_size,
                           TaskType task_type);

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

/**
 * @brief Undo the letterbox padding and scaling applied during preprocessing, mapping bounding
 *        box coordinates back to the original (pre-letterbox) image space.
 * @param boxes Bounding boxes in YOLO model-output coordinate space; modified in-place.
 * @param input_yolo_size Spatial dimensions of the YOLO model input (e.g., 640x640).
 * @param original_image_size Dimensions of the original image before letterboxing was applied.
 * @param letterboxed_image_size Dimensions of the image after letterboxing.
 */
void RemoveLetterboxOffset(std::vector<BoundingBox>& boxes,
                           const cv::Size& input_yolo_size,
                           const cv::Size& original_image_size,
                           const cv::Size& letterboxed_image_size);

/**
 * @brief Extract the prototype mask tensor (second ONNX output) from segmentation inference
 *        results into a flat OpenCV Mat for downstream mask assembly.
 * @param mask Output Mat of shape [32, proto_height * proto_width] holding the flattened prototype masks.
 * @param inference_output_tensor All output tensors from the ORT session; the second tensor (index 1)
 *        must be the prototype mask tensor with shape [1, 32, H, W].
 * @param proto_height Populated with the spatial height of the prototype mask grid (e.g., 160).
 * @param proto_width Populated with the spatial width of the prototype mask grid (e.g., 160).
 */
void GetMasksFromTensor(cv::Mat& mask,
                        std::vector<Ort::Value>& inference_output_tensor,
                        int& proto_height,
                        int& proto_width);

/**
 * @brief Compute per-instance segmentation masks by linearly combining prototype masks with
 *        each box's mask coefficients. Stores the combined masks both in the output Mat and
 *        in the @c mask field of each BoundingBox.
 * @param masks Output Mat of shape [num_boxes, proto_height * proto_width]; each row contains
 *        the flattened mask for the corresponding detection.
 * @param raw_proto_masks Flattened prototype mask tensor of shape [32, proto_height * proto_width]
 *        as returned by GetMasksFromTensor.
 * @param boxes Detected bounding boxes whose mask_coefficients are used; each box's @c mask field
 *        is updated with its 2D mask of size (proto_height x proto_width).
 * @param original_image_size Dimensions of the original input image (width, height).
 * @param proto_height Height of the prototype mask grid (e.g., 160).
 * @param proto_width Width of the prototype mask grid (e.g., 160).
 */
void ProcessYoloMasks(cv::Mat& masks,
                      const cv::Mat& raw_proto_masks,
                      std::vector<BoundingBox>& boxes,
                      int proto_height,
                      int proto_width);

/**
 * @brief Finalise per-instance masks by applying sigmoid activation, resizing to the model
 *        input resolution, cropping to each bounding box region, resizing to the box extents,
 *        and thresholding to produce a binary mask stored in each BoundingBox.
 * @param boxes Detected bounding boxes; each box's @c mask field is updated with a binary mask
 *        (values 0 or 1) sized to the bounding box dimensions.
 * @param proto_masks Mat of shape [num_boxes, proto_height * proto_width] as produced by
 *        ProcessYoloMasks.
 * @param proto_height Height of the prototype mask grid (e.g., 160).
 * @param proto_width Width of the prototype mask grid (e.g., 160).
 * @param model_input_size Spatial dimensions of the YOLO model input (e.g., 640x640); used when
 *        resizing masks before cropping to box coordinates.
 */
void ProcessYoloBoxes(std::vector<BoundingBox>& boxes,
                      const cv::Mat& proto_masks,
                      int proto_height,
                      int proto_width,
                      const cv::Size& model_input_size);

constexpr int INPUT_IMAGE_SIZE = 640;  ///< Default input image size for YOLO models.

}  // namespace yolo_onnx