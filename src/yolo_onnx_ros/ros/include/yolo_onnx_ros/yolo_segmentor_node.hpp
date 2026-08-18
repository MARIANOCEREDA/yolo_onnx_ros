
#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <unordered_set>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <thread>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "yolo_onnx/YoloORTSegmentor.hpp"

namespace yolo_onnx_ros
{

/**
 * @brief Output payload produced by the inference worker.
 */
struct InferenceResult
{
  /** @brief Frame id copied from the input image header. */
  std::string frame_id;
  /** @brief Detected bounding boxes for the frame. */
  std::vector<yolo_onnx::BoundingBox> detections;
  /** @brief Timestamp copied from the input image header. */
  rclcpp::Time timestamp;
};

/**
 * @brief Alias used by lifecycle callbacks in this node.
 */
using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

/**
 * @brief Mapping from parameter string to detector device type.
 */
inline const std::map<std::string, yolo_onnx::DeviceType> device_type_map = {
    {"CPU", yolo_onnx::DeviceType::CPU}, {"GPU", yolo_onnx::DeviceType::GPU}};

/**
 * @brief ROS 2 lifecycle node that performs asynchronous YOLO ONNX inference.
 *
 * The node stores the latest incoming image, runs inference in a dedicated worker
 * thread, and publishes detection messages from a dedicated publisher thread.
 */
class YoloSegmentorNode : public rclcpp_lifecycle::LifecycleNode
{
 public:
  /**
   * @brief Construct a new YoloSegmentorNode.
   * @param options ROS 2 node options.
   */
  explicit YoloSegmentorNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

  /**
   * @brief Configure resources used by the node.
   * @param state Previous lifecycle state.
   * @return Lifecycle transition result.
   */
  CallbackReturn on_configure(const rclcpp_lifecycle::State&);

  /**
   * @brief Activate publishers/subscriptions and start worker threads.
   * @param state Previous lifecycle state.
   * @return Lifecycle transition result.
   */
  CallbackReturn on_activate(const rclcpp_lifecycle::State& state);

  /**
   * @brief Deactivate communication interfaces and stop worker threads.
   * @param state Previous lifecycle state.
   * @return Lifecycle transition result.
   */
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state);

  /**
   * @brief Release configured resources.
   * @param state Previous lifecycle state.
   * @return Lifecycle transition result.
   */
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State& state);

  /**
   * @brief Handle shutdown transition and stop all background work.
   * @param state Previous lifecycle state.
   * @return Lifecycle transition result.
   */
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State& state);

  /**
   * @brief Store the latest image received from the subscribed topic.
   * @param msg Input image message.
   */
  void ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg);

  /**
   * @brief Inference worker loop.
   *
   * Waits for new input images, executes model inference, and updates the latest
   * detection result until a stop is requested.
   *
   * @param stop_token Cooperative cancellation token from std::jthread.
   */
  void RunInferenceAsync(std::stop_token stop_token);

  /**
   * @brief Convert detector bounding boxes to a ROS 2 detection array message.
   * @param detections Detector outputs.
   * @param detection_msg Output ROS message to populate.
   */
  static void ToVisionMsgsDetections(
      const std::vector<yolo_onnx::BoundingBox>& detections,
      vision_msgs::msg::Detection2DArray& detection_msg);

  /**
   * @brief Create a binary segmentation mask.
   * @param image_size Size of the output image (should match the original input image).
   * @param detections Detector outputs with per-instance masks.
   * @return Single-channel (MONO8) image: 255 for segmented pixels, 0 for background.
   */
  static cv::Mat CreateBinaryMask(
      const cv::Size& image_size,
      const std::vector<yolo_onnx::BoundingBox>& detections);

 private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp_lifecycle::LifecyclePublisher<vision_msgs::msg::Detection2DArray>::SharedPtr
      detection_pub_;
  rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>::SharedPtr mask_pub_;
  std::shared_ptr<yolo_onnx::YoloORTSegmentor> segmentor_;

  std::string model_path_;
  std::string device_type_str_;
  yolo_onnx::DeviceType device_type_;

  // Input Image multi-threading safety variables
  std::atomic<bool> new_image_available_{false};
  sensor_msgs::msg::Image::SharedPtr latest_image_;
  mutable std::mutex input_image_mutex_;
  std::condition_variable input_image_cv_;

  std::jthread inference_thread_;

  // Outputs
  std::vector<yolo_onnx::BoundingBox> latest_detections_;
};

}  // namespace yolo_onnx_ros