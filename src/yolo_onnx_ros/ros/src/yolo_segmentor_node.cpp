#include "yolo_onnx_ros/yolo_segmentor_node.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <chrono>
#include <numeric>

namespace yolo_onnx_ros
{
YoloSegmentorNode::YoloSegmentorNode(const rclcpp::NodeOptions& options)
  : rclcpp_lifecycle::LifecycleNode("inference_node", options)
{
  this->declare_parameter<std::string>("model_path");
  this->declare_parameter<std::string>("device_type");
  this->declare_parameter<std::string>("input_image_topic", "/image_rect");
  this->declare_parameter<std::string>("output_mask_topic", "/segmentation/mask");
  this->declare_parameter<std::string>("output_detections_topic", "/segmentation/detections");
  this->declare_parameter<bool>("display_latency", false);
}

CallbackReturn YoloSegmentorNode::on_configure(const rclcpp_lifecycle::State&)
{
  RCLCPP_INFO(get_logger(), "Configuring YoloSegmentorNode...");
  model_path_ = this->get_parameter("model_path").as_string();
  device_type_str_ = this->get_parameter("device_type").as_string();

  if (model_path_.empty() || device_type_str_.empty())
  {
    RCLCPP_ERROR(get_logger(), "Both 'model_path' and 'device_type' parameters must be set.");
    return CallbackReturn::FAILURE;
  }

  return CallbackReturn::SUCCESS;
}

CallbackReturn YoloSegmentorNode::on_activate(const rclcpp_lifecycle::State&)
{
  if (!device_type_map.count(device_type_str_))
  {
    RCLCPP_ERROR(get_logger(), "Invalid device type: %s", device_type_str_.c_str());
    return CallbackReturn::FAILURE;
  }

  device_type_ = device_type_map.at(device_type_str_);

  auto device = yolo_onnx::Device(device_type_);

  try
  {
    segmentor_ = std::make_shared<yolo_onnx::YoloORTSegmentor>(model_path_, device);
  }
  catch (const std::exception& error)
  {
    RCLCPP_ERROR(get_logger(), "Unable to load model '%s': %s", model_path_.c_str(), error.what());
    return CallbackReturn::FAILURE;
  }

  if (!detection_pub_)
  {
    auto output_detections_topic = this->get_parameter("output_detections_topic").as_string();
    detection_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(output_detections_topic, 10);
  }
  detection_pub_->on_activate();

  if (!mask_pub_)
  {
    auto output_mask_topic = this->get_parameter("output_mask_topic").as_string();
    mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>(output_mask_topic, 10);
  }
  mask_pub_->on_activate();

  auto input_image_topic = this->get_parameter("input_image_topic").as_string();
  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    input_image_topic, 10, std::bind(&YoloSegmentorNode::ImageCallback, this, std::placeholders::_1));

  display_latency_ = this->get_parameter("display_latency").as_bool();

  inference_thread_ = std::jthread([this](std::stop_token stop_token) { RunInferenceAsync(stop_token); });

  RCLCPP_INFO(get_logger(), "Activated YoloSegmentorNode...");
  return CallbackReturn::SUCCESS;
}

CallbackReturn YoloSegmentorNode::on_deactivate(const rclcpp_lifecycle::State&)
{
  inference_thread_.request_stop();
  input_image_cv_.notify_all();

  if (inference_thread_.joinable())
  {
    inference_thread_.join();
  }

  image_sub_.reset();
  if (detection_pub_)
  {
    detection_pub_->on_deactivate();
  }
  if (mask_pub_)
  {
    mask_pub_->on_deactivate();
  }

  RCLCPP_INFO(get_logger(), "Deactivated YoloSegmentorNode...");
  return CallbackReturn::SUCCESS;
}

CallbackReturn YoloSegmentorNode::on_cleanup(const rclcpp_lifecycle::State&)
{
  inference_thread_.request_stop();
  input_image_cv_.notify_all();

  if (inference_thread_.joinable())
  {
    inference_thread_.join();
  }

  image_sub_.reset();
  detection_pub_.reset();
  mask_pub_.reset();
  segmentor_.reset();
  RCLCPP_INFO(get_logger(), "YoloSegmentorNode cleaned up.");
  return CallbackReturn::SUCCESS;
}

CallbackReturn YoloSegmentorNode::on_shutdown(const rclcpp_lifecycle::State&)
{
  inference_thread_.request_stop();
  input_image_cv_.notify_all();

  if (inference_thread_.joinable())
  {
    inference_thread_.join();
  }

  RCLCPP_INFO(get_logger(), "Shutting down YoloSegmentorNode...");
  return CallbackReturn::SUCCESS;
}

void YoloSegmentorNode::ToVisionMsgsDetections(const std::vector<yolo_onnx::BoundingBox>& detections,
                                               vision_msgs::msg::Detection2DArray& detection_msg)
{
  detection_msg.detections.clear();
  for (const auto& detection : detections)
  {
    vision_msgs::msg::Detection2D detection2d;
    vision_msgs::msg::BoundingBox2D bbox;
    vision_msgs::msg::Pose2D pose;
    pose.position.x = detection.x + detection.width / 2.0f;
    pose.position.y = detection.y + detection.height / 2.0f;
    pose.theta = 0.0f;  // Assuming no rotation information is available
    bbox.center = pose;
    bbox.size_x = detection.width;
    bbox.size_y = detection.height;

    detection2d.bbox = bbox;

    vision_msgs::msg::ObjectHypothesisWithPose result;
    result.hypothesis.class_id = std::to_string(detection.class_id);
    result.hypothesis.score = detection.confidence;

    detection2d.results.push_back(result);
    detection_msg.detections.push_back(detection2d);
  }
}

cv::Mat YoloSegmentorNode::CreateBinaryMask(const cv::Size& image_size,
                                            const std::vector<yolo_onnx::BoundingBox>& detections)
{
  cv::Mat output(image_size, CV_8UC1, cv::Scalar(0));

  for (const auto& detection : detections)
  {
    if (detection.mask.empty())
    {
      continue;
    }

    int x = std::max(0, static_cast<int>(detection.x));
    int y = std::max(0, static_cast<int>(detection.y));
    int w = std::min(static_cast<int>(detection.width), output.cols - x);
    int h = std::min(static_cast<int>(detection.height), output.rows - y);

    if (w <= 0 || h <= 0)
    {
      continue;
    }

    cv::Mat mask_resized;
    if (detection.mask.rows != h || detection.mask.cols != w)
    {
      cv::resize(detection.mask, mask_resized, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);
    }
    else
    {
      mask_resized = detection.mask;
    }

    cv::Mat roi = output(cv::Rect(x, y, w, h));
    for (int r = 0; r < h; ++r)
    {
      for (int c = 0; c < w; ++c)
      {
        if (mask_resized.at<float>(r, c) > 0.5f)
        {
          roi.at<uint8_t>(r, c) = 255;
        }
      }
    }
  }

  return output;
}

void YoloSegmentorNode::ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  {
    std::lock_guard<std::mutex> lock(input_image_mutex_);
    latest_image_ = std::move(msg);
    new_image_available_ = true;
  }
  input_image_cv_.notify_one();
}

void YoloSegmentorNode::RunInferenceAsync(std::stop_token stop_token)
{
  while (rclcpp::ok() && !stop_token.stop_requested())
  {
    sensor_msgs::msg::Image::SharedPtr input_image_msg{nullptr};
    {
      std::unique_lock<std::mutex> lock(input_image_mutex_);
      input_image_cv_.wait(lock,
                           [this, &stop_token]
                           { return new_image_available_.load() || !rclcpp::ok() || stop_token.stop_requested(); });

      if (stop_token.stop_requested())
      {
        break;
      }

      if (!new_image_available_ || !latest_image_)
      {
        continue;
      }

      input_image_msg = std::move(latest_image_);
      new_image_available_ = false;
    }

    // Start timing only after a frame is dequeued, so idle wait time is excluded.
    const auto start_inference_time = std::chrono::steady_clock::now();

    cv::Mat input_inference_image;
    try
    {
      input_inference_image = cv_bridge::toCvCopy(input_image_msg, sensor_msgs::image_encodings::BGR8)->image;
    }
    catch (const cv_bridge::Exception& error)
    {
      RCLCPP_ERROR(get_logger(), "Unable to convert input image to BGR8: %s", error.what());
      continue;
    }

    try
    {
      segmentor_->Segment(input_inference_image, latest_detections_);
    }
    catch (const std::exception& error)
    {
      RCLCPP_ERROR(get_logger(), "Inference failed: %s", error.what());
      continue;
    }
    const auto& detections = latest_detections_;

    auto detection_msg = std::make_unique<vision_msgs::msg::Detection2DArray>();
    detection_msg->header.stamp = input_image_msg->header.stamp;
    detection_msg->header.frame_id = input_image_msg->header.frame_id;

    this->ToVisionMsgsDetections(detections, *detection_msg);
    if (detection_pub_ && detection_pub_->is_activated())
    {
      if (display_latency_)
      {
        // Processing time: dequeue -> detections ready (convert + preprocess + inference + postprocess).
        const auto processing_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_inference_time).count();
        // End-to-end: image capture stamp -> now (includes transport, queueing and upstream nodes).
        const double end_to_end_ms = (this->now() - rclcpp::Time(input_image_msg->header.stamp)).seconds() * 1e3;

        inference_times_ms_.push_back(processing_ms);
        if (inference_times_ms_.size() > 5)
        {
          inference_times_ms_.erase(inference_times_ms_.begin());
        }
        const double average_processing_ms =
            std::accumulate(inference_times_ms_.begin(), inference_times_ms_.end(), 0.0) / inference_times_ms_.size();
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Processing: %.1f ms (avg of last %zu) | end-to-end: %.1f ms",
                             average_processing_ms, inference_times_ms_.size(), end_to_end_ms);
      }
      detection_pub_->publish(std::move(detection_msg));
    }

    // Publish class-coloured segmentation mask (black background)
    if (mask_pub_ && mask_pub_->is_activated())
    {
      cv::Mat binary_mask = CreateBinaryMask(input_inference_image.size(), detections);

      auto mask_msg = std::make_unique<sensor_msgs::msg::Image>();
      mask_msg->header = input_image_msg->header;
      mask_msg->height = binary_mask.rows;
      mask_msg->width = binary_mask.cols;
      mask_msg->encoding = sensor_msgs::image_encodings::MONO8;
      mask_msg->is_bigendian = false;
      mask_msg->step = static_cast<sensor_msgs::msg::Image::_step_type>(binary_mask.step);
      mask_msg->data.assign(binary_mask.datastart, binary_mask.dataend);

      mask_pub_->publish(std::move(mask_msg));
    }
  }
}

}  // namespace yolo_onnx_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(yolo_onnx_ros::YoloSegmentorNode)