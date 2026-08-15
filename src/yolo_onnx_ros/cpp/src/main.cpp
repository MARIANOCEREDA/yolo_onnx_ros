#include "yolo_onnx/YoloORTDetector.hpp"

int main(int argc, char** argv)
{
  std::string model_path = "/home/dev/ws/src/yolo_onnx_ros/models/yolo11n-det.onnx";
  cv::Mat input_image = cv::imread("/home/dev/ws/src/yolo_onnx_ros/images/test_image.jpeg");

  yolo_onnx::Device device(yolo_onnx::DeviceType::CPU);
  yolo_onnx::TaskType task_type = yolo_onnx::TaskType::DETECTION;

  yolo_onnx::YoloORTDetector yolo_onnx(model_path, device, task_type);

  std::vector<yolo_onnx::BoundingBox> bounding_boxes;
  yolo_onnx.Detect(input_image, bounding_boxes);
  return 0;
}