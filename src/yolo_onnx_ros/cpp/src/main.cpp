#include "yolo_onnx/YoloORTDetector.hpp"

static const std::array<std::string, 80> kCocoClassNames = {
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
  "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
  "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
  "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
  "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
  "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
  "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

int main(int argc, char** argv)
{
  std::string model_path = "/home/dev/ws/src/yolo_onnx_ros/models/yolo11n-det.onnx";
  std::string input_image_path = argc > 1 ? argv[1] : "";

  if (input_image_path.empty())
  {
    std::cerr << "Usage: " << argv[0] << " <path_to_input_image>" << std::endl;
    return 1;
  }

  yolo_onnx::Device device(yolo_onnx::DeviceType::CPU);
  yolo_onnx::TaskType task_type = yolo_onnx::TaskType::DETECTION;

  yolo_onnx::YoloORTDetector yolo_detector(model_path, device, task_type);

  std::vector<yolo_onnx::BoundingBox> bounding_boxes;
  cv::Mat input_image = cv::imread(input_image_path);
  if (input_image.empty())
  {
    std::cerr << "Failed to read image from: " << input_image_path << std::endl;
    return 1;
  }

  yolo_detector.Detect(input_image, bounding_boxes);

  // Draw detections on a copy of the input image.
  cv::Mat vis_image = input_image.clone();
  for (const auto& box : bounding_boxes)
  {
    const std::string& class_name =
      (box.class_id >= 0 && box.class_id < static_cast<int>(kCocoClassNames.size()))
        ? kCocoClassNames[box.class_id]
        : "unknown";

    const std::string label =
      class_name + " " + std::to_string(static_cast<int>(box.confidence * 100)) + "%";

    cv::rectangle(vis_image, box.bbox, cv::Scalar(0, 255, 0), 2);

    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    const cv::Point label_origin(box.bbox.x, std::max(box.bbox.y - 4, text_size.height));
    cv::rectangle(vis_image,
                  label_origin + cv::Point(0, baseline),
                  label_origin + cv::Point(text_size.width, -text_size.height),
                  cv::Scalar(0, 255, 0), cv::FILLED);
    cv::putText(vis_image, label, label_origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
  }

  cv::imshow("YOLO Detections", vis_image);
  cv::waitKey(0);

  return 0;
}