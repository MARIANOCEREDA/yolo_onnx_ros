#include "yolo_onnx/YoloORTDetector.hpp"
#include "yolo_onnx/YoloORTSegmentor.hpp"

static const std::array<std::string, 80> kCocoClassNames = {
  "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
  "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
  "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
  "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
  "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
  "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
  "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
  "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
  "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
  "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
  "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
  "teddy bear",     "hair drier", "toothbrush"};

static cv::Scalar ClassColor(int class_id)
{
  cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar((class_id * 37) % 180, 200, 255));
  cv::Mat bgr;
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  const auto& p = bgr.at<cv::Vec3b>(0, 0);
  return cv::Scalar(p[0], p[1], p[2]);
}

static void DrawLabeledBox(cv::Mat& image, const yolo_onnx::BoundingBox& box, const cv::Scalar& color)
{
  const std::string& class_name = (box.class_id >= 0 && box.class_id < static_cast<int>(kCocoClassNames.size()))
                                    ? kCocoClassNames[box.class_id]
                                    : "unknown";

  const std::string label = class_name + " " + std::to_string(static_cast<int>(box.confidence * 100)) + "%";

  cv::rectangle(image, box.bbox, color, 2);

  int baseline = 0;
  const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
  const cv::Point label_origin(box.bbox.x, std::max(box.bbox.y - 4, text_size.height));
  cv::rectangle(image,
                label_origin + cv::Point(0, baseline),
                label_origin + cv::Point(text_size.width, -text_size.height),
                color,
                cv::FILLED);
  cv::putText(image, label, label_origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
}

int main(int argc, char** argv)
{
  if (argc < 5)
  {
    std::cerr << "Usage: " << argv[0] << " <model_path> <path_to_input_image> <detection|segmentation> <n_inferences>" << std::endl;
    return 1;
  }

  const std::string model_path = argv[1];
  const std::string input_image_path = argv[2];
  const std::string task = argv[3];
  const int n_inferences = std::stoi(argv[4]);  // Number of inferences to run for timing

  cv::Mat input_image = cv::imread(input_image_path);
  if (input_image.empty())
  {
    std::cerr << "Failed to read image from: " << input_image_path << std::endl;
    return 1;
  }

  yolo_onnx::Device device(yolo_onnx::DeviceType::CPU, 0);  // Use GPU device 0. Change to CPU if needed.
  std::vector<yolo_onnx::BoundingBox> bounding_boxes;
  cv::Mat vis_image = input_image.clone();

  auto start_time = std::chrono::high_resolution_clock::now();
  auto end_time = std::chrono::high_resolution_clock::now();

  if (task == "segmentation")
  {
    yolo_onnx::YoloORTSegmentor segmentor(model_path, device, yolo_onnx::TaskType::SEGMENTATION);

    for (int i = 0; i < n_inferences; ++i)
    {
      start_time = std::chrono::high_resolution_clock::now();
      segmentor.Segment(input_image, bounding_boxes);
      end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
      std::cout << "Inference and post-processing time: " << elapsed.count() << " ms" << std::endl;
    }

    // Build a colored mask overlay for all instances.
    cv::Mat overlay = vis_image.clone();
    for (const auto& box : bounding_boxes)
    {
      if (box.mask.empty())
      {
        continue;
      }
      cv::Rect safe_bbox = box.bbox & cv::Rect(0, 0, vis_image.cols, vis_image.rows);
      if (safe_bbox.width <= 0 || safe_bbox.height <= 0)
      {
        continue;
      }
      cv::Mat mask_resized;
      cv::resize(box.mask, mask_resized, safe_bbox.size(), 0, 0, cv::INTER_LINEAR);
      cv::Mat full_mask = cv::Mat::zeros(vis_image.size(), CV_8U);
      cv::Mat mask_u8;
      mask_resized.convertTo(mask_u8, CV_8U, 255.0);
      mask_u8.copyTo(full_mask(safe_bbox));
      overlay.setTo(ClassColor(box.class_id), full_mask);
    }
    cv::addWeighted(vis_image, 0.6, overlay, 0.4, 0, vis_image);

    // Draw bounding boxes and labels on top.
    for (const auto& box : bounding_boxes)
    {
      DrawLabeledBox(vis_image, box, ClassColor(box.class_id));
    }

    cv::imshow("YOLO Segmentation", vis_image);
  }
  else
  {
    yolo_onnx::YoloORTDetector detector(model_path, device, yolo_onnx::TaskType::DETECTION);

    for (int i = 0; i < n_inferences; ++i)
    {
      start_time = std::chrono::high_resolution_clock::now();
      detector.Detect(input_image, bounding_boxes);
      end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
      std::cout << "Inference and post-processing time: " << elapsed.count() << " ms" << std::endl;
    }

    for (const auto& box : bounding_boxes)
    {
      DrawLabeledBox(vis_image, box, cv::Scalar(0, 255, 0));
    }

    cv::imshow("YOLO Detections", vis_image);
  }
  cv::waitKey(0);
  return 0;
}