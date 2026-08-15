#pragma once

#include <onnxruntime_cxx_api.h>

#include "yolo_onnnx/YoloORTInterface.hpp"

using namespace yolo_onnx::yolo_ort_interface;

namespace yolo_onnx
{

class YoloSegmentor : public YoloORTInterface
{
 public:
  YoloSegmentor(const std::string& model_path, const std::string& device, const std::string& task_type);
  ~YoloSegmentor();

  void Segment(const cv::Mat& input_image, std::vector<cv::Rect>& bounding_boxes);
};
}  // namespace yolo_onnx