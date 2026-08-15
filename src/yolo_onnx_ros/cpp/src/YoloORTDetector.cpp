#include "yolo_onnx/YoloORTDetector.hpp"

#include <onnxruntime_cxx_api.h>

using namespace yolo_onnx::yolo_ort_interface;

namespace yolo_onnx
{

void YoloORTDetector::Detect(const cv::Mat& input_image, std::vector<BoundingBox>& bounding_boxes)
{
  Preprocess(input_image, input_tensor_);

  auto& session = GetSession();
  auto& input_node_names = GetInputNames();
  auto& output_node_names = GetOutputNames();

  std::vector<Ort::Value> output_tensors =
    session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor_, 1, output_node_names.data(), 1);
  
  auto tensor_shape = GetTensorShape(output_tensors.front());

  // Postprocess(input_image, output_tensors);
}

void YoloORTDetector::Preprocess(const cv::Mat& input_image, Ort::Value& input_tensor)
{
  // 1. Letterbox image
  LetterboxImage(input_image, letterboxed_image_, cv::Size(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE));

  // 2. Convert from BGR to RGB
  cv::cvtColor(letterboxed_image_, letterboxed_image_, cv::COLOR_BGR2RGB);

  // 3. From image to blob (NCHW) format — stored in the member so the pointer stays valid
  ImageToBlob(letterboxed_image_, input_blob_, input_blob_shape_);

  // 4. Create ONNX Runtime tensor from blob
  auto device = GetDevice();
  BlobToONNXTensor(input_blob_, input_tensor, input_blob_shape_, device);
}

void YoloORTDetector::Postprocess(const cv::Mat& input_image, std::vector<Ort::Value>& output_tensor)
{
  cv::Mat blob;
  OutputTensorToBlob(output_tensor, blob);
}

}  // namespace yolo_onnx