#include <iostream>
#include <moondream.h>
#include <onnxruntime_cxx_api.h>

int main() {
  // try {
  //     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "native");
  //     Ort::SessionOptions session_options;
  //     session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

  //     const char* model_path = "model.onnx";
  //     Ort::Session session(env, model_path, session_options);

  //     std::cout << "ONNX model loaded successfully." << std::endl;

  //     std::vector<int64_t> input_dims = {1, 3}; // example shape
  //     std::vector<float> input_tensor_values{1., -1., 42.}; // fill with data

  //     Ort::MemoryInfo memory_info =
  //     Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  //     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
  //         memory_info, input_tensor_values.data(),
  //         input_tensor_values.size(), input_dims.data(), input_dims.size());

  //     const char* input_names[] = {"input"};
  //     const char* output_names[] = {"output"};

  //     auto output_tensors = session.Run(Ort::RunOptions{nullptr},
  //         input_names, &input_tensor, 1,
  //         output_names, 1);

  //     float* output_data = output_tensors[0].GetTensorMutableData<float>();

  //     std::cout << "Output tensor data: ";
  //     for (size_t i = 0; i <
  //     output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount(); ++i) {
  //         std::cout << output_data[i] << " ";
  //     }

  // } catch (const Ort::Exception& e) {
  //     std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
  //     return 1;
  // }

  using namespace moondream;
  Tokenizer tokenizer("../moondream-mobile/assets/models/tokenizer.json");

  auto encoded = tokenizer.encode("Hello, world!");
  std::cout << "Encoded tokens: ";
  for (const auto &id : encoded) {
    std::cout << id << " ";
  }
  std::cout << std::endl;

  auto decoded = tokenizer.decode(encoded);
  std::cout << "Decoded text: " << decoded << std::endl;

  return 0;
}
