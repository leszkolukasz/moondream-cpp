#include <iostream>
#include <moondream/moondream.h>
#include <onnxruntime_cxx_api.h>
#include <xtensor/io/xio.hpp>

int main() {
  // try {
  //   Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "native");
  //   Ort::SessionOptions session_options;
  //   session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

  //   const char *model_path = "./data/vision_encoder.onnx";
  //   Ort::Session session(env, model_path, session_options);

  //   std::cout << "ONNX model loaded successfully." << std::endl;

  //   std::vector<int64_t> input_dims = {5, 3, 378, 378}; // example shape
  //   // std::vector<float> input_tensor_values{1., -1., 42.}; // fill with
  //   data xt::xarray<float> input_tensor_values = xt::ones<float>(input_dims);

  //   Ort::MemoryInfo memory_info =
  //       Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  //   Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
  //       memory_info, input_tensor_values.data(), input_tensor_values.size(),
  //       input_dims.data(), input_dims.size());

  //   const char *input_names[] = {"input"};
  //   const char *output_names[] = {"output"};

  //   Ort::Value *input_values = new Ort::Value[1];
  //   input_values[0] = std::move(input_tensor);
  //   // std::vector<Ort::Value> input_values;
  //   // input_values[0] = std::move(input_tensor);

  //   auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names,
  //                                     input_values, 1, output_names, 1);

  //   std::cout <<
  //   output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount()
  //             << " elements in output tensor." << std::endl;

  //   // float *output_data = output_tensors[0].GetTensorMutableData<float>();

  //   // std::cout << "Output tensor data: ";
  //   // for (size_t i = 0;
  //   //      i <
  //   output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
  //   //      ++i) {
  //   //   std::cout << output_data[i] << " ";
  //   // }

  // } catch (const Ort::Exception &e) {
  //   std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
  //   return 1;
  // }

  using namespace moondream;
  // Tokenizer tokenizer("../moondream-mobile/assets/models/tokenizer.json");

  // auto encoded = tokenizer.encode("Hello, world!");
  // std::cout << "Encoded tokens: ";
  // for (const auto &id : encoded) {
  //   std::cout << id << " ";
  // }
  // std::cout << std::endl;

  // auto decoded = tokenizer.decode(encoded);
  // std::cout << "Decoded text: " << decoded << std::endl;

  // auto image = load_image("frieren.jpg");

  // std::cout << "Image loaded with shape: " << image.shape()[0] << "x"
  //           << image.shape()[1] << "x" << image.shape()[2] << std::endl;

  // auto resized = resize_tensor(xt::cast<float>(image), 100, 200);

  // std::cout << "Image resized with shape: " << resized.shape()[0] << "x"
  //           << resized.shape()[1] << "x" << resized.shape()[2] << std::endl;

  // save_image(xt::cast<uint8_t>(resized), "output.jpg");

  // xt::xarray<float> ones = xt::ones<float>({1, 3});
  // xt::xarray<float> zeros = xt::zeros<float>({1, 3});

  // std::vector<xt::xarray<float>> vec = {ones, zeros};

  // auto stacked = concat_vector(vec, 1);

  // std::cout << stacked << std::endl;

  Moondream dream("/media/whistleroosh/Universal/projects/moondream-cpp/data");
  // auto res = dream.encode_image(
  //     "/media/whistleroosh/Universal/projects/moondream-cpp/frieren.jpg");
  // std::cout << xt::adapt(res->kv_cache->shape()) << std::endl;

  dream.caption(
      "/media/whistleroosh/Universal/projects/moondream-cpp/frieren.jpg",
      "normal", 100);

  return 0;
}
