#pragma once

#include <fstream>
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>

#include <xtensor/containers/xarray.hpp>

namespace moondream {
inline nlohmann::json load_json(const std::string &config_path) {
  std::ifstream input(config_path);
  if (!input.is_open()) {
    throw std::runtime_error("Unable to open JSON file.");
  }
  nlohmann::json j;
  input >> j;

  return j;
}

inline std::unique_ptr<Ort::Session>
load_ONNX(const std::string &model_path, const Ort::SessionOptions &options) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "moondream");
  return std::make_unique<Ort::Session>(env, model_path.c_str(), options);
}

inline std::vector<Ort::Value>
run_onnx(Ort::Session &session, const std::vector<std::string> &inputs,
         Ort::Value *input_values, const std::vector<std::string> &outputs) {
  Ort::RunOptions run_options{nullptr};

  std::vector<const char *> input_names(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    input_names[i] = inputs[i].c_str();
  }

  std::vector<const char *> output_names(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    output_names[i] = outputs[i].c_str();
  }

  return session.Run(run_options, input_names.data(), input_values,
                     inputs.size(), output_names.data(), outputs.size());
}

template <typename T> inline Ort::Value to_ort_value(xt::xarray<T> &tensor) {
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<int64_t> shape(tensor.shape().size());
  for (size_t i = 0; i < tensor.shape().size(); ++i) {
    shape[i] = static_cast<int64_t>(tensor.shape()[i]);
  }

  std::cout << tensor.size() << " " << shape.size() << std::endl;

  Ort::Value value = Ort::Value::CreateTensor<T>(
      memory_info, tensor.data(), tensor.size(), shape.data(), shape.size());
  return value;
}
} // namespace moondream