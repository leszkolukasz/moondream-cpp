#pragma once

#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

#include <moondream/constants.h>
#include <moondream/image.h>
#include <moondream/tokenizer.h>

namespace moondream {

inline Ort::Session loadONNX(const std::string &model_path,
                             const Ort::SessionOptions &options) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "moondream");
  return Ort::Session(env, model_path.c_str(), options);
}

// class Moondream {
// public:
//   static std::unique_ptr<Moondream> load(const std::string &model_path) {
//     auto instance = std::make_unique<Moondream>();

//     Ort::SessionOptions options;
//     options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
//     options.SetLogSeverityLevel(3);
//     options.DisableMemPattern();

//     instance->coord_encoder =
//         loadONNX(model_path + "/coord_encoder.onnx", options);
//     instance->coord_decoder =
//         loadONNX(model_path + "/coord_decoder.onnx", options);
//     instance->size_encoder =
//         loadONNX(model_path + "/size_encoder.onnx", options);
//     instance->size_decoder =
//         loadONNX(model_path + "/size_decoder.onnx", options);
//     instance->text_encoder =
//         loadONNX(model_path + "/text_encoder.onnx", options);
//     instance->text_decoder =
//         loadONNX(model_path + "/text_decoder.onnx", options);
//     instance->vision_encoder =
//         loadONNX(model_path + "/vision_encoder.onnx", options);
//     instance->vision_projection =
//         loadONNX(model_path + "/vision_projection.onnx", options);

//     instance->config = loadModelConfig(model_path + "/config.json");

//     auto initial_kv = loadJSON(model_path + "/initial_kv_cache.json");
//     instance->initial_kv_cache =
//         xt::adapt(reinterpret_cast<const float
//         *>(initial_kv["data"].get_ptr()),
//                   initial_kv["shape"].size(), xt::no_ownership(),
//                   std::vector<std::size_t>(initial_kv["shape"].begin(),
//                                            initial_kv["shape"].end()));

//     instance->tokenizer =
//         Tokenizer::fromConfig(loadJSON(model_path + "/tokenizer.json"));

//     return instance;
//   }

//   EncodedImage encodeImage(const std::string &image_uri) const {
//     auto image = loadImage(image_uri);
//     auto image_tensor = imageToTensor(image);

//     float scale = static_cast<float>(MAX_IMAGE_SIZE) /
//                   std::max(image.width, image.height);
//     if (scale < 1.0f) {
//       int target_width = static_cast<int>(image.width * scale);
//       int target_height = static_cast<int>(image.height * scale);
//       image_tensor = resize(image_tensor, target_width, target_height);
//     }

//     auto [patches, patch_template] = createPatches(image_tensor);
//     patches = xt::swapaxes(xt::swapaxes(patches, 2, 3), 1, 2);

//     auto result = runONNX(vision_encoder, {{"input", toTensor(patches)}});
//     auto patch_emb = fromTensor(result["output"]);
//     patch_emb = processPatchEmbeddings(patch_emb, patch_template);
//     patch_emb.reshape({1, 729, 1440});

//     result = runONNX(vision_projection, {{"input", toTensor(patch_emb)}});
//     auto input_emb = fromTensor(result["output"]);

//     result = runONNX(text_decoder, {{"input_embeds", toTensor(input_emb)},
//                                     {"kv_cache",
//                                     toTensor(initial_kv_cache)}});

//     auto new_kv = fromTensor(result["new_kv_cache"]);

//     auto kv = concatenateAlongAxis({initial_kv_cache, new_kv}, 4);
//     return EncodedImage{kv};
//   }

//   std::string generate(xt::xarray<float> input_embeds,
//                        const EncodedImage &encoded_image,
//                        int max_tokens) const {
//     int kv_size = encoded_image.kv_cache.shape()[4];
//     int input_len = input_embeds.shape()[1];

//     auto kv_cache = prepareKVCache(encoded_image.kv_cache);
//     std::string text;

//     for (int generated = 0; generated < max_tokens; ++generated) {
//       auto sliced = slice(kv_cache, 0, kv_size);
//       auto result =
//           runONNX(text_decoder, {{"input_embeds", toTensor(input_embeds)},
//                                  {"kv_cache", toTensor(sliced)}});

//       auto logits = fromTensor(result["logits"]);
//       int next_token = argmax(logits);

//       text += tokenizer.decode({next_token});

//       auto encoded = runONNX(
//           text_encoder, {{"input_ids",
//           toTensor(int64Tensor({next_token}))}});

//       input_embeds = fromTensor(encoded["input_embeds"]);
//       assignSlice(kv_cache, kv_size, input_len,
//                   fromTensor(result["new_kv_cache"]));
//       kv_size += input_len;
//       input_len = 1;
//     }

//     return text;
//   }

//   std::string caption(const std::string &image_uri,
//                       const std::vector<int64_t> &prompt,
//                       int max_tokens = 50) const {
//     auto encoded =
//         runONNX(text_encoder, {{"input_ids",
//         toTensor(int64Tensor(prompt))}});

//     auto input_embeds = fromTensor(encoded["input_embeds"]);
//     auto image = encodeImage(image_uri);
//     return generate(input_embeds, image, max_tokens);
//   }

// private:
//   Ort::Session vision_encoder;
//   Ort::Session vision_projection;
//   Ort::Session text_encoder;
//   Ort::Session text_decoder;
//   Ort::Session size_encoder;
//   Ort::Session size_decoder;
//   Ort::Session coord_encoder;
//   Ort::Session coord_decoder;

//   xt::xarray<float> initial_kv_cache;
//   Tokenizer tokenizer;
//   ModelConfig config;

//   xt::xarray<float> prepareKVCache(const xt::xarray<float> &src) const {
//     auto new_shape = src.shape();
//     new_shape[4] = CONTEXT_WINDOW;
//     auto new_kv = xt::zeros<float>(new_shape);
//     assignSlice(new_kv, 0, src.shape()[4], src);
//     return new_kv;
//   }

//   xt::xarray<int64_t> int64Tensor(const std::vector<int64_t> &values) const {
//     return xt::adapt(values);
//   }

//   int argmax(const xt::xarray<float> &logits) const {
//     return static_cast<int>(xt::argmax(logits)[1]);
//   }

//   xt::xarray<float> slice(const xt::xarray<float> &arr, int start,
//                           int end) const {
//     auto view = xt::view(arr, xt::all(), xt::all(), xt::all(), xt::all(),
//                          xt::range(start, end), xt::all());
//     return view;
//   }
// };

} // namespace moondream
