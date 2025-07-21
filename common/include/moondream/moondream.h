#pragma once

#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <xtensor/io/xio.hpp>
#include <xtensor/io/xnpy.hpp>

#include <moondream/constants.h>
#include <moondream/image.h>
#include <moondream/tokenizer.h>
#include <moondream/utils.h>

namespace moondream {

class Moondream {
public:
  explicit Moondream(const std::string &model_path) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    coord_encoder = std::make_unique<Ort::Session>(
        load_ONNX(model_path + "/coord_encoder.onnx", options));
    coord_decoder = std::make_unique<Ort::Session>(
        load_ONNX(model_path + "/coord_decoder.onnx", options));
    size_encoder = std::make_unique<Ort::Session>(
        load_ONNX(model_path + "/size_encoder.onnx", options));
    size_decoder = std::make_unique<Ort::Session>(
        load_ONNX(model_path + "/size_decoder.onnx", options));
    text_encoder = std::make_unique<Ort::Session>(
        load_ONNX(model_path + "/text_encoder.onnx", options));
    text_decoder = std::make_unique<Ort::Session>(
        load_ONNX(model_path + "/text_decoder.onnx", options));
    vision_encoder = std::make_unique<Ort::Session>(
        load_ONNX(model_path + "/vision_encoder.onnx", options));
    vision_projection = std::make_unique<Ort::Session>(
        load_ONNX(model_path + "/vision_projection.onnx", options));

    config = std::make_unique<nlohmann::json>(
        load_json(model_path + "/config.json"));

    initial_kv_cache = std::make_unique<xt::xarray<float>>(
        xt::load_npy<float>(model_path + "/initial_kv_cache.npy"));

    tokenizer = std::make_unique<Tokenizer>(model_path + "/tokenizer.json");
  }

  inline EncodedImage *encode_image(const std::string &image_uri) const {
    auto image = load_image(image_uri);
    auto height = image.shape()[0];
    auto width = image.shape()[1];

    float scale = static_cast<float>(MAX_IMAGE_SIZE) / std::max(width, height);
    if (scale < 1.0f) {
      int target_width = static_cast<int>(static_cast<float>(width) * scale);
      int target_height = static_cast<int>(static_cast<float>(height) * scale);

      std::cout << "Resizing image from " << width << "x" << height << " to "
                << target_width << "x" << target_height << " with scale "
                << scale << std::endl;

      image = resize_tensor(image, target_width, target_height);
    }

    auto [patches, patch_template] = create_patches(image);
    patches =
        xt::moveaxis(patches, 3, 1); // (num patches, 3, patchSize, patchSize)

    // There is segmentation fault without this line. WHY???
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "moondream");

    Ort::Value *input_values = new Ort::Value[1];
    input_values[0] = std::move(to_ort_value<float>(patches));

    auto result =
        run_onnx(*vision_encoder, {"input"}, input_values, {"output"});

    // auto result = runONNX(vision_encoder, {{"input",
    // toTensor(patches)}}); auto patch_emb = fromTensor(result["output"]);
    // patch_emb = processPatchEmbeddings(patch_emb, patch_template);
    // patch_emb.reshape({1, 729, 1440});

    // result = runONNX(vision_projection, {{"input",
    // toTensor(patch_emb)}}); auto input_emb =
    // fromTensor(result["output"]);

    // result = runONNX(text_decoder, {{"input_embeds",
    // toTensor(input_emb)},
    //                                 {"kv_cache",
    //                                 toTensor(initial_kv_cache)}});

    // auto new_kv = fromTensor(result["new_kv_cache"]);

    // auto kv = concatenateAlongAxis({initial_kv_cache, new_kv}, 4);
    // return EncodedImage{kv};
    return nullptr;
  }

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

private:
  std::unique_ptr<Ort::Session> vision_encoder;
  std::unique_ptr<Ort::Session> vision_projection;
  std::unique_ptr<Ort::Session> text_encoder;
  std::unique_ptr<Ort::Session> text_decoder;
  std::unique_ptr<Ort::Session> size_encoder;
  std::unique_ptr<Ort::Session> size_decoder;
  std::unique_ptr<Ort::Session> coord_encoder;
  std::unique_ptr<Ort::Session> coord_decoder;

  std::unique_ptr<xt::xarray<float>> initial_kv_cache;
  std::unique_ptr<Tokenizer> tokenizer;
  std::unique_ptr<nlohmann::json> config;

  //   xt::xarray<float> prepareKVCache(const xt::xarray<float> &src) const {
  //     auto new_shape = src.shape();
  //     new_shape[4] = CONTEXT_WINDOW;
  //     auto new_kv = xt::zeros<float>(new_shape);
  //     assignSlice(new_kv, 0, src.shape()[4], src);
  //     return new_kv;
  //   }

  //   xt::xarray<int64_t> int64Tensor(const std::vector<int64_t> &values) const
  //   {
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
};

} // namespace moondream
