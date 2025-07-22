#pragma once

#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <xtensor/io/xio.hpp>
#include <xtensor/io/xnpy.hpp>
#include <xtensor/misc/xsort.hpp>

#include <moondream/constants.h>
#include <moondream/image.h>
#include <moondream/tokenizer.h>
#include <moondream/utils.h>

namespace moondream {

class Moondream {
public:
  inline explicit Moondream(const std::string &model_path) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    env = std::make_unique<Ort::Env>(
        Ort::Env(ORT_LOGGING_LEVEL_WARNING, "moondream"));

    coord_encoder = std::make_unique<Ort::Session>(
        load_ONNX(model_path + "/coord_encoder.onnx", *env, options));
    coord_decoder = std::make_unique<Ort::Session>(
        load_ONNX(model_path + "/coord_decoder.onnx", *env, options));
    size_encoder = std::make_unique<Ort::Session>(
        load_ONNX(model_path + "/size_encoder.onnx", *env, options));
    size_decoder = std::make_unique<Ort::Session>(
        load_ONNX(model_path + "/size_decoder.onnx", *env, options));
    text_encoder = std::make_unique<Ort::Session>(
        load_ONNX(model_path + "/text_encoder.onnx", *env, options));
    text_decoder = std::make_unique<Ort::Session>(
        load_ONNX(model_path + "/text_decoder.onnx", *env, options));
    vision_encoder = std::make_unique<Ort::Session>(
        load_ONNX(model_path + "/vision_encoder.onnx", *env, options));
    vision_projection = std::make_unique<Ort::Session>(
        load_ONNX(model_path + "/vision_projection.onnx", *env, options));

    config = std::make_unique<nlohmann::json>(
        load_json(model_path + "/config.json"));

    initial_kv_cache = std::make_unique<xt::xarray<float>>(
        xt::load_npy<float>(model_path + "/initial_kv_cache.npy"));

    tokenizer = std::make_unique<Tokenizer>(model_path + "/tokenizer.json");
  }

  inline EncodedImage encode_image(const std::string &image_uri) const {
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

      image = resize_tensor(image, target_height, target_width);
    }

    auto [patches, patch_template] = create_patches(image);
    patches = xt::transpose(
        patches, {0, 3, 1, 2}); // (num patches, 3, patchSize, patchSize)

    Ort::Value *input_values = new Ort::Value[2];
    input_values[0] = std::move(to_ort_value<float>(patches));

    auto result =
        run_onnx(*vision_encoder, {"input"}, input_values, {"output"});

    auto patch_emb = from_ort_value(result.at(0)); // (num patches, 729, 720)
    patch_emb = process_patch_embeddings(patch_emb, patch_template);
    patch_emb = patch_emb.reshape({1, 729, 1440});

    input_values[0] = std::move(to_ort_value<float>(patch_emb));
    result = run_onnx(*vision_projection, {"input"}, input_values, {"output"});

    auto &input_emb = result.at(0); // (1, 729, 1024)
    input_values[0] = std::move(input_emb);
    input_values[1] = std::move(
        to_ort_value<float>(*initial_kv_cache)); // (24, 2, 1, 16, 1, 64)

    result = run_onnx(*text_decoder, {"input_embeds", "kv_cache"}, input_values,
                      {"new_kv_cache"});

    auto new_kv_cache = from_ort_value(result.at(0)); // (24, 2, 1, 16, 729, 64)
    auto kv_cache = xt::concatenate(xt::xtuple(*initial_kv_cache, new_kv_cache),
                                    4); // (24, 2, 1, 16, 730, 64)

    return EncodedImage(
        std::make_unique<xt::xarray<float>>(std::move(kv_cache)));
  }

  inline std::string
  generate(xt::xarray<float> input_embeds, // (1, seq_len, 1024)
           const EncodedImage &encoded_image, int max_tokens) const {
    int kv_size = encoded_image.kv_cache->shape()[4];
    int input_len = input_embeds.shape()[1];

    auto kv_cache = prepare_kv_cache(encoded_image);
    std::string text;

    auto input_embeds_ort = to_ort_value<float>(input_embeds);

    for (int generated = 0; generated < max_tokens; ++generated) {
      Ort::Value *input_values = new Ort::Value[2];
      input_values[0] = std::move(input_embeds_ort);

      xt::xarray<float> relevant_kv_cache = xt::xarray<float>::from_shape(
          {24, 2, 1, 16, static_cast<unsigned long>(kv_size), 64});
      relevant_kv_cache = xt::view(kv_cache, xt::all(), xt::all(), xt::all(),
                                   xt::all(), xt::range(0, kv_size), xt::all());

      input_values[1] = std::move(to_ort_value<float>(relevant_kv_cache));

      auto result = run_onnx(*text_decoder, {"input_embeds", "kv_cache"},
                             input_values, {"logits", "new_kv_cache"});

      auto logits = from_ort_value(result.at(0)); // (1, 51200)
      auto kv_cache_update = from_ort_value(result.at(1));

      xt::view(kv_cache, xt::all(), xt::all(), xt::all(), xt::all(),
               xt::range(kv_size, kv_size + input_len), xt::all()) =
          kv_cache_update;

      kv_size += input_len;

      auto next_token = static_cast<int>(xt::argmax(logits, 1)(0));

      // EOS
      if (next_token == 50256)
        break;

      text += tokenizer->decode({next_token});

      auto input_ids = xt::xarray<int64_t>::from_shape({1, 1});
      input_ids(0, 0) = next_token;

      input_values[0] = std::move(to_ort_value<int64_t>(input_ids));
      result = run_onnx(*text_encoder, {"input_ids"}, input_values,
                        {"input_embeds"});

      input_embeds_ort = std::move(result.at(0));
      input_len = 1;

      std::cout << text << "\n";
    }

    return text;
  }

  inline std::string caption(const std::string &image_uri,
                             const std::string &length,
                             int max_tokens = 50) const {
    auto prompt = config->at("templates")
                      .at("caption")
                      .at(length)
                      .get<std::vector<int>>();

    xt::xarray<int64_t> input_ids =
        xt::xarray<int64_t>::from_shape({1, prompt.size()});

    for (size_t i = 0; i < prompt.size(); ++i) {
      input_ids(0, i) = prompt[i];
    }

    auto input_ids_ort = to_ort_value<int64_t>(input_ids);
    auto result = run_onnx(*text_encoder, {"input_ids"}, &input_ids_ort,
                           {"input_embeds"});

    auto input_embeds = from_ort_value(result.at(0));

    auto image = encode_image(image_uri);
    return generate(input_embeds, image, max_tokens);
  }

private:
  std::unique_ptr<Ort::Env> env;

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

  inline xt::xarray<float> prepare_kv_cache(const EncodedImage &src) const {
    auto old_shape = src.kv_cache->shape();

    std::vector<std::size_t> new_shape(old_shape.begin(), old_shape.end());
    new_shape[4] = CONTEXT_WINDOW;

    xt::xarray<float> new_kv = xt::zeros<float>(new_shape);

    xt::view(new_kv, xt::all(), xt::all(), xt::all(), xt::all(),
             xt::range(0, old_shape[4]), xt::all()) = *(src.kv_cache);

    return new_kv;
  }
};

} // namespace moondream
