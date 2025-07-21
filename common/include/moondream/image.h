#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <moondream/constants.h>
#include <moondream/tensor.h>

namespace moondream {

struct EncodedImage {
  xt::xarray<float> kv_cache; // (layer, K/V, batch, head, seq_len, dim)
};

inline xt::xarray<uint8_t> load_image(const std::string &image_path) {
  int width, height, channels;
  unsigned char *data =
      stbi_load(image_path.c_str(), &width, &height, &channels, 3);
  if (!data)
    throw std::runtime_error("Failed to load image");

  std::vector<size_t> shape = {static_cast<size_t>(height),
                               static_cast<size_t>(width),
                               static_cast<size_t>(channels)};
  auto image = xt::adapt(data, height * width * channels,
                         xt::acquire_ownership(), shape);

  return image;
}

inline void save_image(const xt::xarray<uint8_t> &input,
                       const std::string &filename, int quality = 90) {
  int height = static_cast<int>(input.shape()[0]);
  int width = static_cast<int>(input.shape()[1]);
  int channels = static_cast<int>(input.shape()[2]);

  int success = stbi_write_jpg(filename.c_str(), width, height, channels,
                               input.data(), quality);

  if (success == 0)
    throw std::runtime_error("Failed to write JPEG file");
}

inline xt::xarray<float> normalize(const xt::xarray<float> &arr,
                                   float mean = 0.5f, float std = 0.5f) {
  return (arr / 255.0f - mean) / std;
}

inline xt::xarray<float> adaptiveAvgPooling2D(const xt::xarray<float> &input,
                                              std::pair<int, int> outputSize) {
  int inputHeight = input.shape()[0];
  int inputWidth = input.shape()[1];
  int channels = input.shape()[2];
  int outputHeight = outputSize.first;
  int outputWidth = outputSize.second;

  int stride_h = inputHeight / outputHeight;
  int stride_w = inputWidth / outputWidth;
  int kernel_h = inputHeight - (outputHeight - 1) * stride_h;
  int kernel_w = inputWidth - (outputWidth - 1) * stride_w;

  xt::xarray<float> output =
      xt::zeros<float>({outputHeight, outputWidth, channels});

  for (int i = 0; i < outputHeight; ++i) {
    for (int j = 0; j < outputWidth; ++j) {
      int h_start = i * stride_h;
      int w_start = j * stride_w;
      int h_end = h_start + kernel_h;
      int w_end = w_start + kernel_w;

      auto patch = xt::view(input, xt::range(h_start, h_end),
                            xt::range(w_start, w_end), xt::all());
      auto avg = xt::mean(patch, {0, 1});

      xt::view(output, xt::all(), xt::all(), xt::all()) = avg;
    }
  }

  return output;
}

// Input: (height, width, 3) tensor
// Output: patches (num patches, patchSize, patchSize, 3), template (rows, cols)
std::pair<xt::xarray<float>, std::pair<int, int>>
create_patches(const xt::xarray<float> &image, int patch_size) {
  int height = static_cast<int>(image.shape()[0]);
  int width = static_cast<int>(image.shape()[1]);
  int channels = static_cast<int>(image.shape()[2]);

  std::pair<int, int> selectedTemplate = {1, 1};
  std::vector<std::pair<int, int>> candidateTemplates = {
      {1, 2}, {2, 1}, {2, 2}};
  std::vector<xt::xarray<float>> patches;

  xt::xarray<float> global_patch =
      normalize(resize_tensor(image, patch_size, patch_size));
  patches.push_back(global_patch);

  if (std::max(width, height) >= static_cast<int>(patch_size * 1.4f)) {
    float aspectRatio = static_cast<float>(height) / static_cast<float>(width);
    selectedTemplate = *std::min_element(
        candidateTemplates.begin(), candidateTemplates.end(),
        [aspectRatio](const std::pair<int, int> &a,
                      const std::pair<int, int> &b) {
          float diffA = std::abs(static_cast<float>(a.first) /
                                     static_cast<float>(a.second) -
                                 aspectRatio);
          float diffB = std::abs(static_cast<float>(b.first) /
                                     static_cast<float>(b.second) -
                                 aspectRatio);
          return diffA < diffB;
        });

    int patchHeight = height / selectedTemplate.first;
    int patchWidth = width / selectedTemplate.second;

    for (int row = 0; row < selectedTemplate.first; ++row) {
      for (int col = 0; col < selectedTemplate.second; ++col) {
        int rowStart = row * patchHeight;
        int rowEnd = (row + 1) * patchHeight;
        int colStart = col * patchWidth;
        int colEnd = (col + 1) * patchWidth;

        auto cropped = xt::view(image, xt::range(rowStart, rowEnd),
                                xt::range(colStart, colEnd), xt::all());

        xt::xarray<float> patch =
            normalize(resize_tensor(cropped, patch_size, patch_size));
        patches.push_back(patch);
      }
    }
  }

  auto stacked = stack_vector(patches, 0);
  return {stacked, selectedTemplate};
}

// Input: (num patches, 729, 720)
// Output: (729, 2*720)
inline xt::xarray<float>
process_patch_embeddings(xt::xarray<float> patch_emb,
                         std::pair<int, int> patch_template) {
  xt::xarray<float> global_patch_emb = xt::reshape_view(
      xt::view(patch_emb, 0, xt::all(), xt::all()), {1, 729, 720});

  if (patch_template.first == 1 && patch_template.second == 1) {
    return concat_vector({global_patch_emb, global_patch_emb}, 1);
  }

  int seq_len = patch_emb.shape()[1];
  int w = static_cast<int>(std::sqrt(seq_len));

  assert(w * w == 729);

  std::vector<xt::xarray<float>> rows;
  for (int r = 0; r < patch_template.first; ++r) {
    std::vector<xt::xarray<float>> row;
    for (int c = 0; c < patch_template.second; ++c) {
      int idx = r * patch_template.second + c;
      auto patch = xt::reshape_view(
          xt::view(patch_emb, idx, xt::all(), xt::all()), {1, w, w, 720});
      row.push_back(patch);
    }
    rows.push_back(concat_vector(row, 1));
  }

  auto grid = concat_vector(rows, 0);
  grid = adaptiveAvgPooling2D(grid, {w, w});
  grid.reshape({w * w, 720});

  return xt::concatenate(xt::xtuple(global_patch_emb, grid), 1);
}

} // namespace moondream
