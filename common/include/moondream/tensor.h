#pragma once

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xdynamic_view.hpp>
#include <xtensor/views/xview.hpp>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize2.h>

namespace moondream {

// input: (H, W, C)
inline xt::xarray<float> resize_tensor(const xt::xarray<float> &input,
                                       int target_height, int target_width) {
  int input_height = static_cast<int>(input.shape()[0]);
  int input_width = static_cast<int>(input.shape()[1]);
  int channels = static_cast<int>(input.shape()[2]);

  int in_stride = input_width * channels * sizeof(float);
  int out_stride = target_width * channels * sizeof(float);

  float *resized = stbir_resize_float_linear(
      input.data(), input_width, input_height, in_stride, nullptr, target_width,
      target_height, out_stride, STBIR_RGB);

  if (!resized)
    throw std::runtime_error("stbir_resize_float_linear failed");

  std::vector<size_t> shape = {static_cast<size_t>(target_height),
                               static_cast<size_t>(target_width),
                               static_cast<size_t>(channels)};
  return xt::adapt(resized, target_height * target_width * channels,
                   xt::acquire_ownership(), shape);
}

// template <class T>
inline xt::xarray<float> stack_vector(const std::vector<xt::xarray<float>> &vec,
                                      size_t axis = 0) {
  if (vec.empty())
    throw std::runtime_error("Cannot stack an empty vector");

  auto old_rank = vec[0].dimension();
  auto new_rank = old_rank + 1;
  auto old_shape = vec[0].shape();

  std::vector<size_t> shape(new_rank);
  shape[axis] = vec.size();

  for (size_t i = 0, j = 0; i < old_rank; ++i, ++j) {
    if (i == axis) {
      j++;
    }
    shape[j] = old_shape[i];
  }

  auto output = xt::xarray<float>::from_shape(shape);

  for (size_t k = 0; k < vec.size(); ++k) {
    xt::xdynamic_slice_vector sleft;
    xt::xdynamic_slice_vector sright;

    for (size_t i = 0; i < new_rank; ++i) {
      if (i == axis) {
        sleft.push_back(
            xt::range(static_cast<int>(k), static_cast<int>(k + 1)));
        sright.push_back(xt::newaxis());
      } else {
        sleft.push_back(xt::all());
        sright.push_back(xt::all());
      }
    }

    xt::dynamic_view(output, sleft) = xt::dynamic_view(vec[k], sright);
  }

  return output;
}

inline xt::xarray<float>
concat_vector(const std::vector<xt::xarray<float>> &vec, size_t axis = 0) {
  if (vec.empty())
    throw std::runtime_error("Cannot concatenate an empty vector");

  auto rank = vec[0].dimension();
  auto old_shape = vec[0].shape();
  auto new_shape = old_shape;
  new_shape[axis] = 0;

  for (const auto &arr : vec) {
    new_shape[axis] += arr.shape()[axis];
  }

  auto output = xt::xarray<float>::from_shape(new_shape);

  for (size_t k = 0, offset = 0; k < vec.size(); ++k) {
    xt::xdynamic_slice_vector sleft;

    for (size_t i = 0; i < rank; ++i) {
      if (i == axis) {
        sleft.push_back(xt::range(offset, offset + vec[k].shape()[i]));
        offset += vec[k].shape()[i];
      } else {
        sleft.push_back(xt::all());
      }
    }

    xt::dynamic_view(output, sleft) = vec[k];
  }

  return output;
};
} // namespace moondream