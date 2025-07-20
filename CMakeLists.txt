cmake_minimum_required(VERSION 3.14)
project(moondream_cpp)

include("$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")

add_subdirectory(external/onnxruntime)
add_subdirectory(common)
add_subdirectory(desktop)
# add_subdirectory(mobile)
