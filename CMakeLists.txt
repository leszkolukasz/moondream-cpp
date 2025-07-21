cmake_minimum_required(VERSION 3.14)
project(moondream_cpp)

include("$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")

find_package(xtensor CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_subdirectory(external/onnxruntime)
add_subdirectory(external/stb)
add_subdirectory(common)
add_subdirectory(desktop)
# add_subdirectory(mobile)
