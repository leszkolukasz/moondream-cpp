cmake_minimum_required(VERSION 3.14)
project(moondream_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include("$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")

add_subdirectory(external/onnxruntime)
add_subdirectory(external/stb)
add_subdirectory(common)

if (NOT ANDROID)
    add_subdirectory(desktop)
else()
    add_subdirectory(mobile)
endif()
