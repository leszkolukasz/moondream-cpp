add_library(moondream INTERFACE)

find_package(xtensor CONFIG REQUIRED)

target_include_directories(moondream INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(moondream INTERFACE
    onnxruntime
    xtensor::xtensor
)
