add_library(moondream INTERFACE)

target_include_directories(moondream INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(moondream INTERFACE
    onnxruntime
    xtensor
    nlohmann_json
)
