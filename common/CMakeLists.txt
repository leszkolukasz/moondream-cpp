add_library(moondream INTERFACE)

target_include_directories(moondream INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(xtensor CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

target_link_libraries(moondream INTERFACE
    stb
    onnxruntime
    xtensor
    nlohmann_json
)
