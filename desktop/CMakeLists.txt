cmake_minimum_required(VERSION 3.14)

project(moondream_desktop)

add_executable(moondream_desktop
    main.cpp
)

target_link_libraries(moondream_desktop
    onnxruntime
    moondream
)
