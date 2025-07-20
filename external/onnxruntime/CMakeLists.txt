add_library(onnxruntime INTERFACE)

if(ANDROID)
    target_include_directories(onnxruntime INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/android/include)

    target_link_libraries(onnxruntime INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/android/lib/arm64-v8a/libonnxruntime.so)
elseif(UNIX)
    target_include_directories(onnxruntime INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/linux/include)

    target_link_libraries(onnxruntime INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/linux/lib/libonnxruntime.so)
endif()
