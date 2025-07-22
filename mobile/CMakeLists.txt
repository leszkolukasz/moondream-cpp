cmake_minimum_required(VERSION 3.16)

project(moondream_mobile VERSION 0.1 LANGUAGES CXX)

find_package(Qt6 REQUIRED COMPONENTS Quick Concurrent)

qt_standard_project_setup(REQUIRES 6.8)

qt_add_executable(appmoondream_mobile
    main.cpp
)

qt_add_qml_module(appmoondream_mobile
    URI moondream_mobile
    VERSION 1.0
    QML_FILES
        Main.qml
        SOURCES moondreamwrapper.h moondreamwrapper.cpp
)

# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
set_target_properties(appmoondream_mobile PROPERTIES
#    MACOSX_BUNDLE_GUI_IDENTIFIER com.example.appmoondream_mobile
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

target_link_libraries(appmoondream_mobile
    PRIVATE moondream
            Qt6::Quick
            Qt6::Concurrent
)

include(GNUInstallDirs)
install(TARGETS appmoondream_mobile
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
