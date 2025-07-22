.PHONY: all build_linux run_linux clean

CMAKE_CONFIGURE=cmake --preset
CMAKE_BUILD=cmake --build
CMAKE_DIR=.

all: build_linux

build_linux:
	$(CMAKE_CONFIGURE) linux-debug
	$(CMAKE_BUILD) $(CMAKE_DIR)/build/linux-debug

run_linux: build_linux
	@./build/linux-debug/desktop/moondream_desktop

build_android:
	$(CMAKE_CONFIGURE) android-debug \
		-DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake
	$(CMAKE_BUILD) $(CMAKE_DIR)/build/android-debug
run_android: build_android
	@adb install -r build/android-debug/app/build/outputs/apk/debug/app-debug.apk

format:
	git ls-files '*.cpp' '*.h' ':!external/*' | xargs clang-format -i

clean:
	rm -rf build
