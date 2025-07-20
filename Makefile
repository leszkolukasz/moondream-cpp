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
clean:
	rm -rf build
