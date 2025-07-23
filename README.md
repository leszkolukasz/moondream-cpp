## Moondream C++

This is a C++ library/Android app that allows for inference of the [Moondream](https://github.com/vikhyat/moondream) model.

> [!NOTE]
> This is a C++ port of NodeJS version [moondream-nodejs](https://github.com/leszkolukasz/moondream-nodejs) howerver it is better as it is more performant and actually runs on Android.

## Model files

The weights are from the `.mf` file which, at least in the past, could be found in the moondream repo. Folder `scripts` contains a `unpack.py` script that unpacks `.mf` file into `.onnx`, `.json`, `.npy` files. File `convert.py` converts those files to format that can be run in C++/Android that is:

- `.onnx`, `.npy` operations are cast to Float32

> [!NOTE]
> This is leftover from the NodeJS version, I should probably check if Float16 works in C++.

# Building

Requirements:

- CMake 3.14+
- Vcpkg
- Android SDK/NDK (for mobile)
- Qt 6 (for mobile)

There are two main CMake projects in this repository:

- `desktop` - for desktop inference (no GUI)
- `mobile` - for Android inference

Make sure to set `VCPKG_ROOT` e.g. `$HOME/.local/share/vcpkg`. Then install all dependencies using Vcpkg:

```bash
vcpkg install
```

Dependencies for onnxruntime and stb are already included in `external` folder.

## Desktop

Run command:

```bash
make build_linux
```

There is no CUDA support for now.

## Mobile

Open the project in Qt Creator (choose main CMakeLists.txt). Set the Android kit and build `appmoondream_mobile` target from `mobile` folder.

You may need to update paths to `libssl_3.so` and `libcrypto_3.so` in `mobile/CMakeLists.txt` if they are not in `$HOME/Android/Sdk`.

# Usage

## Desktop

In `desktop/main.cpp` update paths to the folder with model files and the image you want to caption. The default path is `./data` and the image is `./frieren.jpg`.

Run the built binary:

```bash
make run_linux
```

Example output:

```
Resizing image from 773x767 to 768x762 with scale 0.993532
Chosen template: 2x2
Caption:  The image depicts a young girl with light blue hair and elf-like features, wearing a white and green elf costume. She is holding a large, golden-brown pastry, possibly a donut, in her hands. The girl is seated at a table, which has a dark brown surface, and is wearing a white and green elf costume. The background shows a restaurant or cafe setting with a dark-colored wall and a partially visible red-brown table. The girl's costume and the pastry suggest a possible connection to a food-related event or activity.
```

## Mobile

1. Start app in Qt Creator.
2. Click on `Load Model`. It will download the model files from the repository.
3. Click on `Pick Image` to select an image from the gallery.
4. Choose captioning mode (`short` or `normal`).
5. Click on `Caption Image` to get the caption (streaming supported).

### Demo

https://github.com/user-attachments/assets/faad04c8-2120-441f-8f6d-acc6bc091c1a
