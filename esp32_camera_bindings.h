// Bindgen entry point for the esp32-camera *extra component* (see
// `[package.metadata."esp-idf-sys"]` in Cargo.toml).
//
// esp-idf-sys compiles the `espressif/esp32-camera` remote component (fetched by the
// IDF component manager) and runs bindgen over this file, appending the generated
// bindings as the `esp_idf_sys::camera` module. The include path for `esp_camera.h`
// (and its `esp_jpeg` dependency) is provided by the C build itself.
#pragma once

#include "esp_camera.h"
