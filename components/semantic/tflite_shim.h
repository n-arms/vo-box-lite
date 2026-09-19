// C shim over the esp-tflite-micro C++ interpreter (bindgen entry point, see
// Cargo.toml `[package.metadata.esp-idf-sys]`). The interpreter is a process
// singleton: load once, then invoke in a loop.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Build an interpreter over `model` (flatbuffer, kept alive by the caller)
// reusing `arena` as TFLite-Micro scratch. Returns 0 on success, <0 otherwise.
int semantic_model_load(const uint8_t *model, size_t model_len, uint8_t *arena,
                        size_t arena_len);

// Input tensor byte size (0 before a successful load).
size_t semantic_input_size(void);
// Input tensor data pointer (NULL before a successful load). The caller writes
// the int8 input here before each invoke.
int8_t *semantic_input_data(void);

// Number of model outputs (0 before a successful load).
size_t semantic_output_count(void);
// Output `i` byte size / data pointer / int8 quant params (0 or NULL if out of
// range). yolov8n has two outputs: boxes and class scores.
size_t semantic_output_size(size_t i);
int8_t *semantic_output_data(size_t i);
float semantic_output_scale(size_t i);
int32_t semantic_output_zero_point(size_t i);

// Run one inference. Returns 0 on success.
int semantic_invoke(void);

// Human-readable last error (never NULL).
const char *semantic_last_error(void);

#ifdef __cplusplus
}
#endif
