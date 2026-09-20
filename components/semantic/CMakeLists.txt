idf_component_register(
    SRCS "tflite_shim.cc"
    INCLUDE_DIRS "."
    REQUIRES espressif__esp-tflite-micro esp_timer
)
