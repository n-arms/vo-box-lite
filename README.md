# vo-box-lite

Visual localization on an extremely resource-constrained ESP32-S3.

**Offline**: the S3 streams 640x480 grayscale frames over WiFi (SoftAP). An external device (like a laptop), computes a CALC embedding per frame, builds a 3D feature map with COLMAP, and uploads the map back over TCP: intrinsics + per-frame embedding + 3D points.

**Online**: each frame runs through a 7-level pyramid (6:5 downscales) of FAST-12 -> 5x5 box blur -> rBRIEF, is embedded with CALC, matched against the nearest map frame (Hamming MNN + Lowe's ratio), and posed with PnP RANSAC.

## Layout
- `src/` — `no_std` feature/geometry lib + `src/bin/main.rs` esp-idf firmware.
- `scripts/` — host tools: map receive/build, host extractor, matcher crate, embeddings.

## Build & run
```bash
. /home/north/export-esp.sh   # esp toolchain on PATH (once per shell)
cargo run                     # build + flash + serial monitor
```
Map run (join the `vo-box` AP first):
```bash
python3 scripts/receive_map.py --duration 20 --interval 1000
```
See `CONTRIBUTING.md` for workflows.
