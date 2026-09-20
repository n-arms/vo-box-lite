# Contributing

## Architecture
All algorithm modules are no_std and allocation-free (although main allocates).

| File | Purpose |
|---|---|
| `lib.rs` | Crate root. |
| `pyramid.rs` | 7-level scale pyramid, per level FAST-12 -> 5x5 blur -> rBRIEF, keypoints projected back to level-0 pixels. |
| `fast.rs` | FAST-12 detector: scalar ID3-tree detect + score + non-max suppression, and the `ee` submodule (S3 SIMD detector). |
| `fast12_ee.rs` | S3 EE/PIE SIMD FAST-12 scan: 16 px/lane 3-of-4-cardinal heuristic + scalar pattern-tree confirm. Bit-identical to the scalar detector. |
| `fast12_trees.rs` | **Generated** naive 16-pixel ID3 trees used by `fast.rs`. |
| `fast12_cardinal_trees.rs` | **Generated** pattern trees for the SIMD heuristic's confirm step. |
| `blur.rs` | 5x5 separable box blur, clamped borders; EE/PIE SIMD on-device. |
| `downscale.rs` | Fixed-ratio downsamplers. |
| `rbrief.rs` | Rotation-aware BRIEF descriptor: ORB intensity-centroid orientation + 256 rotated learned pairs (no libm). |
| `matcher.rs` | Brute-force Hamming BRIEF descriptor matcher.. |
| `localize.rs` | Matches query pyramid features to one map frame's points, then recovers pose via PnP. |
| `ransac.rs` | Linear-DLT PnP + fixed-iteration RANSAC, plus LM pose refinement. |
| `camera.rs` | Safe wrapper over the esp32-camera driver (OV3660). |
| `semantic.rs` | int8 calc8 embedder (esp-tflite-micro + esp-nn). |
| `bin/main.rs` | Firmware entry. |

Generated files (`fast12_trees.rs`, `fast12_cardinal_trees.rs`) come from
`scripts/gen_fast12_scalar.py`; regenerate, never hand-edit.

