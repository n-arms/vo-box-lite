#!/usr/bin/env python3
"""Offline calc8 place-recognition embeddings for map building (laptop side).

Runs models/calc8.tflite (CALC encoder, int8) on each gray map frame exactly the
way the device would: truncating 4x4 block mean (== src/downscale.rs, matching
src/semantic.rs) of the 640x480 frame -> 160x120 uint8 -> 1064-byte embedding.
Imported lazily by receive_map's build phase (kept out of the receiver path).
"""

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO / "models" / "calc8.tflite"
EMB_DIM = 1064
IN_W, IN_H = 160, 120  # model input WxH; 640x480 / 4


def downscale_4x4(gray: np.ndarray) -> np.ndarray:
    """Truncating mean of each non-overlapping 4x4 block (== downscale_4x4)."""
    h, w = gray.shape
    dh, dw = h // 4, w // 4
    blocks = gray[: dh * 4, : dw * 4].astype(np.uint16).reshape(dh, 4, dw, 4)
    return (blocks.sum(axis=(1, 3)) >> 4).astype(np.uint8)


def compute_embeddings(bmp_paths, model_path: Path = DEFAULT_MODEL) -> dict:
    """-> {stem: bytes[1064]} for each gray BMP (dims must be 4x the model input)."""
    import tensorflow as tf
    from PIL import Image

    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"{model_path} not found — copy it from "
            f"calc-quant/quant/calc_int8.tflite"
        )
    interp = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    embs = {}
    for p in bmp_paths:
        with Image.open(p) as im:
            gray = np.asarray(im.convert("L"), dtype=np.uint8)
        small = downscale_4x4(gray)
        if small.shape != (IN_H, IN_W):
            raise ValueError(
                f"{p.name}: {gray.shape[1]}x{gray.shape[0]} -> "
                f"{small.shape[1]}x{small.shape[0]}, expected {IN_W}x{IN_H}"
            )
        interp.set_tensor(inp["index"], small[None, :, :, None])
        interp.invoke()
        emb = interp.get_tensor(out["index"])[0].astype(np.uint8)
        assert emb.size == EMB_DIM
        embs[p.stem] = emb.tobytes()
    return embs


def load_embeddings_dir(d) -> dict:
    """{stem: bytes[1064]} from a dir of per-frame .npy (uint8, 1064)."""
    embs = {}
    for p in sorted(Path(d).glob("*.npy")):
        arr = np.load(p).astype(np.uint8)
        if arr.size != EMB_DIM:
            raise ValueError(f"{p.name}: {arr.size} values, expected {EMB_DIM}")
        embs[p.stem] = arr.tobytes()
    return embs
