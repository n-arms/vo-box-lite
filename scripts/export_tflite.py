#!/usr/bin/env python3
"""Export the int8 yolov8n .tflite loaded into flash by the `semantic` task.

yolov8n.onnx (fp32)
  --split head-->  yolov8n_split.onnx (2 outputs: boxes, scores)
  --onnx2tf-->  float32 .tflite  --onnx2tf -oiqt (calibrated)-->  full-int8 .tflite

The output is copied to models/yolov8n.tflite (int8, 2 outputs), which
scripts/cargo_run.sh flashes to the `model` partition.

The head split exists because the stock graph Concat's boxes (0..640) and class
scores (0..1) into one output; a single int8 scale then flattens the scores to
zero. Separate outputs get separate scales.

Run it with the onnx2tf venv created next to this script:

  python3 -m venv --system-site-packages scripts/.tflite-venv
  scripts/.tflite-venv/bin/pip install --no-deps onnx2tf==1.26.3 \
      onnx_graphsurgeon sng4onnx tf-keras
  scripts/.tflite-venv/bin/python scripts/export_tflite.py

Two onnx2tf quirks are handled here:
  * It fails on the *QDQ* int8 .onnx at the DFL-head Concat (layout inference),
    so we convert the fp32 graph and quantize it with -oiqt instead.
  * Its default validation snippet downloads a release asset that now 404s; we
    pre-place a dummy at the expected CWD filename so np.load succeeds.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

IMGSZ = 640


def build_calibration(image_dir: Path, out_npy: Path, count: int) -> None:
    """(count, 640, 640, 3) float32 in [0,1], letterboxed like inference."""
    paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg", ".webp"}
    )[:count]
    if not paths:
        sys.exit(f"error: no calibration images in {image_dir}")
    samples = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        scale = min(IMGSZ / w, IMGSZ / h)
        nw, nh = round(w * scale), round(h * scale)
        im = im.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (IMGSZ, IMGSZ), (114, 114, 114))
        canvas.paste(im, ((IMGSZ - nw) // 2, (IMGSZ - nh) // 2))
        samples.append(np.asarray(canvas, dtype=np.float32) / 255.0)
    np.save(out_npy, np.stack(samples))
    print(f"[calib] {len(paths)} imgs -> {out_npy}")


def split_head(onnx_path: Path, out_path: Path) -> Path:
    """Expose the head Concat's box and class tensors as two graph outputs."""
    import onnx
    from onnx import shape_inference

    m = shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    g = m.graph
    producers = {o: n for n in g.node for o in n.output}
    concat = next((producers.get(o.name) for o in g.output
                   if producers.get(o.name) is not None and
                   producers[o.name].op_type == "Concat"), None)
    if concat is None or len(concat.input) != 2:
        sys.exit("error: could not find the 2-input head Concat in the ONNX graph")
    vi = {v.name: v for v in g.value_info}
    names = list(concat.input)  # [boxes (1,4,8400), scores (1,80,8400)]
    for n in names:
        if n not in vi:
            sys.exit(f"error: no shape info for head tensor {n!r}")
    g.node.remove(concat)
    del g.output[:]
    for n in names:
        o = g.output.add()
        o.name = n
        o.type.CopyFrom(vi[n].type)
    onnx.save(m, str(out_path))
    shapes = [tuple(d.dim_value for d in vi[n].type.tensor_type.shape.dim) for n in names]
    print(f"[split] {onnx_path.name} -> {out_path.name} outputs {shapes}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default="models/yolov8n.onnx", help="fp32 ONNX (yolo_quant.py output)")
    ap.add_argument("--split-onnx", default="models/yolov8n_split.onnx", help="2-output ONNX to write")
    ap.add_argument("--out", default="models/yolov8n.tflite", help="int8 tflite to write")
    ap.add_argument("--work", default="models/tflite_int8", help="onnx2tf output dir")
    ap.add_argument("--images", default="map_run/bmps", help="calibration image dir")
    ap.add_argument("--max-calib", type=int, default=16, help="calibration images")
    ap.add_argument("--onnx2tf", default="", help="onnx2tf executable (default: next to this Python)")
    args = ap.parse_args()
    if not args.onnx2tf:
        args.onnx2tf = str(Path(sys.executable).with_name("onnx2tf"))

    root = Path(__file__).resolve().parent.parent
    onnx = root / args.onnx
    if not onnx.exists():
        sys.exit(f"error: {onnx} missing — run scripts/yolo_quant.py first")
    onnx = split_head(onnx, root / args.split_onnx)
    work = root / args.work
    work.mkdir(parents=True, exist_ok=True)

    calib = root / "models" / "calib_640.npy"
    build_calibration(root / args.images, calib, args.max_calib)

    # onnx2tf reads this fixed filename from CWD for its accuracy check.
    dummy = root / "calibration_image_sample_data_20x128x128x3_float32.npy"
    if not dummy.exists():
        np.save(dummy, np.zeros((20, 128, 128, 3), dtype=np.float32))

    cmd = [
        args.onnx2tf,
        "-i", str(onnx),
        "-o", str(work),
        "-oiqt",
        "-cind", "images", str(calib), "[[[[0,0,0]]]]", "[[[[1,1,1]]]]",
    ]
    print("[onnx2tf]", " ".join(cmd))
    subprocess.run(cmd, cwd=root, check=True)

    produced = sorted(work.glob(f"{onnx.stem}*full_integer_quant.tflite"))
    if not produced:
        sys.exit(f"error: no full-integer tflite under {work}")
    out = root / args.out
    out.write_bytes(produced[-1].read_bytes())
    print(f"[done] {produced[-1].name} -> {out} ({out.stat().st_size} B)")


if __name__ == "__main__":
    main()
