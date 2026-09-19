#!/usr/bin/env python3
"""Pull yolov8n, int8-quantize it, and run detection over a directory of images.

Pipeline (all cached under --model-dir):
  yolov8n.pt  --ultralytics-->  yolov8n.onnx (fp32)  --ORT static QDQ-->  yolov8n.int8.onnx

Detection preprocesses each image with an aspect-preserving letterbox resize
(scale to fit 640x640, pad the rest), runs the int8 ONNX model, decodes the
YOLOv8 head + NMS itself, then writes annotated copies to <images>/annotated/.

Usage:
  python3 scripts/yolo_quant.py                       # reads test-images/
  python3 scripts/yolo_quant.py --images some/dir --out some/ann
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".ppm", ".pgm"}
IMGSZ = 640
PAD_COLOR = 114


# ---------------------------------------------------------------- preprocessing
def letterbox(img, imgsz=IMGSZ, color=PAD_COLOR):
    """Aspect-preserving resize to fit imgsz, then pad to a square.

    Returns (canvas, scale, pad_x, pad_y) so boxes can be mapped back.
    """
    w, h = img.size
    scale = min(imgsz / w, imgsz / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (imgsz, imgsz), (color, color, color))
    pad_x, pad_y = (imgsz - nw) // 2, (imgsz - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def to_input(img):
    """PIL RGB -> float32 NCHW in [0,1] (YOLOv8 expects RGB)."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None]


# ---------------------------------------------------------------- model pulling
def get_models(model_dir, name="yolov8n"):
    """Return (pt, fp32_onnx, int8_onnx) paths in the model cache."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    return (
        model_dir / f"{name}.pt",
        model_dir / f"{name}.onnx",
        model_dir / f"{name}.int8.onnx",
    )


def load_and_export(pt, fp32):
    """Pull the .pt (auto-download), return class names, export fp32 ONNX."""
    from ultralytics import YOLO

    print(f"[model] pulling/loading {pt} ...")
    model = YOLO(str(pt))  # ultralytics downloads the asset if missing
    if not fp32.exists():
        out = Path(model.export(format="onnx", imgsz=IMGSZ, opset=13, simplify=True, dynamic=False))
        if out.resolve() != fp32.resolve():
            out.replace(fp32)
        print(f"[model] exported {fp32}")
    else:
        print(f"[model] reuse {fp32}")
    return model.names


# ---------------------------------------------------------------- quantization
class _CalibReader:
    """Feeds letterboxed test images through the quantizer for MinMax ranges."""

    def __init__(self, arrays, input_name):
        self._arrays = arrays
        self._input_name = input_name
        self._i = 0

    def get_next(self):
        if self._i >= len(self._arrays):
            return None
        a = self._arrays[self._i]
        self._i += 1
        return {self._input_name: a}


def quantize(fp32, int8, calib_arrays, input_name):
    if int8.exists():
        print(f"[model] reuse {int8}")
        return
    import onnx
    from onnxruntime.quantization import (
        CalibrationMethod,
        quantize_dynamic,
        quantize_static,
        QuantFormat,
        QuantType,
    )

    # The YOLOv8 graph's single output is a Concat of boxes (0..640) and class
    # scores (0..1); quantizing it gives one scale and flattens the classes to
    # zero, so keep the output node(s) in float.
    graph = onnx.load(str(fp32)).graph
    producers = {o: n.name for n in graph.node for o in n.output}
    exclude = [producers[o.name] for o in graph.output if o.name in producers]

    print(f"[model] int8-quantizing {fp32.name} (static QDQ, {len(calib_arrays)} calib imgs, excl {exclude})")
    try:
        quantize_static(
            str(fp32),
            str(int8),
            _CalibReader(calib_arrays, input_name),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            calibrate_method=CalibrationMethod.MinMax,
            nodes_to_exclude=exclude,
        )
    except Exception as e:  # some ops unsupported by static QDQ -> int8 weights only
        print(f"[model] static failed ({e}); falling back to dynamic int8 weights")
        quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    print(f"[model] wrote {int8}")


# ---------------------------------------------------------------- decode + draw
def decode(output, conf_thres, iou_thres, scale, pad_x, pad_y, orig_w, orig_h):
    """YOLOv8 head (1,84,8400)/(1,8400,84) -> list of (xyxy, class, conf)."""
    import cv2

    pred = np.asarray(output)[0]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T  # -> (8400, 84)
    boxes_xywh = pred[:, :4]
    scores = pred[:, 4:]
    cls = scores.argmax(1)
    conf = scores[np.arange(len(scores)), cls]
    keep = conf >= conf_thres
    boxes_xywh, conf, cls = boxes_xywh[keep], conf[keep], cls[keep]
    if len(boxes_xywh) == 0:
        return []
    # cx,cy,w,h (letterbox space) -> xyxy, then undo letterbox
    xy = np.empty_like(boxes_xywh)
    xy[:, 0] = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2 - pad_x) / scale
    xy[:, 1] = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2 - pad_y) / scale
    xy[:, 2] = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2 - pad_x) / scale
    xy[:, 3] = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2 - pad_y) / scale
    xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, orig_w)
    xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, orig_h)
    idx = cv2.dnn.NMSBoxes(xy.tolist(), conf.tolist(), conf_thres, iou_thres)
    if len(idx) == 0:
        return []
    idx = np.asarray(idx).reshape(-1)
    return [(xy[i], int(cls[i]), float(conf[i])) for i in idx]


def draw(img, dets, names, color=(0, 255, 0)):
    out = img.copy()
    d = ImageDraw.Draw(out)
    for box, cls, conf in dets:
        x1, y1, x2, y2 = box
        label = f"{names.get(cls, cls)} {conf:.2f}"
        d.rectangle([x1, y1, x2, y2], outline=color, width=2)
        d.text((x1 + 2, max(0, y1 - 12)), label, fill=color)
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default="test-images", help="input image directory")
    ap.add_argument("--out", default=None, help="output dir (default <images>/annotated)")
    ap.add_argument("--model-dir", default="models", help="cache for yolov8n.pt/.onnx")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--max-calib", type=int, default=32, help="calibration images")
    args = ap.parse_args()

    img_dir = Path(args.images)
    if not img_dir.is_dir():
        sys.exit(f"error: no image directory {img_dir!r} (create it and add some images)")
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not paths:
        sys.exit(f"error: no images in {img_dir} (looked for {sorted(IMG_EXTS)})")
    out_dir = Path(args.out) if args.out else img_dir / "annotated"
    out_dir.mkdir(parents=True, exist_ok=True)

    # model: pull -> export -> int8 quantize (calibrated on the input images)
    pt, fp32, int8 = get_models(args.model_dir)
    names = load_and_export(pt, fp32)

    calib = []
    for p in paths[: args.max_calib]:
        img = Image.open(p).convert("RGB")
        canvas, _, _, _ = letterbox(img)
        calib.append(to_input(canvas))
    import onnx
    import onnxruntime as ort
    input_name = onnx.load(str(fp32)).graph.input[0].name
    quantize(fp32, int8, calib, input_name)

    sess = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    print(f"[run] {len(paths)} images -> {out_dir}")

    total = 0
    for p in paths:
        img = Image.open(p).convert("RGB")
        canvas, scale, px, py = letterbox(img)
        out = sess.run(None, {input_name: to_input(canvas)})[0]
        dets = decode(out, args.conf, args.iou, scale, px, py, img.width, img.height)
        draw(img, dets, names).save(out_dir / p.name)
        total += len(dets)
        label = ", ".join(f"{names.get(c, c)} {q:.2f}" for _, c, q in dets) or "-"
        print(f"  {p.name}: {len(dets)} det(s)  {label}")
    print(f"[done] {total} detections, annotated in {out_dir}")


if __name__ == "__main__":
    main()
