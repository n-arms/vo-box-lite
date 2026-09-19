#!/usr/bin/env python3
"""Semantic-task viewer: connect to the ESP's SoftAP TCP server and save every
SEM1 record (RGB565 frame + yolov8n bounding boxes) as <out>/frame_NNNNNN.bmp
(clean), frame_NNNNNN_marked.bmp (boxes drawn) and frame_NNNNNN.csv (coords).

The ESP's semantic task (src/semantic.rs) spins up the SoftAP "vo-box" and
streams one record per inference; this script just reads and saves. Stdlib only
so it runs from a Windows Python over the WSL UNC path (see receive_frames.py).
"""

import argparse
import socket
import struct
import sys
from pathlib import Path

MAGIC_SEM = b"SEM1"
FMT_RGB565 = 2
# SEM1 header after the u32 length: magic(4) + fmt(1) + w(2) + h(2) + ndet(2).
HEADER_BYTES = 11
DET_BYTES = 21  # class u8 + 5 x f32
BOX_COLOR = (0, 255, 0)  # BGR: green
BOX_THICK = 2

# COCO class ids -> names (the int8 yolov8n head emits one of these 80).
COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


def class_name(cid: int) -> str:
    return COCO_NAMES[cid] if 0 <= cid < len(COCO_NAMES) else f"class_{cid}"


def read_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"connection closed mid-record ({len(buf)}/{n} bytes)")
        buf.extend(chunk)
    return bytes(buf)


def read_record(conn: socket.socket):
    """Block until one full SEM1 record; return (w, h, rgb565, dets) with dets a
    list of (class_id, x, y, w, h, score) in source 640x480 pixels."""
    (n,) = struct.unpack("<I", read_exact(conn, 4))
    payload = read_exact(conn, n)
    if payload[:4] != MAGIC_SEM:
        raise ValueError(f"bad magic {payload[:4]!r} — stream out of sync")
    fmt = payload[4]
    if fmt != FMT_RGB565:
        raise ValueError(f"bad pixel format {fmt} (want {FMT_RGB565} RGB565)")
    w, h, ndet = struct.unpack("<HHH", payload[5:HEADER_BYTES])
    off = HEADER_BYTES
    pixels = payload[off:off + w * h * 2]
    off += w * h * 2
    dets = []
    for _ in range(ndet):
        (cid,) = struct.unpack("<B", payload[off:off + 1])
        x, y, dw, dh, score = struct.unpack("<fffff", payload[off + 1:off + DET_BYTES])
        off += DET_BYTES
        dets.append((cid, x, y, dw, dh, score))
    return w, h, pixels, dets


def _bmp_headers(w: int, h: int, pixel_bytes: int) -> bytes:
    """24-bit BMP file header + DIB header (bottom-up, 4-B padded rows)."""
    palette_off = 14 + 40
    file_size = palette_off + pixel_bytes
    header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, palette_off)
    info = struct.pack(
        "<IiiHHIIiiII", 40, w, h, 1, 24, 0, pixel_bytes, 2835, 2835, 0, 0,
    )
    return header + info


def rgb565_to_bgr_rows(w: int, h: int, rgb565: bytes) -> tuple:
    """Decode RGB565 -> a bottom-up 24-bit BGR pixel buffer + its row stride.
    Uses a 256-entry low/high-byte expansion table to keep it pure stdlib."""
    stride = w * 3 + (4 - (w * 3) % 4) % 4
    data = bytearray(stride * h)
    px = memoryview(rgb565)
    for y in range(h):
        dst = (h - 1 - y) * stride
        base = y * w * 2
        o = dst
        for x in range(base, base + w * 2, 2):
            v = px[x] | (px[x + 1] << 8)
            r5 = (v >> 11) & 0x1F
            g6 = (v >> 5) & 0x3F
            b5 = v & 0x1F
            data[o] = (b5 << 3) | (b5 >> 2)
            data[o + 1] = (g6 << 2) | (g6 >> 4)
            data[o + 2] = (r5 << 3) | (r5 >> 2)
            o += 3
    return data, stride


def _set_px(data: bytearray, stride: int, w: int, h: int, x: int, y: int, color) -> None:
    if 0 <= x < w and 0 <= y < h:
        o = (h - 1 - y) * stride + x * 3
        data[o] = color[0]
        data[o + 1] = color[1]
        data[o + 2] = color[2]


def draw_box(data: bytearray, stride: int, w: int, h: int, det) -> None:
    """Draw one detection's rectangle outline (2 px thick) on the BGR buffer.
    `det` is (class_id, x, y, w, h, score) in top-left-origin source pixels."""
    _, x, y, dw, dh, _ = det
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(w - 1, int(round(x + dw)))
    y1 = min(h - 1, int(round(y + dh)))
    for t in range(BOX_THICK):
        for xx in range(x0, x1 + 1):
            _set_px(data, stride, w, h, xx, y0 + t, BOX_COLOR)
            _set_px(data, stride, w, h, xx, y1 - t, BOX_COLOR)
        for yy in range(y0, y1 + 1):
            _set_px(data, stride, w, h, x0 + t, yy, BOX_COLOR)
            _set_px(data, stride, w, h, x1 - t, yy, BOX_COLOR)


def write_bmp(path: Path, w: int, h: int, data: bytearray) -> None:
    path.write_bytes(_bmp_headers(w, h, len(data)) + bytes(data))


def write_csv(path: Path, dets) -> None:
    lines = ["class_id,class_name,x,y,w,h,score"]
    for cid, x, y, dw, dh, score in dets:
        lines.append(f"{cid},{class_name(cid)},{x:.2f},{y:.2f},{dw:.2f},{dh:.2f},{score:.4f}")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--host", default="192.168.71.1",
                    help="AP IP from the ESP boot log (default 192.168.71.1)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--out", default="semantic_out", help="output directory")
    ap.add_argument("--once", action="store_true", help="save one frame and exit")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"connecting to {args.host}:{args.port} ...")
    count = 0
    with socket.create_connection((args.host, args.port), timeout=10) as conn:
        print(f"connected — saving frames to {out}/ (Ctrl-C to stop)")
        while True:
            w, h, pixels, dets = read_record(conn)
            if len(pixels) != w * h * 2:
                print(f"frame {count}: bad pixel payload ({len(pixels)} B vs {w}x{h} RGB565), skipping")
                continue
            count += 1
            stem = out / f"frame_{count:06d}"
            data, stride = rgb565_to_bgr_rows(w, h, pixels)
            write_bmp(Path(str(stem) + ".bmp"), w, h, data)
            marked = bytearray(data)
            for det in dets:
                draw_box(marked, stride, w, h, det)
            write_bmp(Path(str(stem) + "_marked.bmp"), w, h, marked)
            write_csv(Path(str(stem) + ".csv"), dets)
            labels = ", ".join(f"{class_name(c)} {s:.2f}" for c, _, _, _, _, s in dets) or "-"
            print(f"saved {stem}.bmp + _marked.bmp + .csv ({w}x{h}, {len(dets)} dets: {labels})")
            if args.once:
                return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
