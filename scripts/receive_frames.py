#!/usr/bin/env python3
"""Receive the vo-box-lite **map mode** stream and save each frame as a
grayscale BMP + a feature CSV (slam-exp grey-features format, ready for the
COLMAP map pipeline).

Wire format — one record per processed frame, as sent by src/bin/main.rs
map_server()/build_record():

    VOX2 (frame + pyramid features):
        u32 LE  n        bytes after this field (= 11 + w*h + 41*nfeat)
        4 B     magic    b"VOX2"
        u8      format   esp32-camera pixformat id (3 = PIXFORMAT_GRAYSCALE)
        u16 LE  width    (level-0, i.e. the original frame)
        u16 LE  height
        u16 LE  nfeat
        w*h B   raw gray pixels (the original image frame)
        nfeat x feature records:
            u8      level      pyramid level the keypoint came from (0..6)
            f32 LE  x          level-0 pixel x (local keypoint * 1.2^level)
            f32 LE  y          level-0 pixel y
            32 B    descriptor 256-bit rBRIEF (8 x u32 LE, memory order)

    VOX1 (legacy, raw frame only — old firmware):
        u32 LE  n        (= 9 + w*h) | b"VOX1" | fmt(1) | w(2) | h(2) | pixels

Usage:
    python3 receive_frames.py --once          # save one frame, then exit
    python3 receive_frames.py                 # save frames until Ctrl-C
    python3 receive_frames.py --host 192.168.71.1 --port 5000 --out frames

Each saved frame produces <out>/frame_NNNNNN.bmp and <out>/frame_NNNNNN.csv
(rows: x,y,<64 hex chars> — feature index = CSV row, like grey-features/).

The ESP accepts ONE persistent TCP connection — kill any leftover `nc` first.
"""

import argparse
import socket
import struct
import sys
from pathlib import Path

MAGIC_VOX1 = b"VOX1"
MAGIC_VOX2 = b"VOX2"
FMT_GRAYSCALE = 3  # esp32-camera PIXFORMAT_GRAYSCALE


def read_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"connection closed mid-record ({len(buf)}/{n} bytes)")
        buf.extend(chunk)
    return bytes(buf)


def parse_vox2(payload: bytes):
    """Parse a VOX2 payload (after magic): returns (fmt, w, h, pixels, feats)
    where feats = list of (level, x, y, desc_bytes)."""
    fmt = payload[0]
    w, h, nfeat = struct.unpack("<HHH", payload[1:7])
    off = 7
    pixels = payload[off:off + w * h]
    off += w * h
    feats = []
    for _ in range(nfeat):
        (level,) = struct.unpack("<B", payload[off:off + 1])
        (x, y) = struct.unpack("<ff", payload[off + 1:off + 9])
        desc = payload[off + 9:off + 41]
        off += 41
        feats.append((level, x, y, desc))
    return fmt, w, h, pixels, feats


def read_record(conn: socket.socket):
    """Block until one full record arrives; returns (kind, payload-after-magic)
    with kind "VOX1" or "VOX2"."""
    (n,) = struct.unpack("<I", read_exact(conn, 4))
    payload = read_exact(conn, n)
    magic = payload[:4]
    if magic == MAGIC_VOX1:
        return "VOX1", payload[4:]
    if magic == MAGIC_VOX2:
        return "VOX2", payload[4:]
    raise ValueError(f"bad magic {magic!r} — stream out of sync")


def write_gray_bmp(path: Path, w: int, h: int, pixels: bytes) -> None:
    """8-bit grayscale BMP: 14 B file hdr + 40 B info hdr + 256-entry palette
    + bottom-up rows, each padded to a multiple of 4 bytes."""
    row_pad = (4 - (w % 4)) % 4
    stride = w + row_pad
    rows = [pixels[y * w:(y + 1) * w] + b"\x00" * row_pad for y in reversed(range(h))]
    pixel_data = b"".join(rows)

    palette_off = 14 + 40
    file_size = palette_off + 1024 + len(pixel_data)
    header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, palette_off)
    info = struct.pack(
        "<IiiHHIIiiII", 40, w, h, 1, 8, 0, len(pixel_data), 2835, 2835, 256, 0
    )
    palette = b"".join(bytes((i, i, i, 0)) for i in range(256))
    path.write_bytes(header + info + palette + pixel_data)


def write_feature_csv(path: Path, feats) -> None:
    """slam-exp grey-features format: `x.xx,y.yy,<64 hex>` per row (no header;
    feature index = row number). x/y are level-0 pixel coords from the S3."""
    lines = []
    for (_level, x, y, desc) in feats:
        hexd = desc.hex()
        lines.append(f"{x:.2f},{y:.2f},{hexd}")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.71.1", help="AP IP from the ESP boot log (default 192.168.71.1)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--out", default="frames", help="output directory (created if missing)")
    ap.add_argument("--once", action="store_true", help="save one frame and exit")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"connecting to {args.host}:{args.port} ...")
    with socket.create_connection((args.host, args.port), timeout=10) as conn:
        print("connected — waiting for map records (Ctrl-C to stop)")
        count = 0
        while True:
            kind, payload = read_record(conn)
            if kind == "VOX1":
                fmt = payload[0]
                w, h = struct.unpack("<HH", payload[1:5])
                pixels = payload[5:]
                if fmt != FMT_GRAYSCALE or len(pixels) != w * h:
                    print(f"frame {count}: bad VOX1 (fmt {fmt}, {len(pixels)}px vs {w}x{h}), skipping")
                    continue
                count += 1
                write_gray_bmp(out / f"frame_{count:06d}.bmp", w, h, pixels)
                print(f"saved {out}/frame_{count:06d}.bmp ({w}x{h}) [legacy VOX1]")
            else:  # VOX2
                fmt, w, h, pixels, feats = parse_vox2(payload)
                if fmt != FMT_GRAYSCALE or len(pixels) != w * h:
                    print(f"frame {count}: bad VOX2 (fmt {fmt}), skipping")
                    continue
                count += 1
                stem = out / f"frame_{count:06d}"
                write_gray_bmp(Path(str(stem) + ".bmp"), w, h, pixels)
                write_feature_csv(Path(str(stem) + ".csv"), feats)
                print(
                    f"saved {stem}.bmp ({w}x{h}) + .csv ({len(feats)} features)"
                )
            if args.once:
                return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
