#!/usr/bin/env python3
"""Receive the vo-box-lite frame stream and save frames as grayscale BMPs.

Wire format — one record per frame, as sent by src/bin/main.rs send_frame():
    u32 LE  n        bytes following this field (= 9 + w*h)
    4 B     magic    b"VOX1"
    u8      format   esp32-camera pixformat id (3 = PIXFORMAT_GRAYSCALE)
    u16 LE  width
    u16 LE  height
    n-9 B   raw pixel bytes (grayscale: 1 byte/px, row-major)

Usage:
    python3 receive_frames.py --once          # save a single frame, then exit
    python3 receive_frames.py                 # save frames until Ctrl-C
    python3 receive_frames.py --host 192.168.71.1 --port 5000 --out frames

The ESP accepts ONE persistent TCP connection — kill any leftover `nc` first.
"""

import argparse
import socket
import struct
import sys
from pathlib import Path

MAGIC = b"VOX1"
FMT_GRAYSCALE = 3  # esp32-camera PIXFORMAT_GRAYSCALE
# Fmt byte maps to camera pixformat ids: 1=YUV422 2=YUV420 3=GRAYSCALE 4=JPEG ...


def read_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"connection closed mid-record ({len(buf)}/{n} bytes)")
        buf.extend(chunk)
    return bytes(buf)


def read_record(conn: socket.socket) -> tuple[int, int, int, bytes]:
    """Block until one full frame record arrives; return (fmt, w, h, pixels)."""
    (n,) = struct.unpack("<I", read_exact(conn, 4))
    payload = read_exact(conn, n)
    if payload[:4] != MAGIC:
        raise ValueError(f"bad magic {payload[:4]!r} (expected {MAGIC!r}) — stream out of sync")
    fmt = payload[4]
    w, h = struct.unpack("<HH", payload[5:9])
    pixels = payload[9:]
    if len(pixels) != w * h:
        raise ValueError(f"frame size mismatch: {len(pixels)} px vs {w}x{h}")
    return fmt, w, h, pixels


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
        print("connected — waiting for frames (Ctrl-C to stop)")
        count = 0
        while True:
            fmt, w, h, pixels = read_record(conn)
            if fmt != FMT_GRAYSCALE:
                print(f"frame {count}: unsupported pixel format id {fmt} (only grayscale=3 handled), skipping")
                continue
            count += 1
            path = out / f"frame_{count:06d}.bmp"
            write_gray_bmp(path, w, h, pixels)
            print(f"saved {path} ({w}x{h}, {len(pixels)} px)")
            if args.once:
                return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
