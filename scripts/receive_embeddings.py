#!/usr/bin/env python3
"""Collect a semantic run over the ESP's SoftAP TCP stream (no start command).
Saves each EMB1 record under <work>/: frames/*.bmp, embeddings/*.npy (raw uint8)
and manifest.csv. Stdlib-only, so it runs from a Windows Python over WSL UNC.
"""

import argparse
import csv
import shutil
import socket
import struct
import sys
import time
from pathlib import Path

MAGIC_EMB = b"EMB1"
FMT_GRAYSCALE = 3
# EMB1 header after the u32 length:
# magic(4) + fmt(1) + w(2) + h(2) + ndim(2) + emb_scale(f32) + emb_zp(i32).
HEADER_BYTES = 19


def read_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"connection closed mid-record ({len(buf)}/{n} bytes)")
        buf.extend(chunk)
    return bytes(buf)


def read_record(conn: socket.socket):
    """Block until one full EMB1 record; return (w, h, pixels, scale, zp, emb_u8)
    where pixels is 8-bit grayscale (w*h) and emb_u8 is the raw quantized
    descriptor (ndim uint8 values)."""
    (n,) = struct.unpack("<I", read_exact(conn, 4))
    payload = read_exact(conn, n)
    if payload[:4] != MAGIC_EMB:
        raise ValueError(f"bad magic {payload[:4]!r} — stream out of sync")
    fmt = payload[4]
    if fmt != FMT_GRAYSCALE:
        raise ValueError(f"bad pixel format {fmt} (want {FMT_GRAYSCALE} grayscale)")
    w, h, ndim = struct.unpack("<HHH", payload[5:11])
    scale, zp = struct.unpack("<fi", payload[11:HEADER_BYTES])
    off = HEADER_BYTES
    pixels = payload[off:off + w * h]
    off += w * h
    emb_u8 = payload[off:off + ndim]
    if len(pixels) != w * h or len(emb_u8) != ndim:
        raise ValueError(f"short record: {len(pixels)} px, {len(emb_u8)} desc bytes")
    return w, h, pixels, scale, zp, emb_u8


def gray_bmp_bytes(w: int, h: int, pixels: bytes) -> bytes:
    """8-bit grayscale BMP (bottom-up, 4-B padded rows, 256-entry gray palette)."""
    stride = w + (4 - w % 4) % 4
    data = bytearray(stride * h)
    for y in range(h):
        src = y * w
        dst = (h - 1 - y) * stride
        data[dst:dst + w] = pixels[src:src + w]
    palette_off = 14 + 40 + 256 * 4
    file_size = palette_off + len(data)
    header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, palette_off)
    info = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 8, 0, len(data), 2835, 2835, 256, 0)
    palette = bytes(v for i in range(256) for v in (i, i, i, 0))
    return header + info + palette + bytes(data)


def npy_u8_bytes(values: bytes) -> bytes:
    """Minimal v1.0 .npy writer (little-endian uint8, 1-D). `values` stays raw
    quantized — dequantize with the scale/zp recorded in manifest.csv."""
    descr = f"{{'descr': '|u1', 'fortran_order': False, 'shape': ({len(values)},), }}"
    header = descr.encode("latin1")
    # Pad so magic(6) + version(2) + hlen(2) + header is a multiple of 64, and
    # the header ends with a newline (numpy's format requires both).
    pad = 64 - ((10 + len(header) + 1) % 64)
    if pad == 64:
        pad = 0
    header = header + b" " * pad + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + values


def fresh_dirs(work: Path) -> None:
    """Wipe + recreate per-run output dirs under `work` (a fresh capture)."""
    for sub in ("frames", "embeddings"):
        d = work / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    (work / "manifest.csv").unlink(missing_ok=True)


def receive(args, work: Path) -> int:
    """Connect and save every streamed EMB1 pair until --count/--duration (or
    forever when both are 0), returning the number saved."""
    frames, embs = work / "frames", work / "embeddings"
    deadline = time.monotonic() + args.duration if args.duration > 0 else None

    print(f"connecting to {args.host}:{args.port} ...")
    with socket.create_connection((args.host, args.port), timeout=15) as conn:
        # Blocking reads: the ESP streams continuously and a per-recv timeout
        # could fire MID-record and desync the parser.
        conn.settimeout(None)
        # Only wipe the previous run once we're actually connected (a failed
        # connect must not destroy the last good capture in <work>).
        fresh_dirs(work)
        print(f"connected — saving pairs to {work}/ "
              f"({'until ' + str(args.duration) + 's' if deadline else 'Ctrl-C to stop'}"
              f"{', ' + str(args.count) + ' frames' if args.count else ''})")
        count = 0
        with open(work / "manifest.csv", "w", newline="") as mf:
            wtr = csv.writer(mf)
            wtr.writerow(["name", "w", "h", "ndim", "scale", "zp"])
            mf.flush()
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                if args.count and count >= args.count:
                    break
                w, h, pixels, scale, zp, emb_u8 = read_record(conn)
                count += 1
                stem = f"frame_{count:06d}"
                (frames / f"{stem}.bmp").write_bytes(gray_bmp_bytes(w, h, pixels))
                (embs / f"{stem}.npy").write_bytes(npy_u8_bytes(emb_u8))
                wtr.writerow([stem, w, h, len(emb_u8), repr(scale), zp])
                mf.flush()
                print(f"[{time.strftime('%H:%M:%S')}] {stem}: {w}x{h}, "
                      f"dim {len(emb_u8)} (scale {scale:.8f}, zp {zp}) — {count} saved")
    return count


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--host", default="192.168.71.1",
                    help="AP IP from the ESP boot log (default 192.168.71.1)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--work", default="semantic_run",
                    help="working dir for frames/, embeddings/, manifest.csv")
    ap.add_argument("--duration", type=int, default=0,
                    help="stop after this many seconds (0 = run until Ctrl-C)")
    ap.add_argument("--count", type=int, default=0,
                    help="stop after this many frames (0 = no limit)")
    ap.add_argument("--once", action="store_true", help="save one frame and exit")
    args = ap.parse_args()
    if args.once:
        args.count = 1

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    try:
        count = receive(args, work)
    except KeyboardInterrupt:
        # Summarise whatever landed on disk; the manifest is flushed per row.
        count = len(list((work / "frames").glob("frame_*.bmp"))) if (work / "frames").exists() else 0
        print(f"\ninterrupted after {count} pair(s)")
        return 0
    except (ConnectionRefusedError, socket.timeout, OSError, ValueError) as e:
        print(f"!! receive failed: {e}", file=sys.stderr)
        print(f"   is the ESP flashed + powered, and are you joined to its SoftAP? "
              f"(AP IP in the boot log)", file=sys.stderr)
        return 1
    print(f"\nwrote {count} pair(s) -> {work}/frames, {work}/embeddings, {work}/manifest.csv")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
