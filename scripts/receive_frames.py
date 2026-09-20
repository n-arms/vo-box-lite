#!/usr/bin/env python3
"""Map-mode viewer: send STRT to kick a timed map run off the ESP, then save
every VOX2 frame as <out>/frame_NNNNNN.bmp (+ _marked.bmp overlay and .csv
features) until the VOXD done record ends the run. Full builder: receive_map.py.
"""

import argparse
import socket
import struct
import sys
from pathlib import Path

MAGIC_VOX1 = b"VOX1"
MAGIC_VOX2 = b"VOX2"
MAGIC_VOXD = b"VOXD"
MAGIC_STRT = b"STRT"  # laptop -> ESP: kick the map task off
MAGIC_MAP_UPLOAD = b"MAP2"  # laptop -> ESP: upload the built localization map
                             # (per-frame {1064 B embedding, points})
MAGIC_MAPK = b"MAPK"  # ESP -> laptop: ack (u32 n_frames, u32 n_points)
FMT_GRAYSCALE = 3  # esp32-camera PIXFORMAT_GRAYSCALE
DOT_RADIUS = 2     # feature marker radius in px

# VOX2 timing footer (firmware with WiFi perf data): appended after the last
# feature descriptor — 3 stage u32s (capture / 4x4-downscale / build, µs)
# then per pyramid level (7): 6 phase u32s (fast/score/nms/blur/rbrief/ds65,
# µs) + a corners u16, all LE. Phase order must match build_record in
# src/bin/main.rs. Old-firmware records have no footer (parse_vox2 tolerates).
VOX2_FOOTER_BYTES = 3 * 4 + 7 * (6 * 4 + 2)
VOX2_FOOTER_FMT = "<III" + "6IH" * 7
VOX2_PHASES = ("fast", "score", "nms", "blur", "rbrief", "ds65")


def read_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"connection closed mid-record ({len(buf)}/{n} bytes)")
        buf.extend(chunk)
    return bytes(buf)


def parse_vox2(payload: bytes):
    """Parse a VOX2 payload (after magic): returns (fmt, w, h, pixels, feats,
    timings) where feats = list of (level, x, y, desc_bytes) and timings = a
    dict with the per-frame µs breakdown (None on old firmware without it).
    x/y are level-0 pixels."""
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
    timings = None
    if len(payload) - off >= VOX2_FOOTER_BYTES:
        vals = struct.unpack_from(VOX2_FOOTER_FMT, payload, off)
        timings = {
            "capture_us": vals[0],
            "ds4_us": vals[1],
            "build_us": vals[2],
            "levels": [],
        }
        for l in range(7):
            base = 3 + l * 7
            level = {ph: vals[base + k] for k, ph in enumerate(VOX2_PHASES)}
            level["corners"] = vals[base + 6]
            timings["levels"].append(level)
    return fmt, w, h, pixels, feats, timings


def format_timings(t: dict) -> str:
    """One-line µs breakdown from a parsed VOX2 timing footer ('' if None).
    Stage sums across all pyramid levels; mirrors the ESP's serial per-frame
    log minus the send/pace stages (those are WiFi-side and not in-record)."""
    if t is None:
        return ""
    sums = {ph: sum(l[ph] for l in t["levels"]) for ph in VOX2_PHASES}
    pyr = sum(sums.values())
    return (f"cap {t['capture_us']}us ds4 {t['ds4_us']}us pyr {pyr}us "
            f"(fast {sums['fast']} score {sums['score']} nms {sums['nms']} "
            f"blur {sums['blur']} rbrief {sums['rbrief']} ds65 {sums['ds65']}) "
            f"build {t['build_us']}us")


def format_per_level(t: dict) -> str:
    """Per-pyramid-level phase total + NMS survivors ('' if None)."""
    if t is None:
        return ""
    parts = []
    for l, lvl in enumerate(t["levels"]):
        tot = sum(lvl[ph] for ph in VOX2_PHASES)
        parts.append(f"L{l} {tot}us/{lvl['corners']}feats blur={lvl['blur']}")
    return "per level: " + ", ".join(parts)


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
    if magic == MAGIC_VOXD:
        # payload-after-magic = u32 frames, u32 total features (LE)
        frames, features = struct.unpack("<II", payload[4:12])
        return "VOXD", (frames, features)
    if magic == MAGIC_MAPK:
        # payload-after-magic = u32 n_frames, u32 n_points stored by the ESP
        n_frames, n_points = struct.unpack("<II", payload[4:12])
        return "MAPK", (n_frames, n_points)
    raise ValueError(f"bad magic {magic!r} — stream out of sync")


def send_start(conn: socket.socket, duration_s: int = 30, interval_ms: int = 1000) -> None:
    """Kick the ESP's map task off: `u32 LE n` (= 12) | b"STRT" | u32 duration_s
    | u32 interval_ms. The MCU idles until this arrives (see src/bin/main.rs)."""
    body = MAGIC_STRT + struct.pack("<II", duration_s, interval_ms)
    conn.sendall(struct.pack("<I", len(body)) + body)


def _bmp_headers(w: int, h: int, pixel_bytes: int, bpp: int) -> bytes:
    """Common 24/8-bit BMP file header + DIB header. bpp = 24 or 8."""
    palette_off = 14 + 40 if bpp == 24 else 14 + 40 + 1024
    file_size = palette_off + pixel_bytes
    header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, palette_off)
    info = struct.pack(
        "<IiiHHIIiiII", 40, w, h, 1, bpp, 0, pixel_bytes, 2835, 2835,
        256 if bpp == 8 else 0, 0,
    )
    return header + info


def write_gray_bmp(path: Path, w: int, h: int, pixels: bytes) -> None:
    """8-bit grayscale BMP (legacy VOX1 frames): 14 B file hdr + 40 B info hdr
    + 256-entry palette + bottom-up rows, each padded to a multiple of 4 B."""
    row_pad = (4 - (w % 4)) % 4
    stride = w + row_pad
    rows = [pixels[y * w:(y + 1) * w] + b"\x00" * row_pad for y in reversed(range(h))]
    pixel_data = b"".join(rows)
    palette = b"".join(bytes((i, i, i, 0)) for i in range(256))
    path.write_bytes(_bmp_headers(w, h, len(pixel_data), 8) + palette + pixel_data)


def feature_dots(w: int, h: int, feats) -> set:
    """Level-0 pixel positions of every feature, as a set of (x, y) ints.
    Border keypoints are dropped on the S3, but clamp defensively anyway."""
    dots = set()
    for (_level, x, y, _desc) in feats:
        cx, cy = int(round(x)), int(round(y))
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        for dy in range(-DOT_RADIUS, DOT_RADIUS + 1):
            for dx in range(-DOT_RADIUS, DOT_RADIUS + 1):
                if dx * dx + dy * dy <= DOT_RADIUS * DOT_RADIUS:
                    px, py = cx + dx, cy + dy
                    if 0 <= px < w and 0 <= py < h:
                        dots.add((px, py))
    return dots


def write_marked_bmp(path: Path, w: int, h: int, gray: bytes, feats) -> None:
    """24-bit BMP of the frame with each feature drawn as a red dot (BGR
    (0,0,255)) at its level-0 pixel position. Bottom-up rows, 4-B padded."""
    dots = feature_dots(w, h, feats)
    row_pad = (4 - (w * 3) % 4) % 4
    stride = w * 3 + row_pad
    pixel_data = bytearray()
    for y in reversed(range(h)):
        row = bytearray(stride)
        base = y * w
        for x in range(w):
            v = gray[base + x]
            if (x, y) in dots:
                row[x * 3:x * 3 + 3] = b"\x00\x00\xff"  # red
            else:
                row[x * 3] = v  # B
                row[x * 3 + 1] = v  # G
                row[x * 3 + 2] = v  # R
        pixel_data += row
    path.write_bytes(_bmp_headers(w, h, len(pixel_data), 24) + bytes(pixel_data))


def write_feature_csv(path: Path, feats) -> None:
    """slam-exp grey-features format: `x.xx,y.yy,<64 hex>` per row (no header;
    feature index = row number). x/y are level-0 pixel coords from the S3."""
    lines = []
    for (_level, x, y, desc) in feats:
        lines.append(f"{x:.2f},{y:.2f},{desc.hex()}")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.71.1", help="AP IP from the ESP boot log (default 192.168.71.1)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--out", default="frames", help="output directory (created if missing)")
    ap.add_argument("--once", action="store_true", help="save one frame and exit")
    ap.add_argument("--duration", type=int, default=30,
                    help="map run length in seconds (sent in the STRT command)")
    ap.add_argument("--interval", type=int, default=1000,
                    help="ms between streamed frames (0 = max rate)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"connecting to {args.host}:{args.port} ...")
    with socket.create_connection((args.host, args.port), timeout=10) as conn:
        # The ESP idles until told to run: kick a map run off, then save
        # frames until it ends with the VOXD done record.
        send_start(conn, args.duration, args.interval)
        print(f"STRT sent: {args.duration}s run at one frame per {args.interval} ms "
              f"— waiting for frames (Ctrl-C stops early)")
        count = 0
        while True:
            kind, payload = read_record(conn)
            if kind == "VOXD":
                frames, features = payload
                print(f"done-mapping record: {frames} frames / {features} features "
                      f"({count} saved here); run complete")
                return 0
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
                fmt, w, h, pixels, feats, timings = parse_vox2(payload)
                if fmt != FMT_GRAYSCALE or len(pixels) != w * h:
                    print(f"frame {count}: bad VOX2 (fmt {fmt}), skipping")
                    continue
                count += 1
                stem = out / f"frame_{count:06d}"
                # Original frame (clean) + labeled frame (features as red dots)
                # + feature CSV.
                write_gray_bmp(Path(str(stem) + ".bmp"), w, h, pixels)
                write_marked_bmp(Path(str(stem) + "_marked.bmp"), w, h, pixels, feats)
                write_feature_csv(Path(str(stem) + ".csv"), feats)
                print(
                    f"saved {stem}.bmp + {stem}_marked.bmp ({w}x{h}, "
                    f"{len(feats)} features as red dots) + .csv"
                )
                if timings:
                    print(f"  {format_timings(timings)}")
                    print(f"  {format_per_level(timings)}")
            if args.once:
                return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
