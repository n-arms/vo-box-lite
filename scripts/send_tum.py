#!/usr/bin/env python3
"""Feed image + IMU records to the DATA board (bench chain front end).

Connects to the DATA board's SoftAP TCP (`vo-box-data`, port 5000) and sends
one length-prefixed record at a time, waiting for the board's ACK1 before
sending the next. DATA only reads TCP while it has no parked frame, so
one-record-in-flight keeps progress unambiguous and makes resends trivial
(VO dedups by SEQ). This script never talks to the VO board.

Records (all ints LE; `u32 n` = bytes after it; one shared SEQ space):
    TUM1 image: n | b"TUM1" | u32 seq | u64 t_us | u16 w | u16 h | w*h gray bytes
    IMU1 batch: n | b"IMU1" | u32 seq | u64 t0_us | u16 nsamp | u16 dt_us
                | nsamp x 6xf32 (ax,ay,az,wx,wy,wz, m/s^2 + rad/s)
`t_us` is the image capture time on the same clock as the IMU `t0_us`
(EuRoC nanoseconds/1000; TUM `rgb.txt` seconds*1e6; 0 if unknown) so the VO
board can anchor a fused fix at the exact frame time.
Ack (DATA -> laptop, after VO consumed the record):
    b"ACK1" | u32 LE seq | u32 LE crc32(whole record incl. length prefix)

Sources (mutually exclusive):
    --dataset   TUM dir (rgb.txt + rgb/): images only, legacy path
    --raw-dir   pre-converted 640x480 *.bit frames: images only, stdlib-only
    --euroc     EuRoC ASL zip or mav0 dir: cam0 images (undistorted
                752x480 -> 640x480) interleaved with IMU1 batches by
                timestamp, one shared seq. Images every --image-period
                (default 5 s: VO fixes are rare, IMU is the stream).

    python3 scripts/send_tum.py --count 40                 # TUM dataset
    python3 scripts/send_tum.py --raw-dir work/raw         # pre-converted .bit
    python3 scripts/send_tum.py --euroc V1_01_easy.zip     # EuRoC replay
    python3 scripts/send_tum.py --euroc V1_01_easy.zip --dry-run --count 2
"""

import argparse
import binascii
import bisect
import csv
import re
import socket
import struct
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

MAGIC_TUM1 = b"TUM1"
MAGIC_IMU1 = b"IMU1"
MAGIC_ACK1 = b"ACK1"
WIDTH = 640
HEIGHT = 480
FRAME_BYTES = WIDTH * HEIGHT
RECORD_BYTES = 4 + 20 + FRAME_BYTES  # len prefix + TUM1 header (magic/seq/t_us/w/h)
DEFAULT_DATASET = str(REPO / "rgbd_dataset_freiburg1_xyz")

# EuRoC cam0 geometry (MT9M034 752x480). Output keeps square pixels via a
# 640-wide center crop (x off 56), so fu/fv carry over and only cu shifts.
EUROC_W, EUROC_H = 752, 480
EUROC_CROP_X = (EUROC_W - WIDTH) // 2
IMU_MAX_SAMP = 64  # must match data_board.rs


def read_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"connection closed mid-record ({len(buf)}/{n} bytes)")
        buf.extend(chunk)
    return bytes(buf)


def read_rgb_txt(path: Path):
    """-> [(timestamp, relpath)] from rgb.txt, skipping '#' comments."""
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ts, rel = line.split()[:2]
            entries.append((float(ts), rel))
    return entries


def to_gray(raw: bytes, label: str) -> bytes:
    if len(raw) != FRAME_BYTES:
        raise SystemExit(
            f"{label}: expected {FRAME_BYTES} gray bytes for {WIDTH}x{HEIGHT}, "
            f"got {len(raw)}"
        )
    return raw


def iter_frames(args):
    """Yield (kind, timestamp|None, label, blob) lazily. blob = gray bytes."""
    if args.raw_dir:
        files = sorted(Path(args.raw_dir).glob("*.bit"))[::args.stride]
        if not files:
            raise SystemExit(f"no *.bit files under {args.raw_dir}")
        for i, p in enumerate(files):
            if args.count is not None and i >= args.count:
                break
            yield "TUM1", None, p.name, to_gray(p.read_bytes(), str(p))
        return

    dataset = Path(args.dataset)
    entries = read_rgb_txt(dataset / "rgb.txt")[::args.stride]
    if not entries:
        raise SystemExit(f"no frames selected from {dataset / 'rgb.txt'}")

    # PIL only for the dataset path, so --raw-dir stays stdlib-only.
    from PIL import Image

    for i, (ts, rel) in enumerate(entries):
        if args.count is not None and i >= args.count:
            break
        img = Image.open(dataset / rel).convert("L")
        if img.size != (WIDTH, HEIGHT):
            img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)
        yield "TUM1", ts, rel, to_gray(img.tobytes(), rel)


# ---------------------------------------------------------- EuRoC source ----

class Euroc:
    """mav0/ tree from an ASL zip or an extracted dir."""

    def __init__(self, path: Path):
        p = Path(path)
        self.zf = None
        if p.is_file() and p.suffix == ".zip":
            self.zf = zipfile.ZipFile(p)
            names = self.zf.namelist()
            if "mav0/cam0/data.csv" not in names:
                inner = sorted({n.split("/")[1] for n in names
                                if n.count("/") >= 1 and n.endswith(".zip")})
                hint = f" inner zips: {inner}" if inner else ""
                raise SystemExit(
                    f"{p} has no mav0/ tree (looks like the outer bundle)."
                    f" Point --euroc at an inner ASL zip.{hint}")
            self.root = None
        elif (p / "mav0").is_dir():
            self.root = p / "mav0"
        elif p.name == "mav0" and p.is_dir():
            self.root = p
        else:
            raise SystemExit(f"--euroc: {p} is neither an ASL zip nor a mav0 dir")

    def _local(self, name: str) -> Path:
        if name.startswith("mav0/"):
            name = name[len("mav0/"):]
        return self.root / name

    def read_text(self, name: str) -> str:
        if self.zf is not None:
            return self.zf.read(name).decode()
        return self._local(name).read_text()

    def read_bytes(self, name: str) -> bytes:
        if self.zf is not None:
            return self.zf.read(name)
        return self._local(name).read_bytes()


def parse_floats(text: str, key: str, n: int):
    m = re.search(key + r":\s*\[([^\]]+)\]", text)
    if not m:
        raise SystemExit(f"sensor.yaml: '{key}' not found")
    vals = [float(x) for x in m.group(1).split(",")]
    if len(vals) != n:
        raise SystemExit(f"sensor.yaml: '{key}' has {len(vals)} vals, want {n}")
    return vals


def euroc_calib(src: Euroc, cam: int):
    """(fu, fv, cu_out, cv, k1, k2, p1, p2) for the 640x480 center crop."""
    t = src.read_text(f"mav0/cam{cam}/sensor.yaml")
    fu, fv, cu, cv = parse_floats(t, "intrinsics", 4)
    res = parse_floats(t, "resolution", 2)
    if [int(res[0]), int(res[1])] != [EUROC_W, EUROC_H]:
        raise SystemExit(f"cam{cam} resolution {res} != 752x480, update crop")
    k1, k2, p1, p2 = parse_floats(t, "distortion_coefficients", 4)
    return fu, fv, cu - EUROC_CROP_X, cv, k1, k2, p1, p2


def undistort_maps(calib):
    """Per-output-pixel source coords (forward distort model, exact)."""
    import numpy as np
    fu, fv, cu, cv, k1, k2, p1, p2 = calib
    us, vs = np.meshgrid(np.arange(WIDTH, dtype=np.float32),
                         np.arange(HEIGHT, dtype=np.float32))
    x = (us - cu) / fu
    y = (vs - cv) / fv
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return (fu * xd + cu + EUROC_CROP_X).astype(np.float32), \
           (fv * yd + cv).astype(np.float32)


def remap_bilinear(src, mx, my):
    """src: HxW uint8 -> 480x640 uint8; out-of-bounds reads as 0."""
    import numpy as np
    h, w = src.shape
    x0 = np.floor(mx).astype(np.int32)
    y0 = np.floor(my).astype(np.int32)
    fx = (mx - x0).astype(np.float32)
    fy = (my - y0).astype(np.float32)
    ok = (x0 >= 0) & (x0 + 1 < w) & (y0 >= 0) & (y0 + 1 < h)
    x0c = np.clip(x0, 0, w - 2)
    y0c = np.clip(y0, 0, h - 2)
    s = src.astype(np.float32)
    a = s[y0c, x0c]
    b = s[y0c, x0c + 1]
    c = s[y0c + 1, x0c]
    d = s[y0c + 1, x0c + 1]
    out = (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
           + c * (1 - fx) * fy + d * fx * fy)
    return (np.where(ok, out, 0)).astype(np.uint8)


def read_euroc_csv(src: Euroc, name: str):
    rows = [ln.split(",") for ln in src.read_text(name).splitlines()
            if ln and not ln.startswith("#")]
    return rows


def iter_euroc(args):
    """Yield (kind, ts_seconds, label, blob): IMU1 batches (t0_us, dt_us,
    [(ax,ay,az,wx,wy,wz)]) interleaved with undistorted TUM1 images."""
    import numpy as np
    from PIL import Image

    src = Euroc(Path(args.euroc))
    calib = euroc_calib(src, args.cam)
    mx, my = undistort_maps(calib)

    cams = [(int(ts), fn) for ts, fn in
            (ln.split(",")[:2] for ln in src.read_text(f"mav0/cam{args.cam}/data.csv")
             .splitlines() if ln and not ln.startswith("#"))]
    cams = cams[::args.stride]
    if not cams:
        raise SystemExit("no cam rows selected")

    imu_rows = read_euroc_csv(src, "mav0/imu0/data.csv")
    imu_ts = np.array([int(r[0]) for r in imu_rows])  # ns
    imu_w = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in imu_rows])
    imu_a = np.array([[float(r[4]), float(r[5]), float(r[6])] for r in imu_rows])

    # Image times: first frame, then every --image-period dataset seconds.
    period_ns = int(args.image_period * 1e9)
    sel, last = [], None
    for ts, fn in cams:
        if last is None or ts - last >= period_ns:
            sel.append((ts, fn))
            last = ts
            if args.count is not None and len(sel) >= args.count:
                break
    if not sel:
        raise SystemExit("no images selected (check --image-period/--count)")

    ptr = bisect.bisect_left(imu_ts, sel[0][0])
    for ts, fn in sel:
        while ptr < len(imu_ts) and imu_ts[ptr] < ts:
            take = min(args.imu_batch, IMU_MAX_SAMP, len(imu_ts) - ptr)
            sl = slice(ptr, ptr + take)
            t0_us = int(imu_ts[ptr] // 1000)
            dt_us = 5000 if take == 1 else max(100, round(
                (imu_ts[ptr + take - 1] - imu_ts[ptr]) / (take - 1) / 1000))
            samp = [(float(imu_a[i, 0]), float(imu_a[i, 1]), float(imu_a[i, 2]),
                     float(imu_w[i, 0]), float(imu_w[i, 1]), float(imu_w[i, 2]))
                    for i in range(ptr, ptr + take)]
            yield ("IMU1", imu_ts[ptr] / 1e9, f"imu@{t0_us}",
                   (t0_us, dt_us, samp))
            ptr += take
        raw = np.asarray(Image.open(
            BytesIO(src.read_bytes(f"mav0/cam{args.cam}/data/{fn}"))).convert("L"))
        if raw.shape != (EUROC_H, EUROC_W):
            raise SystemExit(f"{fn}: shape {raw.shape} != 480x752")
        gray = remap_bilinear(raw, mx, my).tobytes()
        yield "TUM1", ts / 1e9, fn, to_gray(gray, fn)


def build_record(kind: str, seq: int, t_us: int, blob) -> bytes:
    if kind == "TUM1":
        payload = MAGIC_TUM1 + struct.pack("<IQHH", seq, t_us, WIDTH, HEIGHT) + blob
    elif kind == "IMU1":
        t0_us, dt_us, samp = blob
        flat = [v for s in samp for v in s]
        payload = (MAGIC_IMU1
                   + struct.pack("<IQHH", seq, t0_us, len(samp), dt_us)
                   + struct.pack(f"<{len(flat)}f", *flat))
    else:
        raise SystemExit(f"unknown record kind {kind}")
    return struct.pack("<I", len(payload)) + payload


def connect(args) -> socket.socket:
    conn = socket.create_connection((args.host, args.port), timeout=args.connect_timeout)
    conn.settimeout(None)  # sends block on backpressure; recv timeout set per-ack
    print(f"connected to {args.host}:{args.port}")
    return conn


def send_one(conn: socket.socket, record: bytes, seq: int, ack_timeout: float) -> None:
    """Send one record and validate the ACK1 (raises on any failure)."""
    conn.settimeout(None)
    conn.sendall(record)
    conn.settimeout(ack_timeout if ack_timeout > 0 else None)
    ack = read_exact(conn, 12)
    if ack[:4] != MAGIC_ACK1:
        raise ConnectionError(f"bad ack magic {ack[:4]!r}")
    (ack_seq,) = struct.unpack("<I", ack[4:8])
    (ack_crc,) = struct.unpack("<I", ack[8:12])
    if ack_seq != seq:
        raise ConnectionError(f"ack seq {ack_seq} != sent {seq}")
    want = binascii.crc32(record) & 0xFFFFFFFF
    if ack_crc != want:
        raise ConnectionError(f"ack crc {ack_crc:08x} != local {want:08x}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--dataset", default=None,
                     help="TUM dataset dir (rgb.txt + rgb/)")
    src.add_argument("--raw-dir", default=None,
                     help="dir of pre-converted 640x480 *.bit frames (sorted)")
    src.add_argument("--euroc", default=None,
                     help="EuRoC ASL zip or mav0 dir (images + IMU, interleaved)")
    ap.add_argument("--count", type=int, default=None,
                    help="max IMAGES to send (IMU batches ride along)")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth frame")
    ap.add_argument("--image-period", type=float, default=5.0,
                    help="dataset seconds between sent images (euroc only)")
    ap.add_argument("--imu-batch", type=int, default=10,
                    help="IMU samples per IMU1 record (euroc only)")
    ap.add_argument("--cam", type=int, default=0, choices=[0, 1],
                    help="EuRoC camera (default 0)")
    ap.add_argument("--start-seq", type=int, default=1, help="first SEQ value")
    ap.add_argument("--host", default="192.168.71.1",
                    help="DATA SoftAP IP from the boot log (default 192.168.71.1)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--connect-timeout", type=float, default=10.0)
    ap.add_argument("--ack-timeout", type=float, default=0.0,
                    help="seconds to wait for ACK1; 0 = block forever (default)")
    ap.add_argument("--retry-delay", type=float, default=1.0,
                    help="seconds between reconnect attempts")
    ap.add_argument("--once", action="store_true", help="send one image and exit")
    ap.add_argument("--manifest", default="sent.csv",
                    help="write seq,timestamp,label,kind per sent record")
    ap.add_argument("--dry-run", action="store_true",
                    help="build+check records and write the manifest, no socket")
    args = ap.parse_args()

    if args.once:
        args.count = 1
    if args.dataset is None and args.raw_dir is None and args.euroc is None:
        args.dataset = DEFAULT_DATASET
    if args.euroc is not None and args.imu_batch > IMU_MAX_SAMP:
        raise SystemExit(f"--imu-batch max is {IMU_MAX_SAMP} (data_board.rs)")

    gen = iter_euroc(args) if args.euroc is not None else iter_frames(args)
    manifest_file = None
    manifest_csv = None
    if args.manifest:
        manifest_file = open(args.manifest, "w", newline="")
        manifest_csv = csv.writer(manifest_file)
        manifest_csv.writerow(["seq", "timestamp", "label", "kind"])

    seq = args.start_seq
    sent = 0
    n_images = 0
    total = args.count if args.count is not None else "?"
    conn = None
    item = None  # held across retries so a resend reuses the SAME record/seq
    try:
        while True:
            if args.count is not None and n_images >= args.count:
                break
            if item is None:
                try:
                    item = next(gen)
                except StopIteration:
                    break
            kind, ts, label, blob = item
            t_us = 0 if ts is None else int(round(ts * 1e6))

            if args.dry_run:
                record = build_record(kind, seq, t_us, blob)
                want = binascii.crc32(record) & 0xFFFFFFFF
                if kind == "TUM1":
                    assert len(record) == RECORD_BYTES, len(record)
                print(f"[{sent + 1}/{total}] seq={seq} {kind} {label} "
                      f"{len(record)} B crc={want:08x} (dry-run)")
            else:
                t0 = time.monotonic()
                try:
                    if conn is None:
                        conn = connect(args)
                    record = build_record(kind, seq, t_us, blob)
                    send_one(conn, record, seq, args.ack_timeout)
                except (OSError, ConnectionError) as e:
                    if conn is not None:
                        conn.close()
                        conn = None
                    print(f"seq={seq} {kind} {label}: {e} — retrying in "
                          f"{args.retry_delay:.1f}s", file=sys.stderr)
                    time.sleep(args.retry_delay)
                    continue  # resend the SAME record/seq
                print(f"[{sent + 1}/{total}] seq={seq} {kind} ts={ts} {label} "
                      f"acked ({time.monotonic() - t0:.2f}s)")

            if manifest_csv is not None:
                manifest_csv.writerow([seq, "" if ts is None else ts, label, kind])
                manifest_file.flush()
            item = None
            seq += 1
            sent += 1
            if kind == "TUM1":
                n_images += 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
    finally:
        if conn is not None:
            conn.close()
        if manifest_file is not None:
            manifest_file.close()

    print(f"sent {sent} record(s), seq {args.start_seq}..{seq - 1}"
          + (f" -> {args.manifest}" if args.manifest else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
