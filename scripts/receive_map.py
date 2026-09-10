#!/usr/bin/env python3
"""Map-run driver: send STRT to kick a run off, save the streamed VOX2 frames
+ features under <work>/ (bmps/, marked/, features/), then on the VOXD done
record build map.txt + map_report.txt (match -> COLMAP). --rebuild: no ESP.
"""

import argparse
import shutil
import socket
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# receive_frames is stdlib-only; build deps import lazily in build_from_work
# so the receive phase can also run from a Windows Python.
import receive_frames as rf  # record parsing + BMP/CSV writers

MIN_FEATURES = 20      # frames with fewer features are dropped before matching
MIN_USABLE_FRAMES = 3  # below this, COLMAP has no chance — give up gracefully


def fresh_dirs(work: Path) -> None:
    """Wipe + recreate per-run output dirs under `work` (a fresh run)."""
    for sub in ("bmps", "marked", "features", "colmap_work"):
        d = work / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    for f in ("matches.csv", "map.txt", "map_report.txt", "positions.csv"):
        (work / f).unlink(missing_ok=True)
    (work / "colmap_work" / "positions.csv").unlink(missing_ok=True)


def receive_run(args, work: Path) -> None:
    """Connect, send the STRT start command, and save every streamed frame
    (raises on connect failure)."""
    print(f"connecting to {args.host}:{args.port} ...")
    with socket.create_connection((args.host, args.port), timeout=15) as conn:
        # Blocking stream reads: the MCU closes the connection at the end of
        # the run (after the VOXD record), so EOF is the reliable terminator.
        # A per-recv timeout could fire MID-record and desync the parser.
        conn.settimeout(None)
        # Only wipe the previous run once we're actually connected (a failed
        # connect must not destroy the last good capture in <work>).
        fresh_dirs(work)
        bmps, features, marked = work / "bmps", work / "features", work / "marked"
        # The MCU idles until commanded: send STRT to kick the map task off.
        rf.send_start(conn, args.duration, args.interval)
        print(f"STRT sent: {args.duration}s run at one frame per {args.interval} ms "
              "— waiting for frames (Ctrl-C ends early and builds what we have)")
        idx = 0
        while True:
            try:
                kind, payload = rf.read_record(conn)
            except rf.ConnectionError:
                print("!! connection closed before the done record — building "
                      "with the frames received so far")
                break
            if kind == "VOXD":
                frames, features_total = payload
                print(f"done-mapping record: ESP streamed {frames} frames / "
                      f"{features_total} features; received {idx} here")
                if frames != idx:
                    print(f"  !! count mismatch (ESP {frames} vs saved {idx}) — "
                          f"check the connection log")
                break
            if kind == "VOX1":
                print("  !! legacy VOX1 frame — this firmware predates the map "
                      "task; update src/bin/main.rs")
                continue
            # VOX2
            fmt, w, h, pixels, feats, timings = rf.parse_vox2(payload)
            if fmt != rf.FMT_GRAYSCALE or len(pixels) != w * h:
                print(f"  !! bad VOX2 record (fmt {fmt}) — skipping")
                continue
            idx += 1
            stem = f"IMG{idx:04d}"
            rf.write_gray_bmp(bmps / f"{stem}.bmp", w, h, pixels)
            rf.write_feature_csv(features / f"{stem}.csv", feats)
            rf.write_marked_bmp(marked / f"{stem}_marked.bmp", w, h, pixels, feats)
            print(f"[{time.strftime('%H:%M:%S')}] {stem}: {w}x{h}, {len(feats)} "
                  f"features ({idx} so far)")
            if timings:
                # Per-frame perf over WiFi (the ESP's console UART dies when a
                # station joins, so the breakdown rides on the record).
                print(f"    {rf.format_timings(timings)}")
                print(f"    {rf.format_per_level(timings)}")


def build_from_work(work: Path, args) -> int:
    """Phase 2: match + COLMAP + map.txt + report, from whatever frames are in
    <work>. Build deps import here (not at module load) so the receive phase can
    also run from a Windows Python without them. Returns process exit code."""
    import numpy as np
    import match_features as mf  # numpy Hamming matcher
    import colmap_map            # pycolmap reconstruction glue
    import write_map             # map.txt writer

    bmps, features = work / "bmps", work / "features"
    bmp_files = sorted(bmps.glob("IMG*.bmp"))
    if not bmp_files:
        print(f"no frames in {bmps} — nothing to build", file=sys.stderr)
        return 1

    report = []          # lines of map_report.txt (also printed as we go)
    def note(line=""):
        report.append(line)
        print(line)

    # ---- 1. drop feature-starved frames (< min features). Matches are
    # computed afterwards, so nothing else needs filtering --------------------
    counts = {}
    for b in bmp_files:
        csv_path = features / f"{b.stem}.csv"
        if not csv_path.exists():
            counts[b.stem] = 0
            continue
        with open(csv_path) as f:
            n = sum(1 for line in f if line.strip())
        counts[b.stem] = n
    dropped = {s: n for s, n in counts.items() if n < args.min_features}
    if dropped:
        note(f"!! dropped {len(dropped)} frame(s) with < {args.min_features} "
             f"features: " + ", ".join(f"{s}({n})" for s, n in sorted(dropped.items())))
        for s in dropped:
            for p in (bmps / f"{s}.bmp", features / f"{s}.csv"):
                p.unlink(missing_ok=True)
    bmp_files = sorted(bmps.glob("IMG*.bmp"))  # survivors only
    usable = sorted(counts.keys() - dropped.keys())
    if len(usable) < MIN_USABLE_FRAMES:
        print(f"only {len(usable)} usable frame(s) (need >= {MIN_USABLE_FRAMES}) — "
              f"cannot build a map. Point the camera at a textured scene and "
              f"move it while the run streams.", file=sys.stderr)
        return 1

    # dims: all frames in one run share them (loader cross-checks)
    from PIL import Image
    dims = {}
    for b in bmp_files:
        with Image.open(bmps / b.name) as im:
            dims[b.stem] = im.size
    uniq = set(dims.values())
    if len(uniq) != 1:
        print(f"  !! frames have mixed dims {uniq} — expected one uniform run",
              file=sys.stderr)
        return 1
    w, h = uniq.pop()
    note(f"frames: {len(usable)} usable (of {len(bmp_files)} saved) @ {w}x{h}")
    counts_u = {s: counts[s] for s in usable}
    fs = np.array(list(counts_u.values()))
    lo = sorted(counts_u.items(), key=lambda kv: kv[1])[0]
    note(f"features/frame: median {int(np.median(fs))}, min {int(fs.min())} "
         f"({lo[0]}), max {int(fs.max())}; total {int(fs.sum())}")

    # ---- 1. cross-image matching -------------------------------------------
    t0 = time.time()
    rows, mstats = mf.match_all(features)
    if mstats.get("error"):
        print(f"matching failed: {mstats['error']}", file=sys.stderr)
        return 1
    mf.write_matches_csv(work / "matches.csv", rows)
    note(f"matching: {mstats['total_matches']} matches across "
         f"{mstats['pairs_with_matches']}/{mstats['pairs_total']} pairs "
         f"({time.time()-t0:.1f}s)")
    if mstats["total_matches"]:
        note(f"  best-match Hamming dist: min {mstats['match_dist_min']}, "
             f"median {mstats['match_dist_median']}, max {mstats['match_dist_max']}")
        note(f"  ambiguous matches (2nd candidate also within 100 bits): "
             f"{mstats['ambiguous']} ({100*mstats['ambiguous_frac']:.1f}%)")
    if mstats["pairs_with_matches"] == 0:
        print("no image pair matched at all — nothing for COLMAP. Lower "
              "FAST_THRESHOLD on the ESP or point at more texture.",
              file=sys.stderr)
        return 1

    # ---- 2. COLMAP reconstruction ------------------------------------------
    stats = {}
    try:
        images, features_map, matches = colmap_map.load_images_features_matches(
            str(bmps), str(features), str(work / "matches.csv"),
            expected_size=(w, h),
        )
        positions = colmap_map.reconstruct_feature_positions(
            images, features_map, matches,
            image_dir=str(bmps),
            workdir=str(work / "colmap_work"),
            stats=stats,
        )
    except Exception as e:  # colmap_map raises RuntimeError/ValueError/...
        print(f"!! COLMAP failed: {e}", file=sys.stderr)
        print("   hints: keep the camera moving (baseline between frames), aim "
              "at textured scenes, check marked/ for feature coverage.",
              file=sys.stderr)
        return 1

    name_by_id = {img["image_id"]: img["name"] for img in images}
    pos_csv = work / "colmap_work" / "positions.csv"
    import csv as csv_mod
    with open(pos_csv, "w", newline="") as f:
        wtr = csv_mod.writer(f)
        wtr.writerow(["image", "feature_idx", "x", "y", "z"])
        for (img_id, feat_idx), xyz in sorted(positions.items()):
            wtr.writerow([name_by_id[img_id], feat_idx, *np.round(xyz, 6)])
    total_features = sum(len(v) for v in features_map.values())
    note(f"COLMAP: verified {stats.get('pairs_verified', '?')}/{len(matches)} "
         f"image pairs; triangulated {len(positions)}/{total_features} features")

    # ---- 3. map.txt + quality report ---------------------------------------
    try:
        summary = write_map.build_map_file(
            work / "colmap_work", features, work / "map.txt")
    except Exception as e:
        print(f"!! map assembly failed: {e}", file=sys.stderr)
        return 1

    cam_init = stats.get("camera_initial")
    cam_ref = summary["camera_params"]
    if cam_init is not None and len(cam_ref) == len(cam_init):
        f0, cx0, cy0 = cam_init[0], cam_init[1], cam_init[2]
        f, cx, cy, k1 = cam_ref
        note(f"camera SIMPLE_RADIAL (focal guessed {f0:.1f} @ {cx0:.0f},{cy0:.0f} "
             f"-> refined f={f:.2f} cx={cx:.2f} cy={cy:.2f} k1={k1:.4f})")
    note(f"registered images: {summary['n_images']}/{len(images)}")
    note(f"map points: {summary['n_map_points']} "
         f"(from {summary['n_points3d']} COLMAP 3D points; "
         f"skipped {summary['skipped']})")
    if summary["point_reproj_err"]:
        e = summary["point_reproj_err"]
        note(f"3D-point reprojection error: mean {e['mean']:.2f} px, "
             f"median {e['median']:.2f} px, p95 {e['p95']:.2f} px")

    # ---- write the report file ---------------------------------------------
    header = [f"vo-box map run report — {time.strftime('%Y-%m-%d %H:%M:%S')}",
              f"work dir: {work}"]
    (work / "map_report.txt").write_text("\n".join(header + [""] + report) + "\n")
    print(f"\nwrote {work / 'map.txt'} and {work / 'map_report.txt'}")
    return 0


# COLMAP camera model name -> on-wire id (must match CAMERA_MODEL_* in
# src/bin/main.rs). Only SIMPLE_RADIAL is produced by colmap_map.py.
CAMERA_MODEL_IDS = {"SIMPLE_RADIAL": 1}


def load_map_txt(path: Path):
    """Parse map.txt -> (model_id, [f, cx, cy, k1], [(x, y, z, desc_bytes)])."""
    model_id = params = None
    points = []
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t or t[0].startswith("#"):
                continue
            if t[0] == "CAMERA":
                if t[1] not in CAMERA_MODEL_IDS:
                    raise ValueError(f"unsupported camera model {t[1]!r} in {path}")
                model_id = CAMERA_MODEL_IDS[t[1]]
                params = [float(v) for v in t[2:6]]
                if len(params) != 4:
                    raise ValueError(f"expected 4 SIMPLE_RADIAL params, got {t[2:]}")
            elif t[0] == "POINT":
                desc = bytes.fromhex(t[4])
                if len(desc) != 32:
                    raise ValueError(f"descriptor is {len(desc)} B, expected 32")
                points.append((float(t[1]), float(t[2]), float(t[3]), desc))
    if model_id is None:
        raise ValueError(f"no CAMERA line in {path}")
    if not points:
        raise ValueError(f"no POINT lines in {path}")
    return model_id, params, points


def upload_map(args, map_path: Path) -> None:
    """Send the built map (intrinsics + 3D points + descriptors) to the ESP
    (MAPU) and wait for its MAPK ack. Fresh connection: the run's was dropped."""
    model_id, params, points = load_map_txt(map_path)
    body = bytearray(rf.MAGIC_MAPU)
    body.append(model_id)
    body += struct.pack("<4f", *params)
    body += struct.pack("<I", len(points))
    for (x, y, z, desc) in points:
        body += struct.pack("<3f", x, y, z) + desc
    print(f"uploading map: {len(points)} points, camera model {model_id}, "
          f"params {['%.4g' % p for p in params]}, {len(body) + 4} bytes "
          f"-> {args.host}:{args.port}")
    with socket.create_connection((args.host, args.port), timeout=15) as conn:
        conn.settimeout(60)  # ESP ack after it has read + stored the points
        conn.sendall(struct.pack("<I", len(body)) + bytes(body))
        kind, payload = rf.read_record(conn)
        if kind != "MAPK":
            raise ConnectionError(f"expected MAPK ack, got {kind}")
        print(f"map accepted: ESP stored {payload} points — firmware is now in "
              f"localize mode (idle until the next map run)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.71.1",
                    help="AP IP from the ESP boot log (default 192.168.71.1)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--work", default="map_run",
                    help="working dir for frames, features, COLMAP, map.txt")
    ap.add_argument("--rebuild", action="store_true",
                    help="skip receiving; rebuild map.txt from <work>'s saved "
                         "frames + features (no ESP needed)")
    ap.add_argument("--min-features", type=int, default=MIN_FEATURES,
                    help="drop frames with fewer features than this")
    ap.add_argument("--duration", type=int, default=300,
                    help="map run length in seconds (sent in the STRT command; "
                         "default 300 = 5 min)")
    ap.add_argument("--interval", type=int, default=1000,
                    help="ms between streamed frames (0 = max rate; sent in STRT)")
    ap.add_argument("--no-upload", action="store_true",
                    help="skip uploading the built map to the ESP")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    if args.rebuild:
        return build_from_work(work, args)

    try:
        receive_run(args, work)
    except KeyboardInterrupt:
        print("\ninterrupted during receive — building with what we have")
    except (ConnectionRefusedError, socket.timeout, OSError) as e:
        print(f"!! cannot reach the ESP at {args.host}:{args.port} ({e}) — "
              f"is it flashed + powered, and are you joined to its SoftAP?",
              file=sys.stderr)
        return 1
    n = len(list((work / "bmps").glob("IMG*.bmp")))
    if n == 0:
        print("no frames received — nothing to build", file=sys.stderr)
        return 1
    if n < MIN_USABLE_FRAMES:
        print(f"warning: only {n} frame(s) — COLMAP needs overlapping views; "
              f"building anyway")
    rc = build_from_work(work, args)
    if rc == 0 and not args.no_upload:
        # Push the built map back to the ESP (it then idles in localize mode).
        # --rebuild never uploads (that path is for building without an ESP).
        try:
            upload_map(args, work / "map.txt")
        except (OSError, ValueError) as e:
            print(f"!! map upload failed: {e}", file=sys.stderr)
            print(f"   map.txt is saved at {work / 'map.txt'}", file=sys.stderr)
            return 1
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
