#!/usr/bin/env python3
"""Map-run driver: send STRT to kick a run off, save the streamed raw VOX2
frames under <work>/ (raw/, bmps/, marked/), re-extract features on the laptop
with the S3's own Rust extractor, then on the VOXD done record build map.txt +
map_report.txt (match -> COLMAP). --rebuild: no ESP.
"""

import argparse
import json
import shutil
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# receive_frames is stdlib-only; build deps import lazily in build_from_work
# so the receive phase can also run from a Windows Python.
import receive_frames as rf  # record parsing + BMP/CSV writers

MIN_FEATURES = 20      # frames with fewer features are dropped before matching
MIN_USABLE_FRAMES = 3  # below this, COLMAP has no chance — give up gracefully
MIN_FRAME_POINTS = 8   # a map frame needs this many triangulated points (PnP min)

# Host feature extraction (scripts/extract_host.rs): compiled with plain
# `rustc +stable` (no cargo / no ESP deps) and run over the raw/ frames. It
# #[path]-includes the exact S3 source modules, so the descriptors are identical.
EXTRACT_HOST_SRC = Path(__file__).resolve().parent / "extract_host.rs"

# Rust matcher (scripts/matcher/): rayon + AVX2, byte-identical to
# match_features.py but ~100x faster. Built with cargo (+stable) + an explicit
# host --target (the repo's .cargo/config.toml targets xtensa for the ESP).
MATCHER_CRATE = Path(__file__).resolve().parent / "matcher"
MATCHER_EXE = MATCHER_CRATE / "target" / "x86_64-unknown-linux-gnu" / "release" / "vo-matcher"


def fresh_dirs(work: Path) -> None:
    """Wipe + recreate per-run output dirs under `work` (a fresh run)."""
    for sub in ("raw", "bmps", "marked", "features", "embeddings", "colmap_work"):
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
        raw = work / "raw"
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
            (raw / f"{stem}.bit").write_bytes(pixels)
            rf.write_gray_bmp(bmps / f"{stem}.bmp", w, h, pixels)
            rf.write_feature_csv(features / f"{stem}.csv", feats)
            rf.write_marked_bmp(marked / f"{stem}_marked.bmp", w, h, pixels, feats)
            print(f"[{time.strftime('%H:%M:%S')}] {stem}: {w}x{h}, {len(feats)} "
                  f"features from ESP — laptop extracts ({idx} so far)")
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

    # ---- 0. laptop-side feature extraction ---------------------------------
    # The firmware streams raw frames only (0 features on the wire), so
    # re-extract here with the exact S3 Rust extractor: map descriptors then
    # match the on-device localize queries bit-for-bit. Older captures (or a
    # --rebuild of one) have no raw/ and keep the CSVs saved during receive.
    raw_dir = work / "raw"
    if raw_dir.is_dir() and any(raw_dir.glob("*.bit")):
        from PIL import Image
        with Image.open(bmp_files[0]) as im:
            fw, fh = im.size
        try:
            extract_features_host(work, fw, fh, note,
                                  getattr(args, "threshold", None))
        except RuntimeError as e:
            print(f"!! {e}", file=sys.stderr)
            return 1
    else:
        note("no raw/ frames — using the features saved during receive")

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

    # ---- 1b. KLT keyframe selection (opt-in: bench --keyframes) ------------
    # Non-keyframes are dropped entirely (bmp + features unlinked): COLMAP
    # never sees them and no map points come from them.
    if getattr(args, "keyframes", False):
        import keyframes as kf
        keys, kinfo = kf.select_keyframes(
            bmps, usable,
            drift_px=getattr(args, "klt_drift", kf.DEFAULT_DRIFT_PX),
            min_tracked=getattr(args, "klt_min_tracked", kf.DEFAULT_MIN_TRACKED))
        for s in usable:
            if s not in set(keys):
                for p in (bmps / f"{s}.bmp", features / f"{s}.csv"):
                    p.unlink(missing_ok=True)
        bmp_files = sorted(bmps.glob("IMG*.bmp"))
        usable = keys
        note(f"keyframes: {len(keys)} kept (KLT drift >= "
             f"{getattr(args, 'klt_drift', kf.DEFAULT_DRIFT_PX):g}px or tracked < "
             f"{getattr(args, 'klt_min_tracked', kf.DEFAULT_MIN_TRACKED)})")
        for (s, reason, n_ok, drift) in kinfo:
            note(f"  {s}: {reason}")
        if len(usable) < MIN_USABLE_FRAMES:
            print(f"only {len(usable)} keyframe(s) (need >= {MIN_USABLE_FRAMES}) — "
                  f"lower --klt-drift / --klt-min-tracked.", file=sys.stderr)
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

    # ---- 1c. offline calc8 place-recognition embeddings --------------------
    # One 1064-B descriptor per frame with the device-identical preprocessing
    # (truncating 4x4 block mean -> 160x120). Runs BEFORE matching: the
    # candidate pairs below need them, and map assembly reuses them.
    import embed_host
    t0 = time.time()
    try:
        embs = embed_host.compute_embeddings(bmp_files)
    except (FileNotFoundError, ValueError) as e:
        print(f"!! embedding failed: {e}", file=sys.stderr)
        return 1
    emb_dir = work / "embeddings"
    emb_dir.mkdir(exist_ok=True)
    for stem, b in embs.items():
        np.save(emb_dir / f"{stem}.npy", np.frombuffer(b, dtype=np.uint8))
    note(f"embeddings: {len(embs)} frames @ {embed_host.EMB_DIM} B "
         f"({time.time()-t0:.1f}s, calc8 offline)")

    # ---- 2. cross-image matching (Rust: rayon + AVX2) ----------------------
    # With keyframes, only the candidate pairs are matched (temporal window +
    # top-k embedding neighbours, both causal); otherwise every pair.
    t0 = time.time()
    max_distance = getattr(args, "max_distance", mf.MATCH_MAX_DISTANCE)
    lowe_ratio = getattr(args, "lowe_ratio", mf.LOWE_RATIO)
    cand_pairs = None
    if getattr(args, "keyframes", False):
        import keyframes as kf
        window = getattr(args, "match_window", kf.DEFAULT_WINDOW)
        topk = getattr(args, "match_topk", kf.DEFAULT_TOPK)
        cand_pairs = sorted(kf.candidate_pairs(usable, embs,
                                                window=window, topk=topk))
        note(f"candidate pairs: {len(cand_pairs)} ({window}-frame window + "
             f"top-{topk} embedding)")
        if not cand_pairs:
            print("no candidate pairs — need >= 2 keyframes with a nonzero "
                  "window/topk.", file=sys.stderr)
            return 1
    try:
        mstats = match_features_rust(work, features, max_distance, lowe_ratio,
                                      pairs=cand_pairs)
    except RuntimeError as e:
        # No rust toolchain: the numpy matcher is byte-identical, just slower.
        print(f"!! {e} — falling back to the numpy matcher", file=sys.stderr)
        rows, mstats = mf.match_all(features, max_distance=max_distance,
                                    lowe_ratio=lowe_ratio, pairs=cand_pairs)
        mf.write_matches_csv(work / "matches.csv", rows)
    if mstats.get("error"):
        print(f"matching failed: {mstats['error']}", file=sys.stderr)
        return 1
    note(f"matching: {mstats['total_matches']} matches across "
         f"{mstats['pairs_with_matches']}/{mstats['pairs_total']} pairs "
         f"({time.time()-t0:.1f}s)")
    if mstats["total_matches"]:
        note(f"  best-match Hamming dist: min {mstats['match_dist_min']}, "
             f"median {mstats['match_dist_median']}, max {mstats['match_dist_max']}")
        note(f"  ambiguous matches (2nd candidate also within {max_distance} bits): "
             f"{mstats['ambiguous']} ({100*mstats['ambiguous_frac']:.1f}%)")
    if mstats["pairs_with_matches"] == 0:
        print("no image pair matched at all — nothing for COLMAP. Lower "
              "FAST_THRESHOLD on the ESP or point at more texture.",
              file=sys.stderr)
        return 1

    # ---- 3. COLMAP reconstruction ------------------------------------------
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
            camera_params=getattr(args, "camera_params", None),
            refine_intrinsics=getattr(args, "refine_intrinsics", True),
            min_num_inliers=getattr(args, "min_num_inliers", 3),
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

    # ---- 4. map.txt + quality report ---------------------------------------
    try:
        summary = write_map.build_map_file(
            work / "colmap_work", features, work / "map.txt",
            embeddings=embs, min_points=MIN_FRAME_POINTS)
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
    note(f"map frames: {summary['n_frames']} with >= {MIN_FRAME_POINTS} "
         f"triangulated points (of {len(embs)} embedded)")
    note(f"map points: {summary['n_map_points']} per-frame observations "
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


def ensure_extract_host(work: Path) -> Path:
    """Compile scripts/extract_host.rs with the host stable toolchain unless a
    cached binary is newer than the harness and every included source module."""
    exe = work / "extract_host"
    src_dir = EXTRACT_HOST_SRC.parent.parent / "src"
    newest = EXTRACT_HOST_SRC.stat().st_mtime
    # The harness `#[path]`-includes these; rebuild if any changed.
    for p in src_dir.glob("*.rs"):
        newest = max(newest, p.stat().st_mtime)
    if exe.exists() and exe.stat().st_mtime >= newest:
        return exe
    print(f"compiling {EXTRACT_HOST_SRC.name} (rustc +stable) ...")
    try:
        subprocess.run(
            ["rustc", "+stable", "--edition", "2021", "-O",
             str(EXTRACT_HOST_SRC), "-o", str(exe)],
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "rustc not found — laptop feature extraction needs a Rust toolchain "
            "(run the build in WSL, or `rustup toolchain install stable`)"
        )
    return exe


def extract_features_host(work: Path, w: int, h: int, note,
                          threshold=None) -> None:
    """Run the S3's own extractor on every raw/<IMG>.bit -> features/<IMG>.csv.
    `threshold` overrides pyramid::FAST_THRESHOLD (the extractor's default)."""
    exe = ensure_extract_host(work)
    note(f"laptop feature extraction: {w}x{h}, same Rust extractor as the S3"
         + (f", FAST threshold {threshold}" if threshold else ""))
    cmd = [str(exe), str(work / "raw"), str(work / "features"), str(w), str(h)]
    if threshold:
        cmd.append(str(threshold))
    subprocess.run(cmd, check=True)


def ensure_matcher() -> Path:
    """Build the rayon+AVX2 matcher crate (stable host toolchain) if stale.
    Raises RuntimeError if cargo is unavailable (caller falls back to numpy)."""
    newest = (MATCHER_CRATE / "Cargo.toml").stat().st_mtime
    for p in (MATCHER_CRATE / "src").rglob("*.rs"):
        newest = max(newest, p.stat().st_mtime)
    if MATCHER_EXE.exists() and MATCHER_EXE.stat().st_mtime >= newest:
        return MATCHER_EXE
    print("compiling vo-matcher (cargo +stable, release) ...")
    try:
        subprocess.run(
            ["cargo", "+stable", "build", "--release",
             "--target", "x86_64-unknown-linux-gnu"],
            cwd=MATCHER_CRATE, check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "cargo not found — the Rust matcher needs a Rust toolchain "
            "(run the build in WSL, or `rustup toolchain install stable`)"
        )
    return MATCHER_EXE


def match_features_rust(work: Path, features: Path, max_distance,
                        lowe_ratio, pairs=None) -> dict:
    """Run the Rust matcher -> <work>/matches.csv + stats dict (same images /
    pair ordering / semantics as match_features.match_all). `pairs` is an
    optional list of (stem_a, stem_b) to match; None = every pair."""
    exe = ensure_matcher()
    stats_path = work / "matches_stats.json"
    cmd = [str(exe), str(features), str(work / "matches.csv"), str(stats_path),
           str(max_distance), str(lowe_ratio)]
    if pairs is not None:
        pairs_path = work / "candidate_pairs.csv"
        with open(pairs_path, "w") as f:
            for (a, b) in pairs:
                f.write(f"{a},{b}\n")
        cmd.append(str(pairs_path))
    subprocess.run(cmd, check=True)
    return json.loads(stats_path.read_text())


# COLMAP camera model name -> on-wire id (must match CAMERA_MODEL_* in
# src/bin/main.rs). Only SIMPLE_RADIAL is produced by colmap_map.py.
CAMERA_MODEL_IDS = {"SIMPLE_RADIAL": 1}
EMBEDDING_BYTES = 1064  # calc8 descriptor (must match EMBEDDING_DIM in main.rs)


def load_map_txt(path: Path):
    """Parse map.txt -> (model_id, [f, cx, cy, k1], frames) where frames is a
    list of (embedding_bytes, [(x, y, z, desc_bytes), ...])."""
    model_id = params = None
    frames = []
    cur = None
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
            elif t[0] == "FRAME":
                emb = bytes.fromhex(t[2])
                if len(emb) != EMBEDDING_BYTES:
                    raise ValueError(f"embedding is {len(emb)} B, expected "
                                     f"{EMBEDDING_BYTES}")
                cur = (emb, [])
                frames.append(cur)
            elif t[0] == "POINT":
                if cur is None:
                    raise ValueError(f"POINT before any FRAME in {path}")
                desc = bytes.fromhex(t[4])
                if len(desc) != 32:
                    raise ValueError(f"descriptor is {len(desc)} B, expected 32")
                cur[1].append((float(t[1]), float(t[2]), float(t[3]), desc))
    if model_id is None:
        raise ValueError(f"no CAMERA line in {path}")
    if not frames:
        raise ValueError(f"no FRAME lines in {path}")
    return model_id, params, frames


def upload_map(args, map_path: Path) -> None:
    """Send the built map (intrinsics + per-frame embedding + its points) to
    the ESP (MAP2) and wait for its MAPK ack. Fresh connection: run's dropped."""
    model_id, params, frames = load_map_txt(map_path)
    body = bytearray(rf.MAGIC_MAP_UPLOAD)
    body.append(model_id)
    body += struct.pack("<4f", *params)
    body += struct.pack("<I", len(frames))
    n_points = 0
    for emb, points in frames:
        body += emb
        body += struct.pack("<I", len(points))
        for (x, y, z, desc) in points:
            body += struct.pack("<3f", x, y, z) + desc
        n_points += len(points)
    print(f"uploading map: {len(frames)} frames / {n_points} points, camera "
          f"model {model_id}, params {['%.4g' % p for p in params]}, "
          f"{len(body) + 4} bytes -> {args.host}:{args.port}")
    with socket.create_connection((args.host, args.port), timeout=15) as conn:
        conn.settimeout(60)  # ESP ack after it has read + stored the frames
        conn.sendall(struct.pack("<I", len(body)) + bytes(body))
        kind, payload = rf.read_record(conn)
        if kind != "MAPK":
            raise ConnectionError(f"expected MAPK ack, got {kind}")
        n_frames, n_points = payload
        print(f"map accepted: ESP stored {n_frames} frames / {n_points} points "
              f"— firmware is now in localize mode (idle until the next map run)")


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
