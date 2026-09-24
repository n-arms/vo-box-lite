#!/usr/bin/env python3
"""Benchmark the vo-box feature/SfM pipeline on a TUM RGB-D sequence.

Takes the first N frames listed in <dataset>/rgb.txt, converts each RGB PNG to
the 8-bit gray raw/ + bmps/ layout a map run produces, then runs the same build
as `receive_map.py --rebuild` (host extractor -> Hamming match -> pycolmap SfM
-> map.txt) and prints a map-health report. If <dataset>/groundtruth.txt exists
the reconstruction is similarity-aligned to it, so pose quality is measured
against the benchmark's own ground truth.

    python3 scripts/bench_freiburg.py                    # first 40 frames
    python3 scripts/bench_freiburg.py --stride 4         # every 4th (bigger baseline)
    python3 scripts/bench_freiburg.py --report-only      # skip rebuild, re-report
"""

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import receive_frames as rf      # 8-bit gray BMP writer (stdlib-only)
import receive_map               # the real build_from_work path
import match_features as mf      # matcher defaults (max distance / Lowe ratio)

# TUM fr1 (freiburg1) published RGB calibration. The benchmark's model is
# 5-parameter but our pipeline uses COLMAP SIMPLE_RADIAL (single radial term),
# so k1 = d0 is an approximation.
TUM_FR1_FX = 517.306408
TUM_FR1_CX = 318.643040
TUM_FR1_CY = 255.313989
TUM_FR1_K1 = 0.262383


def read_rgb_txt(path: Path):
    """-> [(timestamp, relpath)] from rgb.txt, skipping '#' comments."""
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ts, rel = line.split()[:2]
            entries.append((float(ts), rel))
    return entries


def select_frames(dataset: Path, count: int, stride: int):
    entries = read_rgb_txt(dataset / "rgb.txt")
    sel = entries[::stride][:count]
    if not sel:
        raise SystemExit(f"no frames selected from {dataset / 'rgb.txt'}")
    return sel


def extract_frames(dataset: Path, work: Path, sel) -> None:
    """Convert the selected RGB frames to raw/IMG####.bit + bmps/IMG####.bmp."""
    raw, bmps = work / "raw", work / "bmps"
    for d in (raw, bmps):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    size = None
    for i, (_ts, rel) in enumerate(sel, start=1):
        img = Image.open(dataset / rel).convert("L")
        if size is None:
            size = img.size
        elif img.size != size:
            raise SystemExit(f"{rel}: {img.size} != {size} (dims must be uniform)")
        gray = np.asarray(img, dtype=np.uint8).tobytes()
        stem = f"IMG{i:04d}"
        (raw / f"{stem}.bit").write_bytes(gray)
        rf.write_gray_bmp(bmps / f"{stem}.bmp", size[0], size[1], gray)


def load_gt(dataset: Path, timestamps):
    """Nearest-sample ground-truth (x, y, z) for each frame timestamp."""
    gt_path = dataset / "groundtruth.txt"
    if not gt_path.is_file():
        return None
    rows = np.array([[float(x) for x in l.split()]
                     for l in gt_path.read_text().splitlines()
                     if l.strip() and not l.startswith("#")])
    if rows.size == 0:
        return None
    return np.array([rows[np.argmin(abs(rows[:, 0] - t))][1:4] for t in timestamps])


def similarity_align(src, dst):
    """Least-squares similarity (scale, R, t) mapping `src` onto `dst`
    (Umeyama). Monocular SfM has no scale, so alignment must absorb it."""
    ms, md = src.mean(0), dst.mean(0)
    sc, dc = src - ms, dst - md
    u, d, vt = np.linalg.svd(dc.T @ sc / len(src))
    sign = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, sign]) @ vt
    scale = float(np.trace(np.diag(d) @ np.diag([1.0, 1.0, sign]))
                  / ((sc ** 2).sum() / len(src)))
    return (scale * (rot @ sc.T)).T + md, scale


def health_report(work: Path, dataset: Path, timestamps, span_s: float,
                  keyframed: bool = False) -> int:
    """Map-health metrics from the finished COLMAP model + a ground-truth
    trajectory comparison if the dataset ships one."""
    import write_map

    recon = write_map.load_best_model(work / "colmap_work")
    if recon.num_cameras() != 1:
        print(f"!! expected 1 camera, found {recon.num_cameras()}")
        return 1
    cam = list(recon.cameras.values())[0]
    f, cx, cy, k1 = (list(map(float, cam.params)) + [0.0] * 4)[:4]

    n_sel = len(timestamps)
    imgs = sorted(recon.images.values(), key=lambda im: im.name)
    per_img = Counter()
    track_lens, dup_tracks = [], 0
    for p in recon.points3D.values():
        ids = [e.image_id for e in p.track.elements]
        track_lens.append(len(ids))
        if len(ids) != len(set(ids)):
            dup_tracks += 1
        for image_id in ids:
            per_img[image_id] += 1
    errs = np.asarray([p.error for p in recon.points3D.values() if p.has_error])

    print("\n===== map health =====")
    print(f"frames selected      : {n_sel} ({span_s:.1f}s of video)")
    n_map = n_sel
    if keyframed:
        # Non-keyframes were deleted from bmps/ before the build; COLMAP only
        # ever saw the survivors, so judge registration against those.
        n_map = len(list((work / "bmps").glob("IMG*.bmp")))
        print(f"keyframes kept (KLT) : {n_map}/{n_sel}")
    print(f"registered by COLMAP : {recon.num_images()}/{n_map} "
          f"({100 * recon.num_images() / n_map:.0f}%)")
    print(f"3D points            : {recon.num_points3D()}")
    if errs.size:
        print(f"point reproj error   : mean {errs.mean():.2f} px, median "
              f"{np.median(errs):.2f} px, p95 {np.percentile(errs, 95):.2f} px")
    if track_lens:
        t = np.asarray(track_lens)
        print(f"track length         : mean {t.mean():.1f}, median {int(np.median(t))}, "
              f"max {t.max()}")
        print(f"track merging        : {dup_tracks}/{len(t)} points "
              f"({100 * dup_tracks / len(t):.0f}%) have >1 observation from the same "
              f"image (matcher over-merge)")
    if per_img:
        c = np.asarray(list(per_img.values()))
        print(f"points/image         : median {int(np.median(c))}, "
              f"min {int(c.min())}, max {int(c.max())}")

    # --- trajectory vs ground truth (similarity-aligned) ---------------------
    traj_ratio = None
    if imgs:
        est = np.array([im.projection_center() for im in imgs])
        gt_all = load_gt(dataset, timestamps)
        if gt_all is not None:
            gt = gt_all[[int(Path(im.name).stem[3:]) - 1 for im in imgs]]
            aligned, scale = similarity_align(est, gt)
            derr = np.linalg.norm(aligned - gt, axis=1)
            gt_path = np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1))
            traj_ratio = float(derr.mean() / max(gt_path, 1e-9))
            print(f"trajectory vs GT     : mean {derr.mean() * 1000:.1f} mm, "
                  f"max {derr.max() * 1000:.1f} mm over a {gt_path * 100:.1f} cm GT path "
                  f"({100 * traj_ratio:.1f}%), scale {scale:.4g}")
            print(f"  est path/straight  : "
                  f"{np.sum(np.linalg.norm(np.diff(est, axis=0), axis=1)) / max(np.linalg.norm(est.max(0) - est.min(0)), 1e-9):.2f} "
                  f"(GT {np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1)) / max(np.linalg.norm(gt.max(0) - gt.min(0)), 1e-9):.2f})")

    print("intrinsics vs TUM fr1 : "
          f"f {f:.1f} ({100 * (f - TUM_FR1_FX) / TUM_FR1_FX:+.1f}%), "
          f"cx {cx:.1f} ({cx - TUM_FR1_CX:+.1f} px), "
          f"cy {cy:.1f} ({cy - TUM_FR1_CY:+.1f} px), k1 {k1:+.4f}")
    if abs(cx - 320.0) < 0.5 and abs(cy - 240.0) < 0.5:
        print("  (cx/cy are the fixed image center — COLMAP does not refine the "
              "principal point, so the GT offset is expected)")

    reg = recon.num_images() / n_map
    reproj = float(errs.mean()) if errs.size else 99.0
    tr = traj_ratio if traj_ratio is not None else 0.0
    if reg >= 0.9 and reproj < 2.0 and tr < 0.05:
        verdict = "healthy"
    elif reg < 0.7 or reproj > 4.0 or tr > 0.20:
        verdict = "DEGENERATE"
    else:
        verdict = "marginal"
    print(f"verdict              : {verdict}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(REPO / "rgbd_dataset_freiburg1_xyz"),
                    help="dir holding rgb.txt + rgb/ (and groundtruth.txt)")
    ap.add_argument("--count", type=int, default=40, help="number of frames")
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth listed frame (wider baseline)")
    ap.add_argument("--work", default=str(REPO / "bench_freiburg1_xyz"),
                    help="working dir for frames/features/COLMAP/map.txt")
    ap.add_argument("--min-features", type=int, default=receive_map.MIN_FEATURES,
                    help="drop frames with fewer features than this")
    ap.add_argument("--threshold", type=int, default=40,
                    help="FAST threshold (pyramid::FAST_THRESHOLD is 10, which "
                         "saturates MAX_FEATURES=4096 on rich RGB-D imagery)")
    ap.add_argument("--max-distance", type=int, default=mf.MATCH_MAX_DISTANCE,
                    help="max matching Hamming distance (pipeline default 100)")
    ap.add_argument("--lowe-ratio", type=float, default=mf.LOWE_RATIO,
                    help="Lowe ratio test (pipeline default 0.8)")
    ap.add_argument("--fixed-intrinsics", action="store_true",
                    help="use the TUM fr1 calibration as-is (COLMAP does not "
                         "refine focal/radial); default is COLMAP-derived")
    ap.add_argument("--fixed-k1", type=float, default=None,
                    help="override k1 for --fixed-intrinsics (default TUM d0)")
    ap.add_argument("--no-keyframes", dest="keyframes", action="store_false",
                    help="disable KLT keyframe selection + candidate-pair matching "
                         "(default: enabled)")
    ap.add_argument("--klt-drift", type=float, default=20.0,
                    help="new keyframe when median KLT drift from the last "
                         "keyframe reaches this many px (default 20)")
    ap.add_argument("--klt-min-tracked", type=int, default=150,
                    help="new keyframe when fewer than this many KLT points "
                         "still track (default 150)")
    ap.add_argument("--match-window", type=int, default=30,
                    help="match each keyframe against this many previous "
                         "keyframes (default 30)")
    ap.add_argument("--match-topk", type=int, default=15,
                    help="also match each keyframe against this many most "
                         "similar prior keyframes by embedding (default 15)")
    ap.add_argument("--report-only", action="store_true",
                    help="skip conversion + build, just re-run the health report")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.dataset)
    sel = select_frames(dataset, args.count, args.stride)
    span = sel[-1][0] - sel[0][0]

    cam_params, refine = None, True
    if args.fixed_intrinsics:
        k1 = TUM_FR1_K1 if args.fixed_k1 is None else args.fixed_k1
        cam_params = [TUM_FR1_FX, TUM_FR1_CX, TUM_FR1_CY, k1]
        refine = False

    if not args.report_only:
        extract_frames(dataset, work, sel)
        print(f"converted {len(sel)} frames from {dataset} -> {work} "
              f"({span:.1f}s span, stride {args.stride})")
        print(f"FAST t={args.threshold}, Hamming <= {args.max_distance}, "
              f"Lowe {args.lowe_ratio}, "
              + ("fixed intrinsics " + str([round(p, 4) for p in cam_params])
                 if cam_params else "COLMAP-derived intrinsics"))
        rc = receive_map.build_from_work(work, argparse.Namespace(
            min_features=args.min_features, threshold=args.threshold,
            max_distance=args.max_distance, lowe_ratio=args.lowe_ratio,
            camera_params=cam_params, refine_intrinsics=refine,
            keyframes=args.keyframes, klt_drift=args.klt_drift,
            klt_min_tracked=args.klt_min_tracked,
            match_window=args.match_window, match_topk=args.match_topk))
        if rc != 0:
            return rc
    keyframed = args.keyframes and not args.report_only
    if args.report_only:
        # A previous keyframed build deleted non-keyframes from bmps/; detect
        # that from the saved report so the denominator stays correct.
        try:
            keyframed = "keyframes:" in (work / "map_report.txt").read_text()
        except OSError:
            keyframed = False
    return health_report(work, dataset, [t for t, _ in sel], span,
                         keyframed=keyframed)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
