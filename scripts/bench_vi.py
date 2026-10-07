#!/usr/bin/env python3
"""Offline BENCH_VI: replay EuRoC through the S3's VO+EKF loop, no hardware.

Phases (default: all four, in order):
  map   build a localization map from EuRoC cam0 frames (host extractor + COLMAP)
        -> <map-work>/map.txt + bmps/ + features/ + embeddings/
  prep  write the replay inputs the S3 would receive over the wire
        -> <replay-work>/stream.bin (length-prefixed TUM1/IMU1 records),
           embeddings.bin (one 1064-B calc8 embedding per TUM1, stream order),
           gt.csv (EuRoC groundtruth as t_us x y z qw qx qy qz)
  run   compile + run scripts/bench_vi_offline.rs -> <work>/trajectory.csv
  ate   similarity/Rigid-align the trajectory to groundtruth, print + save ATE

Defaults emulate the S3: the VO fix is applied `--spi-us + --vo-latency-us`
after the image (the laptop VO is much faster than the board's ~1.16 s), and
IMU is fused continuously in between. The map lives in a monocular COLMAP
gauge, so the ATE is reported both similarity-aligned (scale absorbed, the
fair trajectory-shape number) and rigid (scale=1, shows the gauge/scale gap).

    python3 scripts/bench_vi.py                      # all phases
    python3 scripts/bench_vi.py --phase prep         # rebuild replay inputs
    python3 scripts/bench_vi.py --phase ate          # re-score an existing run
"""

import argparse
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import receive_frames as rf          # 8-bit gray BMP writer
import receive_map                   # the real build_from_work path
import send_tum as st                # EuRoC zip/mav0 reader + undistort
import embed_host                    # calc8 host embeddings

DEFAULT_EUROC = "/tmp/vicon_room1/V1_01_easy/V1_01_easy.zip"
EMB_DIM = 1064
CAM_W, CAM_H = 640, 480


# ------------------------------------------------------------ EuRoC helpers --

def resolve_euroc(path) -> Path:
    """Accept an ASL zip, a mav0/ dir, or a dir containing exactly one zip."""
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        if (p / "mav0").is_dir():
            return p
        zips = sorted(p.glob("*.zip"))
        if zips:
            return zips[0]
    raise SystemExit(f"--euroc: {path} is not a zip / mav0 dir")


def cam_rows(src, cam):
    rows = [ln.split(",")[:2] for ln in
            src.read_text(f"mav0/cam{cam}/data.csv").splitlines()
            if ln and not ln.startswith("#")]
    return [(int(ts), fn) for ts, fn in rows]


def load_imu(src):
    rows = st.read_euroc_csv(src, "mav0/imu0/data.csv")
    ts = np.array([int(r[0]) for r in rows], dtype=np.int64)
    a = np.array([[float(r[4]), float(r[5]), float(r[6])] for r in rows], np.float32)
    w = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in rows], np.float32)
    return ts, a, w


def cam_imu_extrinsic(src, cam):
    """EuRoC T_BS -> (R_cam_body 3x3, t_cam_body 3): camera<-body rotation
    and the camera center expressed in the body/IMU frame."""
    import re
    text = src.read_text(f"mav0/cam{cam}/sensor.yaml")
    m = re.search(r"T_BS:.*?data:\s*\[([^\]]+)\]", text, re.S)
    vals = [float(x) for x in m.group(1).split(",")]
    T = np.array(vals).reshape(4, 4)
    return T[:3, :3].T, T[:3, 3]


def frame_gray(src, calib, mx, my, cam, fn) -> bytes:
    import numpy as np
    from PIL import Image
    from io import BytesIO
    raw = np.asarray(Image.open(
        BytesIO(src.read_bytes(f"mav0/cam{cam}/data/{fn}"))).convert("L"))
    if raw.shape != (st.EUROC_H, st.EUROC_W):
        raise SystemExit(f"{fn}: shape {raw.shape} != {st.EUROC_H}x{st.EUROC_W}")
    return st.remap_bilinear(raw, mx, my).tobytes()


def write_gt(src, out: Path) -> None:
    lines = [ln for ln in
             src.read_text("mav0/state_groundtruth_estimate0/data.csv").splitlines()
             if ln and not ln.startswith("#")]
    with open(out, "w") as f:
        f.write("# t_us px py pz qw qx qy qz\n")
        for ln in lines:
            p = ln.split(",")
            t_us = int(p[0]) // 1000
            vals = [float(x) for x in p[1:8]]  # px,py,pz,qw,qx,qy,qz
            f.write(f"{t_us} " + " ".join(f"{v:.9g}" for v in vals) + "\n")


# ------------------------------------------------------------------ map phase

def build_map(args) -> int:
    src = st.Euroc(resolve_euroc(args.euroc))
    calib = st.euroc_calib(src, args.cam)
    mx, my = st.undistort_maps(calib)
    rows = cam_rows(src, args.cam)
    sel = rows[args.map_offset::args.map_stride][:args.map_count]
    if len(sel) < 3:
        raise SystemExit(f"only {len(sel)} map frames selected")

    work = Path(args.map_work)
    if work.exists() and args.force_map:
        shutil.rmtree(work)
    raw, bmps = work / "raw", work / "bmps"
    for d in (raw, bmps):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    for i, (_ts, fn) in enumerate(sel, start=1):
        gray = frame_gray(src, calib, mx, my, args.cam, fn)
        stem = f"IMG{i:04d}"
        (raw / f"{stem}.bit").write_bytes(gray)
        rf.write_gray_bmp(bmps / f"{stem}.bmp", CAM_W, CAM_H, gray)
    print(f"map: {len(sel)} frames from {args.euroc} @ stride "
          f"{args.map_stride} offset {args.map_offset} -> {work}")

    from argparse import Namespace
    # EuRoC cam0 intrinsics are known and the frames are already undistorted,
    # so freeze the pinhole camera (f, cx, cy, k1=0) -> metrically-shaped SfM.
    if args.map_refine:
        cam_params, refine = None, True
    else:
        fx, _fv, cu, cv = calib[0], calib[1], calib[2], calib[3]
        cam_params, refine = [fx, cu, cv, 0.0], False
        print(f"map: fixed intrinsics f={fx:.3f} cx={cu:.3f} cy={cv:.3f} k1=0")
    rc = receive_map.build_from_work(work, Namespace(
        min_features=args.min_features, threshold=args.threshold,
        max_distance=args.max_distance, lowe_ratio=args.lowe_ratio,
        camera_params=cam_params, refine_intrinsics=refine,
        min_num_inliers=args.min_num_inliers,
        keyframes=args.keyframes, klt_drift=20.0, klt_min_tracked=150,
        match_window=30, match_topk=15))
    if rc == 0:
        compute_map_scale(work, sel, src)
    return rc


def _fit_sim3(S, D):
    """Least-squares similarity mapping S onto D; returns (scale, R, t)."""
    ms, md = S.mean(0), D.mean(0)
    sc, dc = S - ms, D - md
    u, d, vt = np.linalg.svd(dc.T @ sc / len(S))
    sign = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, sign]) @ vt
    scale = float(np.trace(np.diag(d) @ np.diag([1.0, 1.0, sign]))
                  / ((sc ** 2).sum() / len(S)))
    return scale, rot, md - scale * (rot @ ms)


def compute_map_scale(work, sel, src, thresh_m=0.2) -> float:
    """Monocular SfM is scale-free; measure the map's m-per-unit against the
    EuRoC groundtruth with a RANSAC similarity (a few wrongly-registered
    frames otherwise skew a plain least-squares scale by orders of magnitude)."""
    import pycolmap
    lines = [l for l in
             src.read_text("mav0/state_groundtruth_estimate0/data.csv").splitlines()
             if l and not l.startswith("#")]
    gt_t = np.array([int(l.split(",")[0]) // 1000 for l in lines], np.int64)
    gt_p = np.array([[float(x) for x in l.split(",")[1:4]] for l in lines])
    recon = pycolmap.Reconstruction(str(Path(work) / "colmap_work" / "sparse" / "best"))
    C, G = [], []
    for im in recon.images.values():
        k = int(Path(im.name).stem[3:]) - 1
        i = int(np.argmin(np.abs(gt_t - sel[k][0] // 1000)))
        C.append(im.projection_center())
        G.append(gt_p[i])
    C, G = np.array(C), np.array(G)
    rng = np.random.default_rng(0)
    best_inl, best = -1, None
    for _ in range(6000):
        sub = rng.choice(len(C), 6, replace=False)
        s, R, t = _fit_sim3(C[sub], G[sub])
        n = int((np.linalg.norm((s * (R @ C.T).T + t) - G, axis=1) < thresh_m).sum())
        if n > best_inl:
            best_inl, best = n, (s, R, t)
    s, R, t = best
    err = np.linalg.norm((s * (R @ C.T).T + t) - G, axis=1)
    (Path(work) / "map_scale.txt").write_text(f"{s:.9g}\n")
    # Full map->GT similarity p_gt = s*R*p_map + t; the pose-prior harness
    # needs R,t to express a GT prior in the (scaled) map frame.
    (Path(work) / "map_sim3.txt").write_text(
        " ".join(f"{v:.9g}" for v in [s, *R.ravel(), *t]) + "\n")
    # The map frame has an arbitrary orientation; gravity in that frame (for
    # the EKF) is R^T * GT-frame gravity, GT being z-up.
    g_map = R.T @ np.array([0.0, 0.0, -9.81])
    (Path(work) / "map_gravity.txt").write_text(
        " ".join(f"{v:.9g}" for v in g_map) + "\n")
    np.savetxt(Path(work) / "map_scale_inliers.csv",
               np.column_stack([range(1, len(C) + 1), err]), delimiter=",",
               header="stem_index,gt_err_m", comments="")
    print(f"map scale: 1 map unit = {s:.6g} m; RANSAC inliers(<{thresh_m} m) "
          f"{best_inl}/{len(C)} frames; mean err over inliers "
          f"{err[err < thresh_m].mean() * 1000:.1f} mm")
    return s


# ----------------------------------------------------------------- prep phase

def compute_embedder():
    import tensorflow as tf
    if not embed_host.DEFAULT_MODEL.is_file():
        raise SystemExit(f"{embed_host.DEFAULT_MODEL} not found — copy it from calc-quant")
    interp = tf.lite.Interpreter(model_path=str(embed_host.DEFAULT_MODEL), num_threads=1)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    def embed(gray: bytes) -> bytes:
        g = np.frombuffer(gray, np.uint8).reshape(CAM_H, CAM_W)
        small = embed_host.downscale_4x4(g)
        interp.set_tensor(inp["index"], small[None, :, :, None])
        interp.invoke()
        e = interp.get_tensor(out["index"])[0].astype(np.uint8)
        assert e.size == EMB_DIM
        return e.tobytes()

    return embed


def prep(args) -> int:
    if not args.query_stride:
        args.query_stride = max(1, round(args.image_period * 20))  # EuRoC is 20 Hz

    src = st.Euroc(resolve_euroc(args.euroc))
    calib = st.euroc_calib(src, args.cam)
    mx, my = st.undistort_maps(calib)
    rows = cam_rows(src, args.cam)
    qsel = rows[args.query_offset::args.query_stride][:args.query_count]
    if len(qsel) < 2:
        raise SystemExit(f"only {len(qsel)} query frames selected")
    imu_ts, imu_a, imu_w = load_imu(src)

    work = Path(args.replay_work)
    if work.exists():
        shutil.rmtree(work)
    (work / "frames").mkdir(parents=True)
    embed = compute_embedder()

    # Interleave IMU1 batches with TUM1 images exactly as send_tum.iter_euroc.
    ptr = int(np.searchsorted(imu_ts, qsel[0][0]))
    stream = bytearray()
    embs = bytearray()
    imgs = []
    seq = 1
    n_imu = 0

    def emit_imu(upto_ns: int, batch: int):
        nonlocal ptr, n_imu, seq, stream
        while ptr < len(imu_ts) and imu_ts[ptr] < upto_ns:
            take = min(batch, 64, len(imu_ts) - ptr)
            sl = slice(ptr, ptr + take)
            t0_us = int(imu_ts[ptr]) // 1000
            if take == 1:
                dt_us = 5000
            else:
                dt_us = max(100, round((imu_ts[ptr + take - 1] - imu_ts[ptr])
                                       / (take - 1) / 1000))
            flat = np.concatenate([imu_a[sl], imu_w[sl]], axis=1).ravel()
            payload = (b"IMU1" + struct.pack("<IQHH", seq, t0_us, take, dt_us)
                       + flat.astype("<f4").tobytes())
            stream += struct.pack("<I", len(payload)) + payload
            ptr += take
            n_imu += take
            seq += 1

    for ts, fn in qsel:
        emit_imu(ts, args.imu_batch)
        gray = frame_gray(src, calib, mx, my, args.cam, fn)
        stem = f"IMG{len(imgs) + 1:04d}"
        (work / "frames" / f"{stem}.bit").write_bytes(gray)
        payload = (b"TUM1" + struct.pack("<IQHH", seq, ts // 1000, CAM_W, CAM_H)
                   + gray)
        stream += struct.pack("<I", len(payload)) + payload
        embs += embed(gray)
        imgs.append((stem, ts))
        seq += 1
    # trailing IMU so the final state propagates past the last fix.
    emit_imu(int(qsel[-1][0] + args.trail_s * 1e9), args.imu_batch)

    (work / "stream.bin").write_bytes(bytes(stream))
    (work / "embeddings.bin").write_bytes(bytes(embs))
    (work / "imgs.csv").write_text("stem,t_us\n"
                                   + "".join(f"{s},{t // 1000}\n" for s, t in imgs))
    write_gt(src, work / "gt.csv")
    r_cb, t_cb = cam_imu_extrinsic(src, args.cam)
    np.savetxt(work / "imu_cam_rot.txt",
               np.concatenate([r_cb.ravel(), t_cb]).reshape(1, 12), fmt="%.9g")
    span = (imgs[-1][1] - imgs[0][1]) / 1e9
    print(f"prep: {len(imgs)} query images over {span:.1f}s + {n_imu} IMU samples "
          f"({seq - 1} records, {len(stream) / 1e6:.1f} MB) -> {work}")
    return 0


# ------------------------------------------------------------------ run phase

def compile_harness(work: Path) -> Path:
    src = REPO / "scripts" / "bench_vi_offline.rs"
    exe = work / "bench_vi_offline"
    newest = src.stat().st_mtime
    for p in (REPO / "src").glob("*.rs"):
        newest = max(newest, p.stat().st_mtime)
    if exe.exists() and exe.stat().st_mtime >= newest:
        return exe
    print("compiling bench_vi_offline.rs (rustc +stable) ...")
    subprocess.run(["rustc", "+stable", "--edition", "2021", "-O",
                    str(src), "-o", str(exe)], check=True)
    return exe


def run(args) -> int:
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    exe = compile_harness(work)
    map_path = Path(args.map_work) / "map.txt"
    if not map_path.is_file():
        raise SystemExit(f"{map_path} missing — run the map phase first")
    scale = args.map_scale
    if scale is None:
        scale_file = Path(args.map_work) / "map_scale.txt"
        scale = float(scale_file.read_text()) if scale_file.is_file() else 1.0
    grav = "0,0,-9.81"
    grav_file = Path(args.map_work) / "map_gravity.txt"
    if grav_file.is_file():
        grav = ",".join(grav_file.read_text().split())
    cmd = [str(exe), "--map", str(map_path), "--replay", str(args.replay_work),
           "--out", str(work / "trajectory.csv"),
           "--vo-latency-us", str(args.vo_latency_us), "--spi-us", str(args.spi_us),
           "--seed", str(args.seed), "--fast-threshold", str(args.threshold),
           "--map-scale", repr(scale), "--gravity", grav,
           "--prior", args.prior, "--matcher", args.matcher,
           "--topk", str(args.topk), "--window-px", repr(args.window_px),
           "--max-kf-angle", repr(args.max_kf_angle),
           "--prior-noise-pos", repr(args.prior_noise_pos),
           "--prior-noise-att", repr(args.prior_noise_att)]
    sim3_file = Path(args.map_work) / "map_sim3.txt"
    if sim3_file.is_file():
        cmd += ["--map-sim3", str(sim3_file)]
    elif args.prior == "gt":
        raise SystemExit(f"--prior gt needs {sim3_file} — run --phase scale")
    return subprocess.run(cmd).returncode


# ------------------------------------------------------------------ ate phase

def umeyama(src, dst, with_scale: bool):
    """Least-squares (similarity or rigid) align src -> dst; returns aligned."""
    ms, md = src.mean(0), dst.mean(0)
    sc, dc = src - ms, dst - md
    u, d, vt = np.linalg.svd(dc.T @ sc / len(src))
    sign = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, sign]) @ vt
    if with_scale:
        scale = float(np.trace(np.diag(d) @ np.diag([1.0, 1.0, sign]))
                      / ((sc ** 2).sum() / len(src)))
    else:
        scale = 1.0
    return (scale * (rot @ sc.T)).T + md, scale


def ate(args) -> int:
    work = Path(args.work)
    traj_p = work / "trajectory.csv"
    if not traj_p.is_file():
        raise SystemExit(f"{traj_p} missing — run the run phase first")
    tr = np.loadtxt(traj_p)
    est_t = tr[:, 0]
    est_p = tr[:, 1:4]
    gt = np.loadtxt(args.replay_work + "/gt.csv")
    gt_t, gt_p = gt[:, 0], gt[:, 1:4]
    # nearest GT position for each estimate timestamp
    idx = np.searchsorted(gt_t, est_t).clip(1, len(gt_t) - 1)
    lo = np.abs(gt_t[idx - 1] - est_t) < np.abs(gt_t[idx] - est_t)
    ref = gt_p[idx - lo]

    out = ["", "===== BENCH_VI offline ATE =====",
           f"trajectory samples : {len(est_t)} over "
           f"{(est_t[-1] - est_t[0]) / 1e6:.1f}s",
           f"est path length    : {np.linalg.norm(np.diff(est_p, axis=0), axis=1).sum():.3f} m",
           f"GT path length     : {np.linalg.norm(np.diff(ref, axis=0), axis=1).sum():.3f} m"]
    for name, with_scale in (("similarity (sim3)", True), ("rigid (se3)", False)):
        aligned, scale = umeyama(est_p, ref, with_scale)
        err = np.linalg.norm(aligned - ref, axis=1)
        out.append(f"{name:18s}: RMSE {err.mean() * 1000:8.1f} mm, "
                   f"mean {err.mean() * 1000:8.1f} mm, max {err.max() * 1000:8.1f} mm"
                   + (f", scale {scale:.4g}" if with_scale else ""))
    (work / "ate.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))
    # save the similarity-aligned trajectory next to the GT it was scored against
    aligned, _ = umeyama(est_p, ref, True)
    np.savetxt(work / "trajectory_aligned.csv",
               np.column_stack([est_t / 1e6, aligned, ref]),
               header="t_s est_x est_y est_z gt_x gt_y gt_z", comments="")
    return 0


# ------------------------------------------------------------------ driver --

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="all",
                    choices=["all", "map", "scale", "prep", "run", "ate"])
    ap.add_argument("--euroc", default=DEFAULT_EUROC)
    ap.add_argument("--cam", type=int, default=0, choices=[0, 1])
    ap.add_argument("--work", default=str(REPO / "bench_vi"),
                    help="top-level output dir (trajectory + ATE)")
    ap.add_argument("--map-work", default=None, help="map dir (default <work>/map)")
    ap.add_argument("--replay-work", default=None, help="replay dir (default <work>/replay)")
    # map selection / matching
    ap.add_argument("--map-stride", type=int, default=10, help="0.5 Hz over EuRoC 20 Hz")
    ap.add_argument("--map-offset", type=int, default=300,
                    help="skip the rotation-only lead-in (~15 s)")
    ap.add_argument("--map-count", type=int, default=300)
    ap.add_argument("--force-map", action="store_true", help="wipe + rebuild the map")
    ap.add_argument("--map-refine", action="store_true",
                    help="let COLMAP refine intrinsics (default: freeze EuRoC cam0)")
    ap.add_argument("--no-keyframes", dest="keyframes", action="store_false",
                    help="disable KLT keyframe + candidate-pair matching (default: on)")
    ap.add_argument("--keyframes", dest="keyframes", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--threshold", type=int, default=20, help="FAST threshold (map + query)")
    ap.add_argument("--min-features", type=int, default=receive_map.MIN_FEATURES)
    ap.add_argument("--min-num-inliers", type=int, default=20,
                    help="COLMAP min inliers to register/bootstrap an image "
                         "(pipeline default 3 is too permissive for EuRoC)")
    ap.add_argument("--max-distance", type=int, default=100)
    ap.add_argument("--lowe-ratio", type=float, default=0.8)
    # query selection / replay timing
    ap.add_argument("--image-period", type=float, default=5.0,
                    help="dataset seconds between query images")
    ap.add_argument("--query-stride", type=int, default=None,
                    help="override image-period (20 Hz frames)")
    ap.add_argument("--query-offset", type=int, default=307,
                    help="first query frame index (offset so queries are not map frames)")
    ap.add_argument("--query-count", type=int, default=30)
    ap.add_argument("--imu-batch", type=int, default=10)
    ap.add_argument("--trail-s", type=float, default=5.0, help="trailing IMU after last image")
    ap.add_argument("--vo-latency-us", type=int, default=1_160_000,
                    help="simulated S3 VO compute time")
    ap.add_argument("--spi-us", type=int, default=100_000, help="simulated image transfer")
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=0x1234_5678_9abc_def0)
    ap.add_argument("--map-scale", type=float, default=None,
                    help="override the map's m/unit (default: <map-work>/map_scale.txt)")
    # pose-prior keyframe selection + windowed matching (laptop baseline)
    ap.add_argument("--prior", choices=["ekf", "gt"], default="ekf",
                    help="pose prior for keyframe search (ekf = online state, "
                         "gt = groundtruth ablation/upper bound)")
    ap.add_argument("--matcher", choices=["brute", "windowed"], default="brute",
                    help="brute = embedding top-1 + full match (legacy); "
                         "windowed = pose-prior keyframe + local-window matching")
    ap.add_argument("--topk", type=int, default=1,
                    help="keyframes whose points are unioned for matching")
    ap.add_argument("--window-px", type=float, default=80.0,
                    help="matching search box around the prior-projected point "
                         "(L0 px); 15 is too tight for the ~0.1 m / ~1 deg map "
                         "alignment floor")
    ap.add_argument("--max-kf-angle", type=float, default=30.0,
                    help="attitude gate: keep keyframes within this angle of the prior")
    ap.add_argument("--prior-noise-pos", type=float, default=0.0,
                    help="GT-prior position noise sigma (m); 0 = oracle")
    ap.add_argument("--prior-noise-att", type=float, default=0.0,
                    help="GT-prior attitude noise sigma (deg); 0 = oracle")
    args = ap.parse_args()

    args.map_work = args.map_work or str(Path(args.work) / "map")
    args.replay_work = args.replay_work or str(Path(args.work) / "replay")

    if args.phase in ("all", "map"):
        rc = build_map(args)
        if rc:
            return rc
    if args.phase == "scale":
        src = st.Euroc(resolve_euroc(args.euroc))
        rows = cam_rows(src, args.cam)
        sel = rows[args.map_offset::args.map_stride][:args.map_count]
        compute_map_scale(args.map_work, sel, src)
        return 0
    if args.phase in ("all", "prep"):
        rc = prep(args)
        if rc:
            return rc
    if args.phase in ("all", "run"):
        rc = run(args)
        if rc:
            return rc
    if args.phase in ("all", "ate"):
        rc = ate(args)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
