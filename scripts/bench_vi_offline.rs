//! Offline BENCH_VI replay: the S3's VO + EKF loop on the laptop, no board.
//!
//! Consumes a prepared replay directory (see scripts/bench_vi.py):
//!   stream.bin      length-prefixed TUM1 (image) + IMU1 (10-sample batch)
//!                   records in dataset order, exactly as send_tum.py emits.
//!   embeddings.bin  one 1064-B calc8 embedding per TUM1, in stream order.
//!   gt.csv          groundtruth pose stream (for scoring, not used here).
//! and a map.txt (`CAMERA` + `FRAME`/`POINT` groups, as write_map.py emits).
//!
//! It mirrors `src/bin/vo_replay.rs` link_task + vo_task in one thread:
//! IMU is fused continuously; a TUM1 snapshots the EKF at `t_us`, runs VO
//! (pyramid + top-1 embedding + RANSAC PnP), and the fix is applied
//! `--spi-us + --vo-latency-us` later by rewind + correct + replay — the
//! simulated S3 latency, since the laptop VO is much faster. Output is the
//! corrected `(t, pose)` stream as TUM CSV.
//!
//! The modules are `#[path]`-included from `src/`, so this runs the same code
//! as the firmware (host scalar/mirror kernels; no esp-idf, no TFLM).
//!
//! Build: rustc +stable --edition 2021 -O scripts/bench_vi_offline.rs -o bench_vi_offline
//! Run:   benches_vi_offline --map m.txt --replay dir --out traj.csv [flags]

#![allow(dead_code)]

#[path = "../src/blur.rs"]
mod blur;
#[path = "../src/downscale.rs"]
mod downscale;
#[path = "../src/fast.rs"]
mod fast;
#[path = "../src/rbrief.rs"]
mod rbrief;
#[path = "../src/pyramid.rs"]
mod pyramid;
#[path = "../src/matcher.rs"]
mod matcher;
#[path = "../src/ransac.rs"]
mod ransac;
#[path = "../src/ekf.rs"]
mod ekf;
#[path = "../src/localize.rs"]
mod localize;

use std::env;
use std::fs;
use std::process::exit;

use localize::MapPoint;
use ransac::Rng as _;

// ---- constants mirrored from src/bin/vo_replay.rs --------------------------
const CAM_W: usize = 640;
const CAM_H: usize = 480;
const EMBEDDING_DIM: usize = 1064;
const MAX_MAP_FRAMES: usize = 4096;
const MAX_MAP_POINTS: usize = 400_000;
const MAX_PRIOR_CANDIDATES: usize = 8192;
const FAST_THRESHOLD: i32 = 40; // TUM/EuRoC bench default (device localize uses 10)
// EKF fusion tuning, copied verbatim from vo_replay.rs.
const R_POS_VAR: f32 = 0.01;
const R_ATT_VAR: f32 = 0.002;
const R_VEL_VAR: f32 = 0.01; // (0.1 m/s)^2 VO-differenced velocity variance
const GAP_RESYNC_US: u64 = 500_000;
const COV_EVERY: usize = 10;

type ImuSample = (u64, [f32; 3], [f32; 3]);

// ------------------------------------------------------------------- CLI ----

/// Where the pose prior comes from: the online filter state (honest) or the
/// groundtruth sim3-transformed into the map frame (oracle ablation).
#[derive(Clone, Copy, PartialEq, Debug)]
enum PriorMode {
    Ekf,
    Gt,
}

/// `Brute` = legacy embedding top-1 + full match. `Windowed` = hybrid:
/// pose-prior keyframe search + brute match over the prior-projected points.
#[derive(Clone, Copy, PartialEq, Debug)]
enum MatchMode {
    Brute,
    Windowed,
}

struct Opts {
    map: String,
    replay: String,
    out: String,
    vo_latency_us: u64,
    spi_us: u64,
    seed: u64,
    fast_threshold: i32,
    map_scale: f32,
    gravity: [f32; 3],
    est_gravity: bool,
    r_pos: f32,
    r_att: f32,
    r_vel: f32,
    min_inliers: usize,
    max_vo_speed: f32,
    max_att_resid_deg: f32,
    fuse_attitude: bool,
    reopen_ba_at: usize,
    reopen_ba_std: f32,
    ba_floor_std: f32,
    res_fb_gain: f32,
    res_fb_alpha: f32,
    matcher: MatchMode,
    prior: PriorMode,
    topk: usize,
    window_px: f32,
    max_kf_angle: f32,
    prior_noise_pos: f32,
    prior_noise_att_deg: f32,
    map_sim3: Option<String>,
    dedup_px: f32,
    bucket_px: f32,
    bucket_k: usize,
    bucket_min: usize,
}

fn parse_args() -> Opts {
    let a: Vec<String> = env::args().collect();
    let mut o = Opts {
        map: String::new(),
        replay: String::new(),
        out: "trajectory.csv".to_string(),
        vo_latency_us: 1_160_000,
        spi_us: 100_000,
        seed: 0x1234_5678_9abc_def0,
        fast_threshold: FAST_THRESHOLD,
        map_scale: 1.0,
        gravity: [0.0, 0.0, -9.81],
        est_gravity: false,
        r_pos: R_POS_VAR,
        r_att: R_ATT_VAR,
        r_vel: R_VEL_VAR,
        // Fix-rejection gates off by default: the firmware has no gating, and
        // absolute inlier counts are regime-dependent (a 1 s fix with 28
        // inliers is good; a 5 s outlier has ~10). Opt in via the flags.
        min_inliers: 0,
        max_vo_speed: 0.0,
        max_att_resid_deg: 0.0,
        fuse_attitude: true,
        reopen_ba_at: 0,
        reopen_ba_std: 0.5,
        ba_floor_std: 0.0,
        res_fb_gain: 0.0,
        res_fb_alpha: 0.5,
        matcher: MatchMode::Brute,
        prior: PriorMode::Ekf,
        topk: 1,
        // 15 px (the plan default) misses the correspondences on this map: the
        // map<->GT similarity leaves ~0.1 m / ~1 deg, i.e. ~30 px of projected
        // offset, so the whole correct set falls outside the box. 80 px is the
        // smallest value that keeps windowed+ekf within noise of brute.
        window_px: 80.0,
        max_kf_angle: 30.0,
        prior_noise_pos: 0.0,
        prior_noise_att_deg: 0.0,
        map_sim3: None,
        dedup_px: 0.0,
        bucket_px: 0.0,
        bucket_k: 2,
        bucket_min: 150,
    };
    let mut i = 1;
    while i < a.len() {
        let val = || a.get(i + 1).cloned().unwrap_or_default();
        match a[i].as_str() {
            "--map" => {
                o.map = val();
                i += 2;
            }
            "--replay" => {
                o.replay = val();
                i += 2;
            }
            "--out" => {
                o.out = val();
                i += 2;
            }
            "--vo-latency-us" => {
                o.vo_latency_us = val().parse().expect("bad --vo-latency-us");
                i += 2;
            }
            "--spi-us" => {
                o.spi_us = val().parse().expect("bad --spi-us");
                i += 2;
            }
            "--seed" => {
                o.seed = val().parse().expect("bad --seed");
                i += 2;
            }
            "--fast-threshold" => {
                o.fast_threshold = val().parse().expect("bad --fast-threshold");
                i += 2;
            }
            "--map-scale" => {
                o.map_scale = val().parse().expect("bad --map-scale");
                i += 2;
            }
            "--gravity" => {
                let v: Vec<f32> = val().split(',').map(|x| x.parse().expect("bad --gravity")).collect();
                o.gravity = [v[0], v[1], v[2]];
                i += 2;
            }
            "--est-gravity" => {
                o.est_gravity = true;
                i += 1;
            }
            "--r-pos" => {
                o.r_pos = val().parse().expect("bad --r-pos");
                i += 2;
            }
            "--r-att" => {
                o.r_att = val().parse().expect("bad --r-att");
                i += 2;
            }
            "--r-vel" => {
                o.r_vel = val().parse().expect("bad --r-vel");
                i += 2;
            }
            "--min-inliers" => {
                o.min_inliers = val().parse().expect("bad --min-inliers");
                i += 2;
            }
            "--max-vo-speed" => {
                o.max_vo_speed = val().parse().expect("bad --max-vo-speed");
                i += 2;
            }
            "--max-att-resid-deg" => {
                o.max_att_resid_deg = val().parse().expect("bad --max-att-resid-deg");
                i += 2;
            }
            "--no-attitude" => {
                o.fuse_attitude = false;
                i += 1;
            }
            "--reopen-ba-at" => {
                o.reopen_ba_at = val().parse().expect("bad --reopen-ba-at");
                i += 2;
            }
            "--reopen-ba-std" => {
                o.reopen_ba_std = val().parse().expect("bad --reopen-ba-std");
                i += 2;
            }
            "--ba-floor-std" => {
                o.ba_floor_std = val().parse().expect("bad --ba-floor-std");
                i += 2;
            }
            "--res-fb-gain" => {
                o.res_fb_gain = val().parse().expect("bad --res-fb-gain");
                i += 2;
            }
            "--res-fb-alpha" => {
                o.res_fb_alpha = val().parse().expect("bad --res-fb-alpha");
                i += 2;
            }
            "--matcher" => {
                o.matcher = match val().as_str() {
                    "brute" => MatchMode::Brute,
                    "windowed" => MatchMode::Windowed,
                    other => {
                        eprintln!("bad --matcher {other}");
                        exit(2);
                    }
                };
                i += 2;
            }
            "--prior" => {
                o.prior = match val().as_str() {
                    "ekf" => PriorMode::Ekf,
                    "gt" => PriorMode::Gt,
                    other => {
                        eprintln!("bad --prior {other}");
                        exit(2);
                    }
                };
                i += 2;
            }
            "--topk" => {
                o.topk = val().parse().expect("bad --topk");
                i += 2;
            }
            "--window-px" => {
                o.window_px = val().parse().expect("bad --window-px");
                i += 2;
            }
            "--max-kf-angle" => {
                o.max_kf_angle = val().parse().expect("bad --max-kf-angle");
                i += 2;
            }
            "--prior-noise-pos" => {
                o.prior_noise_pos = val().parse().expect("bad --prior-noise-pos");
                i += 2;
            }
            "--prior-noise-att" => {
                o.prior_noise_att_deg = val().parse().expect("bad --prior-noise-att");
                i += 2;
            }
            "--map-sim3" => {
                o.map_sim3 = Some(val());
                i += 2;
            }
            "--dedup-px" => {
                o.dedup_px = val().parse().expect("bad --dedup-px");
                i += 2;
            }
            "--bucket-px" => {
                o.bucket_px = val().parse().expect("bad --bucket-px");
                i += 2;
            }
            "--bucket-k" => {
                o.bucket_k = val().parse().expect("bad --bucket-k");
                i += 2;
            }
            "--bucket-min" => {
                o.bucket_min = val().parse().expect("bad --bucket-min");
                i += 2;
            }
            "--help" | "-h" => {
                eprintln!("usage: bench_vi_offline --map m.txt --replay dir --out traj.csv \\
                    [--vo-latency-us N] [--spi-us N] [--seed N] [--fast-threshold N] [--map-scale S]");
                exit(0);
            }
            other => {
                eprintln!("unknown arg {other}");
                exit(2);
            }
        }
    }
    if o.map.is_empty() || o.replay.is_empty() {
        eprintln!("--map and --replay are required");
        exit(2);
    }
    o
}

// ------------------------------------------------------------ map.txt I/O ----

struct LocalMap {
    params: [f32; 4], // SIMPLE_RADIAL f, cx, cy, k1
    frames: Vec<MapFrame>,
}

struct MapFrame {
    embedding: [u8; EMBEDDING_DIM],
    /// Camera center in the map frame (map.txt `# POSE`).
    pos: [f32; 3],
    /// Camera orientation, world->camera, [w,x,y,z] (map.txt `# POSE`).
    quat: [f32; 4],
    has_pose: bool,
    points: Vec<MapPoint>,
}

impl localize::PosedFrame for MapFrame {
    fn has_pose(&self) -> bool {
        self.has_pose
    }
    fn pos(&self) -> [f32; 3] {
        self.pos
    }
    fn quat(&self) -> [f32; 4] {
        self.quat
    }
    fn points(&self) -> &[MapPoint] {
        &self.points
    }
}

fn hex_val(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        b'A'..=b'F' => c - b'A' + 10,
        _ => panic!("bad hex char"),
    }
}

fn decode_hex(s: &[u8], out: &mut [u8]) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = (hex_val(s[2 * i]) << 4) | hex_val(s[2 * i + 1]);
    }
}

fn parse_map_txt(txt: &str) -> LocalMap {
    let mut params: Option<[f32; 4]> = None;
    let mut frames: Vec<MapFrame> = Vec::new();
    let mut total = 0usize;
    for line in txt.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.is_empty() {
            continue;
        }
        // `# POSE <stem> qx qy qz qw cx cy cz` (world->camera q + center C).
        // A comment so the on-device parser ignores it.
        if t[0] == "#" && t.len() == 10 && t[1] == "POSE" {
            let q = [
                t[6].parse().unwrap(), // qw
                t[3].parse().unwrap(), // qx
                t[4].parse().unwrap(), // qy
                t[5].parse().unwrap(), // qz
            ];
            let f = frames.last_mut().expect("POSE before FRAME");
            f.pos = [t[7].parse().unwrap(), t[8].parse().unwrap(), t[9].parse().unwrap()];
            f.quat = q;
            f.has_pose = true;
            continue;
        }
        if t[0].starts_with('#') {
            continue;
        }
        match t[0] {
            "CAMERA" => {
                let mut p = [0f32; 4];
                for (i, v) in p.iter_mut().enumerate() {
                    *v = t[2 + i].parse().expect("bad camera param");
                }
                params = Some(p);
            }
            "FRAME" => {
                assert!(frames.len() < MAX_MAP_FRAMES, "too many map frames");
                let mut emb = [0u8; EMBEDDING_DIM];
                decode_hex(t[2].as_bytes(), &mut emb);
                frames.push(MapFrame {
                    embedding: emb,
                    pos: [0.0; 3],
                    quat: [1.0, 0.0, 0.0, 0.0],
                    has_pose: false,
                    points: Vec::new(),
                });
            }
            "POINT" => {
                total += 1;
                assert!(total <= MAX_MAP_POINTS, "too many map points");
                let cur = frames.last_mut().expect("POINT before FRAME");
                let xyz = [
                    t[1].parse().unwrap(),
                    t[2].parse().unwrap(),
                    t[3].parse().unwrap(),
                ];
                let mut raw = [0u8; 32];
                decode_hex(t[4].as_bytes(), &mut raw);
                let mut desc = [0u32; 8];
                for (w, d) in desc.iter_mut().enumerate() {
                    *d = u32::from_le_bytes(raw[w * 4..w * 4 + 4].try_into().unwrap());
                }
                cur.points.push(MapPoint { xyz, desc });
            }
            _ => panic!("unknown map record {}", t[0]),
        }
    }
    LocalMap { params: params.expect("no CAMERA line"), frames }
}

// ------------------------------------------------ pose prior + keyframes ----

// Quaternion conventions in this file are `[w, x, y, z]`.
fn q_norm(q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n > 0.0 {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

fn q_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn q_conj(q: [f32; 4]) -> [f32; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

fn q_to_mat(q: [f32; 4]) -> [[f32; 3]; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

/// Camera pose prior in the (scaled) map frame; `quat` is camera->world.
#[derive(Clone, Copy)]
struct Prior {
    pos: [f32; 3],
    quat: [f32; 4],
}

// ---- GT prior (map frame) ----

/// Similarity mapping map units -> GT meters: `p_gt = s * R * p_map + t`.
#[derive(Clone, Copy)]
struct Sim3 {
    s: f32,
    r: [[f32; 3]; 3],
    t: [f32; 3],
}

fn load_sim3(path: &str) -> Sim3 {
    let txt = fs::read_to_string(path).expect("read map_sim3.txt");
    let v: Vec<f32> = txt.split_whitespace().map(|x| x.parse().unwrap()).collect();
    assert_eq!(v.len(), 13, "map_sim3.txt must hold 13 numbers");
    Sim3 {
        s: v[0],
        r: [[v[1], v[2], v[3]], [v[4], v[5], v[6]], [v[7], v[8], v[9]]],
        t: [v[10], v[11], v[12]],
    }
}

/// One GT pose sample: `[w,x,y,z]` camera->world in the EuRoC GT frame.
struct GtSample {
    t_us: u64,
    p: [f32; 3],
    q: [f32; 4],
}

fn load_gt(path: &str) -> Vec<GtSample> {
    let txt = fs::read_to_string(path).expect("read gt.csv");
    let mut out = Vec::new();
    for line in txt.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // t_us must stay integer: f32 cannot hold ~1.4e15 us exactly.
        let mut it = line.split_whitespace();
        let t_us: u64 = it.next().unwrap().parse().unwrap();
        let v: Vec<f32> = it.map(|x| x.parse().unwrap()).collect();
        // t_us px py pz qw qx qy qz
        out.push(GtSample {
            t_us,
            p: [v[0], v[1], v[2]],
            q: [v[3], v[4], v[5], v[6]],
        });
    }
    out
}

fn nearest_gt(gt: &[GtSample], t_us: u64) -> &GtSample {
    let pos = gt.partition_point(|s| s.t_us < t_us);
    if pos == 0 {
        return &gt[0];
    }
    if pos >= gt.len() {
        return &gt[gt.len() - 1];
    }
    if t_us - gt[pos - 1].t_us <= gt[pos].t_us - t_us {
        &gt[pos - 1]
    } else {
        &gt[pos]
    }
}

/// Standard normal via Box-Muller (host std, so ln/cos are available).
fn gauss(rng: &mut ransac::Xorshift64) -> f32 {
    let u1 = (rng.next_u32() as f32 + 1.0) / (u32::MAX as f32 + 1.0);
    let u2 = (rng.next_u32() as f32) / (u32::MAX as f32 + 1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// GT pose at `t_us` expressed in the (scaled) map frame, optionally perturbed.
/// `gt.csv` carries the IMU/body state, so the camera<-body extrinsic
/// (`imu_rot`, `t_cam_body`) is applied before the map-frame transform.
#[allow(clippy::too_many_arguments)]
fn gt_prior(
    gt: &[GtSample],
    sim3: &Sim3,
    map_scale: f32,
    t_us: u64,
    imu_rot: &[[f32; 3]; 3],
    t_cam_body: [f32; 3],
    noise_pos: f32,
    noise_att_deg: f32,
    rng: &mut ransac::Xorshift64,
) -> Prior {
    let g = nearest_gt(gt, t_us);
    // Body->world, camera center in world, camera->world.
    let r_bw = q_to_mat(g.q);
    let pc = mat3_vec(&r_bw, t_cam_body);
    let p_gt = [g.p[0] + pc[0], g.p[1] + pc[1], g.p[2] + pc[2]];
    let r_c2w_gt = mat_mul3(&r_bw, &transpose3(imu_rot));
    // p_gt = s R p_units + t and points were scaled to `map_scale`:
    // p = (map_scale / s) R^T (p_gt - t).
    let k = map_scale / sim3.s;
    let d = [p_gt[0] - sim3.t[0], p_gt[1] - sim3.t[1], p_gt[2] - sim3.t[2]];
    let rt = transpose3(&sim3.r);
    let p = mat3_vec(&rt, d);
    let mut pos = [p[0] * k, p[1] * k, p[2] * k];
    // camera->world in the map frame = R^T * camera->world in GT frame.
    let r_c2w = mat_mul3(&rt, &r_c2w_gt);
    let mut quat = ekf::quat_from_mat(r_c2w);
    if noise_pos > 0.0 {
        for v in &mut pos {
            *v += noise_pos * gauss(rng);
        }
    }
    if noise_att_deg > 0.0 {
        let s = noise_att_deg * std::f32::consts::PI / 180.0;
        let dphi = [s * gauss(rng), s * gauss(rng), s * gauss(rng)];
        quat = q_norm(q_mul(quat, ekf::quat_from_rotvec(dphi)));
    }
    Prior { pos, quat }
}

// --------------------------------------------------------- stream parsing ----

fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn u64le(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}
fn f32le(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

/// One IMU1 record: `n | "IMU1" | u32 seq | u64 t0 | u16 nsamp | u16 dt_us | samples`.
fn parse_imu(rec: &[u8]) -> Vec<ImuSample> {
    let t0 = u64le(rec, 8);
    let nsamp = u16::from_le_bytes(rec[16..18].try_into().unwrap()) as usize;
    let dt = u16::from_le_bytes(rec[18..20].try_into().unwrap()) as u64;
    let mut out = Vec::with_capacity(nsamp);
    for i in 0..nsamp {
        let o = 20 + 24 * i;
        out.push((
            t0 + i as u64 * dt,
            [f32le(rec, o), f32le(rec, o + 4), f32le(rec, o + 8)],
            [f32le(rec, o + 12), f32le(rec, o + 16), f32le(rec, o + 20)],
        ));
    }
    out
}

// ---------------------------------------------------------------- EKF loop ----

struct Snap {
    ekf: ekf::Ekf,
    time_us: Option<u64>,
    traj_len: usize,
}

struct Pending {
    t_img: u64,
    t_fix: u64,
    pnp: Option<ransac::PnpResult>,
}

struct Harness {
    ekf: ekf::Ekf,
    time_us: Option<u64>,
    snap: Option<Snap>,
    imu_buf: Vec<ImuSample>,
    traj: Vec<(u64, [f32; 3], [f32; 4])>,
    gravity_estimated: bool,
    est_gravity: bool,
    r_pos: f32,
    r_att: f32,
    r_vel: f32,
    min_inliers: usize,
    max_vo_speed: f32,
    max_att_resid_deg: f32,
    fuse_attitude: bool, // false = position-only fixes (+ one-time q init)
    reopen_ba_at: usize,
    reopen_ba_std: f32,
    ba_floor_std: f32, // after every accept, P_ba_ii = max(P_ba_ii, floor^2).
                      // Stops the bias covariance collapsing onto a wrong value
                      // (0 = disabled). Matters at sparse fix rates: a 0.45 bias
                      // costs 0.5*0.45*dt^2 per interval (~5.6m at 5s, 0.2m at 1s).
    res_fb_gain: f32,  // residual-feedback gain (0 = off): pre-correct each
    res_fb_alpha: f32, // attitude fix by gain * low-passed past residuals
    res_lp: [f32; 3],  // (body-frame rotvec). Online, no GT: learns the slow
                        // map-twist half of fix errors (+0.48 lag-1 autocorr).
    prev_fix: Option<(u64, [f32; 3])>, // (t_img, camera center) of last good fix
    n_accept: usize,
    initialized: bool, // false until the first accepted fix: IMU is buffered
                       // but not fused (identity-q0 propagation is pure leak)
}

fn flush_cov(
    e: &mut ekf::Ekf,
    acc_a: &mut [f32; 3],
    acc_w: &mut [f32; 3],
    n: &mut usize,
    dt_sum: &mut f32,
) {
    if *n == 0 {
        return;
    }
    let inv = 1.0 / *n as f32;
    for k in 0..3 {
        acc_a[k] *= inv;
        acc_w[k] *= inv;
    }
    e.propagate_covariance(*acc_a, *acc_w, e.body_to_world(), *dt_sum);
    *acc_a = [0.0; 3];
    *acc_w = [0.0; 3];
    *n = 0;
    *dt_sum = 0.0;
}

impl Harness {
    fn new(est_gravity: bool) -> Self {
        Harness {
            ekf: ekf::Ekf::new(
                [0.0; 3],
                [1.0, 0.0, 0.0, 0.0],
                ekf::EkfNoise::default(),
                [10.0, 1.0, 0.5, 0.1, 0.01],
            ),
            time_us: None,
            snap: None,
            imu_buf: Vec::new(),
            traj: Vec::new(),
            gravity_estimated: false,
            est_gravity,
            r_pos: R_POS_VAR,
            r_att: R_ATT_VAR,
            r_vel: R_VEL_VAR,
            min_inliers: 0,
            max_vo_speed: 0.0,
            max_att_resid_deg: 0.0,
            fuse_attitude: true,
            prev_fix: None,
            n_accept: 0,
            initialized: false,
            reopen_ba_at: 0,
            reopen_ba_std: 0.5,
            ba_floor_std: 0.0,
            res_fb_gain: 0.0,
            res_fb_alpha: 0.5,
            res_lp: [0.0; 3],
        }
    }

    /// Split-rate fuse, byte-for-byte the logic in vo_replay.rs
    /// (`fuse_imu_samples` + `flush_cov`); `emit` appends the pose stream.
    fn fuse(&mut self, samples: &[ImuSample], emit: bool) {
        if !self.initialized {
            // Hold: advance the clock so post-init dt is sane, buffer (done
            // by the caller) but fuse nothing — no attitude to project with.
            if let Some(&(t, _, _)) = samples.last() {
                self.time_us = Some(t);
            }
            return;
        }
        let mut acc_a = [0.0; 3];
        let mut acc_w = [0.0; 3];
        let (mut n, mut dt_sum) = (0usize, 0.0f32);
        for &(t, a, w) in samples {
            let dt = match self.time_us {
                Some(last) if t > last && t - last <= GAP_RESYNC_US => (t - last) as f32 / 1e6,
                Some(last) => {
                    flush_cov(&mut self.ekf, &mut acc_a, &mut acc_w, &mut n, &mut dt_sum);
                    eprintln!("ekf: IMU gap, re-anchoring clock (t={t} last={last})");
                    self.time_us = Some(t);
                    continue;
                }
                None => {
                    self.time_us = Some(t);
                    continue;
                }
            };
            if let Some((ac, wc)) = self.ekf.propagate_nominal(a, w, dt) {
                for k in 0..3 {
                    acc_a[k] += ac[k];
                    acc_w[k] += wc[k];
                }
                n += 1;
                dt_sum += dt;
                if emit {
                    self.traj.push((t, self.ekf.p, self.ekf.q));
                }
                if n == COV_EVERY {
                    flush_cov(&mut self.ekf, &mut acc_a, &mut acc_w, &mut n, &mut dt_sum);
                }
            }
            self.time_us = Some(t);
        }
        flush_cov(&mut self.ekf, &mut acc_a, &mut acc_w, &mut n, &mut dt_sum);
    }

    /// Rewind to the snapshot, fuse the fix, replay the buffered IMU.
    fn apply_fix(&mut self, p: &Pending) {
        let Some(snap) = self.snap.take() else {
            eprintln!("VO_DROP fix with no snapshot (stale)");
            return;
        };
        self.ekf = snap.ekf;
        self.time_us = snap.time_us;
        self.traj.truncate(snap.traj_len);
        if let Some(pnp) = p.pnp {
            let c = camera_center(&pnp.r, &pnp.t);
            let mut qm = ekf::quat_from_mat(transpose3(&pnp.r));
            // Residual feedback (disturbance observer): pre-correct qm by
            // gain * low-passed past body-frame residuals, learning the slow
            // map-twist half of fix errors online (no GT). Then LP-update
            // with the residual the filter is about to see.
            let mut res_now = [0.0; 3];
            // Armed after 3 accepts: early residuals are init transient
            // (identity start), not map twist — learning them poisons the LP.
            if self.res_fb_gain > 0.0 && self.fuse_attitude && self.n_accept >= 3 {
                qm = ekf::quat_correct_body(qm, [
                    -self.res_fb_gain * self.res_lp[0],
                    -self.res_fb_gain * self.res_lp[1],
                    -self.res_fb_gain * self.res_lp[2],
                ]);
                res_now = ekf::attitude_residual(self.ekf.q, qm);
            }
            // Gates (all logged; rejected fixes still replay IMU but change
            // nothing, and don't advance prev_fix). Attitude gate arms only
            // after 3 accepts so init (identity q0) can't lock itself out.
            let mut reject: Option<String> = None;
            if pnp.inlier_count < self.min_inliers {
                reject = Some(format!("inliers {}<{}", pnp.inlier_count, self.min_inliers));
            }
            if reject.is_none() && self.max_vo_speed > 0.0 {
                if let Some((t_prev, p_prev)) = self.prev_fix {
                    let dt = (p.t_img.saturating_sub(t_prev)) as f32 / 1e6;
                    if dt > 0.0 {
                        let d = ((c[0]-p_prev[0]).powi(2) + (c[1]-p_prev[1]).powi(2) + (c[2]-p_prev[2]).powi(2)).sqrt();
                        if d / dt > self.max_vo_speed {
                            reject = Some(format!("speed {:.2}>{:.2} m/s", d / dt, self.max_vo_speed));
                        }
                    }
                }
            }
            if reject.is_none() && self.max_att_resid_deg > 0.0 && self.n_accept >= 3 {
                let dot = (self.ekf.q[0]*qm[0] + self.ekf.q[1]*qm[1] + self.ekf.q[2]*qm[2] + self.ekf.q[3]*qm[3]).abs().min(1.0);
                // No trig (no libm): reject if dot < cos(th/2), small-angle
                // cos(x)~1-x^2/2 with x = th/2.
                let th = self.max_att_resid_deg * 3.14159 / 180.0;
                if dot < 1.0 - th * th / 8.0 {
                    reject = Some(format!("att>{:.0}deg", self.max_att_resid_deg));
                }
            }
            if let Some(reason) = reject {
                eprintln!("VO_REJECT t_img={} inl={} ({})", p.t_img, pnp.inlier_count, reason);
            } else {
            if self.fuse_attitude {
            self.ekf.correct_pose(
                c,
                qm,
                self.r_pos,
                self.r_att,
            );
            } else {
                if self.n_accept == 0 {
                    self.ekf.q = qm; // one-time attitude init, never fused after
                }
                self.ekf.correct_position(c, self.r_pos);
            }
            // Pseudo velocity measurement: finite difference of consecutive
            // VO camera centers. Skipped on the first fix / failed fixes /
            // absurd dt; r_vel<=0 disables.
            if self.r_vel > 0.0 {
                if let Some((t_prev, p_prev)) = self.prev_fix {
                    let dt = (p.t_img.saturating_sub(t_prev)) as f32 / 1e6;
                    if dt >= 0.5 && dt <= 15.0 {
                        let v_meas = [
                            (c[0] - p_prev[0]) / dt,
                            (c[1] - p_prev[1]) / dt,
                            (c[2] - p_prev[2]) / dt,
                        ];
                        let rv = self.ekf.correct_velocity(v_meas, self.r_vel);
                        eprintln!("  vel pseudo dt={dt:.1}s meas=({:.2},{:.2},{:.2}) res={:?}",
                                  v_meas[0], v_meas[1], v_meas[2], rv);
                    }
                }
            }
            self.prev_fix = Some((p.t_img, c));
            self.n_accept += 1;
            self.initialized = true;
            if self.res_fb_gain > 0.0 && self.fuse_attitude && self.n_accept >= 3 {
                for k in 0..3 {
                    self.res_lp[k] += self.res_fb_alpha * (res_now[k] - self.res_lp[k]);
                }
            }
            if self.reopen_ba_at > 0 && self.n_accept == self.reopen_ba_at {
                for i in 0..3 {
                    self.ekf.p_cov[9 + i][9 + i] += self.reopen_ba_std * self.reopen_ba_std;
                }
                eprintln!("  P_ba reopened (+{:.2}^2); ba={:?}", self.reopen_ba_std, self.ekf.ba);
            }
            if self.ba_floor_std > 0.0 {
                for i in 0..3 {
                    let f = self.ba_floor_std * self.ba_floor_std;
                    if self.ekf.p_cov[9 + i][9 + i] < f {
                        self.ekf.p_cov[9 + i][9 + i] = f;
                    }
                }
            }
            }
        }
        // First fix: estimate world gravity from the mean specific force in the
        // buffer + the just-corrected attitude. The COLMAP map frame's
        // orientation is only as good as the SfM gauge, so anchor g to the IMU.
        if !self.gravity_estimated && !self.imu_buf.is_empty() && self.est_gravity {
            let n = self.imu_buf.len() as f32;
            let mut ma = [0.0f32; 3];
            for s in &self.imu_buf {
                for k in 0..3 {
                    ma[k] += s.1[k];
                }
            }
            let rw = self.ekf.body_to_world();
            let gc = [-ma[0] / n, -ma[1] / n, -ma[2] / n];
            self.ekf.g = mat3_vec(&rw, gc);
            self.gravity_estimated = true;
            eprintln!("  estimated g_map from IMU: ({:.3},{:.3},{:.3}) |g|={:.3}",
                      self.ekf.g[0], self.ekf.g[1], self.ekf.g[2],
                      (self.ekf.g[0] * self.ekf.g[0] + self.ekf.g[1] * self.ekf.g[1]
                       + self.ekf.g[2] * self.ekf.g[2]).sqrt());
        }
        let buf = std::mem::take(&mut self.imu_buf);
        self.fuse(&buf, true);
        let c = p.pnp.map(|r| camera_center(&r.r, &r.t));
        let qlog = p.pnp.map(|r| ekf::quat_from_mat(transpose3(&r.r)));
        eprintln!(
            "VO_FIX t_img={} traj={} inliers={} meas_c={:?} meas_q={:?} ekf_p=({:.2},{:.2},{:.2}) v=({:.2},{:.2},{:.2})",
            p.t_img, self.traj.len(), p.pnp.map_or(0, |r| r.inlier_count),
            c.map(|c| [c[0] as f64, c[1] as f64, c[2] as f64]),
            qlog.map(|q| [q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64]),
            self.ekf.p[0], self.ekf.p[1], self.ekf.p[2],
            self.ekf.v[0], self.ekf.v[1], self.ekf.v[2]
        );
        eprintln!("  ba={:?} bg={:?}", self.ekf.ba, self.ekf.bg);
    }
}

fn transpose3(m: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]]
}

fn mat_mul3(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut r = [[0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}

fn mat3_vec(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn camera_center(r: &[[f32; 3]; 3], t: &[f32; 3]) -> [f32; 3] {    [
        -(r[0][0] * t[0] + r[1][0] * t[1] + r[2][0] * t[2]),
        -(r[0][1] * t[0] + r[1][1] * t[1] + r[2][1] * t[2]),
        -(r[0][2] * t[0] + r[1][2] * t[1] + r[2][2] * t[2]),
    ]
}

fn cosine(a: &[u8], b: &[u8]) -> f32 {
    let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f32, *y as f32);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let d = na.sqrt() * nb.sqrt();
    if d <= f32::EPSILON { 0.0 } else { dot / d }
}

struct VoScratch {
    arena: Vec<u8>,
    work: Vec<u8>,
    vcol: Vec<u16>,
    corners: Vec<fast::Corner>,
    scores: Vec<i32>,
    rowidx: Vec<usize>,
    nms: Vec<fast::Corner>,
    cells: Vec<u32>,
    cand: Vec<pyramid::Candidate>,
    feats: Vec<pyramid::Feature>,
    frame: Vec<u8>,
    best_idx: Vec<u32>,
    best_dist: Vec<u32>,
    second_dist: Vec<u32>,
    point_query: Vec<u32>,
    point_dist: Vec<u32>,
    matches: Vec<matcher::Match>,
    corrs: Vec<ransac::Correspondence>,
    mask: Vec<bool>,
    cands: Vec<localize::Candidate>,
    cand_query: Vec<u32>,
    cand_dist: Vec<u32>,
}

impl VoScratch {
    fn new() -> Self {
        let nf = pyramid::MAX_FEATURES;
        VoScratch {
            arena: vec![0u8; pyramid::arena_bytes(CAM_W, CAM_H)],
            work: vec![0u8; CAM_W * CAM_H],
            vcol: vec![0u16; CAM_W],
            corners: vec![fast::Corner { x: 0, y: 0 }; pyramid::CORNERS_RAW_MAX],
            scores: vec![0i32; pyramid::CORNERS_RAW_MAX],
            rowidx: vec![usize::MAX; CAM_H],
            nms: vec![fast::Corner { x: 0, y: 0 }; pyramid::CORNERS_RAW_MAX],
            cells: vec![0u32; pyramid::bucket_cells(CAM_W, CAM_H) * pyramid::BUCKET_K],
            cand: vec![pyramid::Candidate::default(); pyramid::CAND_MAX],
            feats: vec![pyramid::Feature::default(); nf],
            frame: vec![0u8; CAM_W * CAM_H],
            best_idx: vec![0u32; nf],
            best_dist: vec![0u32; nf],
            second_dist: vec![0u32; nf],
            point_query: vec![0u32; MAX_MAP_POINTS],
            point_dist: vec![0u32; MAX_MAP_POINTS],
            matches: vec![matcher::Match::default(); nf],
            corrs: vec![ransac::Correspondence { world: [0.0; 3], xn: 0.0, yn: 0.0 }; nf],
            mask: vec![false; nf],
            cands: vec![localize::Candidate::default(); MAX_PRIOR_CANDIDATES],
            cand_query: vec![0u32; MAX_PRIOR_CANDIDATES],
            cand_dist: vec![u32::MAX; MAX_PRIOR_CANDIDATES],
        }
    }
}

fn now_us_dummy() -> u64 {
    0
}

/// One localization attempt's diagnostics for the log line.
struct VoOut {
    nfeat: usize,
    nfeat_raw: usize,
    matches: usize,
    cos1: f32,
    cos2: f32,
    pnp: Option<ransac::PnpResult>,
    best: usize,
    reproj: f32,
    kf_ids: Vec<usize>,
    survivors: usize,
    infrustum: usize,
    winmatches: usize,
    bootstrap: bool,
}

impl VoOut {
    fn failed(nfeat: usize) -> Self {
        VoOut {
            nfeat,
            nfeat_raw: nfeat,
            matches: 0,
            cos1: 0.0,
            cos2: 0.0,
            pnp: None,
            best: 0,
            reproj: -1.0,
            kf_ids: Vec::new(),
            survivors: 0,
            infrustum: 0,
            winmatches: 0,
            bootstrap: false,
        }
    }
}

/// Top-1 map frame by embedding cosine (also returns cos1/cos2 for logging).
fn top1_cosine(emb: &[u8; EMBEDDING_DIM], map: &LocalMap) -> (usize, f32, f32) {
    let mut best = 0usize;
    let (mut c1, mut c2) = (f32::MIN, f32::MIN);
    for (i, f) in map.frames.iter().enumerate() {
        let c = cosine(emb, &f.embedding);
        if c > c1 {
            c2 = c1;
            c1 = c;
            best = i;
        } else if c > c2 {
            c2 = c;
        }
    }
    (best, c1, c2.max(0.0))
}

/// Feature-count reduction config (BENCH_VI offline experiment only; the S3
/// would apply the same rules pre-describe to save rBRIEF time — here we
/// filter post-extract, which yields the identical feature set since
/// descriptors don't depend on which other features survive).
#[derive(Clone, Copy)]
struct FilterCfg {
    dedup_px: f32,  // cross-level dedup radius, L0 px (0 = off)
    bucket_px: f32, // per-cell top-K cap cell size, L0 px (0 = off)
    bucket_k: usize,
    bucket_min: usize, // skip bucketing when deduped count is below this
}

/// Level-`l` image slice for re-scoring: L0 is the frame, L1+ are the arena
/// regions in forward order (mirrors `extract_pyramid`'s layout).
fn level_img<'a>(
    frame: &'a [u8],
    arena: &'a [u8],
    l: usize,
) -> (&'a [u8], usize, usize) {
    if l == 0 {
        return (frame, CAM_W, CAM_H);
    }
    let mut off = 0usize;
    for k in 1..l {
        let (cw, ch) = pyramid::level_dims(CAM_W, CAM_H, k);
        off += cw * ch;
    }
    let (cw, ch) = pyramid::level_dims(CAM_W, CAM_H, l);
    (&arena[off..off + cw * ch], cw, ch)
}

/// Exact FAST score of one extracted feature: invert the L0 projection to
/// level-local coords and run the same `fast12_score` the extractor used.
fn feature_score(
    f: &pyramid::Feature,
    frame: &[u8],
    arena: &[u8],
    thr: i32,
    tmp: &mut [i32; 1],
) -> i32 {
    let l = f.level as usize;
    let (img, cw, ch) = level_img(frame, arena, l);
    let s = pyramid::SCALE.powi(l as i32);
    let lx = (f.x / s).round() as isize;
    let ly = (f.y / s).round() as isize;
    if lx < 3 || ly < 3 || lx + 3 >= cw as isize || ly + 3 >= ch as isize {
        return thr - 1; // cannot happen for real features; drop-first
    }
    let c = [fast::Corner { x: lx as usize, y: ly as usize }];
    if fast::fast12_score(img, cw, &c, thr, tmp) == 1 {
        tmp[0]
    } else {
        thr - 1
    }
}

/// Dedup (finest-level wins: `feats` is level-ordered) + per-cell top-K by
/// score, in place, order-preserving. Returns (kept, deduped) counts.
fn reduce_features(
    feats: &mut [pyramid::Feature],
    frame: &[u8],
    arena: &[u8],
    thr: i32,
    cfg: FilterCfg,
) -> (usize, usize) {
    let n = feats.len();
    let mut keep = vec![true; n];
    if cfg.dedup_px > 0.0 {
        let r2 = cfg.dedup_px * cfg.dedup_px;
        for i in 0..n {
            if !keep[i] {
                continue;
            }
            let (xi, yi) = (feats[i].x, feats[i].y);
            for j in (i + 1)..n {
                if !keep[j] {
                    continue;
                }
                let dx = feats[j].x - xi;
                let dy = feats[j].y - yi;
                if dx * dx + dy * dy < r2 {
                    keep[j] = false;
                }
            }
        }
    }
    let n1 = keep.iter().filter(|&&k| k).count();
    if cfg.bucket_px > 0.0 && cfg.bucket_k > 0 && n1 >= cfg.bucket_min {
        let b = cfg.bucket_px;
        let nx = ((CAM_W as f32 / b).ceil() as usize).max(1);
        let ny = ((CAM_H as f32 / b).ceil() as usize).max(1);
        let mut counts = vec![0usize; nx * ny];
        let mut order: Vec<usize> = (0..n).filter(|&i| keep[i]).collect();
        let mut scores = vec![0i32; n];
        let mut tmp = [0i32; 1];
        for &i in &order {
            scores[i] = feature_score(&feats[i], frame, arena, thr, &mut tmp);
        }
        order.sort_by(|&a, &c| {
            scores[c].cmp(&scores[a]).then_with(|| feats[a].level.cmp(&feats[c].level))
        });
        let mut keep2 = vec![false; n];
        for i in order {
            let gx = ((feats[i].x / b) as usize).min(nx - 1);
            let gy = ((feats[i].y / b) as usize).min(ny - 1);
            let cell = gy * nx + gx;
            if counts[cell] < cfg.bucket_k {
                counts[cell] += 1;
                keep2[i] = true;
            }
        }
        keep = keep2;
    }
    let mut w = 0usize;
    for r in 0..n {
        if keep[r] {
            feats[w] = feats[r];
            w += 1;
        }
    }
    (w, n1)
}

/// The S3 `vo_task` body: pyramid -> (brute: embedding top-1 | windowed:
/// pose-prior keyframes) -> match + PnP.
fn run_vo(
    gray: &[u8],
    emb: &[u8; EMBEDDING_DIM],
    map: &LocalMap,
    cam: &ransac::Camera,
    opts: &ransac::PnpOptions,
    rng: &mut ransac::Xorshift64,
    s: &mut VoScratch,
    thr: i32,
    mode: MatchMode,
    prior_in: Option<Prior>,
    topk: usize,
    max_kf_angle: f32,
    filt: FilterCfg,
) -> VoOut {
    s.frame.copy_from_slice(gray);
    let nraw = pyramid::extract_pyramid(
        &s.frame, CAM_W, CAM_H, thr, &mut s.arena, &mut s.work, &mut s.vcol,
        &mut s.corners, &mut s.scores, &mut s.rowidx, &mut s.nms, &mut s.cells,
        &mut s.cand, &mut s.feats, None,
    );
    // Offline feature-count experiment (identity when filters are off).
    let (nfeat, _) = reduce_features(
        &mut s.feats[..nraw],
        &s.frame,
        &s.arena,
        thr,
        filt,
    );
    if map.frames.is_empty() {
        return VoOut::failed(nfeat);
    }

    if mode == MatchMode::Brute {
        let (best, cos1, cos2) = top1_cosine(emb, map);
        let mut ls = localize::LocalizeScratch {
            mb: matcher::MatchBuffers {
                best_idx: &mut s.best_idx,
                best_dist: &mut s.best_dist,
                second_dist: &mut s.second_dist,
                point_query: &mut s.point_query,
                point_dist: &mut s.point_dist,
            },
            matches: &mut s.matches,
            corrs: &mut s.corrs,
            mask: &mut s.mask,
        };
        let stats = localize::localize_frame(
            &s.feats[..nfeat], &map.frames[best].points, cam, opts, rng, &mut ls,
            now_us_dummy,
        );
        return VoOut {
            nfeat,
            nfeat_raw: nraw,
            matches: stats.matches,
            cos1,
            cos2,
            pnp: stats.pnp,
            best,
            reproj: stats.pnp.map_or(-1.0, |r| r.mean_reproj_error_px),
            kf_ids: vec![best],
            survivors: 1,
            infrustum: 0,
            winmatches: stats.matches,
            bootstrap: false,
        };
    }

    // Hybrid (BENCH_VI): strong prior -> attitude/frustum keyframes + windowed
    // match; weak/absent prior -> embedding top-K brute.
    if let Some(prior) = prior_in {
        let p = localize::PosePrior { pos: prior.pos, quat: prior.quat };
        let mut ids = [0usize; localize::MAX_KEYFRAMES];
        let sel = localize::select_keyframes(
            &p, &map.frames, cam, CAM_W, CAM_H, max_kf_angle.to_radians(), topk, &mut ids,
        );
        if sel.n > 0 {
            let mut ws = localize::PriorScratch {
                cands: &mut s.cands,
                cand_query: &mut s.cand_query,
                cand_dist: &mut s.cand_dist,
                best_idx: &mut s.best_idx,
                best_dist: &mut s.best_dist,
                second_dist: &mut s.second_dist,
                corrs: &mut s.corrs,
                mask: &mut s.mask,
            };
            let stats = localize::localize_prior(
                &s.feats[..nfeat], &map.frames, &ids[..sel.n], &p, cam, CAM_W, CAM_H,
                opts, rng, &mut ws, now_us_dummy,
            );
            return VoOut {
                nfeat,
                nfeat_raw: nraw,
                matches: stats.matches,
                cos1: 0.0,
                cos2: 0.0,
                pnp: stats.pnp,
                best: ids[0],
                reproj: stats.pnp.map_or(-1.0, |r| r.mean_reproj_error_px),
                kf_ids: ids[..sel.n].to_vec(),
                survivors: sel.survivors,
                infrustum: sel.infrustum,
                winmatches: stats.matches,
                bootstrap: false,
            };
        }
    }
    // Embedding top-K brute fallback (plan branch 2): rank by cosine, match
    // each, keep the PnP with the most inliers.
    let mut ranked: Vec<(f32, usize)> =
        map.frames.iter().enumerate().map(|(i, f)| (cosine(emb, &f.embedding), i)).collect();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(topk.max(1));
    let cos1 = ranked.first().map_or(0.0, |x| x.0);
    let cos2 = ranked.get(1).map_or(0.0, |x| x.0);
    let mut best: Option<(localize::LocalizeStats, usize)> = None;
    for &(_, fi) in &ranked {
        let mut ls = localize::LocalizeScratch {
            mb: matcher::MatchBuffers {
                best_idx: &mut s.best_idx,
                best_dist: &mut s.best_dist,
                second_dist: &mut s.second_dist,
                point_query: &mut s.point_query,
                point_dist: &mut s.point_dist,
            },
            matches: &mut s.matches,
            corrs: &mut s.corrs,
            mask: &mut s.mask,
        };
        let st = localize::localize_frame(
            &s.feats[..nfeat], &map.frames[fi].points, cam, opts, rng, &mut ls, now_us_dummy,
        );
        let sc = st.pnp.map_or(st.pnp_best.inlier_count, |p| p.inlier_count);
        let better = best.as_ref().map_or(true, |(b, _)| {
            sc > b.pnp.map_or(b.pnp_best.inlier_count, |p| p.inlier_count)
        });
        if better {
            best = Some((st, fi));
        }
    }
    let (stats, fi) = best.expect("map has frames");
    VoOut {
        nfeat,
        nfeat_raw: nraw,
        matches: stats.matches,
        cos1,
        cos2,
        pnp: stats.pnp,
        best: fi,
        reproj: stats.pnp.map_or(-1.0, |r| r.mean_reproj_error_px),
        kf_ids: vec![fi],
        survivors: 0,
        infrustum: 0,
        winmatches: stats.matches,
        bootstrap: prior_in.is_none(),
    }
}

fn main() {
    let o = parse_args();
    let mut map = parse_map_txt(&fs::read_to_string(&o.map).expect("read map"));
    // Monocular COLMAP is scale-free; rescale the (metric-shaped) map to meters.
    if o.map_scale != 1.0 {
        for f in &mut map.frames {
            for p in &mut f.points {
                for v in &mut p.xyz {
                    *v *= o.map_scale;
                }
            }
            for v in &mut f.pos {
                *v *= o.map_scale;
            }
        }
    }
    let stream = fs::read(format!("{}/stream.bin", o.replay)).expect("read stream.bin");
    let emb_bytes = fs::read(format!("{}/embeddings.bin", o.replay)).expect("read embeddings.bin");
    // cam0 <- IMU body rotation (EuRoC T_BS); identity if absent. The EKF
    // state is the camera body frame, so IMU samples are rotated into it.
    let imu_vals: Vec<f32> = fs::read_to_string(format!("{}/imu_cam_rot.txt", o.replay))
        .ok()
        .map(|s| s.split_whitespace().map(|x| x.parse().unwrap()).collect())
        .unwrap_or_default();
    let imu_rot: [[f32; 3]; 3] = if imu_vals.len() >= 9 {
        [[imu_vals[0], imu_vals[1], imu_vals[2]],
         [imu_vals[3], imu_vals[4], imu_vals[5]],
         [imu_vals[6], imu_vals[7], imu_vals[8]]]
    } else {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    };
    // Camera center in the body frame; zero for 9-value (rotation-only) files.
    let t_cam_body: [f32; 3] = if imu_vals.len() >= 12 {
        [imu_vals[9], imu_vals[10], imu_vals[11]]
    } else {
        [0.0; 3]
    };
    eprintln!("imu cam<-body rot = {:?}", imu_rot);
    eprintln!("camera center in body = {:?}", t_cam_body);
    eprintln!(
        "map: {} frames / {} points; replay {} B, {} embeddings; vo_latency={}us spi={}us thr={} seed={:#x}",
        map.frames.len(),
        map.frames.iter().map(|f| f.points.len()).sum::<usize>(),
        stream.len(),
        emb_bytes.len() / EMBEDDING_DIM,
        o.vo_latency_us, o.spi_us, o.fast_threshold, o.seed,
    );

    // ---- pose-prior setup (windowed matcher) ----
    let use_gt = o.matcher == MatchMode::Windowed && o.prior == PriorMode::Gt;
    let gt_data: Vec<GtSample> = if use_gt {
        load_gt(&format!("{}/gt.csv", o.replay))
    } else {
        Vec::new()
    };
    let sim3 = if use_gt {
        Some(load_sim3(o.map_sim3.as_ref().expect(
            "--prior gt requires --map-sim3 (run bench_vi.py --phase scale)",
        )))
    } else {
        None
    };
    if o.matcher == MatchMode::Windowed {
        let missing = map.frames.iter().filter(|f| !f.has_pose).count();
        if missing > 0 {
            eprintln!(
                "ERROR: {missing}/{} map frames lack a `# POSE` line -- rebuild the map with the current write_map.py",
                map.frames.len()
            );
            exit(2);
        }
        eprintln!(
            "windowed: prior={:?} topk={} window={}px max_kf_angle={}deg noise_pos={} noise_att={}deg",
            o.prior, o.topk, o.window_px, o.max_kf_angle, o.prior_noise_pos,
            o.prior_noise_att_deg,
        );
    }
    let cam = ransac::Camera {
        fx: map.params[0],
        fy: map.params[0],
        cx: map.params[1],
        cy: map.params[2],
        k1: map.params[3],
    };
    let opts = ransac::PnpOptions::default();
    let mut rng = ransac::Xorshift64::new(o.seed);
    let mut scratch = VoScratch::new();
    let mut h = Harness::new(o.est_gravity);
    h.r_pos = o.r_pos;
    h.r_att = o.r_att;
    h.r_vel = o.r_vel;
    h.min_inliers = o.min_inliers;
    h.max_vo_speed = o.max_vo_speed;
    h.max_att_resid_deg = o.max_att_resid_deg;
    h.fuse_attitude = o.fuse_attitude;
    h.reopen_ba_at = o.reopen_ba_at;
    h.reopen_ba_std = o.reopen_ba_std;
    h.ba_floor_std = o.ba_floor_std;
    h.res_fb_gain = o.res_fb_gain;
    h.res_fb_alpha = o.res_fb_alpha;
    if !o.fuse_attitude {
        // Attitude gate assumes fused attitude tracks measurements; in
        // position-only mode that premise is intentionally violated.
        h.max_att_resid_deg = 0.0;
    }
    h.ekf.g = o.gravity;
    eprintln!("gravity (map frame) = {:?}", o.gravity);
    let mut pending: Option<Pending> = None;
    let (mut n_img, mut n_fix, mut n_fail) = (0usize, 0usize, 0usize);
    let mut emb_idx = 0usize;
    let mut pos = 0usize;
    let mut next_mark = 0u64;

    while pos + 4 <= stream.len() {
        let n = u32le(&stream, pos) as usize;
        let rec = &stream[pos + 4..pos + 4 + n];
        pos += 4 + n;
        let magic = &rec[0..4];
        if magic == b"IMU1" {
            let mut samples = parse_imu(rec);
            for s in &mut samples {
                s.1 = mat3_vec(&imu_rot, s.1);
                s.2 = mat3_vec(&imu_rot, s.2);
            }
            if let Some(p) = &pending {
                // The fix lands once the fused dataset clock passes t_fix;
                // whole batches, as on the S3.
                if samples.first().map_or(false, |s| p.t_fix <= s.0) {
                    h.apply_fix(p);
                    n_fix += usize::from(p.pnp.is_some());
                    n_fail += usize::from(p.pnp.is_none());
                    pending = None;
                }
            }
            h.fuse(&samples, true);
            if h.snap.is_some() {
                h.imu_buf.extend_from_slice(&samples);
            }
            if let Some(t) = samples.last().map(|s| s.0) {
                if t >= next_mark {
                    next_mark = t + 5_000_000;
                    if let Some(&(tt, p, q)) = h.traj.last() {
                        eprintln!("VO_IMU t={tt} p=({:.2},{:.2},{:.2}) q=({:.3},{:.3},{:.3},{:.3})",
                                  p[0], p[1], p[2], q[0], q[1], q[2], q[3]);
                    }
                }
            }
        } else if magic == b"TUM1" {
            if let Some(p) = &pending {
                h.apply_fix(p);
                n_fix += usize::from(p.pnp.is_some());
                n_fail += usize::from(p.pnp.is_none());
            }
            let t_img = u64le(rec, 8);
            let w = u16::from_le_bytes(rec[16..18].try_into().unwrap()) as usize;
            let h_ = u16::from_le_bytes(rec[18..20].try_into().unwrap()) as usize;
            assert_eq!((w, h_), (CAM_W, CAM_H), "bad TUM1 dims");
            let gray = &rec[20..20 + CAM_W * CAM_H];
            let mut emb = [0u8; EMBEDDING_DIM];
            emb.copy_from_slice(&emb_bytes[emb_idx * EMBEDDING_DIM..(emb_idx + 1) * EMBEDDING_DIM]);
            emb_idx += 1;
            n_img += 1;
            // snapshot at the image, then run VO (the fix is delayed below).
            let snap = Snap { ekf: h.ekf.clone(), time_us: h.time_us, traj_len: h.traj.len() };
            h.snap = Some(snap);
            h.imu_buf.clear();
            let prior = if o.matcher == MatchMode::Windowed {
                match o.prior {
                    PriorMode::Gt => Some(gt_prior(
                        &gt_data,
                        sim3.as_ref().unwrap(),
                        o.map_scale,
                        t_img,
                        &imu_rot,
                        t_cam_body,
                        o.prior_noise_pos,
                        o.prior_noise_att_deg,
                        &mut rng,
                    )),
                    PriorMode::Ekf => {
                        if h.initialized {
                            Some(Prior { pos: h.ekf.p, quat: h.ekf.q })
                        } else {
                            None // run_vo bootstraps from the embedding top-1
                        }
                    }
                }
            } else {
                None
            };
            let filt = FilterCfg {
                dedup_px: o.dedup_px,
                bucket_px: o.bucket_px,
                bucket_k: o.bucket_k,
                bucket_min: o.bucket_min,
            };
            let r = run_vo(
                gray, &emb, &map, &cam, &opts, &mut rng, &mut scratch,
                o.fast_threshold, o.matcher, prior, o.topk, o.max_kf_angle, filt,
            );
            eprintln!(
                "VO_IMG {n_img} t={t_img} feats={} raw={} matches={} cos1={:.3} cos2={:.3} best={} reproj={:.2} inliers={} kf={:?} surv={} infrustum={} win={} boot={}",
                r.nfeat, r.nfeat_raw, r.matches, r.cos1, r.cos2, r.best, r.reproj,
                r.pnp.map_or(0, |x| x.inlier_count), r.kf_ids, r.survivors,
                r.infrustum, r.winmatches, r.bootstrap,
            );
            pending = Some(Pending {
                t_img,
                t_fix: t_img + o.spi_us + o.vo_latency_us,
                pnp: r.pnp,
            });
        } else {
            eprintln!("unknown magic {:?}", magic);
        }
    }
    if let Some(p) = &pending {
        h.apply_fix(p);
        n_fix += usize::from(p.pnp.is_some());
        n_fail += usize::from(p.pnp.is_none());
    }

    // ---- write trajectory (TUM: t_us tx ty tz qx qy qz qw) ----
    let mut out = String::from("# t_us tx ty tz qx qy qz qw\n");
    for (t, p, q) in &h.traj {
        out.push_str(&format!(
            "{} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}\n",
            t, p[0], p[1], p[2], q[1], q[2], q[3], q[0]
        ));
    }
    fs::write(&o.out, out).expect("write trajectory");
    eprintln!(
        "done: {n_img} images, {n_fix} fixes, {n_fail} failed, {} traj samples -> {}",
        h.traj.len(), o.out
    );
}
