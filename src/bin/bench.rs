//! On-device pyramid bench (BENCH_VI front-end, 4:3 kernels, no embeddings).
//!
//! Bakes a few VGA frames into flash (`include_bytes!`) so nothing has to be
//! streamed, then runs `pyramid::extract_pyramid` a couple of times per frame
//! and prints the per-phase µs split: FAST detect / score / NMS / box blur /
//! rBRIEF (with the angle+sample cycle split) / 4:3 downscale, per level.
//!
//! No map, no calc8 Embedder, no EKF: this measures the part of BENCH_VI that
//! is new and content-dependent (the extraction kernels). Matching + PnP are
//! measured by the full `vo_replay` run once a map is available.
//!
//! Run: `cargo run --bin bench`
//!
//! The baked frames live under `bench_vi/replay/frames/` (the offline BENCH_VI
//! replay inputs). Point `include_bytes!` elsewhere to bench other imagery.

use std::time::{SystemTime, UNIX_EPOCH};

use esp_idf_hal::delay::FreeRtos;
use vo_box_lite::fast::Corner;
use vo_box_lite::pyramid::{self, Feature, PyramidProfile};

// Baked 640x480 grayscale frames (the EuRoC replay query frames).
const FRAMES: [(&str, &[u8]); 3] = [
    (
        "IMG0001",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/bench_vi/replay/frames/IMG0001.bit"
        )),
    ),
    (
        "IMG0002",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/bench_vi/replay/frames/IMG0002.bit"
        )),
    ),
    (
        "IMG0003",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/bench_vi/replay/frames/IMG0003.bit"
        )),
    ),
];

const W: usize = 640;
const H: usize = 480;
/// Repeats per (frame, threshold): "a couple of times".
const REPS: usize = 2;
/// FAST thresholds to bench (device localize default, plus 30: the threshold
/// sweep's sweet spot — ~half the features of 20 at no accuracy cost).
const THRS: [i32; 2] = [pyramid::FAST_THRESHOLD, 30];
/// CPU clock for the CCOUNT -> µs conversion (sdkconfig.defaults = 240 MHz).
const CPU_MHZ: u64 = 240;

fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

fn sum(a: &[u64; pyramid::LEVELS]) -> u64 {
    a.iter().sum()
}

fn report(name: &str, thr: i32, rep: usize, feats: usize, total_us: u64, p: &PyramidProfile) {
    for l in 0..pyramid::LEVELS {
        let (lw, lh) = pyramid::level_dims(W, H, l);
        let ds = if l + 1 < pyramid::LEVELS { p.downscale_us[l] } else { 0 };
        log::info!(
            "  {name} t{thr} r{rep} L{l} {lw}x{lh} kp={:4} kept={:4} fast={:5} score={:4} nms={:3} blur={:5} brief={:6} ds={:5}",
            p.corners[l], p.kept[l], p.fast_us[l], p.score_us[l], p.nms_us[l], p.blur_us[l],
            p.rbrief_us[l], ds,
        );
    }
    let (ang_cyc, smp_cyc) = (sum(&p.rbrief_angle_cyc), sum(&p.rbrief_sample_cyc));
    log::info!(
        "  {name} t{thr} r{rep} TOTAL extract={total_us}us feats={feats} | fast={} score={} nms={} blur={} brief={} ds={} | brief angle={}us/{ang_cyc}cyc sample={}us/{smp_cyc}cyc",
        sum(&p.fast_us), sum(&p.score_us), sum(&p.nms_us), sum(&p.blur_us),
        sum(&p.rbrief_us), sum(&p.downscale_us),
        ang_cyc / CPU_MHZ, smp_cyc / CPU_MHZ,
    );
}

fn run() {
    let mut arena = vec![0u8; pyramid::arena_bytes(W, H)];
    let mut work = vec![0u8; W * H];
    // blur's EE kernel needs a 16-byte-aligned `vcol` (it silently falls back
    // to scalar otherwise). Over-allocate and slice at the aligned offset;
    // `main.rs`/`vo_replay.rs` still use a bare `vec![0u16; w]`, so their blur
    // may be scalar too depending on the heap layout.
    let mut vcol_buf = vec![0u16; W + 8];
    let vcol_off = ((16 - (vcol_buf.as_ptr() as usize & 15)) / 2) % 8;
    let vcol = &mut vcol_buf[vcol_off..vcol_off + W];
    let mut corners = vec![Corner { x: 0, y: 0 }; pyramid::CORNERS_RAW_MAX];
    let mut scores = vec![0i32; pyramid::CORNERS_RAW_MAX];
    let mut rowidx = vec![usize::MAX; H];
    let mut nms = vec![Corner { x: 0, y: 0 }; pyramid::CORNERS_RAW_MAX];
    let mut cells = vec![0u32; pyramid::bucket_cells(W, H) * pyramid::BUCKET_K];
    let mut cand = vec![pyramid::Candidate::default(); pyramid::CAND_MAX];
    let mut feats = vec![Feature::default(); pyramid::MAX_FEATURES];

    log::info!(
        "bench: {} frames {W}x{H}, thr={THRS:?}, {REPS} reps, arena {} B, {} MHz, vcol 16B-aligned: {}",
        FRAMES.len(),
        arena.len(),
        CPU_MHZ,
        vcol.as_ptr() as usize & 15 == 0,
    );
    for (name, img) in FRAMES {
        assert_eq!(img.len(), W * H, "{name} is not {W}x{H}");
        for &thr in &THRS {
            for rep in 0..REPS {
                let mut prof = PyramidProfile::new(now_us);
                let t0 = now_us();
                let n = pyramid::extract_pyramid(
                    img, W, H, thr, &mut arena, &mut work, vcol, &mut corners,
                    &mut scores, &mut rowidx, &mut nms, &mut cells, &mut cand,
                    &mut feats, Some(&mut prof),
                );
                let total = now_us().wrapping_sub(t0);
                report(name, thr, rep, n, total, &prof);
                // Yield after each run so IDLE0 can feed the task WDT (the
                // extraction loop is CPU-bound on this main task, CPU0).
                FreeRtos::delay_ms(5);
            }
        }
    }
    log::info!(
        "bench done; 4:3 SIMD exercised: {}",
        vo_box_lite::downscale::downscale43_simd_used()
    );
}

fn main() {
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();
    run();
}
