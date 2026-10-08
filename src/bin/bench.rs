//! On-device pyramid bench (BENCH_VI front-end, 4:3 kernels, no embeddings).
//!
//! Bakes 10 VGA frames into flash (`include_bytes!`) so nothing has to be
//! streamed, then runs the `Filtered` extractor (dedup + bucket pruning) at
//! FAST threshold 30 several times per frame and prints the average per-phase
//! µs split: FAST detect / score / NMS / box blur / rBRIEF (with the
//! angle+sample cycle split) / 4:3 downscale, per level.
//!
//! No map, no calc8 Embedder, no EKF: this measures the part of BENCH_VI that
//! is new and content-dependent (the extraction kernels).
//!
//! Run: `cargo run --bin bench`
//!
//! The baked frames live under `bench_vi/replay/frames/` (the offline BENCH_VI
//! replay inputs). Point `include_bytes!` elsewhere to bench other imagery.

use std::time::{SystemTime, UNIX_EPOCH};

use esp_idf_hal::delay::FreeRtos;
use vo_box_lite::fast::Corner;
use vo_box_lite::pyramid::{self, Feature, PyramidProfile};

// Baked 640x480 grayscale frames (a random sample of the EuRoC replay frames).
const FRAMES: [(&str, &[u8]); 10] = [
    (
        "IMG0003",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/bench_vi/replay/frames/IMG0003.bit"
        )),
    ),
    (
        "IMG0004",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/bench_vi/replay/frames/IMG0004.bit"
        )),
    ),
    (
        "IMG0005",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/bench_vi/replay/frames/IMG0005.bit"
        )),
    ),
    (
        "IMG0008",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/bench_vi/replay/frames/IMG0008.bit"
        )),
    ),
    (
        "IMG0010",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/bench_vi/replay/frames/IMG0010.bit"
        )),
    ),
    (
        "IMG0015",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/bench_vi/replay/frames/IMG0015.bit"
        )),
    ),
    (
        "IMG0019",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/bench_vi/replay/frames/IMG0019.bit"
        )),
    ),
    (
        "IMG0023",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/bench_vi/replay/frames/IMG0023.bit"
        )),
    ),
    (
        "IMG0025",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/bench_vi/replay/frames/IMG0025.bit"
        )),
    ),
    (
        "IMG0027",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/bench_vi/replay/frames/IMG0027.bit"
        )),
    ),
];

const W: usize = 640;
const H: usize = 480;
/// FAST threshold (query-side sweet spot; only the filtered path is benched).
const THR: i32 = 30;
/// Runs per frame; the per-level breakdown printed after these is the average.
const REPS: usize = 3;
/// Query-side extractor: dedup + bucket pruning.
const MODE: pyramid::ExtractMode = pyramid::ExtractMode::Filtered;
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

/// Accumulate one run's profile into `acc` (reps summed; divided at report).
fn accumulate(acc: &mut PyramidProfile, p: &PyramidProfile) {
    for l in 0..pyramid::LEVELS {
        acc.fast_us[l] += p.fast_us[l];
        acc.score_us[l] += p.score_us[l];
        acc.nms_us[l] += p.nms_us[l];
        acc.blur_us[l] += p.blur_us[l];
        acc.rbrief_us[l] += p.rbrief_us[l];
        acc.rbrief_angle_cyc[l] += p.rbrief_angle_cyc[l];
        acc.rbrief_sample_cyc[l] += p.rbrief_sample_cyc[l];
        acc.downscale_us[l] += p.downscale_us[l];
        acc.corners[l] += p.corners[l];
        acc.kept[l] += p.kept[l];
    }
}

/// Print the average (over `div` reps) per-level breakdown + summary.
fn report(name: &str, feats: usize, total_us: u64, div: usize, p: &PyramidProfile) {
    let d = div as u64;
    for l in 0..pyramid::LEVELS {
        let (lw, lh) = pyramid::level_dims(W, H, l);
        let ds = if l + 1 < pyramid::LEVELS { p.downscale_us[l] / d } else { 0 };
        log::info!(
            "  AVG {name} t{THR} L{l} {lw}x{lh} kp={:4} kept={:4} fast={:5} score={:4} nms={:3} blur={:5} brief={:6} ds={:5}",
            p.corners[l] / div, p.kept[l] / div, p.fast_us[l] / d, p.score_us[l] / d,
            p.nms_us[l] / d, p.blur_us[l] / d, p.rbrief_us[l] / d, ds,
        );
    }
    let (ang_cyc, smp_cyc) = (sum(&p.rbrief_angle_cyc) / d, sum(&p.rbrief_sample_cyc) / d);
    log::info!(
        "  AVG {name} t{THR} extract={}us feats={feats} | fast={} score={} nms={} blur={} brief={} ds={} | brief angle={}us/{ang_cyc}cyc sample={}us/{smp_cyc}cyc",
        total_us / d,
        sum(&p.fast_us) / d,
        sum(&p.score_us) / d,
        sum(&p.nms_us) / d,
        sum(&p.blur_us) / d,
        sum(&p.rbrief_us) / d,
        sum(&p.downscale_us) / d,
        ang_cyc / CPU_MHZ,
        smp_cyc / CPU_MHZ,
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
    let mut dedup = vec![0u16; pyramid::dedup_scratch_len(W, H)];
    let mut feats = vec![Feature::default(); pyramid::MAX_FEATURES];
    // The baked frames live in flash, which the S3 GDMA cannot read; copy each
    // into PSRAM so the DMA downscale path (as in the real pipeline) is used.
    let mut frame = vec![0u8; W * H];

    log::info!(
        "bench: {} frames {W}x{H}, thr={THR}, filter-only, {REPS} reps averaged, arena {} B, {} MHz, vcol 16B-aligned: {}",
        FRAMES.len(),
        arena.len(),
        CPU_MHZ,
        vcol.as_ptr() as usize & 15 == 0,
    );
    for (name, img) in FRAMES {
        assert_eq!(img.len(), W * H, "{name} is not {W}x{H}");
        frame.copy_from_slice(img);
        let mut acc = PyramidProfile::new(now_us);
        let mut total_us = 0u64;
        let mut n = 0usize;
        for rep in 0..REPS {
            let mut prof = PyramidProfile::new(now_us);
            let t0 = now_us();
            n = pyramid::extract_pyramid(
                &frame, W, H, THR, MODE, &mut arena, &mut work, vcol, &mut corners,
                &mut scores, &mut rowidx, &mut nms, &mut cells, &mut cand,
                &mut dedup, &mut feats, Some(&mut prof),
            );
            let dt = now_us().wrapping_sub(t0);
            total_us += dt;
            accumulate(&mut acc, &prof);
            log::info!("  rep {name} r{rep} extract={dt}us feats={n}");
            // Yield after each run so IDLE0 can feed the task WDT (the
            // extraction loop is CPU-bound on this main task, CPU0).
            FreeRtos::delay_ms(5);
        }
        report(name, n, total_us, REPS, &acc);
    }
    log::info!(
        "bench done; EE exercised: blur={} ds43={} fast={} dma43={} fused43={}",
        vo_box_lite::blur::blur_simd_used(),
        vo_box_lite::downscale::downscale43_simd_used(),
        vo_box_lite::fast::ee::fast12_ee_simd_used(),
        vo_box_lite::downscale::dma43_used(),
        vo_box_lite::downscale::fused43_cpu_used()
    );
}

fn main() {
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();
    run();
}
