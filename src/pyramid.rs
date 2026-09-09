//! Seven-level scale pyramid — the `map` task's per-frame extractor.
//!
//! Mirrors slam-exp's `extract_pyramid()`: level 0 is the source frame and
//! levels 1..=6 are the previous level downscaled by the fixed 6:5 ratio
//! (1.2x) with [`downscale::downscale_65`]. Every level runs FAST-12 on the
//! raw level image, 5x5-box-blurs a copy ([`blur::box_blur5x5`], after FAST,
//! before description — extract.c order), then computes rBRIEF descriptors on
//! the blurred copy ([`rbrief::rbrief_descriptor`]); border-band keypoints are
//! dropped (all-zero descriptors never match). Each kept feature's local
//! integer keypoint is projected back to level-0 pixels with `x * 1.2^level`,
//! accumulated exactly like extract.c so descriptors/positions line up with
//! the COLMAP map built from this stream on the laptop.
//!
//! no_std + alloc-free: every buffer is caller-owned (PSRAM on the S3), so the
//! whole pyramid needs no per-level allocations and the buffers are reused
//! across frames. The level-0 frame is only ever read — the caller keeps it
//! pristine as the upload payload.
//!
//! Buffer layout (sized from the level-0 dims):
//!  - `arena`: pixel storage for levels 1..=6, concatenated and written
//!    strictly forward (each downscale writes the next level's region, which
//!    is never read again once that level is processed). >= [`arena_bytes`].
//!  - `work`: box-blur destination AND downscale h-pass scratch — used at
//!    different points of each level, never concurrently; >= `w * h` bytes.
//!  - `vcol`: box-blur running column sums, >= `w` u16 (keep in internal SRAM).
//! -  `corners`: per-level RAW FAST corner scratch, >= [`CORNERS_RAW_MAX`].
//! -  `scores` >= corners.len() i32s, `rowidx` >= the level-0 height (usize
//!    per image row), `nms`: non-max-suppression survivor store.
//! -  `out`: feature store, >= [`MAX_FEATURES`].

use crate::blur::box_blur5x5;
use crate::downscale::downscale_65;
use crate::downscale::downscale_65_size;
use crate::fast;
use crate::rbrief;

/// fast+blur+rbrief runs per frame: level 0 (full size) + 6 downscales.
pub const LEVELS: usize = 7;
/// Fixed 6:5 pyramid ratio (the downscaler's scale per level).
pub const SCALE: f32 = 1.2;
/// FAST corner threshold. Sep 8: 40 -> 20 (smooth webcam scenes gave 0
/// corners), then 20 -> 10 once the capture moved to QXGA 2048x1536 4x4-down-
/// sampled to 512x384 (the 4x4 mean blurs texture, so a weaker threshold is
/// needed to keep corner density up). Keep the future `localize` extractor on
/// the same value as THIS (map descriptors only match query descriptors
/// extracted at the same threshold). slam-exp's C pipeline runs 40 on its own
/// full-res captures — not directly comparable.
pub const FAST_THRESHOLD: i32 = 10;
/// Per-level RAW FAST corner scratch cap. Non-max suppression needs the FULL
/// raw list to suppress exactly (a truncated list only suppresses its top
/// rows), so this must comfortably exceed the raw count: 512x384 content at
/// threshold 10 can yield a few thousand corners on textured scenes.
pub const CORNERS_RAW_MAX: usize = 8192;
/// Per-frame feature cap (= the `out` capacity the caller must provide).
pub const MAX_FEATURES: usize = 4096;

/// One pyramid feature: descriptor + position projected to level-0 pixels.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Feature {
    /// Pyramid level the keypoint was detected on (0 = full size).
    pub level: u8,
    /// Level-0 pixel x (local keypoint x * 1.2^level, f32 like extract.c).
    pub x: f32,
    /// Level-0 pixel y.
    pub y: f32,
    /// 256-bit descriptor (extract.h layout: 8 x u32, bit k of word i = pair
    /// i*32+k). Never all-zero (border keypoints are dropped before this).
    pub desc: rbrief::Descriptor,
}

impl Default for Feature {
    fn default() -> Self {
        Feature {
            level: 0,
            x: 0.0,
            y: 0.0,
            desc: [0; 8],
        }
    }
}

/// Optional per-frame phase timings, filled by [`extract_pyramid`] when a
/// profile is passed (`None` = no profiling: a branch per level, nothing
/// else). One entry per pyramid level (index 0 = full size); `downscale_us[l]`
/// covers the 6:5 downscale INTO level l+1 (so index LEVELS-1 is never
/// written), `corners[l]` is the level's NMS survivor count (the rBRIEF
/// keypoint workload).
///
/// The clock is caller-injected (`now_us`, a monotonic microsecond fn —
/// `std::time::Instant` based on the S3) so this no_std lib stays portable;
/// host builds can pass a fake clock.
#[derive(Debug)]
pub struct PyramidProfile {
    /// Caller's microsecond clock (see the struct docs).
    pub now_us: fn() -> u64,
    /// FAST-12 detect per level.
    pub fast_us: [u64; LEVELS],
    /// FAST corner score per level (binary-search strength, feeds NMS).
    pub score_us: [u64; LEVELS],
    /// Non-max suppression per level.
    pub nms_us: [u64; LEVELS],
    /// 5x5 box blur per level.
    pub blur_us: [u64; LEVELS],
    /// rBRIEF description per level (over the NMS survivors).
    pub rbrief_us: [u64; LEVELS],
    /// 6:5 downscale into the next level (index LEVELS-1 unused).
    pub downscale_us: [u64; LEVELS],
    /// NMS survivors per level.
    pub corners: [usize; LEVELS],
}

impl PyramidProfile {
    pub fn new(now_us: fn() -> u64) -> Self {
        PyramidProfile {
            now_us,
            fast_us: [0; LEVELS],
            score_us: [0; LEVELS],
            nms_us: [0; LEVELS],
            blur_us: [0; LEVELS],
            rbrief_us: [0; LEVELS],
            downscale_us: [0; LEVELS],
            corners: [0; LEVELS],
        }
    }
}

/// Level-`l` pixel dims from level-0 `(w, h)` (6:5 rule applied `l` times).
pub fn level_dims(w: usize, h: usize, l: usize) -> (usize, usize) {
    let mut cw = w;
    let mut ch = h;
    for _ in 0..l {
        cw = downscale_65_size(cw);
        ch = downscale_65_size(ch);
    }
    (cw, ch)
}

/// Arena bytes needed for the downscaled levels 1..=LEVELS-1 of a `w` x `h`
/// frame (concatenated level pixel counts; VGA = 602,075 B).
pub fn arena_bytes(w: usize, h: usize) -> usize {
    let mut sz = 0;
    for l in 1..LEVELS {
        let (cw, ch) = level_dims(w, h, l);
        sz += cw * ch;
    }
    sz
}

/// Run the full pyramid extraction of the raw level-0 `img` (`w` x `h`) at
/// FAST threshold `thr`. Returns the number of features written to `out`
/// (levels in order 0..6, survivors of FAST non-max suppression in scan
/// order, only rBRIEF-valid ones).
///
/// Returns 0 if any buffer is undersized (checked once from the level-0 dims)
/// or the image has no interior (w/h < 7). Scratch: `corners` holds the raw
/// FAST candidates for a level (>= [`CORNERS_RAW_MAX`] so full lists are
/// scored and suppressed exactly), `scores` >= corners.len() i32s, `rowidx`
/// >= `h` usize (one per image row of the largest level), `nms` is the
/// survivor store (survivors past its length are dropped).
///
/// `profile`: optional per-phase/per-level timers, filled for the levels that
/// actually ran (skipped levels stay 0). Pass None to skip profiling.
pub fn extract_pyramid(
    img: &[u8],
    w: usize,
    h: usize,
    thr: i32,
    arena: &mut [u8],
    work: &mut [u8],
    vcol: &mut [u16],
    corners: &mut [fast::Corner],
    scores: &mut [i32],
    rowidx: &mut [usize],
    nms: &mut [fast::Corner],
    out: &mut [Feature],
    mut profile: Option<&mut PyramidProfile>,
) -> usize {
    if img.len() < w * h
        || arena.len() < arena_bytes(w, h)
        || work.len() < w * h
        || vcol.len() < w
        || corners.is_empty()
        || scores.len() < corners.len()
        || rowidx.len() < h
        || nms.is_empty()
        || out.is_empty()
    {
        return 0;
    }

    let mut total = 0;
    let mut scale = 1.0f32; // accumulates *1.2 per level (extract.c order)
    let mut cw = w;
    let mut ch = h;
    let mut cur: &[u8] = img; // level-l raw: img for l==0, else an arena region
    let mut rest: &mut [u8] = arena; // arena not yet claimed by a level
    for l in 0..LEVELS {
        if cw < 7 || ch < 7 || total >= out.len() {
            break;
        }
        total += process_level(
            cur, cw, ch, thr, l as u8, scale, work, vcol, corners, scores, rowidx, nms, out,
            total,
            profile.as_deref_mut(),
        );
        scale *= SCALE;

        if l + 1 == LEVELS {
            break;
        }
        let dw = downscale_65_size(cw);
        let dh = downscale_65_size(ch);
        if dw < 7 || dh < 7 {
            break;
        }
        // Downscale level l -> level l+1 into the next arena region (regions
        // are disjoint and written forward, so `cur` never aliases it). `work`
        // is reused as the h-pass scratch — the level's blur output is dead.
        let (region, tail) = rest.split_at_mut(dw * dh);
        // The 6:5 downscale runs between levels, so it is timed here (not in
        // process_level) when a profile is attached.
        let ds_t0 = profile.as_ref().map(|pr| (pr.now_us)());
        if !downscale_65(cur, cw, ch, work, region) {
            return total;
        }
        if let (Some(pr), Some(ds_t0)) = (profile.as_deref_mut(), ds_t0) {
            pr.downscale_us[l] = (pr.now_us)() - ds_t0;
        }
        cur = &*region;
        rest = tail;
        cw = dw;
        ch = dh;
    }
    total
}

/// FAST-12 (detect -> score -> non-max suppression) + 5x5 blur + rBRIEF on
/// one level (`src`, `cw` x `ch`). NMS runs on the raw level image before the
/// blur (blur output = the rBRIEF image only; the raw level image stays
/// untouched for the downscale that follows). Border-band keypoints are
/// dropped (all-zero descriptor, never matched). Appends to `out[total..]`
/// (up to `out.len()`); returns the number appended. When `profile` is Some,
/// the per-phase timers for this level are filled (fast/score/nms/blur/
/// rbrief; the inter-level downscale is timed by extract_pyramid).
#[allow(clippy::too_many_arguments)]
fn process_level(
    src: &[u8],
    cw: usize,
    ch: usize,
    thr: i32,
    level: u8,
    scale: f32,
    work: &mut [u8],
    vcol: &mut [u16],
    corners: &mut [fast::Corner], // raw candidates scratch
    scores: &mut [i32],
    rowidx: &mut [usize],
    nms: &mut [fast::Corner], // NMS survivor store
    out: &mut [Feature],
    total: usize,
    mut profile: Option<&mut PyramidProfile>,
) -> usize {
    let li = level as usize;
    // Detect -> score -> non-max suppression, timed per phase when a profile
    // is attached. The three calls are exactly what fast::fast12_detect_nonmax
    // does (fast.rs's `detect_nonmax_wrapper_matches_manual` test guards the
    // equivalence); they are split here only so each phase can be timed — keep
    // in sync with that wrapper. The pyramid preconditions keep the scratch
    // sized, so no level is ever truncated/starved by this composition.
    let mut t0 = 0u64; // phase start on the profile clock (0 = not profiling)
    if let Some(pr) = profile.as_deref_mut() {
        t0 = (pr.now_us)();
    }
    let nraw = fast::fast12_detect(src, cw, ch, cw, thr, corners);
    if let Some(pr) = profile.as_deref_mut() {
        pr.fast_us[li] = (pr.now_us)() - t0;
        t0 = (pr.now_us)();
    }
    let mut n = 0usize; // NMS survivors (== fast12_detect_nonmax's result)
    if nraw > 0 {
        let nsc = fast::fast12_score(src, cw, &corners[..nraw.min(corners.len())], thr, scores);
        if let Some(pr) = profile.as_deref_mut() {
            pr.score_us[li] = (pr.now_us)() - t0;
            t0 = (pr.now_us)();
        }
        n = fast::nonmax_suppression(&corners[..nsc], &scores[..nsc], rowidx, nms);
        if let Some(pr) = profile.as_deref_mut() {
            pr.nms_us[li] = (pr.now_us)() - t0;
        }
    }
    // Blur after FAST, before description.
    if let Some(pr) = profile.as_deref_mut() {
        t0 = (pr.now_us)();
    }
    if !box_blur5x5(src, work, cw, ch, vcol) {
        return 0;
    }
    if let Some(pr) = profile.as_deref_mut() {
        pr.blur_us[li] = (pr.now_us)() - t0;
        t0 = (pr.now_us)();
    }
    let mut added = 0;
    let n = n.min(nms.len());
    for i in 0..n {
        if total + added >= out.len() {
            break;
        }
        let kp = nms[i];
        let mut desc = [0u32; 8];
        if rbrief::rbrief_descriptor(work, cw, ch, kp.x, kp.y, &mut desc) {
            out[total + added] = Feature {
                level,
                x: kp.x as f32 * scale,
                y: kp.y as f32 * scale,
                desc,
            };
            added += 1;
        }
    }
    if let Some(pr) = profile.as_deref_mut() {
        pr.rbrief_us[li] = (pr.now_us)() - t0;
        pr.corners[li] = n; // rBRIEF attempts == NMS survivors this level
    }
    added
}
