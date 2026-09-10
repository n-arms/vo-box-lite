//! Seven-level scale pyramid: level 0 = source, 1..6 = 6:5-downscaled. Per level
//! FAST-12 -> box blur -> rBRIEF (border keypoints dropped); positions project to
//! level-0 as `x * 1.2^level` in extract.c order. no_std, caller-owned buffers.

use crate::blur::box_blur5x5;
use crate::downscale::downscale_65;
use crate::downscale::downscale_65_size;
use crate::fast;
use crate::rbrief;

/// fast+blur+rbrief runs per frame: level 0 (full size) + 6 downscales.
pub const LEVELS: usize = 7;
/// Fixed 6:5 pyramid ratio (the downscaler's scale per level).
pub const SCALE: f32 = 1.2;
/// FAST threshold. Lowered 40 -> 20 -> 10 as capture moved to QXGA 4x4-down-
/// sampled VGA (the mean blurs texture). Keep future `localize` extraction on
/// this value: map descriptors only match queries extracted at the same threshold.
pub const FAST_THRESHOLD: i32 = 10;
/// Raw FAST corner scratch cap. NMS needs the FULL raw list to suppress exactly
/// (a truncated list only covers its top rows), so this must comfortably exceed
/// the raw count — a few thousand corners on textured 512x384 scenes.
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

/// Optional per-frame timing profile (None = off). `now_us` is a caller-injected
/// monotonic µs clock so the no_std lib stays portable; `downscale_us[l]` times
/// the 6:5 into level l+1 (index LEVELS-1 unused), `corners[l]` = NMS survivors.
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
    /// rBRIEF orientation (IC_Angle) cycles per level (CCOUNT @ 160 MHz).
    pub rbrief_angle_cyc: [u64; LEVELS],
    /// rBRIEF 256-pair sampling cycles per level (CCOUNT @ 160 MHz).
    pub rbrief_sample_cyc: [u64; LEVELS],
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
            rbrief_angle_cyc: [0; LEVELS],
            rbrief_sample_cyc: [0; LEVELS],
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

/// Extract the whole pyramid from raw level-0 `img` (`w` x `h`) at threshold
/// `thr`; returns the feature count written to `out` (0 if a buffer is undersized
/// or there is no interior, w/h < 7). `profile`: optional per-level timers.
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

/// FAST-12 (detect -> score -> NMS) + 5x5 blur + rBRIEF on one level (`src`,
/// `cw` x `ch`). NMS runs on the raw image (the blur feeds rBRIEF only). Appends
/// to `out[total..]`, returns the count; `profile` fills this level's timers.
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
    // These three calls are exactly fast::fast12_detect_nonmax, split only for
    // per-phase timing (fast.rs's wrapper test guards the equivalence). Detect
    // is the EE SIMD variant; score/NMS stay scalar.
    let mut t0 = 0u64; // phase start on the profile clock (0 = not profiling)
    if let Some(pr) = profile.as_deref_mut() {
        t0 = (pr.now_us)();
    }
    let nraw = fast::ee::fast12_detect_ee(src, cw, ch, cw, thr, corners);
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
    // Split rBRIEF into orientation + sampling and time each per keypoint with
    // the CCOUNT cycle counter (host builds return 0, so the timers stay 0).
    let mut angle_cyc = 0u64;
    let mut sample_cyc = 0u64;
    for i in 0..n {
        if total + added >= out.len() {
            break;
        }
        let kp = nms[i];
        let mut desc = [0u32; 8];
        let c0 = ccount();
        let ang = rbrief::rbrief_angle(work, cw, ch, kp.x, kp.y);
        let c1 = ccount();
        let ok = match ang {
            Some((s, c)) => {
                rbrief::rbrief_samples(work, cw, kp.x, kp.y, s, c, &mut desc);
                true
            }
            None => false,
        };
        let c2 = ccount();
        angle_cyc += c1.wrapping_sub(c0);
        sample_cyc += c2.wrapping_sub(c1);
        if ok {
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
        pr.rbrief_angle_cyc[li] = angle_cyc;
        pr.rbrief_sample_cyc[li] = sample_cyc;
    }
    added
}

/// Xtensa CCOUNT cycle counter (rsr.ccount, 1 instruction) for the rBRIEF
/// phase timers; runs at the CPU clock (160 MHz), host builds return 0.
#[cfg(target_arch = "xtensa")]
fn ccount() -> u64 {
    let c: u32;
    // SAFETY: rsr has no side effects beyond writing the output register.
    unsafe {
        core::arch::asm!("rsr.ccount {0}", out(reg) c, options(nomem, nostack));
    }
    c as u64
}

#[cfg(not(target_arch = "xtensa"))]
fn ccount() -> u64 {
    0
}
