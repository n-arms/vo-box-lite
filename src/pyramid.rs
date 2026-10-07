//! Five-slot scale pyramid: level 0 = source, levels 1..4 are 4:3-downscaled
//! from the previous. All five levels are extracted. Per level: FAST-12 -> box
//! blur -> rBRIEF (border keypoints dropped); positions project to level-0 as
//! `x * (4/3)^level`. no_std, caller-owned buffers.

use crate::blur::box_blur5x5;
use crate::downscale::downscale_43;
use crate::downscale::downscale_43_size;
use crate::fast;
use crate::rbrief;

/// Pyramid slots. Level 0 = the caller's source frame; levels 1..=4 are the
/// downscaled images. All slots run FAST+blur+rBRIEF.
pub const LEVELS: usize = 5;
/// Fixed 4:3 pyramid ratio (the downscaler's scale per level).
pub const SCALE: f32 = 4.0 / 3.0;
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
/// Cross-level dedup radius in level-0 px (0.0 = off). A survivor within this
/// of an already-kept feature (finer levels are processed first, so
/// first-seen wins = finest wins) is dropped before description and costs no
/// rBRIEF. Laptop A/B (1 s replay, EKF prior): 3.0 px takes 1226 -> 574
/// feats/frame at 140 -> 159 mm sim3 ATE.
///
/// Bucketing (`BUCKET_CELL` > 0): at most `BUCKET_K` survivors per cell, by
/// FAST score, are described per frame. Combined with dedup this takes
/// 1226 -> 146 feats at 139 mm ATE (baseline 140 mm) with regime-matched
/// gates (`min-inliers` 20, att 60 deg — absolute gates tuned for the dense
/// regime reject good sparse fixes and spiral the EKF).
///
/// Map + query descriptors must come from the same build: rebuild the map
/// (`receive_map` / `bench_vi` prep) after changing any of these.
pub const DEDUP_PX: f32 = 3.0;
/// Bucket cell pitch in level-0 px (0 = bucketing off). Cells tile the frame;
/// winners compete globally across levels via the caller-owned `cells` scratch
/// (see `bucket_cells`). VGA -> 16x12 = 192 cells.
pub const BUCKET_CELL: usize = 40;
/// Max described survivors per bucket cell.
pub const BUCKET_K: usize = 2;
/// Sparse-frame floor: levels are processed finest-first and bucketing engages
/// once the running NMS-survivor count reaches this; frames ending below it
/// keep every deduped survivor (matches the offline reference, which keys off
/// the final deduped count, except when the count crosses mid-frame).
pub const BUCKET_MIN: usize = 150;

/// Bucket-cell count for a `w` x `h` frame (0 when bucketing is off). Size the
/// `cells` scratch passed to `extract_pyramid` as `bucket_cells(w, h) *
/// BUCKET_K` u32s (VGA: 192 * 2 = 384).
pub const fn bucket_cells(w: usize, h: usize) -> usize {
    if BUCKET_CELL == 0 {
        0
    } else {
        (w + BUCKET_CELL - 1) / BUCKET_CELL * ((h + BUCKET_CELL - 1) / BUCKET_CELL)
    }
}

/// One pyramid feature: descriptor + position projected to level-0 pixels.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Feature {
    /// Pyramid level the keypoint was detected on (0 = full size).
    pub level: u8,
    /// Level-0 pixel x (local keypoint x * (4/3)^level, f32 like extract.c).
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
    /// Described (kept) features per level (post dedup/bucket; == corners
    /// when the filters are off).
    pub kept: [usize; LEVELS],
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
            kept: [0; LEVELS],
        }
    }
}

/// Level-`l` pixel dims from level-0 `(w, h)` (4:3 rule applied `l` times).
pub fn level_dims(w: usize, h: usize, l: usize) -> (usize, usize) {
    let mut cw = w;
    let mut ch = h;
    for _ in 0..l {
        cw = downscale_43_size(cw);
        ch = downscale_43_size(ch);
    }
    (cw, ch)
}

/// Arena bytes needed for the downscaled levels 1..=LEVELS-1 of a `w` x `h`
/// frame (concatenated level pixel counts; VGA = 354,420 B).
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
/// `cells` is the bucket top-K scratch, `bucket_cells(w, h) * BUCKET_K` u32s
/// (may be empty when bucketing is off or undersized — bucketing then
/// degrades to describe-all-deduped, dedup still applies). `cand` holds up to
/// `CAND_MAX` pre-describe survivors (phase 1); smaller buffers silently cap
/// the candidate set (all in-repo callers pass `CAND_MAX`).
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
    cells: &mut [u32],
    cand: &mut [Candidate],
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

    let filtering = DEDUP_PX > 0.0 || bucket_cells(w, h) > 0 && !cells.is_empty();
    if !filtering {
        // Legacy unfiltered path (all consts off, no cells scratch): detect +
        // blur + describe every level exactly as before.
        let mut total = 0;
        let mut scale = 1.0f32; // accumulates *4/3 per level ((4/3)^l)
        let mut cw = w;
        let mut ch = h;
        let mut cur: &[u8] = img;
        let mut rest: &mut [u8] = arena;
        for l in 0..LEVELS {
            if cw < 7 || ch < 7 || total >= out.len() {
                break;
            }
            total += process_level(
                cur, cw, ch, thr, l as u8, scale, work, vcol, corners, scores, rowidx,
                nms, out, total, profile.as_deref_mut(),
            );
            scale *= SCALE;
            if l + 1 == LEVELS {
                break;
            }
            let dw = downscale_43_size(cw);
            let dh = downscale_43_size(ch);
            if dw < 7 || dh < 7 {
                break;
            }
            let (region, tail) = rest.split_at_mut(dw * dh);
            let ds_t0 = profile.as_ref().map(|pr| (pr.now_us)());
            if !downscale_43(cur, cw, ch, work, region) {
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
        return total;
    }
    // Filtered path: phase 1 detects every level into `cand` (no blur, no
    // describe); phase 2 selects globally (dedup + floor + bucket); phase 3
    // blurs only levels that kept winners and describes them. Per-level
    // describe in phase 1 would let later levels add on top of earlier ones
    // so the bucket cap never binds — selection must precede any describe.
    let cells_nx = (w + BUCKET_CELL.max(1) - 1) / BUCKET_CELL.max(1);
    let cells_ny = (h + BUCKET_CELL.max(1) - 1) / BUCKET_CELL.max(1);
    let cells_ok =
        BUCKET_CELL > 0 && BUCKET_K > 0 && cells.len() >= cells_nx * cells_ny * BUCKET_K;
    if cells_ok {
        for c in cells[..cells_nx * cells_ny * BUCKET_K].iter_mut() {
            *c = 0;
        }
    }
    // Phase 1: detect-only + downscale chain (`work` is the h-pass tmp; the
    // level blur outputs are dead until phase 3). `scales[l]` records the
    // exact accumulated projection factor per level.
    let mut nc = 0usize; // candidates in cand[..nc]
    let mut scale = 1.0f32;
    let mut scales = [1.0f32; LEVELS];
    let mut cw = w;
    let mut ch = h;
    let mut cur: &[u8] = img;
    let mut rest: &mut [u8] = arena;
    for l in 0..LEVELS {
        if cw < 7 || ch < 7 || nc >= cand.len() || nc >= scores.len() {
            break;
        }
        scales[l] = scale;
        let n = detect_level(
            cur, cw, ch, thr, corners, scores, rowidx, nms, profile.as_deref_mut(), l,
        );
        if let Some(pr) = profile.as_deref_mut() {
            pr.corners[l] = n;
        }
        // Border-doomed keys die at describe (`rbrief_angle` returns None
        // inside the 20-px band); drop them now so they neither occupy slots
        // nor shield keepable neighbours — the offline post-rbrief set.
        for i in 0..n {
            if nc >= cand.len() || nc >= scores.len() {
                break;
            }
            let kp = nms[i];
            if kp.x < rbrief::HALF_BOUNDARY
                || kp.y < rbrief::HALF_BOUNDARY
                || kp.x + rbrief::HALF_BOUNDARY >= cw
                || kp.y + rbrief::HALF_BOUNDARY >= ch
            {
                continue;
            }
            cand[nc] = Candidate {
                level: l as u8,
                x: kp.x as u16,
                y: kp.y as u16,
                score: scores[i],
                x0: kp.x as f32 * scale,
                y0: kp.y as f32 * scale,
            };
            nc += 1;
        }
        scale *= SCALE;
        if l + 1 == LEVELS {
            break;
        }
        let dw = downscale_43_size(cw);
        let dh = downscale_43_size(ch);
        if dw < 7 || dh < 7 {
            break;
        }
        let (region, tail) = rest.split_at_mut(dw * dh);
        let ds_t0 = profile.as_ref().map(|pr| (pr.now_us)());
        if !downscale_43(cur, cw, ch, work, region) {
            break;
        }
        if let (Some(pr), Some(ds_t0)) = (profile.as_deref_mut(), ds_t0) {
            pr.downscale_us[l] = (pr.now_us)() - ds_t0;
        }
        cur = &*region;
        rest = tail;
        cw = dw;
        ch = dh;
    }
    // Phase 2: global first-wins dedup over cand[..nc] (level-major order is
    // preserved, so finest wins — exactly the offline reference), then the
    // BUCKET_MIN floor and global top-K bucket. Winners flagged in
    // `scores[..nc]` (nc <= scores.len() by the phase-1 cap above).
    let r2 = DEDUP_PX * DEDUP_PX;
    let do_dedup = DEDUP_PX > 0.0;
    let mut n1 = 0usize; // deduped count (compacted in cand)
    for i in 0..nc {
        if do_dedup {
            let (x0, y0) = (cand[i].x0, cand[i].y0);
            let mut dup = false;
            for k in 0..n1 {
                let dx = cand[k].x0 - x0;
                let dy = cand[k].y0 - y0;
                if dx * dx + dy * dy < r2 {
                    dup = true;
                    break;
                }
            }
            if dup {
                continue;
            }
        }
        cand[n1] = cand[i];
        n1 += 1;
    }
    // Winner idx needs 13 packed bits; above that the bucket is skipped
    // (CAND_MAX == 8192 keeps this a formality).
    let do_bucket = cells_ok && n1 >= BUCKET_MIN && n1 <= 0x1FFF;
    let mut nw = 0usize; // winners listed in scores[..nw]
    if do_bucket {
        // Slots pack `((score + 1) << 13) | idx | NEW_BIT`; the tag marks
        // insertions so collection never picks up stale levels (cells were
        // zeroed above, so every tag here is this frame's).
        const NEW_BIT: u32 = 1 << 31;
        for i in 0..n1 {
            let gx = ((cand[i].x0 / BUCKET_CELL as f32) as usize).min(cells_nx - 1);
            let gy = ((cand[i].y0 / BUCKET_CELL as f32) as usize).min(cells_ny - 1);
            let base = (gy * cells_nx + gx) * BUCKET_K;
            let mut v = (((cand[i].score.max(0) + 1) as u32) << 13) | (i as u32) | NEW_BIT;
            for s in 0..BUCKET_K {
                let slot = &mut cells[base + s];
                if v > *slot {
                    core::mem::swap(&mut v, slot);
                }
            }
        }
        // Collect tagged slots (cap: scores scratch), then sort winner idxs
        // ascending to restore candidate (level-major) order for phase 3.
        for slot in cells[..cells_nx * cells_ny * BUCKET_K].iter_mut() {
            if nw >= scores.len() {
                break;
            }
            if *slot & NEW_BIT != 0 {
                *slot &= !NEW_BIT;
                scores[nw] = (*slot & 0x1FFF) as i32;
                nw += 1;
            }
        }
        // Insertion sort (nw <= ncells*K, a few hundred at most).
        for i in 1..nw {
            let v = scores[i];
            let mut j = i;
            while j > 0 && scores[j - 1] > v {
                scores[j] = scores[j - 1];
                j -= 1;
            }
            scores[j] = v;
        }
    } else {
        for i in 0..n1 {
            scores[nw] = i as i32;
            nw += 1;
        }
    }
    // Phase 3: blur each winning level once, describe its winners (level
    // images persist: L0 is `img`, L1+ are the arena regions phase 1 wrote).
    // Winners arrive level-major, so one blur covers each level's run.
    let mut total = 0;
    let mut angle_cyc = 0u64;
    let mut sample_cyc = 0u64;
    let mut kept = [0usize; LEVELS];
    let mut lvl = 255u8; // level currently blurred into work (none)
    let mut cwl = 0usize;
    let mut chl = 0usize;
    let mut seg_t0 = profile.as_ref().map(|pr| (pr.now_us)());
    for wi in 0..nw {
        if total >= out.len() {
            break;
        }
        let c = cand[scores[wi] as usize];
        if c.level != lvl {
            // Close the previous level's rbrief segment.
            if lvl != 255 {
                if let (Some(pr), Some(t0)) = (profile.as_deref_mut(), seg_t0) {
                    pr.rbrief_us[lvl as usize] += (pr.now_us)().wrapping_sub(t0);
                }
            }
            lvl = c.level;
            let (img_l, dw, dh) = level_image(img, &*arena, w, h, lvl as usize);
            cwl = dw;
            chl = dh;
            if let Some(pr) = profile.as_deref_mut() {
                let t0 = (pr.now_us)();
                if !box_blur5x5(img_l, work, cwl, chl, vcol) {
                    break;
                }
                pr.blur_us[lvl as usize] = (pr.now_us)().wrapping_sub(t0);
            } else if !box_blur5x5(img_l, work, cwl, chl, vcol) {
                break;
            }
            seg_t0 = profile.as_ref().map(|pr| (pr.now_us)());
        }
        if describe_survivor(
            work, cwl, chl,
            fast::Corner { x: c.x as usize, y: c.y as usize },
            lvl, scales[lvl as usize], out, total, &mut angle_cyc, &mut sample_cyc,
        ) {
            kept[lvl as usize] += 1;
            total += 1;
        }
    }
    if lvl != 255 {
        if let (Some(pr), Some(t0)) = (profile.as_deref_mut(), seg_t0) {
            pr.rbrief_us[lvl as usize] += (pr.now_us)().wrapping_sub(t0);
        }
    }
    if let Some(pr) = profile.as_deref_mut() {
        for l in 0..LEVELS {
            pr.kept[l] = kept[l];
        }
        pr.rbrief_angle_cyc[0] = angle_cyc;
        pr.rbrief_sample_cyc[0] = sample_cyc;
    }
    total
}

/// Pre-describe survivor: level-local coords + FAST score + level-0 projection.
/// Phase 1 of extraction fills `cand`; phase 2 selects globally; phase 3
/// describes. 20 B each; the caller-owned buffer holds CAND_MAX (160 KB).
#[derive(Clone, Copy, Default)]
pub struct Candidate {
    pub level: u8,
    pub x: u16,
    pub y: u16,
    pub score: i32,
    pub x0: f32,
    pub y0: f32,
}
/// Candidate (pre-describe survivor) cap = the `cand` capacity the caller must
/// provide. Bounds the O(n^2) global dedup (thr-20 EuRoC holds ~3000).
pub const CAND_MAX: usize = 8192;

/// FAST-12 detect -> score -> NMS on one level, with per-survivor scores
/// compacted into `scores[..n]` by one merge walk (NMS output is a
/// raster-ordered subsequence of the raw corners, whose scores sit parallel in
/// the raw score array; the write index never overtakes the read cursor).
/// Returns the survivor count. Shared by the legacy and filtered paths.
#[allow(clippy::too_many_arguments)]
fn detect_level(
    src: &[u8],
    cw: usize,
    ch: usize,
    thr: i32,
    corners: &mut [fast::Corner], // raw candidates scratch
    scores: &mut [i32],
    rowidx: &mut [usize],
    nms: &mut [fast::Corner], // NMS survivor store
    mut profile: Option<&mut PyramidProfile>,
    li: usize,
) -> usize {
    // Exactly fast::fast12_detect_nonmax, split only for per-phase timing
    // (fast.rs's wrapper test guards the equivalence). Detect is the EE SIMD
    // variant; score/NMS stay scalar.
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
    let mut nsc = 0usize; // scored raw corners (merge-walk bound)
    if nraw > 0 {
        nsc = fast::fast12_score(src, cw, &corners[..nraw.min(corners.len())], thr, scores);
        if let Some(pr) = profile.as_deref_mut() {
            pr.score_us[li] = (pr.now_us)() - t0;
            t0 = (pr.now_us)();
        }
        n = fast::nonmax_suppression(&corners[..nsc], &scores[..nsc], rowidx, nms);
        if let Some(pr) = profile.as_deref_mut() {
            pr.nms_us[li] = (pr.now_us)() - t0;
        }
    }
    let n = n.min(nms.len());
    if n > 0 && nsc > 0 {
        let mut p = 0usize;
        for i in 0..n {
            while p < nsc && (corners[p].x != nms[i].x || corners[p].y != nms[i].y) {
                p += 1;
            }
            if p >= nsc {
                scores[i] = thr - 1; // unreachable; drop-first if ever hit
            } else {
                scores[i] = scores[p];
                p += 1;
            }
        }
    }
    n
}

/// FAST-12 (detect -> score -> NMS) + 5x5 blur + rBRIEF on one level (`src`,
/// `cw` x `ch`). NMS runs on the raw image (the blur feeds rBRIEF only). Appends
/// to `out[total..]`, returns the count; `profile` fills this level's timers.
/// Legacy unfiltered path (all filter consts off); the filtered path below
/// detects through `detect_level` and selects globally instead.
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
    let n = detect_level(
        src, cw, ch, thr, corners, scores, rowidx, nms, profile.as_deref_mut(), li,
    );
    // Blur after FAST, before description.
    let mut t0 = 0u64;
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
    // Split rBRIEF into orientation + sampling and time each per keypoint with
    // the CCOUNT cycle counter (host builds return 0, so the timers stay 0).
    let mut added = 0;
    let mut angle_cyc = 0u64;
    let mut sample_cyc = 0u64;
    for i in 0..n {
        if total + added >= out.len() {
            break;
        }
        if describe_survivor(
            work, cw, ch, nms[i], level, scale, out, total + added, &mut angle_cyc,
            &mut sample_cyc,
        ) {
            added += 1;
        }
    }
    if let Some(pr) = profile.as_deref_mut() {
        pr.rbrief_us[li] = (pr.now_us)() - t0;
        pr.corners[li] = n; // rBRIEF attempts == NMS survivors this level
        pr.kept[li] = n;
        pr.rbrief_angle_cyc[li] = angle_cyc;
        pr.rbrief_sample_cyc[li] = sample_cyc;
    }
    added
}

/// Level-`l` raw image for phase 3: L0 is the source frame, L1+ are the arena
/// regions phase 1 wrote (same `level_dims` sequence, so offsets match).
fn level_image<'a>(img: &'a [u8], arena: &'a [u8], w: usize, h: usize, l: usize) -> (&'a [u8], usize, usize) {
    if l == 0 {
        return (&img[..w * h], w, h);
    }
    let mut off = 0usize;
    for k in 1..l {
        let (cw, ch) = level_dims(w, h, k);
        off += cw * ch;
    }
    let (cw, ch) = level_dims(w, h, l);
    (&arena[off..off + cw * ch], cw, ch)
}

/// Describe one NMS survivor into `out[slot]`; returns true if a feature was
/// stored (false = border band, all-zero descriptor dropped). Phase timing is
/// the caller's job (see the CCOUNT split in `process_level`).
#[allow(clippy::too_many_arguments)]
fn describe_survivor(
    work: &[u8],
    cw: usize,
    ch: usize,
    kp: fast::Corner,
    level: u8,
    scale: f32,
    out: &mut [Feature],
    slot: usize,
    angle_cyc: &mut u64,
    sample_cyc: &mut u64,
) -> bool {
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
    *angle_cyc += c1.wrapping_sub(c0);
    *sample_cyc += c2.wrapping_sub(c1);
    if ok {
        out[slot] = Feature {
            level,
            x: kp.x as f32 * scale,
            y: kp.y as f32 * scale,
            desc,
        };
        true
    } else {
        false
    }
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
