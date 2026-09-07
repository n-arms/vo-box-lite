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
//!  - `corners`: per-level FAST scratch, >= [`CORNERS_PER_LEVEL`].
//!  - `out`: feature store, >= [`MAX_FEATURES`].

use crate::blur::box_blur5x5;
use crate::downscale::downscale_65;
use crate::downscale::downscale_65_size;
use crate::fast;
use crate::rbrief;

/// fast+blur+rbrief runs per frame: level 0 (full size) + 6 downscales.
pub const LEVELS: usize = 7;
/// Fixed 6:5 pyramid ratio (the downscaler's scale per level).
pub const SCALE: f32 = 1.2;
/// FAST corner threshold — keep in sync with the future `localize` extractor
/// (and slam-exp's process_images/localize.cpp, which use 40).
pub const FAST_THRESHOLD: i32 = 40;
/// Per-level FAST corner scratch cap (raw detector, no non-max suppression).
pub const CORNERS_PER_LEVEL: usize = 2048;
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
/// (levels in order 0..6, keypoints in scan order, only rBRIEF-valid ones).
///
/// Returns 0 if any buffer is undersized (checked once from the level-0 dims)
/// or the image has no interior (w/h < 7).
pub fn extract_pyramid(
    img: &[u8],
    w: usize,
    h: usize,
    thr: i32,
    arena: &mut [u8],
    work: &mut [u8],
    vcol: &mut [u16],
    corners: &mut [fast::Corner],
    out: &mut [Feature],
) -> usize {
    if img.len() < w * h
        || arena.len() < arena_bytes(w, h)
        || work.len() < w * h
        || vcol.len() < w
        || corners.is_empty()
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
        total += process_level(cur, cw, ch, thr, l as u8, scale, work, vcol, corners, out, total);
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
        if !downscale_65(cur, cw, ch, work, region) {
            return total;
        }
        cur = &*region;
        rest = tail;
        cw = dw;
        ch = dh;
    }
    total
}

/// FAST + 5x5 blur + rBRIEF on one level (`src`, `cw` x `ch`). Border-band
/// keypoints are dropped (all-zero descriptor, never matched). Appends to
/// `out[total..]` (up to `out.len()`); returns the number appended.
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
    corners: &mut [fast::Corner],
    out: &mut [Feature],
    total: usize,
) -> usize {
    let n = fast::fast12_detect(src, cw, ch, cw, thr, corners);
    // Blur after FAST, before description (blur output = the rBRIEF image; the
    // raw level image is untouched for the downscale that follows).
    if !box_blur5x5(src, work, cw, ch, vcol) {
        return 0;
    }
    let mut added = 0;
    let n = n.min(corners.len());
    for i in 0..n {
        if total + added >= out.len() {
            break;
        }
        let kp = corners[i];
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
    added
}
