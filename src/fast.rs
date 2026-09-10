//! FAST-12 corners over a raw grayscale buffer via the naive 16-pixel ID3
//! trees from scripts/gen_fast12_scalar.py (included below); scan, border
//! and corner order match fast.c's fast12_detect(). no_std + alloc-free.

/// Detected corner (image coordinates, top-left origin).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Corner {
    pub x: usize,
    pub y: usize,
}

/// Generated light/dark trees + CIRCLE table (regenerate, do not edit).
mod trees {
    include!("fast12_trees.rs");
}

/// SIMD FAST-12 for the ESP32-S3 (EE heuristic + scalar pattern-tree confirm):
/// same corner set as the scalar fns above; host builds use a scalar mirror
/// of the SIMD lane semantics so the whole scan geometry is testable.
pub mod ee {
    include!("fast12_ee.rs");
}

/// Per-image circle offsets: off[k] = dx + dy*stride from the CIRCLE table.
fn circle_offsets(stride: usize) -> [isize; 16] {
    let mut o = [0isize; 16];
    for (i, &(dx, dy)) in trees::CIRCLE.iter().enumerate() {
        o[i] = dx as isize + dy as isize * stride as isize;
    }
    o
}

/// Exact FAST-12 rule at center index `c` (center value `v`): 12 contiguous
/// circle pixels strictly brighter than v+b, or strictly darker than v-b.
/// Shared by detect and score so both use the same predicate.
#[inline(always)]
fn corner_at(im: &[u8], c: isize, off: &[isize; 16], v: i32, b: i32) -> bool {
    trees::light(im, c, off, v + b) || trees::dark(im, c, off, v - b)
}

/// FAST-12 corners of `im` (row pitch `stride`), row-major, same interior
/// scan as fast.c (y/x in 3..size-3). Needs stride >= w and im.len() >=
/// (h-1)*stride + w. Returns the total corner count; `out` is a store cap.
pub fn fast12_detect(
    im: &[u8],
    w: usize,
    h: usize,
    stride: usize,
    b: i32,
    out: &mut [Corner],
) -> usize {
    if w < 7 || h < 7 || stride < w {
        return 0;
    }
    debug_assert!(im.len() >= (h - 1) * stride + w);
    let off = circle_offsets(stride);
    let mut n = 0;
    for y in 3..h - 3 {
        for x in 3..w - 3 {
            let c = (y * stride + x) as isize;
            let v = im[c as usize] as i32;
            if corner_at(im, c, &off, v, b) {
                if n < out.len() {
                    out[n] = Corner { x, y };
                }
                n += 1;
            }
        }
    }
    n
}

/// FAST score of one corner: the largest threshold at which it is still a corner,
/// binary-searched over [bstart, 255] exactly like fast.c's
/// fast12_corner_score() (bit-identical; `bstart` is the detector's threshold).
#[inline]
fn corner_score(im: &[u8], c: isize, off: &[isize; 16], v: i32, bstart: i32) -> i32 {
    let mut bmin = bstart;
    let mut bmax = 255i32;
    let mut b = (bmin + bmax) / 2;
    loop {
        if corner_at(im, c, off, v, b) {
            bmin = b;
        } else {
            bmax = b;
        }
        if bmin == bmax - 1 || bmin == bmax {
            return bmin;
        }
        b = (bmin + bmax) / 2;
    }
}

/// Scores a raster-ordered corner list (one i32 each), mirroring fast.c's
/// fast12_score(). `corners` must come from [`fast12_detect`] on `im` at `b`
/// (same stride); writes min(len) scores and returns how many.
pub fn fast12_score(
    im: &[u8],
    stride: usize,
    corners: &[Corner],
    b: i32,
    scores: &mut [i32],
) -> usize {
    let n = corners.len().min(scores.len());
    if n == 0 {
        return 0;
    }
    debug_assert!(im.len() >= (corners[n - 1].y * stride + corners[n - 1].x) + 3 * stride + 3);
    let off = circle_offsets(stride);
    for i in 0..n {
        let c = (corners[i].y * stride + corners[i].x) as isize;
        scores[i] = corner_score(im, c, &off, im[c as usize] as i32, b);
    }
    n
}

/// Non-max suppression, a direct port of fast.c's nonmax_suppression(): a corner
/// dies to any 3x3 neighbour of score >= its own. Raster-ordered input (as
/// [`fast12_detect`] emits); `rowidx` >= last corner y + 1, `out` is a store cap.
pub fn nonmax_suppression(
    corners: &[Corner],
    scores: &[i32],
    rowidx: &mut [usize],
    out: &mut [Corner],
) -> usize {
    let sz = corners.len();
    if sz == 0 || scores.len() < sz {
        return 0;
    }
    let last_row = corners[sz - 1].y;
    if rowidx.len() <= last_row {
        return 0;
    }
    // Row index: row_start[y] = index of the first corner on row y, or
    // usize::MAX if that row has none. Corners are raster-ordered so each
    // row's corners are contiguous and one pass suffices.
    for r in rowidx.iter_mut().take(last_row + 1) {
        *r = usize::MAX;
    }
    let mut prev_row = usize::MAX;
    for (i, c) in corners.iter().enumerate() {
        if c.y != prev_row {
            rowidx[c.y] = i;
            prev_row = c.y;
        }
    }
    // Monotonic row cursors (never reset) keep each row's window scan
    // amortized O(1); total cursor movement over the pass is O(n).
    let (mut point_above, mut point_below) = (0usize, 0usize);
    let mut num_nonmax = 0;
    'corner: for i in 0..sz {
        let score = scores[i];
        let pos = corners[i];
        let px = pos.x as isize;
        // Left: raster order => the only same-row left neighbour is i-1.
        if i > 0 {
            let l = corners[i - 1];
            if l.y == pos.y && l.x as isize == px - 1 && scores[i - 1] >= score {
                continue 'corner;
            }
        }
        // Right: the only same-row right neighbour is i+1.
        if i + 1 < sz {
            let r = corners[i + 1];
            if r.y == pos.y && r.x as isize == px + 1 && scores[i + 1] >= score {
                continue 'corner;
            }
        }
        // Above: only if row pos.y-1 exists and has corners.
        if pos.y != 0 && rowidx[pos.y - 1] != usize::MAX {
            // Snap the cursor onto the row above if it fell behind it.
            if corners[point_above].y < pos.y - 1 {
                point_above = rowidx[pos.y - 1];
            }
            // Advance past corners left of the 3-wide window (rows strictly
            // above; stops at the current row at the latest, so in-bounds).
            while corners[point_above].y < pos.y && (corners[point_above].x as isize) < px - 1 {
                point_above += 1;
            }
            let mut j = point_above;
            while corners[j].y < pos.y && corners[j].x as isize <= px + 1 {
                let x = corners[j].x as isize;
                if (x == px - 1 || x == px || x == px + 1) && scores[j] >= score {
                    continue 'corner;
                }
                j += 1;
            }
        }
        // Below: only if row pos.y+1 exists and has corners and the cursor
        // is not past the end of the list.
        if pos.y != last_row && rowidx[pos.y + 1] != usize::MAX && point_below < sz {
            if corners[point_below].y < pos.y + 1 {
                point_below = rowidx[pos.y + 1];
            }
            while point_below < sz
                && corners[point_below].y == pos.y + 1
                && (corners[point_below].x as isize) < px - 1
            {
                point_below += 1;
            }
            let mut j = point_below;
            while j < sz && corners[j].y == pos.y + 1 && corners[j].x as isize <= px + 1 {
                let x = corners[j].x as isize;
                if (x == px - 1 || x == px || x == px + 1) && scores[j] >= score {
                    continue 'corner;
                }
                j += 1;
            }
        }

        if num_nonmax < out.len() {
            out[num_nonmax] = corners[i];
        }
        num_nonmax += 1;
    }
    num_nonmax
}

/// Detect -> score -> NMS in one call (fast.c's fast12_detect_nonmax()): returns
/// the survivor count and stores raster-ordered survivors into `out` (a cap, like
/// [`fast12_detect`]). `corners` must fit the FULL raw list for exact NMS.
pub fn fast12_detect_nonmax(
    im: &[u8],
    w: usize,
    h: usize,
    stride: usize,
    b: i32,
    corners: &mut [Corner],
    scores: &mut [i32],
    rowidx: &mut [usize],
    out: &mut [Corner],
) -> usize {
    if corners.is_empty() || scores.is_empty() || rowidx.is_empty() || out.is_empty() {
        return 0;
    }
    let nraw = fast12_detect(im, w, h, stride, b, corners);
    if nraw == 0 {
        return 0;
    }
    let n = fast12_score(im, stride, &corners[..nraw.min(corners.len())], b, scores);
    nonmax_suppression(
        &corners[..n],
        &scores[..n],
        rowidx,
        out,
    )
}

// ---------------------------------------------------------------- tests ----

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64* so tests need no RNG dependency.
    struct Lcg(u64);
    impl Lcg {
        fn fill(&mut self, buf: &mut [u8]) {
            for b in buf {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.0 = x;
                *b = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u8;
            }
        }
    }

    fn rand_img(w: usize, h: usize, stride: usize, seed: u64) -> Vec<u8> {
        let mut im = vec![0u8; (h - 1) * stride + w];
        Lcg(seed).fill(&mut im);
        im
    }

    /// Independent reference: any of the 16 rotations has 12 contiguous
    /// pixels passing the polarity test (no trees), so it cross-checks the
    /// generated trees and the CIRCLE ordering, not themselves.
    fn has_run(im: &[u8], c: isize, off: &[isize; 16], gt: bool, thr: i32) -> bool {
        for r in 0..16 {
            if (0..12).all(|j| {
                let v = im[(c + off[(r + j) % 16]) as usize] as i32;
                if gt {
                    v > thr
                } else {
                    v < thr
                }
            }) {
                return true;
            }
        }
        false
    }

    fn is_corner(im: &[u8], c: isize, off: &[isize; 16], cb: i32, c_b: i32) -> bool {
        has_run(im, c, off, true, cb) || has_run(im, c, off, false, c_b)
    }

    fn naive_detect(
        im: &[u8],
        w: usize,
        h: usize,
        stride: usize,
        b: i32,
        out: &mut [Corner],
    ) -> usize {
        if w < 7 || h < 7 || stride < w {
            return 0;
        }
        let off = circle_offsets(stride);
        let mut n = 0;
        for y in 3..h - 3 {
            for x in 3..w - 3 {
                let c = (y * stride + x) as isize;
                let v = im[c as usize] as i32;
                if is_corner(im, c, &off, v + b, v - b) {
                    if n < out.len() {
                        out[n] = Corner { x, y };
                    }
                    n += 1;
                }
            }
        }
        n
    }

    /// Same scan order => a cap must truncate both detectors identically.
    fn check(
        im: &[u8],
        w: usize,
        h: usize,
        stride: usize,
        b: i32,
        cap: usize,
        label: &str,
    ) {
        let mut refs = vec![Corner { x: 0, y: 0 }; cap];
        let mut got = vec![Corner { x: 0, y: 0 }; cap];
        let na = naive_detect(im, w, h, stride, b, &mut refs);
        let nc = fast12_detect(im, w, h, stride, b, &mut got);
        assert_eq!(na, nc, "{label}: count");
        assert_eq!(refs, got, "{label}: corners"); // unused tails are both zero-filled
    }

    #[test]
    fn edge_cases() {
        let im = [0u8; 64];
        let mut out = [Corner { x: 0, y: 0 }; 16];
        assert_eq!(fast12_detect(&im, 6, 8, 6, 20, &mut out), 0); // w < 7
        assert_eq!(fast12_detect(&im, 8, 6, 8, 20, &mut out), 0); // h < 7
        assert_eq!(fast12_detect(&im, 8, 8, 4, 20, &mut out), 0); // stride < w
        assert_eq!(fast12_detect(&[0u8; 49], 7, 7, 7, 20, &mut out), 0); // 1 interior px
        let flat = vec![128u8; 640 * 480];
        assert_eq!(fast12_detect(&flat, 640, 480, 640, 40, &mut out), 0);
    }

    #[test]
    fn matches_naive_reference() {
        for (w, h, stride) in [
            (64usize, 48usize, 64usize),
            (97, 63, 97),      // odd dims
            (60, 44, 128),     // padded stride, partial last row
            (7, 7, 7),         // single interior pixel
            (33, 17, 33),
            (127, 95, 127),    // 4px checker, both polarities
        ] {
            let checker = w == 127 && h == 95;
            let im = if checker {
                let mut v = vec![0u8; w * h];
                for (i, p) in v.iter_mut().enumerate() {
                    *p = if ((i % w / 4) + (i / w / 4)) % 2 == 0 { 220 } else { 30 };
                }
                v
            } else {
                let seed = 0x9E37_79B9_7F4A_7C15 ^ ((w as u64) << 32)
                    ^ ((h as u64) << 16) ^ (stride as u64);
                rand_img(w, h, stride, seed)
            };
            for b in [8i32, 20, 40] {
                check(&im, w, h, stride, b, 8192, &format!("{w}x{h} s{stride} b{b}"));
            }
        }
    }

    #[test]
    fn small_buffer_truncates_identically() {
        let im = rand_img(96, 72, 96, 0xDEAD_BEEF);
        const CAP: usize = 5;
        check(&im, 96, 72, 96, 20, CAP, "cap 5");
        let n = fast12_detect(&im, 96, 72, 96, 20, &mut [Corner { x: 0, y: 0 }; CAP]);
        assert!(n > CAP, "need > {CAP} corners, got {n}"); // returned = total
    }

    // ---- score + non-max suppression ----

    /// Independent score reference: largest b in [bstart, 255] at which the
    /// exact 12-contiguous rule (rotation loop, not the trees) still fires.
    fn naive_score(im: &[u8], c: isize, off: &[isize; 16], bstart: i32) -> i32 {
        let v = im[c as usize] as i32;
        for b in (bstart..=255).rev() {
            if is_corner(im, c, off, v + b, v - b) {
                return b;
            }
        }
        bstart // no threshold >= bstart fires (impl returns its lower bound too)
    }

    /// Independent NMS reference: a corner dies if ANY other corner within its
    /// 3x3 neighbourhood has score >= its own (the C >= rule; no raster-order
    /// shortcuts).
    fn naive_nonmax(corners: &[Corner], scores: &[i32]) -> Vec<Corner> {
        let mut out = Vec::new();
        'i: for (i, c) in corners.iter().enumerate() {
            for (j, o) in corners.iter().enumerate() {
                if j == i {
                    continue;
                }
                let dx = o.x as isize - c.x as isize;
                let dy = o.y as isize - c.y as isize;
                if dx.abs() <= 1 && dy.abs() <= 1 && scores[j] >= scores[i] {
                    continue 'i;
                }
            }
            out.push(*c);
        }
        out
    }

    /// Detect (full list), score both ways, NMS both ways — all must agree.
    fn score_nms_case(im: &[u8], w: usize, h: usize, stride: usize, b: i32, label: &str) {
        const CAP: usize = 16384;
        let mut det = vec![Corner { x: 0, y: 0 }; CAP];
        let nraw = fast12_detect(im, w, h, stride, b, &mut det);
        assert!(nraw <= CAP, "{label}: raw {nraw} exceeds CAP");
        let det = &det[..nraw];

        let off = circle_offsets(stride);
        let mut s_impl = vec![0i32; nraw];
        assert_eq!(
            fast12_score(im, stride, det, b, &mut s_impl),
            nraw,
            "{label}: score count"
        );
        let s_naive: Vec<i32> = det
            .iter()
            .map(|c| naive_score(im, (c.y * stride + c.x) as isize, &off, b))
            .collect();
        assert_eq!(s_impl, s_naive, "{label}: scores");

        let mut rowidx = vec![usize::MAX; h]; // h >= last corner row + 1
        let mut nm = vec![Corner { x: 0, y: 0 }; CAP];
        let nnm = nonmax_suppression(det, &s_impl, &mut rowidx, &mut nm);
        let naive = naive_nonmax(det, &s_impl);
        assert_eq!(nnm, naive.len(), "{label}: nms count");
        assert_eq!(&nm[..nnm], naive.as_slice(), "{label}: nms corners");
    }

    #[test]
    fn score_and_nms_match_naive_reference() {
        for (w, h, stride) in [
            (64usize, 48usize, 64usize),
            (97, 63, 97),      // odd dims
            (60, 44, 128),     // padded stride
            (127, 95, 127),    // dense 4x4 dots, strong corners at every t
            (33, 17, 33),
        ] {
            let dots = w == 127 && h == 95;
            let im = if dots {
                // Bright 4x4 blocks on a dark grid: ~2000 corners/level.
                let mut v = vec![0u8; w * h];
                for (i, p) in v.iter_mut().enumerate() {
                    *p = if (i % w % 8) < 4 && (i / w % 8) < 4 { 200 } else { 40 };
                }
                v
            } else {
                let seed = 0x243F_6A88_85A3_08D3 ^ ((w as u64) << 32)
                    ^ ((h as u64) << 16) ^ (stride as u64);
                rand_img(w, h, stride, seed)
            };
            for b in [8i32, 20, 40] {
                score_nms_case(&im, w, h, stride, b, &format!("{w}x{h} s{stride} b{b}"));
            }
        }
    }

    #[test]
    fn nms_edge_cases() {
        let mut rowidx = [usize::MAX; 64];
        let mut out = [Corner { x: 0, y: 0 }; 64];
        // Empty input -> 0 survivors, no writes.
        assert_eq!(nonmax_suppression(&[], &[], &mut rowidx, &mut out), 0);
        // Undersized scores / rowidx -> 0.
        let c1 = [Corner { x: 10, y: 10 }];
        assert_eq!(nonmax_suppression(&c1, &[], &mut rowidx, &mut out), 0);
        let mut short = [usize::MAX; 10]; // last_row 10 needs len >= 11
        assert_eq!(nonmax_suppression(&c1, &[60], &mut short, &mut out), 0);
        // Single corner always survives.
        assert_eq!(nonmax_suppression(&c1, &[60], &mut rowidx, &mut out), 1);
        assert_eq!(out[0], c1[0]);
        // Equal adjacent scores mutually suppress (the C >= rule, verbatim).
        let row = [Corner { x: 10, y: 10 }, Corner { x: 11, y: 10 }];
        assert_eq!(nonmax_suppression(&row, &[60, 60], &mut rowidx, &mut out), 0);
        // Unequal: only the weaker dies (right neighbour >= rule).
        assert_eq!(nonmax_suppression(&row, &[60, 40], &mut rowidx, &mut out), 1);
        assert_eq!(out[0], row[0]);
        assert_eq!(nonmax_suppression(&row, &[40, 60], &mut rowidx, &mut out), 1);
        assert_eq!(out[0], row[1]);
        // Vertical and diagonal neighbours count (3-wide window, dx <= 1).
        let col = [Corner { x: 10, y: 10 }, Corner { x: 10, y: 11 }];
        assert_eq!(nonmax_suppression(&col, &[60, 60], &mut rowidx, &mut out), 0);
        let diag = [Corner { x: 10, y: 10 }, Corner { x: 11, y: 11 }];
        assert_eq!(nonmax_suppression(&diag, &[40, 60], &mut rowidx, &mut out), 1);
        assert_eq!(out[0], diag[1]);
        assert_eq!(nonmax_suppression(&diag, &[60, 60], &mut rowidx, &mut out), 0);
        // Two corners of the same row far apart both survive.
        let far = [Corner { x: 10, y: 10 }, Corner { x: 30, y: 10 }];
        assert_eq!(nonmax_suppression(&far, &[40, 60], &mut rowidx, &mut out), 2);
        // Out store is a cap: survivors past out.len() are counted, not stored.
        let mut tiny = [Corner { x: 0, y: 0 }; 1];
        assert_eq!(
            nonmax_suppression(&far, &[40, 60], &mut rowidx, &mut tiny),
            2
        );
        assert_eq!(tiny[0], far[0]);
    }

    #[test]
    fn detect_nonmax_wrapper_matches_manual() {
        for (w, h, stride) in [(96usize, 72usize, 96usize), (60, 44, 128)] {
            let im = rand_img(w, h, stride, 0xBADC_0FFE ^ ((w as u64) << 32) ^ (h as u64));
            for b in [8i32, 20, 40] {
                let label = format!("{w}x{h} s{stride} b{b}");
                const CAP: usize = 8192;
                let mut corners = vec![Corner { x: 0, y: 0 }; CAP];
                let mut scores = vec![0i32; CAP];
                let mut rowidx = vec![usize::MAX; h];
                let mut out = vec![Corner { x: 0, y: 0 }; CAP];
                let nn = fast12_detect_nonmax(
                    &im, w, h, stride, b, &mut corners, &mut scores, &mut rowidx, &mut out,
                );
                // Manual composition on the same scratch.
                let nraw = fast12_detect(&im, w, h, stride, b, &mut corners);
                assert!(nraw <= CAP, "{label}: raw {nraw}");
                let n = fast12_score(&im, stride, &corners[..nraw], b, &mut scores);
                let n2 = nonmax_suppression(&corners[..n], &scores[..n], &mut rowidx, &mut out);
                assert_eq!(nn, n2, "{label}: wrapper vs manual count");
                assert!(nn > 0, "{label}: expected survivors");
                let naive = naive_nonmax(&corners[..n], &scores[..n]);
                assert_eq!(nn, naive.len(), "{label}: vs naive count");
                assert_eq!(&out[..nn], naive.as_slice(), "{label}: vs naive corners");
            }
        }
    }

    // ---- SIMD EE variant (fast.rs::ee) vs the scalar path ----

    /// ee (raw detect and full score+NMS) must be bit-identical to the scalar
    /// path for every image/stride/threshold (host = scalar mirror of the
    /// SIMD lane semantics; xtensa = the inline-asm kernel).
    fn ee_matches_scalar_case(im: &[u8], w: usize, h: usize, stride: usize, b: i32, label: &str) {
        const CAP: usize = 8192;
        // Raw detect, both paths, same store cap -> identical lists AND totals.
        let mut a = vec![Corner { x: 0, y: 0 }; CAP];
        let mut b2 = vec![Corner { x: 0, y: 0 }; CAP];
        let na = fast12_detect(im, w, h, stride, b, &mut a);
        let nb = ee::fast12_detect_ee(im, w, h, stride, b, &mut b2);
        assert_eq!(na, nb, "{label}: ee raw count");
        assert_eq!(&a[..na.min(CAP)], &b2[..nb.min(CAP)], "{label}: ee raw corners");
        // Full pipeline: detect -> score -> NMS on both paths.
        let mut ca = vec![Corner { x: 0, y: 0 }; CAP];
        let mut sa = vec![0i32; CAP];
        let mut ra = vec![usize::MAX; h.max(1)];
        let mut oa = vec![Corner { x: 0, y: 0 }; CAP];
        let nna = fast12_detect_nonmax(im, w, h, stride, b, &mut ca, &mut sa, &mut ra, &mut oa);
        let mut cb = vec![Corner { x: 0, y: 0 }; CAP];
        let mut sb = vec![0i32; CAP];
        let mut rb = vec![usize::MAX; h.max(1)];
        let mut ob = vec![Corner { x: 0, y: 0 }; CAP];
        let nnb = ee::fast12_detect_nonmax_ee(
            im, w, h, stride, b, &mut cb, &mut sb, &mut rb, &mut ob,
        );
        assert_eq!(nna, nnb, "{label}: ee nonmax count");
        assert_eq!(&oa[..nna], &ob[..nnb], "{label}: ee nonmax corners");
    }

    #[test]
    fn ee_matches_scalar() {
        // Strides >= 16 take the SIMD path; odd/padded widths exercise the
        // right-edge strip; tiny strides take the scalar fallback.
        for (w, h, stride) in [
            (64usize, 48usize, 64usize),
            (97, 63, 97),       // odd width: tail strip + misaligned rows
            (100, 76, 100),     // width%16 = 4
            (60, 44, 128),      // padded stride, width < stride
            (127, 95, 127),     // dense 4x4 dots: many candidates + NMS
            (96, 72, 96),
            (33, 17, 33),
            (7, 7, 7),          // scalar fallback (stride < 16), single px
            (15, 15, 16),       // w < 16 but stride >= 16: strip-only SIMD
            (23, 11, 32),       // padded, h barely above the 3-radius borders
        ] {
            let dots = w == 127 && h == 95;
            let im = if dots {
                let mut v = vec![0u8; w * h];
                for (i, p) in v.iter_mut().enumerate() {
                    *p = if (i % w % 8) < 4 && (i / w % 8) < 4 { 200 } else { 40 };
                }
                v
            } else {
                let seed = 0x0F1E_2D3C_4B5A_6978 ^ ((w as u64) << 32)
                    ^ ((h as u64) << 16) ^ (stride as u64);
                rand_img(w, h, stride, seed)
            };
            for b in [8i32, 20, 40] {
                ee_matches_scalar_case(&im, w, h, stride, b, &format!("{w}x{h} s{stride} b{b}"));
            }
        }
    }

    #[test]
    fn ee_falls_back_out_of_domain() {
        // b outside [1,127] and stride < 16 must delegate to the scalar
        // detector (identical sets), never panic or read out of bounds.
        let im = rand_img(96, 72, 96, 0x5151_5A5A);
        let mut a = vec![Corner { x: 0, y: 0 }; 8192];
        let mut b = vec![Corner { x: 0, y: 0 }; 8192];
        for (bb, lab) in [(0i32, "b=0"), (128, "b=128"), (200, "b=200"), (-5, "b<0")] {
            let na = fast12_detect(&im, 96, 72, 96, bb, &mut a);
            let nb = ee::fast12_detect_ee(&im, 96, 72, 96, bb, &mut b);
            assert_eq!(na, nb, "{lab}: count");
            assert_eq!(&a[..na], &b[..nb], "{lab}: corners");
        }
        let small = rand_img(8, 8, 8, 0x0B0B);
        let na = fast12_detect(&small, 8, 8, 8, 20, &mut a);
        let nb = ee::fast12_detect_ee(&small, 8, 8, 8, 20, &mut b);
        assert_eq!(na, nb, "tiny: count");
        assert_eq!(&a[..na], &b[..nb], "tiny: corners");
    }
}
