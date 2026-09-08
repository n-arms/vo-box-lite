//! Fixed-ratio downsamplers for the pyramid:
//! - [`downscale_65`]: 6:5, byte-identical to slam-exp's `downscale_65_sse`
//!   (separable two-tap filter, weights ×128 with exact `>>7`, 6 px/rows -> 5;
//!   h-pass scratch = u8, dw-strided).
//! - [`downscale_4x4`]: INTER_AREA-style 4x4-block mean (fixed 4:1, `>>4`).
//! Both no_std, alloc-free, u16 math, no clamping; trailing cols/rows beyond
//! the last full block are never read. Caller-owned dst; dst must not alias
//! src/scratch (in-place is unsupported).

/// Weights for the first (earlier) sample of each output phase.
const W1: [u16; 5] = [107, 85, 64, 43, 21];
/// Weights for the second (later) sample; `W2[k] == W1[4-k]`.
const W2: [u16; 5] = [21, 43, 64, 85, 107];

/// Output size for one axis of the fixed 6:5 ratio: `5 * (n / 6)`.
#[inline]
pub const fn downscale_65_size(n: usize) -> usize {
    5 * (n / 6)
}

/// Two-tap weighted average of one output phase: `(W1[k]*a + W2[k]*b) >> 7`.
/// Sum ≤ 255*128 = 32640 fits u16; the plain unsigned shift reproduces the
/// C `>>7` (== `_mm_srai_epi16` on the SIMD version) exactly.
#[inline(always)]
fn phase(k: usize, a: u8, b: u8) -> u8 {
    ((W1[k] * a as u16 + W2[k] * b as u16) >> 7) as u8
}

/// Pass 1 (horizontal): src (sw) -> tmp (dw x sh), each 6-px group -> 5 px,
/// `t[5g+k] = (W1[k]*s[6g+k] + W2[k]*s[6g+k+1]) >> 7`. Sizes pre-validated.
fn hpass(src: &[u8], sw: usize, sh: usize, dw: usize, tmp: &mut [u8]) {
    let groups = dw / 5; // == sw / 6
    for y in 0..sh {
        let row = &src[y * sw..];
        let t = &mut tmp[y * dw..];
        for g in 0..groups {
            let i = 6 * g;
            let o = 5 * g;
            // k = 0..4 unrolled so the compiler folds the constant weights.
            t[o + 0] = phase(0, row[i + 0], row[i + 1]);
            t[o + 1] = phase(1, row[i + 1], row[i + 2]);
            t[o + 2] = phase(2, row[i + 2], row[i + 3]);
            t[o + 3] = phase(3, row[i + 3], row[i + 4]);
            t[o + 4] = phase(4, row[i + 4], row[i + 5]);
        }
    }
}

/// Pass 2 (vertical): tmp (strided dw) -> dst, each 6-row block -> 5 rows,
/// `dst[5b+k][x] = (W1[k]*tmp[6b+k][x] + W2[k]*tmp[6b+k+1][x]) >> 7`. Pre-validated.
fn vpass(tmp: &[u8], dw: usize, dh: usize, dst: &mut [u8]) {
    let blocks = dh / 5; // == sh / 6
    for b in 0..blocks {
        for k in 0..5 {
            let r0 = &tmp[(6 * b + k) * dw..(6 * b + k + 1) * dw];
            let r1 = &tmp[(6 * b + k + 1) * dw..(6 * b + k + 2) * dw];
            let d = &mut dst[(5 * b + k) * dw..(5 * b + k + 1) * dw];
            for (x, px) in d.iter_mut().enumerate() {
                *px = phase(k, r0[x], r1[x]);
            }
        }
    }
}

/// 6:5 downsample src (sw x sh) -> dst (dw x dh) with sizes from the fixed
/// ratio. `tmp` = caller-owned h-pass scratch (dw x sh u8, PSRAM-sized at VGA);
/// dst must not alias src/tmp. False on invalid sizes; < 6 axis: empty output.
pub fn downscale_65(
    src: &[u8],
    sw: usize,
    sh: usize,
    tmp: &mut [u8],
    dst: &mut [u8],
) -> bool {
    let dw = downscale_65_size(sw);
    let dh = downscale_65_size(sh);
    if sw == 0
        || sh == 0
        || src.len() < sw * sh
        || tmp.len() < dw * sh
        || dst.len() < dw * dh
    {
        return false;
    }
    // dw/dh == 0 (axis < 6): hpass/vpass have no groups and no-op.
    hpass(src, sw, sh, dw, tmp);
    vpass(tmp, dw, dh, dst);
    true
}

/// Output size for one axis of the fixed 4:1 ratio (4x4 block -> 1 px): `n / 4`.
#[inline]
pub const fn downscale_4x4_size(n: usize) -> usize {
    n / 4
}

/// INTER_AREA-style 4x4 downsample: `dst[y][x] = mean(src[4y..4y+4][4x..4x+4])`
/// over non-overlapping blocks — the exact-integer-ratio form of OpenCV's
/// `INTER_AREA` (dst dims = `sw/4`, `sh/4`; trailing partial rows/cols are
/// never read, like the 6:5). Truncating mean via `>> 4`: the sum of 16 u8
/// is ≤ 4080 (fits u16) and the exact shift reproduces a C `s / 16` division,
/// matching the truncation convention of the rest of the pipeline. False on
/// invalid sizes; axis < 4: valid empty output, nothing written. No scratch:
/// each output pixel reads its own 4x4 block directly from src.
pub fn downscale_4x4(src: &[u8], sw: usize, sh: usize, dst: &mut [u8]) -> bool {
    let dw = downscale_4x4_size(sw);
    let dh = downscale_4x4_size(sh);
    if sw == 0 || sh == 0 || src.len() < sw * sh || dst.len() < dw * dh {
        return false;
    }
    for y in 0..dh {
        let y0 = 4 * y;
        let r0 = &src[y0 * sw..(y0 + 1) * sw];
        let r1 = &src[(y0 + 1) * sw..(y0 + 2) * sw];
        let r2 = &src[(y0 + 2) * sw..(y0 + 3) * sw];
        let r3 = &src[(y0 + 3) * sw..(y0 + 4) * sw];
        let out = &mut dst[y * dw..(y + 1) * dw];
        for x in 0..dw {
            let i = 4 * x;
            // 16 taps, k = 0..3 unrolled per row (compiler folds the offsets).
            let s = r0[i] as u16
                + r0[i + 1] as u16
                + r0[i + 2] as u16
                + r0[i + 3] as u16
                + r1[i] as u16
                + r1[i + 1] as u16
                + r1[i + 2] as u16
                + r1[i + 3] as u16
                + r2[i] as u16
                + r2[i + 1] as u16
                + r2[i + 2] as u16
                + r2[i + 3] as u16
                + r3[i] as u16
                + r3[i + 1] as u16
                + r3[i + 2] as u16
                + r3[i + 3] as u16;
            out[x] = (s >> 4) as u8;
        }
    }
    true
}

// ---------------------------------------------------------------- tests ----

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent reference: straight from the spec formulas with plain
    /// per-pixel index math (no incremental offsets, no unrolling), so the
    /// production loop structure is cross-checked rather than self-referenced.
    fn naive_downscale_65(src: &[u8], sw: usize, sh: usize) -> Vec<u8> {
        let dw = downscale_65_size(sw);
        let dh = downscale_65_size(sh);
        let g = sw / 6;
        let b = sh / 6;
        // Horizontal pass into a dw x sh u8 tmp (flat, stride dw).
        let mut tmp = vec![0u8; dw * sh];
        for y in 0..sh {
            for gg in 0..g {
                for k in 0..5 {
                    let a = src[y * sw + 6 * gg + k] as u32;
                    let c = src[y * sw + 6 * gg + k + 1] as u32;
                    tmp[y * dw + 5 * gg + k] = ((W1[k] as u32 * a + W2[k] as u32 * c) >> 7) as u8;
                }
            }
        }
        // Vertical pass.
        let mut out = vec![0u8; dw * dh];
        for bb in 0..b {
            for k in 0..5 {
                let y0 = 6 * bb + k;
                for x in 0..dw {
                    let a = tmp[y0 * dw + x] as u32;
                    let c = tmp[(y0 + 1) * dw + x] as u32;
                    out[(5 * bb + k) * dw + x] =
                        ((W1[k] as u32 * a + W2[k] as u32 * c) >> 7) as u8;
                }
            }
        }
        out
    }

    /// Tiny deterministic xorshift so tests need no RNG dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u8
        }
        fn fill(&mut self, buf: &mut [u8]) {
            for b in buf.iter_mut() {
                *b = self.next();
            }
        }
    }

    #[test]
    fn sizes_follow_fixed_ratio() {
        assert_eq!(downscale_65_size(640), 530); // 5 * (640/6) = 5*106
        assert_eq!(downscale_65_size(480), 400); // 5 * (480/6) = 5*80
        assert_eq!(downscale_65_size(530), 440); // 530/6 = 88 -> 440 (chain)
        assert_eq!(downscale_65_size(400), 330); // 400/6 = 66 -> 330
        assert_eq!(downscale_65_size(0), 0);
        assert_eq!(downscale_65_size(1), 0); // < 6: no group
        assert_eq!(downscale_65_size(5), 0);
        assert_eq!(downscale_65_size(6), 5);
        assert_eq!(downscale_65_size(7), 5); // trailing col never read
        assert_eq!(downscale_65_size(11), 5);
        assert_eq!(downscale_65_size(12), 10);
    }

    #[test]
    fn matches_naive_reference() {
        // Trailing columns/rows (sw % 6) vary so the never-read tails are
        // exercised, plus exact-multiple and full-VGA sizes.
        let sizes: &[(usize, usize)] = &[
            (6, 6),
            (6, 7),
            (7, 6),
            (7, 12),
            (11, 11),
            (12, 12),
            (13, 18),
            (30, 42),
            (17, 7), // dw 15, dh 5
            (35, 35),
            (64, 48),
            (640, 480), // VGA -> 530x400
        ];
        for &(sw, sh) in sizes {
            let mut rng = Lcg(0x9E37_79B9_7F4A_7C15 ^ ((sw as u64) << 32) ^ (sh as u64));
            for trial in 0..3 {
                let mut src = vec![0u8; sw * sh];
                rng.fill(&mut src);
                let dw = downscale_65_size(sw);
                let dh = downscale_65_size(sh);
                let mut tmp = vec![0xAAu8; dw * sh];
                let mut dst = vec![0xBBu8; dw * dh];
                assert!(
                    downscale_65(&src, sw, sh, &mut tmp, &mut dst),
                    "size {sw}x{sh}"
                );
                assert_eq!(
                    dst,
                    naive_downscale_65(&src, sw, sh),
                    "size {sw}x{sh} trial {trial}"
                );
            }
        }
    }

    #[test]
    fn constant_image_is_preserved_exactly() {
        // W1[k] + W2[k] == 128 for all k, so c -> (c*128)>>7 == c. Also runs
        // the max-value path (255*128 = 32640, the u16 overflow boundary).
        for c in [0u8, 1, 64, 127, 128, 137, 254, 255] {
            let (sw, sh) = (100, 60);
            let dw = downscale_65_size(sw);
            let dh = downscale_65_size(sh);
            let src = vec![c; sw * sh];
            let mut tmp = vec![0u8; dw * sh];
            let mut dst = vec![0u8; dw * dh];
            assert!(downscale_65(&src, sw, sh, &mut tmp, &mut dst));
            assert!(dst.iter().all(|&p| p == c), "constant {c} not preserved");
        }
    }

    #[test]
    fn chained_levels_stay_in_bounds_and_match() {
        // Mirror of the C harness: repeatedly downscale level 0 -> level n
        // (sizes shrink 640x480 -> 530x400 -> 440x330 -> 365x275 -> ...).
        let mut rng = Lcg(42);
        let (w0, h0) = (640, 480);
        let mut src = vec![0u8; w0 * h0];
        rng.fill(&mut src);
        let (mut sw, mut sh) = (w0, h0);
        let mut naive_img = src.clone();
        for lvl in 1..8 {
            let dw = downscale_65_size(sw);
            let dh = downscale_65_size(sh);
            if dw == 0 || dh == 0 {
                break;
            }
            let mut tmp = vec![0u8; dw * sh];
            let mut dst = vec![0u8; dw * dh];
            assert!(downscale_65(&src, sw, sh, &mut tmp, &mut dst));
            naive_img = naive_downscale_65(&naive_img, sw, sh);
            assert_eq!(dst, naive_img, "level {lvl} ({sw}x{sh} -> {dw}x{dh})");
            src = dst;
            naive_img.truncate(dw * dh);
            (sw, sh) = (dw, dh);
        }
    }

    #[test]
    fn rejects_bad_sizes_and_noops_on_small_axes() {
        let src = [0u8; 36]; // 6x6
        let mut dst = [0u8; 25]; // 5x5
        let mut tmp = [0u8; 30]; // dw*sh = 5*6
        assert!(!downscale_65(&src, 0, 6, &mut tmp, &mut dst)); // zero dims
        assert!(!downscale_65(&src, 6, 0, &mut tmp, &mut dst));
        assert!(!downscale_65(&src, 7, 6, &mut tmp, &mut dst)); // src too small
        assert!(!downscale_65(&src, 6, 6, &mut tmp, &mut [0u8; 24])); // dst too small
        assert!(!downscale_65(&src, 6, 6, &mut [0u8; 29], &mut dst)); // tmp too small
        // sw < 6: dw == 0 -> valid empty output, nothing written.
        let mut d2 = [0xFFu8; 5];
        assert!(downscale_65(&src, 5, 6, &mut tmp, &mut d2));
        assert_eq!(d2, [0xFF; 5]);
    }

    /// Independent reference: plain per-pixel index math (no row slices, no
    /// unrolling), so the production loop structure is cross-checked.
    fn naive_downscale_4x4(src: &[u8], sw: usize, sh: usize) -> Vec<u8> {
        let dw = downscale_4x4_size(sw);
        let dh = downscale_4x4_size(sh);
        let mut out = vec![0u8; dw * dh];
        for y in 0..dh {
            for x in 0..dw {
                let mut s = 0u32;
                for dy in 0..4 {
                    for dx in 0..4 {
                        s += src[(4 * y + dy) * sw + 4 * x + dx] as u32;
                    }
                }
                out[y * dw + x] = (s >> 4) as u8;
            }
        }
        out
    }

    #[test]
    fn area_sizes_follow_fixed_ratio() {
        assert_eq!(downscale_4x4_size(640), 160); // VGA -> 160
        assert_eq!(downscale_4x4_size(480), 120);
        assert_eq!(downscale_4x4_size(4), 1);
        assert_eq!(downscale_4x4_size(7), 1); // trailing col never read
        assert_eq!(downscale_4x4_size(8), 2);
        assert_eq!(downscale_4x4_size(0), 0);
        assert_eq!(downscale_4x4_size(3), 0); // < 4: no block
    }

    #[test]
    fn area_matches_naive_reference() {
        // Trailing rows/cols (dims % 4) vary so the never-read tails are
        // exercised, plus exact multiples and full-VGA size.
        let sizes: &[(usize, usize)] = &[
            (4, 4),
            (4, 7),
            (7, 4),
            (9, 9),
            (16, 12),
            (17, 18),
            (64, 48),
            (640, 480), // VGA -> 160x120
        ];
        for &(sw, sh) in sizes {
            let mut rng = Lcg(0x4D59_5DF4_D0F3_3173 ^ ((sw as u64) << 32) ^ (sh as u64));
            for trial in 0..3 {
                let mut src = vec![0u8; sw * sh];
                rng.fill(&mut src);
                let dw = downscale_4x4_size(sw);
                let dh = downscale_4x4_size(sh);
                let mut dst = vec![0xBBu8; dw * dh];
                assert!(
                    downscale_4x4(&src, sw, sh, &mut dst),
                    "size {sw}x{sh}"
                );
                assert_eq!(
                    dst,
                    naive_downscale_4x4(&src, sw, sh),
                    "size {sw}x{sh} trial {trial}"
                );
            }
        }
    }

    #[test]
    fn area_mean_truncates_not_rounds() {
        // 8x8 checkerboard (alternating 0/255): every 4x4 block holds 8 black
        // + 8 white px, mean = 2040 >> 4 == 127 (truncating, not 128). Also
        // exercises the u16 max path (16 * 255 = 4080).
        let (sw, sh) = (8, 8);
        let src: Vec<u8> = (0..sh)
            .flat_map(|y| (0..sw).map(move |x| if (x + y) % 2 == 0 { 0 } else { 255 }))
            .collect();
        let dw = downscale_4x4_size(sw); // 2
        let dh = downscale_4x4_size(sh);
        let mut dst = vec![0u8; dw * dh];
        assert!(downscale_4x4(&src, sw, sh, &mut dst));
        assert!(dst.iter().all(|&p| p == 127), "got {dst:?}");
    }

    #[test]
    fn area_rejects_bad_sizes_and_noops_on_small_axes() {
        let src = [0u8; 64]; // 8x8
        let mut dst = [0u8; 4]; // 2x2
        assert!(!downscale_4x4(&src, 0, 8, &mut dst)); // zero dims
        assert!(!downscale_4x4(&src, 8, 0, &mut dst));
        assert!(!downscale_4x4(&src, 9, 8, &mut dst)); // src too small
        assert!(!downscale_4x4(&src, 8, 8, &mut [0u8; 3])); // dst too small
        // sw < 4: dw == 0 -> valid empty output, nothing written.
        let mut d2 = [0xFFu8; 1];
        assert!(downscale_4x4(&src, 3, 8, &mut d2));
        assert_eq!(d2, [0xFF]);
    }
}
