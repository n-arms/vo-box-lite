//! Fixed 6:5 downsampler (bit-identical to the C `downscale_65_sse` scheme in
//! the sibling slam-exp pipeline). `no_std`, alloc-free, no SIMD.
//!
//! Every 6 input pixels/rows become 5 output pixels/rows via a separable
//! two-tap filter with weights scaled by 128 and exact `>>7` normalize:
//!
//! ```text
//! out[5g+k] = (W1[k]*in[6g+k] + W2[k]*in[6g+k+1]) >> 7     // horizontal
//! dst[5b+k] = (W1[k]*tmp[6b+k] + W2[k]*tmp[6b+k+1]) >> 7    // vertical
//! W1 = [107, 85, 64, 43, 21]   W2 = [21, 43, 64, 85, 107]  (W2[k] = W1[4-k])
//! ```
//!
//! Integer-only: products fit u16 (max term 255*128 = 32640), the sum is
//! `>>7`'d with no rounding/clamping term, so each pass stores fully
//! normalized u8 (`tmp` is `dw`-strided u8 — no padding needed; the C SSE
//! version's `+16` tail is only for its wide-load overreads).
//!
//! No clamping anywhere: each group taps `in[6g+k+1]` with max index
//! `6G-1 = sw-1`, and the vertical pass max row is `6B-1 = sh-1`, so all
//! reads stay in bounds by construction. Trailing input columns
//! (`sw - 6*(sw/6)`, 0..=5 of them) and trailing bottom rows are never read.
//! `k` never reaches 5 — one group of 6 inputs yields exactly 5 outputs.
//!
//! Per group of 6 inputs the filter reads sample k (weight W1[k]) and
//! sample k+1 (weight W2[k]) — a two-tap interpolator whose weights encode
//! the 5 output phases between the 6 input phases. Since W1[k]+W2[k] = 128
//! for every k, a constant input is preserved exactly.

/// 5-tap weights for the first (earlier) sample of each output phase.
pub const W1: [u16; 5] = [107, 85, 64, 43, 21];
/// 5-tap weights for the second (later) sample; `W2[k] == W1[4-k]`.
pub const W2: [u16; 5] = [21, 43, 64, 85, 107];

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

/// Pass 1: horizontal 6:5 over every source row into `tmp` (strided `dw`,
/// `sh` rows). Each group of 6 input pixels `s[6g..6g+6)` produces the 5
/// output pixels `t[5g+k] = (W1[k]*s[k] + W2[k]*s[k+1]) >> 7`.
///
/// Requires `sw >= 6`, `sh >= 1`, `tmp.len() >= dw*sh` (unchecked — caller
/// validates once in [`downscale_65`]).
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

/// Pass 2: vertical 6:5 over `tmp` (strided `dw`, `sh` rows) into `dst`.
/// Each block of 6 input rows produces 5 output rows
/// `dst[5b+k][x] = (W1[k]*tmp[6b+k][x] + W2[k]*tmp[6b+k+1][x]) >> 7`.
///
/// Requires `dh >= 5` (caller validated), `dst.len() >= dw*dh` (unchecked).
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

/// Fixed 6:5 downsample: `dst` (dw x dh) = `src` (sw x sh) at 5/6 scale,
/// where `dw = downscale_65_size(sw)`, `dh = downscale_65_size(sh)`.
///
/// `tmp` is the horizontal-pass scratch, `dw`-strided over `sh` rows:
/// `downscale_65_size(sw) * sh` bytes (254,400 at VGA 640x480 -> 530x480) —
/// PSRAM-sized, so pass a caller-owned buffer (reused across frames). `dst`
/// needs `dw * dh` bytes and must NOT alias `src`/`tmp` (a fresh downscale
/// target). Output is byte-identical to the slam-exp `downscale_65_sse`.
///
/// Returns `false` on invalid sizes (buffer too small, zero dims). If either
/// axis < 6 the output is empty (dw or dh == 0) and nothing is written.
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
    if dw == 0 || dh == 0 {
        return true; // axis < 6: no 6:5 group exists, valid empty output
    }
    hpass(src, sw, sh, dw, tmp);
    vpass(tmp, dw, dh, dst);
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
}
