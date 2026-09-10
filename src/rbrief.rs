//! Rotation-aware BRIEF (rBRIEF): ORB intensity-centroid orientation + rotated
//! sampling of 256 learned pairs. 1:1 port of extract.c's rbrief_descriptor(),
//! no_std + alloc-free + no libm. Blur the image once (blur.rs box_blur5x5),
//! then describe per keypoint; never match all-zero descriptors.

/// Border-reject half-width 20 = pattern radius ~18.4 (max pair coord ±13) +
/// 1px rounding margin — rotated samples always stay in-bounds.
pub const HALF_BOUNDARY: usize = 20;

/// 256-bit descriptor, extract.h layout: 8 x u32, bit k of word i = pair
/// i*32+k. All-zero = invalid (border-rejected or flat patch); never matched.
pub type Descriptor = [u32; 8];

/// ORB paper's 256 learned pairs (px, py, qx, qy), copied verbatim from
/// extract.c. Pair idx at ORB_PATTERN[idx*4 .. idx*4+4].
pub static ORB_PATTERN: [i32; 256 * 4] = [
    8, -3, 9, 5,
    4, 2, 7, -12,
    -11, 9, -8, 2,
    7, -12, 12, -13,
    2, -13, 2, 12,
    1, -7, 1, 6,
    -2, -10, -2, -4,
    -13, -13, -11, -8,
    -13, -3, -12, -9,
    10, 4, 11, 9,
    -13, -8, -8, -9,
    -11, 7, -9, 12,
    7, 7, 12, 6,
    -4, -5, -3, 0,
    -13, 2, -12, -3,
    -9, 0, -7, 5,
    12, -6, 12, -1,
    -3, 6, -2, 12,
    -6, -13, -4, -8,
    11, -13, 12, -8,
    4, 7, 5, 1,
    5, -3, 10, -3,
    3, -7, 6, 12,
    -8, -7, -6, -2,
    -2, 11, -1, -10,
    -13, 12, -8, 10,
    -7, 3, -5, -3,
    -4, 2, -3, 7,
    -10, -12, -6, 11,
    5, -12, 6, -7,
    5, -6, 7, -1,
    1, 0, 4, -5,
    9, 11, 11, -13,
    4, 7, 4, 12,
    2, -1, 4, 4,
    -4, -12, -2, 7,
    -8, -5, -7, -10,
    4, 11, 9, 12,
    0, -8, 1, -13,
    -13, -2, -8, 2,
    -3, -2, -2, 3,
    -6, 9, -4, -9,
    8, 12, 10, 7,
    0, 9, 1, 3,
    7, -5, 11, -10,
    -13, -6, -11, 0,
    10, 7, 12, 1,
    -6, -3, -6, 12,
    10, -9, 12, -4,
    -13, 8, -8, -12,
    -13, 0, -8, -4,
    3, 3, 7, 8,
    5, 7, 10, -7,
    -1, 7, 1, -12,
    3, -10, 5, 6,
    2, -4, 3, -10,
    -13, 0, -13, 5,
    -13, -7, -12, 12,
    -13, 3, -11, 8,
    -7, 12, -4, 7,
    6, -10, 12, 8,
    -9, -1, -7, -6,
    -2, -5, 0, 12,
    -12, 5, -7, 5,
    3, -10, 8, -13,
    -7, -7, -4, 5,
    -3, -2, -1, -7,
    2, 9, 5, -11,
    -11, -13, -5, -13,
    -1, 6, 0, -1,
    5, -3, 5, 2,
    -4, -13, -4, 12,
    -9, -6, -9, 6,
    -12, -10, -8, -4,
    10, 2, 12, -3,
    7, 12, 12, 12,
    -7, -13, -6, 5,
    -4, 9, -3, 4,
    7, -1, 12, 2,
    -7, 6, -5, 1,
    -13, 11, -12, 5,
    -3, 7, -2, -6,
    7, -8, 12, -7,
    -13, -7, -11, -12,
    1, -3, 12, 12,
    2, -6, 3, 0,
    -4, 3, -2, -13,
    -1, -13, 1, 9,
    7, 1, 8, -6,
    1, -1, 3, 12,
    9, 1, 12, 6,
    -1, -9, -1, 3,
    -13, -13, -10, 5,
    7, 7, 10, 12,
    12, -5, 12, 9,
    6, 3, 7, 11,
    5, -13, 6, 10,
    2, -12, 2, 3,
    3, 8, 4, -6,
    2, 6, 12, -13,
    9, -12, 10, 3,
    -8, 4, -7, 9,
    -11, 12, -4, -6,
    1, 12, 2, -8,
    6, -9, 7, -4,
    2, 3, 3, -2,
    6, 3, 11, 0,
    3, -3, 8, -8,
    7, 8, 9, 3,
    -11, -5, -6, -4,
    -10, 11, -5, 10,
    -5, -8, -3, 12,
    -10, 5, -9, 0,
    8, -1, 12, -6,
    4, -6, 6, -11,
    -10, 12, -8, 7,
    4, -2, 6, 7,
    -2, 0, -2, 12,
    -5, -8, -5, 2,
    7, -6, 10, 12,
    -9, -13, -8, -8,
    -5, -13, -5, -2,
    8, -8, 9, -13,
    -9, -11, -9, 0,
    1, -8, 1, -2,
    7, -4, 9, 1,
    -2, 1, -1, -4,
    11, -6, 12, -11,
    -12, -9, -6, 4,
    3, 7, 7, 12,
    5, 5, 10, 8,
    0, -4, 2, 8,
    -9, 12, -5, -13,
    0, 7, 2, 12,
    -1, 2, 1, 7,
    5, 11, 7, -9,
    3, 5, 6, -8,
    -13, -4, -8, 9,
    -5, 9, -3, -3,
    -4, -7, -3, -12,
    6, 5, 8, 0,
    -7, 6, -6, 12,
    -13, 6, -5, -2,
    1, -10, 3, 10,
    4, 1, 8, -4,
    -2, -2, 2, -13,
    2, -12, 12, 12,
    -2, -13, 0, -6,
    4, 1, 9, 3,
    -6, -10, -3, -5,
    -3, -13, -1, 1,
    7, 5, 12, -11,
    4, -2, 5, -7,
    -13, 9, -9, -5,
    7, 1, 8, 6,
    7, -8, 7, 6,
    -7, -4, -7, 1,
    -8, 11, -7, -8,
    -13, 6, -12, -8,
    2, 4, 3, 9,
    10, -5, 12, 3,
    -6, -5, -6, 7,
    8, -3, 9, -8,
    2, -12, 2, 8,
    -11, -2, -10, 3,
    -12, -13, -7, -9,
    -11, 0, -10, -5,
    5, -3, 11, 8,
    -2, -13, -1, 12,
    -1, -8, 0, 9,
    -13, -11, -12, -5,
    -10, -2, -10, 11,
    -3, 9, -2, -13,
    2, -3, 3, 2,
    -9, -13, -4, 0,
    -4, 6, -3, -10,
    -4, 12, -2, -7,
    -6, -11, -4, 9,
    6, -3, 6, 11,
    -13, 11, -5, 5,
    11, 11, 12, 6,
    7, -5, 12, -2,
    -1, 12, 0, 7,
    -4, -8, -3, -2,
    -7, 1, -6, 7,
    -13, -12, -8, -13,
    -7, -2, -6, -8,
    -8, 5, -6, -9,
    -5, -1, -4, 5,
    -13, 7, -8, 10,
    1, 5, 5, -13,
    1, 0, 10, -13,
    9, 12, 10, -1,
    5, -8, 10, -9,
    -1, 11, 1, -13,
    -9, -3, -6, 2,
    -1, -10, 1, 12,
    -13, 1, -8, -10,
    8, -11, 10, -6,
    2, -13, 3, -6,
    7, -13, 12, -9,
    -10, -10, -5, -7,
    -10, -8, -8, -13,
    4, -6, 8, 5,
    3, 12, 8, -13,
    -4, 2, -3, -3,
    5, -13, 10, -12,
    4, -13, 5, -1,
    -9, 9, -4, 3,
    0, 3, 3, -9,
    -12, 1, -6, 1,
    3, 2, 4, -8,
    -10, -10, -10, 9,
    8, -13, 12, 12,
    -8, -12, -6, -5,
    2, 2, 3, 7,
    10, 6, 11, -8,
    6, 8, 8, -12,
    -7, 10, -6, 5,
    -3, -9, -3, 9,
    -1, -13, -1, 5,
    -3, -7, -3, 4,
    -8, -2, -8, 3,
    4, 2, 12, 12,
    2, -5, 3, 11,
    6, -9, 11, -13,
    3, -1, 7, 12,
    11, -1, 12, 4,
    -3, 0, -3, 6,
    4, -11, 4, 12,
    2, -4, 2, 1,
    -10, -6, -8, 1,
    -13, 7, -11, 1,
    -13, 12, -11, -13,
    6, 0, 11, -13,
    0, -1, 1, 4,
    -13, 3, -9, -2,
    -9, 8, -6, -3,
    -13, -6, -8, -2,
    5, -9, 8, 10,
    2, 7, 3, -9,
    -1, -6, -1, -1,
    9, 5, 11, -2,
    11, -3, 12, -8,
    3, 0, 3, 5,
    -1, 4, 0, 10,
    3, -6, 4, 5,
    -13, 0, -10, 5,
    5, 8, 12, 11,
    8, 9, 9, -6,
    7, -4, 8, -12,
    -10, 4, -10, 9,
    7, 3, 12, 4,
    9, -7, 10, -2,
    7, 0, 12, -2,
    -1, -6, 0, -11,
];

/// IC_Angle radius-15 window, per-row half-widths. Odd + negation-symmetric
/// (an even square window would not track rotation). Keep as-is.
const U_MAX: [i32; 16] = [15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3];
const HALF_PATCH: i32 = 15;

/// libm-free f32 sqrt (f32::sqrt is std-only): bit-trick guess + 3 Newton.
fn sqrt_f32(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut r = f32::from_bits((x.to_bits() + 0x3F80_0000) >> 1);
    for _ in 0..3 {
        r = 0.5 * (r + x / r);
    }
    r
}

/// C `lrintf` round-half-to-even (NOT truncation — a 1px shift flips tests on
/// sharp edges). |x| < 2^23; fraction is exact (Sterbenz), so ties are exact.
fn lrint_half_even(x: f32) -> i32 {
    let a = if x < 0.0 { -x } else { x };
    let t = a as i32; // trunc toward zero
    let f = a - t as f32; // exact fraction
    let r = if f > 0.5 || f == 0.5 && (t & 1) == 1 { t + 1 } else { t };
    if x < 0.0 { -r } else { r }
}

/// (sin, cos) of the patch orientation from the intensity centroid over the
/// circular radius-15 window. `c` = keypoint pixel index (y*w + x) in the
/// blurred image. Integer moments; f32 only from the normalization on.
fn ic_angle(im: &[u8], w: usize, c: usize) -> (f32, f32) {
    let base = c as isize;
    let sw = w as isize;
    let mut m01: i32 = 0;
    let mut m10: i32 = 0;
    for u in -HALF_PATCH..=HALF_PATCH {
        m10 += u * im[(base + u as isize) as usize] as i32; // center row, once
    }
    for v in 1..=HALF_PATCH {
        let d = U_MAX[v as usize];
        let mut v_sum: i32 = 0;
        let row = v as isize * sw;
        for u in -d..=d {
            let uu = u as isize;
            let val_plus = im[(base + uu + row) as usize] as i32;
            let val_minus = im[(base + uu - row) as usize] as i32;
            v_sum += val_plus - val_minus;
            m10 += u * (val_plus + val_minus);
        }
        m01 += v * v_sum;
    }
    let m_sqrt = sqrt_f32((m01 as f32) * (m01 as f32) + (m10 as f32) * (m10 as f32));
    if m_sqrt > 1e-6 {
        ((m01 as f32) / m_sqrt, (m10 as f32) / m_sqrt) // centroid atan2 (ORB)
    } else {
        (0.0, 1.0) // flat patch -> unrotated
    }
}

/// Pair idx -> (px, py, qx, qy).
#[inline(always)]
fn pattern_pair(idx: usize) -> (i32, i32, i32, i32) {
    let o = idx * 4;
    (
        ORB_PATTERN[o],
        ORB_PATTERN[o + 1],
        ORB_PATTERN[o + 2],
        ORB_PATTERN[o + 3],
    )
}

/// Border-checked rBRIEF orientation at (x, y) in the w x h image (stride == w);
/// None in the border band (all-zero descriptor, never matched).
pub fn rbrief_angle(im: &[u8], w: usize, h: usize, x: usize, y: usize) -> Option<(f32, f32)> {
    let ok = x >= HALF_BOUNDARY
        && y >= HALF_BOUNDARY
        && x + HALF_BOUNDARY < w
        && y + HALF_BOUNDARY < h;
    if !ok {
        return None;
    }
    debug_assert!(im.len() >= w * h, "image smaller than w*h");
    Some(ic_angle(im, w, y * w + x))
}

/// 256-pair rotated sampling for an orientation from [`rbrief_angle`]. Writes
/// all 8 words of `desc`.
pub fn rbrief_samples(
    im: &[u8],
    w: usize,
    x: usize,
    y: usize,
    sin_theta: f32,
    cos_theta: f32,
    desc: &mut Descriptor,
) {
    // Rotate a pattern offset by theta (image frame, y-down), translate to
    // the keypoint; round half-even.
    let rot = |dx: i32, dy: i32| {
        (
            lrint_half_even(cos_theta * dx as f32 - sin_theta * dy as f32) + x as i32,
            lrint_half_even(sin_theta * dx as f32 + cos_theta * dy as f32) + y as i32,
        )
    };
    for (i, word) in desc.iter_mut().enumerate() {
        let mut d: u32 = 0;
        for k in 0..32usize {
            let (px, py, qx, qy) = pattern_pair(i * 32 + k);
            let (ax, ay) = rot(px, py);
            let (bx, by) = rot(qx, qy);
            // Strict <: first point darker than second (polarity cancels in
            // Hamming matching).
            if im[ay as usize * w + ax as usize] < im[by as usize * w + bx as usize] {
                d |= 1u32 << k;
            }
        }
        *word = d;
    }
}

/// rBRIEF at (x, y) over the 5x5-box-blurred image (blur.rs); false + zeroed
/// desc in the border band. == [`rbrief_angle`] + [`rbrief_samples`] composed.
pub fn rbrief_descriptor(
    im: &[u8],
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    desc: &mut Descriptor,
) -> bool {
    *desc = [0; 8];
    let (sin_theta, cos_theta) = match rbrief_angle(im, w, h, x, y) {
        Some(a) => a,
        None => return false,
    };
    rbrief_samples(im, w, x, y, sin_theta, cos_theta, desc);
    true
}

// ---------------------------------------------------------------- tests ----

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64* so tests need no RNG dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u8 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u8
        }
        fn fill(&mut self, buf: &mut [u8]) {
            for b in buf.iter_mut() {
                *b = self.next();
            }
        }
    }

    fn rand_img(w: usize, h: usize, seed: u64) -> Vec<u8> {
        let mut im = vec![0u8; w * h];
        Lcg(seed).fill(&mut im);
        im
    }

    /// Independent reference: (v, u) disk moments + rotate-helper sampling,
    /// sharing only the private float helpers — cross-checks loops, packing
    /// and table indexing bit-for-bit.
    fn naive_descriptor(im: &[u8], w: usize, _h: usize, x: usize, y: usize) -> Descriptor {
        let mut m01: i32 = 0;
        let mut m10: i32 = 0;
        let c = (y * w + x) as isize;
        let sw = w as isize;
        for v in -HALF_PATCH..=HALF_PATCH {
            let d = if v == 0 { HALF_PATCH } else { U_MAX[v.unsigned_abs() as usize] };
            for u in -d..=d {
                let p = im[(c + u as isize + v as isize * sw) as usize] as i32;
                m10 += u * p;
                m01 += v * p;
            }
        }
        let m_sqrt = sqrt_f32((m01 as f32) * (m01 as f32) + (m10 as f32) * (m10 as f32));
        let (st, ct) = if m_sqrt > 1e-6 {
            ((m01 as f32) / m_sqrt, (m10 as f32) / m_sqrt)
        } else {
            (0.0, 1.0)
        };
        let rot = |dx: i32, dy: i32| {
            (
                lrint_half_even(ct * dx as f32 - st * dy as f32) + x as i32,
                lrint_half_even(st * dx as f32 + ct * dy as f32) + y as i32,
            )
        };
        let mut desc = [0u32; 8];
        for idx in 0..256usize {
            let (px, py, qx, qy) = pattern_pair(idx);
            let (ax, ay) = rot(px, py);
            let (bx, by) = rot(qx, qy);
            if im[ay as usize * w + ax as usize] < im[by as usize * w + bx as usize] {
                desc[idx / 32] |= 1u32 << (idx % 32);
            }
        }
        desc
    }

    #[test]
    fn pattern_table_sane() {
        assert_eq!(ORB_PATTERN.len(), 1024);
        for v in ORB_PATTERN {
            assert!((-13..=13).contains(&v), "out-of-range pattern coord {v}");
        }
        // px and qx (even indices): no single-sided bias in the learned pairs.
        let sum: i64 = ORB_PATTERN.iter().step_by(2).map(|&v| v as i64).sum();
        assert!(sum.abs() < 1024, "pattern looks biased: {sum}");
    }

    #[test]
    fn border_rejection() {
        let w = 200;
        let h = 150;
        let im = rand_img(w, h, 1);
        let mut d = [0u32; 8];
        // Interior (>= 20, < w-20) is accepted.
        for &(x, y) in &[(20usize, 20usize), (w - 21, h - 21), (100, 75), (20, 100), (100, 20)] {
            assert!(
                rbrief_descriptor(&im, w, h, x, y, &mut d),
                "interior ({x},{y}) rejected"
            );
        }
        // Border band and out-of-range are rejected with zeros.
        for &(x, y) in &[
            (19usize, 100usize),
            (100, 19),
            (w - 20, 100),
            (100, h - 20),
            (0, 0),
            (w - 1, h - 1),
            (w + 5, 10),
        ] {
            d = [0xDEAD_BEEF; 8];
            assert!(
                !rbrief_descriptor(&im, w, h, x, y, &mut d),
                "border ({x},{y}) accepted"
            );
            assert_eq!(d, [0; 8], "border descriptor not zeroed");
        }
        // Image too small to ever have a valid keypoint.
        assert!(!rbrief_descriptor(&im, 39, 39, 10, 10, &mut d));
        assert!(!rbrief_descriptor(&im, 39, 200, 10, 10, &mut d));
    }

    #[test]
    fn flat_image_descriptor_is_zero() {
        let im = vec![128u8; 640 * 480];
        let mut d = [0u32; 8];
        assert!(rbrief_descriptor(&im, 640, 480, 320, 240, &mut d));
        assert_eq!(d, [0; 8], "flat patch: no strict-< test can fire");
    }

    #[test]
    fn matches_naive_reference() {
        for (w, h, seed) in [
            (640usize, 480usize, 0x9E37_79B9_7F4A_7C15u64),
            (320, 240, 0xDEAD_BEEF),
            (97, 63, 0x0123_4567), // odd dims
            (64, 64, 7),
        ] {
            let im = rand_img(w, h, seed);
            // Interior keypoints only (both reject identically anyway).
            for y in [20usize, 21, h / 2, h - 21] {
                for x in [20usize, 21, w / 2, w - 21] {
                    let mut got = [0u32; 8];
                    assert!(rbrief_descriptor(&im, w, h, x, y, &mut got));
                    assert_eq!(got, naive_descriptor(&im, w, h, x, y), "{w}x{h} ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn deterministic_and_textured() {
        // Textured (checker-ish) patch: descriptor must be nonzero and stable.
        let w = 200;
        let h = 200;
        let mut im = vec![0u8; w * h];
        for (i, p) in im.iter_mut().enumerate() {
            *p = if ((i % w / 3) + (i / w / 3)) % 2 == 0 { 220 } else { 30 };
        }
        let mut d1 = [0u32; 8];
        let mut d2 = [0u32; 8];
        assert!(rbrief_descriptor(&im, w, h, 100, 100, &mut d1));
        assert!(rbrief_descriptor(&im, w, h, 100, 100, &mut d2));
        assert_eq!(d1, d2);
        assert!(d1.iter().any(|&w| w != 0), "textured patch gave zero descriptor");
    }
}
