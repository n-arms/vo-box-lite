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

/// Per-image circle offsets: off[k] = dx + dy*stride from the CIRCLE table.
fn circle_offsets(stride: usize) -> [isize; 16] {
    let mut o = [0isize; 16];
    for (i, &(dx, dy)) in trees::CIRCLE.iter().enumerate() {
        o[i] = dx as isize + dy as isize * stride as isize;
    }
    o
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
            if trees::light(im, c, &off, v + b) || trees::dark(im, c, &off, v - b) {
                if n < out.len() {
                    out[n] = Corner { x, y };
                }
                n += 1;
            }
        }
    }
    n
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
}
