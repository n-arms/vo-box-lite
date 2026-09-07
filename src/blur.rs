//! 5x5 box blur over contiguous grayscale frames (u8, row-major): separable
//! sliding windows (~2 loads/px, no SIMD), clamp borders, no_std + alloc-free.
//! `scratch` = one u16 row of `width` (1280 B at VGA — keep in internal SRAM).

/// Full kernel width/height (always 5).
const WINDOW: usize = 5;
/// Kernel half-width (always 2).
const RADIUS: usize = 2;

/// Round-half-up divide by 25, exact for all sums <= 6375 (max 5x5 u8 window):
/// `(s*41943 + 2^19) >> 20`. Mul-shift beats the Xtensa divide at 307k px.
#[inline(always)]
fn div25(s: u16) -> u8 {
    ((s as u32 * 41943 + (1 << 19)) >> 20) as u8
}

/// Clamped 25-tap box at pixel (`y`, `x`). Slow but only used for border rows
/// (≤ 4 per frame) and tiny frames.
fn blur_px_clamped(src: &[u8], w: usize, h: usize, y: usize, x: usize) -> u8 {
    let mut s: u32 = 0;
    let ymax = (h - 1) as i32;
    let xmax = (w - 1) as i32;
    for dy in 0..WINDOW {
        let yy = ((y as i32) - (RADIUS as i32) + (dy as i32)).clamp(0, ymax) as usize;
        let row = &src[yy * w..(yy + 1) * w];
        for dx in 0..WINDOW {
            let xx = ((x as i32) - (RADIUS as i32) + (dx as i32)).clamp(0, xmax) as usize;
            s += row[xx] as u32;
        }
    }
    div25(s as u16)
}

/// Write one whole output row (border row `y`: both axes clamped).
fn fill_row_clamped(src: &[u8], dst: &mut [u8], w: usize, h: usize, y: usize) {
    let out = &mut dst[y * w..(y + 1) * w];
    for x in 0..w {
        out[x] = blur_px_clamped(src, w, h, y, x);
    }
}

/// Seed `vcol` with the column sums of source rows 0..=4 (the window of
/// output row 2). Requires `h >= 5`.
fn vcol_init(src: &[u8], w: usize, vcol: &mut [u16]) {
    for x in 0..w {
        // Cast each byte to u16 BEFORE summing — plain u8 adds would wrap at 256.
        vcol[x] = src[x] as u16
            + src[w + x] as u16
            + src[2 * w + x] as u16
            + src[3 * w + x] as u16
            + src[4 * w + x] as u16;
    }
}

/// Slide the vertical window down one row: drop source row `drop`, add
/// source row `add`. Column sums never leave `0..=1275`, so plain u16
/// arithmetic cannot overflow.
fn vcol_advance(src: &[u8], w: usize, drop: usize, add: usize, vcol: &mut [u16]) {
    let a = &src[drop * w..(drop + 1) * w];
    let b = &src[add * w..(add + 1) * w];
    for x in 0..w {
        vcol[x] += b[x] as u16;
        vcol[x] -= a[x] as u16;
    }
}

/// Horizontal 5-tap sliding sum over the column sums `vcol` -> one output
/// row. Ends are clamped; interior slides `s += v[x+2] - v[x-3]`.
/// Requires `w >= 5`.
fn out_row_from_vcol(vcol: &[u16], w: usize, dst_row: &mut [u8]) {
    let v = vcol;
    // x = 0, 1: windows (0,0,0,1,2) and (0,0,1,2,3).
    dst_row[0] = div25(3 * v[0] + v[1] + v[2]);
    dst_row[1] = div25(2 * v[0] + v[1] + v[2] + v[3]);
    // x = 2: window 0..=4, then slide through x = w-3.
    let mut s: u32 = (v[0] + v[1] + v[2] + v[3] + v[4]) as u32;
    dst_row[2] = div25(s as u16);
    for x in 3..w - 2 {
        // Slide: drop v[x-3], add v[x+2]. s always contains v[x-3] (it is a
        // 5-window sum), so subtract FIRST — a combined `s += v[x+2] - v[x-3]`
        // underflows u32 in debug builds whenever the window shrinks.
        s -= v[x - 3] as u32;
        s += v[x + 2] as u32;
        dst_row[x] = div25(s as u16);
    }
    // x = w-2, w-1: windows (...,w-1,w-1) and (...,w-1,w-1,w-1).
    dst_row[w - 2] = div25(v[w - 4] + v[w - 3] + v[w - 2] + 2 * v[w - 1]);
    dst_row[w - 1] = div25(v[w - 3] + v[w - 2] + 3 * v[w - 1]);
}

/// 5x5 box blur (clamp borders), grayscale `src` -> `dst` (w*h bytes each;
/// src = camera PSRAM fb, dst = second PSRAM buffer). `scratch` >= width u16.
/// False on invalid sizes; success fills `dst[..w*h]`.
pub fn box_blur5x5(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    scratch: &mut [u16],
) -> bool {
    let n = width * height;
    if width == 0 || height == 0 || src.len() < n || dst.len() < n || scratch.len() < width {
        return false;
    }
    // Windows need 5 rows and 5 cols; anything smaller is all border.
    if width < WINDOW || height < WINDOW {
        for y in 0..height {
            fill_row_clamped(src, dst, width, height, y);
        }
        return true;
    }
    let vcol = &mut scratch[..width];
    // Top 2 border rows (their window clamps source row 0) ...
    for y in 0..RADIUS {
        fill_row_clamped(src, dst, width, height, y);
    }
    // ... interior rows 2..=h-3: vertical running sums + horizontal slide.
    vcol_init(src, width, vcol); // window of output row 2 = source rows 0..4
    for y in RADIUS..height - RADIUS {
        out_row_from_vcol(vcol, width, &mut dst[y * width..(y + 1) * width]);
        if y < height - 3 {
            // Next output row's window drops row y-2, adds row y+3.
            vcol_advance(src, width, y - 2, y + 3, vcol);
        }
    }
    // Bottom 2 border rows (window clamps source row h-1).
    for y in height - RADIUS..height {
        fill_row_clamped(src, dst, width, height, y);
    }
    true
}

// ---------------------------------------------------------------- tests ----

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivially independent reference (fresh clamp loops + exact division)
    /// so the sliding-window bookkeeping is cross-checked, not self-referenced.
    fn naive_box_blur(src: &[u8], w: usize, h: usize) -> Vec<u8> {
        let mut out = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut s = 0u32;
                for dy in -2i64..=2 {
                    let yy = ((y as i64 + dy).clamp(0, h as i64 - 1)) as usize;
                    for dx in -2i64..=2 {
                        let xx = ((x as i64 + dx).clamp(0, w as i64 - 1)) as usize;
                        s += src[yy * w + xx] as u32;
                    }
                }
                out[y * w + x] = ((s + 12) / 25) as u8; // round-half-up
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
    fn div25_is_round_half_up_for_all_sums() {
        for s in 0..=25 * 255 {
            assert_eq!(div25(s as u16), ((s + 12) / 25) as u8, "s={s}");
        }
    }

    #[test]
    fn matches_naive_reference() {
        let sizes: &[(usize, usize)] = &[
            (1, 1),      // degenerate both axes
            (1, 7),      // w < 5
            (7, 1),      // h < 5
            (2, 3),      // tiny
            (4, 5),      // h == 5, w < 5 -> all-border path
            (5, 5),      // exactly one interior row/col
            (5, 6),      // one interior col? w == 5, h == 6
            (6, 5),      // one interior row
            (7, 9),      // small interior
            (16, 16),    // regular
            (33, 17),    // odd
            (640, 480),  // VGA camera frame
        ];
        for &(w, h) in sizes {
            let mut rng = Lcg(0x9E37_79B9_7F4A_7C15 ^ ((w as u64) << 32) ^ (h as u64));
            for trial in 0..3 {
                let n = w * h;
                let mut src = vec![0u8; n];
                rng.fill(&mut src);
                let mut dst = vec![0xAAu8; n];
                let mut scratch = vec![0u16; w];
                assert!(
                    box_blur5x5(&src, &mut dst, w, h, &mut scratch),
                    "size {w}x{h}"
                );
                let expect = naive_box_blur(&src, w, h);
                assert_eq!(dst, expect, "size {w}x{h} trial {trial}");
            }
        }
    }

    #[test]
    fn rejects_bad_sizes() {
        let src = [0u8; 4];
        let mut dst = [0u8; 4];
        let mut scratch = [0u16; 2];
        assert!(!box_blur5x5(&src, &mut dst, 0, 2, &mut scratch)); // zero width
        assert!(!box_blur5x5(&src, &mut dst, 2, 0, &mut scratch)); // zero height
        assert!(!box_blur5x5(&src, &mut dst, 3, 2, &mut scratch)); // src too small (n=6 > 4)
        assert!(!box_blur5x5(&src, &mut dst, 2, 3, &mut scratch)); // dst too small (n=6 > 4)
        assert!(!box_blur5x5(&src, &mut dst, 2, 2, &mut [0u16; 1])); // scratch too small
        let mut ok_dst = [0u8; 4];
        assert!(box_blur5x5(&src, &mut ok_dst, 2, 2, &mut scratch)); // valid tiny frame
    }

    #[test]
    fn uniform_image_is_unchanged() {
        let w = 20;
        let h = 20;
        let src = vec![137u8; w * h];
        let mut dst = vec![0u8; w * h];
        let mut scratch = vec![0u16; w];
        assert!(box_blur5x5(&src, &mut dst, w, h, &mut scratch));
        assert_eq!(dst, src);
    }
}
