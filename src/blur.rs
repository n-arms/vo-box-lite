//! 5x5 box blur over contiguous grayscale frames (u8, row-major): separable
//! (vertical running column sums + horizontal 5-tap), clamped borders,
//! no_std + alloc-free. `scratch` = one u16 row of `width` (1280 B at VGA —
//! keep in internal SRAM).
//!
//! The two O(pixels) phases are vectorized with the ESP32-S3 EE/PIE SIMD
//! extension (inline asm, same construction as [`crate::fast`]'s `ee` module):
//!
//!  * [`vcol_advance`] — 16 columns/iteration: unaligned-load two source rows,
//!    widen u8 -> u16 with `ee.vzip.8` (against a zero register), then add/sub
//!    the running column sums with `ee.vadds/vsubs.s16`.
//!  * [`out_row_from_vcol`] — 8 outputs/iteration: build the five shifted
//!    `vcol` windows for the 5-tap sum from two aligned loads with
//!    `ee.srci.2q`, accumulate, then divide by 25 with one `ee.vmul.u16`
//!    (`((s + 12) * 41944) >> 20`, exact round-half-up for every reachable
//!    sum); the 8 bytes are packed with `ee.vunzip.8`.
//!
//! Host builds (`rustc --test`) and a `scratch` that is not 16-byte aligned
//! fall back to an exactly equivalent scalar/mirror path, so the output is
//! bit-identical in every case (the EE and scalar forms differ only in how the
//! same exact integer arithmetic is arranged).

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

/// The SIMD divide: `((s + 12) * 41944) >> 20` is exactly `round-half-up(s/25)`
/// for every reachable 5x5 sum `s <= 6375` (exhaustively verified in tests).
/// `ee.vmul.u16` computes the full 32-bit product, shifts it right by SAR and
/// keeps the low 16 bits, so `vldbc.16` + `wsr.sar` + `vmul.u16` divides 8 u16
/// lanes in one shot. Plain-Rust mirror + host tests; the statics must live in
/// memory because `ee.vldbc.16` loads the broadcast value from an address.
#[inline(always)]
#[cfg_attr(target_arch = "xtensa", allow(dead_code))]
fn div25_simd(s: u16) -> u8 {
    (((s as u32) + 12) * 41944 >> 20) as u8
}

#[cfg(target_arch = "xtensa")]
static SIMD_ROUND12: u16 = 12;
#[cfg(target_arch = "xtensa")]
static SIMD_INV25: u16 = 41944;
/// SAR used by `ee.vmul.u16` in the SIMD divide (the `>> 20` above).
#[cfg(target_arch = "xtensa")]
const SIMD_SAR: usize = 20;

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

// ------------------------------------------------------- SIMD block kernels ----

// `${...}` in the asm templates below are Rust operand placeholders (`q0`..`q7`
// are written literally, as in `fast::ee`: LLVM only names the 8 EE registers,
// and normal Rust codegen never uses them, so they need no clobber list).

/// Write the 8 output bytes of a SIMD block directly (no `copy_from_slice`):
/// at `opt-level="z"` that call is outlined to `index_mut` +
/// `copy_from_slice_impl`, which dominated the loop. Unrolled raw stores keep
/// it to plain `s8i` (the `bytes` array + loop let LLVM call `memcpy` instead).
/// `x + 8` must be in bounds (caller guarantees `x + 7 <= w - 3`).
#[inline(always)]
fn store8(dst_row: &mut [u8], x: usize, words: [u32; 2]) {
    let p = dst_row.as_mut_ptr();
    unsafe {
        *p.wrapping_add(x) = words[0] as u8;
        *p.wrapping_add(x + 1) = (words[0] >> 8) as u8;
        *p.wrapping_add(x + 2) = (words[0] >> 16) as u8;
        *p.wrapping_add(x + 3) = (words[0] >> 24) as u8;
        *p.wrapping_add(x + 4) = words[1] as u8;
        *p.wrapping_add(x + 5) = (words[1] >> 8) as u8;
        *p.wrapping_add(x + 6) = (words[1] >> 16) as u8;
        *p.wrapping_add(x + 7) = (words[1] >> 24) as u8;
    }
}

/// One 8-output horizontal block (outputs `x..x+8` of an interior row):
/// `sum[i] = v[i-2]+v[i-1]+v[i]+v[i+1]+v[i+2]` over `vcol`, divided by 25 and
/// packed into two little-endian u32 words.
///
/// Preconditions (asserted by the caller): `vcol` 16-byte aligned,
/// `2 <= x`, `x + 14 <= vcol.len()`, `x ≡ 2 (mod 8)` (so both loads align).
#[cfg(target_arch = "xtensa")]
#[inline(always)]
fn simd_out_block8(vcol: &[u16], x: usize) -> [u32; 2] {
    use core::arch::asm;
    // `wrapping_add`: the plain `add` carries a `debug_assert` precondition
    // check that LLVM turns into an unconditional call in the dev build.
    let a = vcol.as_ptr().wrapping_add(x - 2) as usize;
    let b = vcol.as_ptr().wrapping_add(x + 6) as usize;
    let c12 = &SIMD_ROUND12 as *const u16 as usize;
    let cm = &SIMD_INV25 as *const u16 as usize;
    let sar = SIMD_SAR;
    let (mut w0, mut w1) = (0u32, 0u32);
    unsafe {
        asm!(
            // q0/q2 = the two aligned 8-u16 windows around the block.
            "ee.vld.128.ip q0, {a}, 0",
            "ee.vld.128.ip q2, {b}, 0",
            "ee.vldbc.16   q3, {c12}",       // q3 = {12}
            "ee.vldbc.16   q7, {cm}",        // q7 = {41944}
            "ee.orq        q5, q0, q0",      // q5 = copy of A (srci.2q shifts in place)
            "ee.orq        q6, q2, q2",      // q6 = copy of B
            "ee.vadds.s16  q4, q0, q3",      // acc = vcol[x-2..] + 12  (V0 + round)
            // Each `srci.2q q6,q5,1` shifts the 32-byte {B:A} right by 2 bytes,
            // yielding V1..V4 (the windows starting at x-1, x, x+1, x+2).
            "ee.srci.2q    q6, q5, 1",
            "ee.vadds.s16  q4, q4, q5",
            "ee.srci.2q    q6, q5, 1",
            "ee.vadds.s16  q4, q4, q5",
            "ee.srci.2q    q6, q5, 1",
            "ee.vadds.s16  q4, q4, q5",
            "ee.srci.2q    q6, q5, 1",
            "ee.vadds.s16  q4, q4, q5",
            "wsr.sar       {sar}",           // ee.vmul.u16 shifts its product by SAR
            "ee.vmul.u16   q4, q4, q7",      // q4 = (sum+12)*41944 >> 20  == /25
            "ee.vunzip.8   q4, q3",          // low 8 bytes = the 8 results
            "ee.movi.32.a  q4, {w0}, 0",
            "ee.movi.32.a  q4, {w1}, 1",
            a = in(reg) a,
            b = in(reg) b,
            c12 = in(reg) c12,
            cm = in(reg) cm,
            sar = in(reg) sar,
            w0 = lateout(reg) w0,
            w1 = lateout(reg) w1,
            options(nostack, readonly),
        );
    }
    [w0, w1]
}

/// Host mirror of [`simd_out_block8`] (same exact arithmetic, no asm).
#[cfg(not(target_arch = "xtensa"))]
#[inline(always)]
fn simd_out_block8(vcol: &[u16], x: usize) -> [u32; 2] {
    let mut bytes = [0u8; 8];
    for (i, byte) in bytes.iter_mut().enumerate() {
        let s = vcol[x - 2 + i] as u32
            + vcol[x - 1 + i] as u32
            + vcol[x + i] as u32
            + vcol[x + 1 + i] as u32
            + vcol[x + 2 + i] as u32;
        *byte = div25_simd(s as u16);
    }
    [
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
    ]
}

/// One 16-column vertical block: `vcol[x+i] += add[x+i] - drop[x+i]` for `i`
/// in `0..16`. `add`/`drop` are the source rows (any alignment);
/// `&vcol[x..x+16]` must be 16-byte aligned. Taking the full slices + offset
/// (rather than `&mut vcol[x..x+16]`) keeps `index_mut` out of the loop.
#[cfg(target_arch = "xtensa")]
#[inline(always)]
fn simd_vcol_block16(vcol: &mut [u16], x: usize, add: &[u8], drop: &[u8]) {
    use core::arch::asm;
    let ap = add.as_ptr().wrapping_add(x) as usize;
    let dp = drop.as_ptr().wrapping_add(x) as usize;
    let vl = vcol.as_ptr().wrapping_add(x) as usize; // 16-byte aligned
    let vh = vl + 16;
    unsafe {
        asm!(
            // Unaligned 16-byte window of each source row: aligned block +
            // `src.q` selects [ptr, ptr+16) using the SAR set by `usar`.
            "ee.ld.128.usar.ip q1, {ap}, 16",
            "ee.vld.128.ip      q7, {ap}, -16",
            "ee.src.q           q1, q1, q7",
            "ee.ld.128.usar.ip q4, {dp}, 16",
            "ee.vld.128.ip      q7, {dp}, -16",
            "ee.src.q           q4, q4, q7",
            // Widen 16 x u8 -> 16 x u16 with a zero register (vzip.8 with a
            // zero operand interleaves the byte lanes into u16 lanes).
            "ee.zero.q          q0",
            "ee.vzip.8          q1, q0",     // q1 = add[0..8] (u16), q0 = add[8..16]
            "ee.zero.q          q7",
            "ee.vzip.8          q4, q7",     // q4 = drop[0..8], q7 = drop[8..16]
            // Running sums: values stay in 0..=1530, far from s16 saturation.
            "ee.vld.128.ip q5, {vl}, 0",
            "ee.vld.128.ip q6, {vh}, 0",
            "ee.vadds.s16 q5, q5, q1",
            "ee.vsubs.s16 q5, q5, q4",
            "ee.vadds.s16 q6, q6, q0",
            "ee.vsubs.s16 q6, q6, q7",
            "ee.vst.128.ip q5, {vl}, 0",
            "ee.vst.128.ip q6, {vh}, 0",
            ap = in(reg) ap,
            dp = in(reg) dp,
            vl = in(reg) vl,
            vh = in(reg) vh,
            options(nostack),
        );
    }
}

/// Host mirror of [`simd_vcol_block16`]. `vadds`/`vsubs.s16` cannot saturate
/// here (sums <= 1275, and `vcol` always still contains the dropped row), so
/// wrapping u16 arithmetic is exact.
#[cfg(not(target_arch = "xtensa"))]
#[inline(always)]
fn simd_vcol_block16(vcol: &mut [u16], x: usize, add: &[u8], drop: &[u8]) {
    for i in 0..16 {
        vcol[x + i] = vcol[x + i]
            .wrapping_add(add[x + i] as u16)
            .wrapping_sub(drop[x + i] as u16);
    }
}

/// Whether the EE path is usable for a `vcol` with this base pointer. The EE
/// kernel needs the 16-byte-aligned `vld`/`vst`; host builds can ignore it
/// (the mirror has no alignment requirement).
#[inline(always)]
fn simd_aligned(vcol: *const u16) -> bool {
    #[cfg(target_arch = "xtensa")]
    {
        (vcol as usize) & 15 == 0
    }
    #[cfg(not(target_arch = "xtensa"))]
    {
        let _ = vcol;
        true
    }
}

/// Slide the vertical window down one row: drop source row `drop`, add
/// source row `add`. Column sums never leave `0..=1275`, so plain u16
/// arithmetic cannot overflow. EE path in 16-column blocks, scalar tail.
fn vcol_advance(src: &[u8], w: usize, drop: usize, add: usize, vcol: &mut [u16]) {
    let a = &src[drop * w..(drop + 1) * w];
    let b = &src[add * w..(add + 1) * w];
    let mut x = 0usize;
    if simd_aligned(vcol.as_ptr()) {
        while x + 16 <= w {
            // The EE load reads the two 16-byte aligned blocks covering the
            // source window: the first can start up to 15 bytes before it and
            // the second up to 15 bytes past it. Skip the block if either
            // leaves `src` (first row / last row only).
            #[cfg(target_arch = "xtensa")]
            {
                let start = src.as_ptr() as usize;
                let end = start + src.len();
                let pa = b.as_ptr() as usize + x;
                let pd = a.as_ptr() as usize + x;
                if (pa & !15) < start
                    || (pa & !15) + 32 > end
                    || (pd & !15) < start
                    || (pd & !15) + 32 > end
                {
                    break;
                }
            }
            simd_vcol_block16(vcol, x, b, a);
            x += 16;
        }
    }
    for xi in x..w {
        vcol[xi] += b[xi] as u16;
        vcol[xi] -= a[xi] as u16;
    }
}

/// Horizontal 5-tap sum over the column sums `vcol` -> one output row. Ends
/// are clamped (scalar); the interior runs the EE 8-output blocks.
/// Requires `w >= 5`.
fn out_row_from_vcol(vcol: &[u16], w: usize, dst_row: &mut [u8]) {
    let v = vcol;
    // x = 0, 1: windows (0,0,0,1,2) and (0,0,1,2,3).
    dst_row[0] = div25(3 * v[0] + v[1] + v[2]);
    dst_row[1] = div25(2 * v[0] + v[1] + v[2] + v[3]);
    // Interior: EE blocks while a full block plus its trailing context is
    // available, scalar for the remainder.
    let mut x = 2usize;
    if simd_aligned(vcol.as_ptr()) {
        while x + 14 <= w {
            let words = simd_out_block8(vcol, x);
            store8(dst_row, x, words);
            x += 8;
        }
    }
    while x <= w - 3 {
        // Direct 5-window (the sliding form is no longer tracked here).
        let s = v[x - 2] as u32
            + v[x - 1] as u32
            + v[x] as u32
            + v[x + 1] as u32
            + v[x + 2] as u32;
        dst_row[x] = div25(s as u16);
        x += 1;
    }
    // x = w-2, w-1: windows (...,w-1,w-1) and (...,w-1,w-1,w-1).
    dst_row[w - 2] = div25(v[w - 4] + v[w - 3] + v[w - 2] + 2 * v[w - 1]);
    dst_row[w - 1] = div25(v[w - 3] + v[w - 2] + 3 * v[w - 1]);
}

/// 5x5 box blur (clamp borders), grayscale `src` -> `dst` (w*h bytes each;
/// src = camera PSRAM fb, dst = second PSRAM buffer). `scratch` >= width u16
/// (for the EE path, also 16-byte aligned). False on invalid sizes; success
/// fills `dst[..w*h]`.
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
    fn simd_div25_matches_div25_for_all_sums() {
        for s in 0..=25 * 255 {
            assert_eq!(
                div25_simd(s as u16),
                div25(s as u16),
                "s={s} (the EE vmul.u16 divide must be exact)"
            );
        }
    }

    /// The EE block builds the five shifted windows from two aligned loads via
    /// `srci.2q`; check it equals the direct 5-tap sum on random interiors.
    #[test]
    fn simd_out_block8_matches_direct_windows() {
        let mut rng = Lcg(0x1234_5678_9abc_def0);
        for w in [16usize, 17, 23, 32, 33, 640] {
            let mut vcol = vec![0u16; w];
            for _ in 0..4 {
                for v in vcol.iter_mut() {
                    *v = rng.next() as u16 * 5; // <= 1275
                }
                let mut x = 2usize;
                while x + 14 <= w {
                    let words = simd_out_block8(&vcol, x);
                    let bytes: Vec<u8> = words
                        .iter()
                        .flat_map(|word| word.to_le_bytes())
                        .collect();
                    for i in 0..8 {
                        let s = vcol[x - 2 + i] as u32
                            + vcol[x - 1 + i] as u32
                            + vcol[x + i] as u32
                            + vcol[x + 1 + i] as u32
                            + vcol[x + 2 + i] as u32;
                        assert_eq!(bytes[i], div25(s as u16), "w={w} x={x} i={i}");
                    }
                    x += 8;
                }
            }
        }
    }

    /// 16-byte-aligned scratch so the EE-dispatch branch is the one tested
    /// (host mirrors have no alignment requirement, but production does).
    fn aligned_scratch(w: usize) -> (Vec<u16>, usize) {
        let buf = vec![0u16; w + 8];
        let base = buf.as_ptr() as usize;
        let off = ((16 - (base & 15)) & 15) / 2;
        (buf, off)
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
            (16, 16),    // first full EE blocks
            (17, 16),    // odd width
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
                let (mut scratch, off) = aligned_scratch(w);
                let vcol = &mut scratch[off..off + w];
                assert!(
                    box_blur5x5(&src, &mut dst, w, h, vcol),
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
        let (mut scratch, off) = aligned_scratch(w);
        assert!(box_blur5x5(&src, &mut dst, w, h, &mut scratch[off..off + w]));
        assert_eq!(dst, src);
    }
}
