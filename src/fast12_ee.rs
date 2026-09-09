// SIMD FAST-12 for the ESP32-S3 (fast.rs::ee): vector 3-of-4-cardinal heuristic
// (16 px/lane, EE/PIE inline asm) + scalar pattern-tree confirm (the generated
// fast12_cardinal_trees.rs). Corner set is bit-identical to the scalar detector.

use super::{circle_offsets, corner_at, Corner};

mod cardinal {
    include!("fast12_cardinal_trees.rs");
}

/// 0x80 byte for the u8->signed compare flip (loaded by the kernel).
#[cfg_attr(not(target_arch = "xtensa"), allow(dead_code))]
static SIGN_BYTE: u8 = 0x80;

/// 16-byte aligned per-group scratch: slots 0..3 = the four flipped pk window
/// vectors saved by the light pass for the dark pass, slot 4 = the spilled
/// c_bx bound. 80 bytes total; every slot is 16-aligned.
#[cfg_attr(not(target_arch = "xtensa"), allow(dead_code))]
#[repr(align(16))]
struct Scratch([u8; 80]);

/// Candidate vector of one 16-lane group as four u32s (word w = lanes
/// 4w..4w+4, one byte per lane: 0xFF = heuristic candidate). The a_* args are
/// byte indices into `im` with 16 readable bytes each (caller guarantees).
fn group_words(
    im: &[u8],
    a_c: usize,
    a_p0: usize,
    a_p4: usize,
    a_p8: usize,
    a_p12: usize,
    b: u8,
    scr: &mut Scratch,
) -> [u32; 4] {
    group_words_impl(im, a_c, a_p0, a_p4, a_p8, a_p12, b, scr)
}

/// Host mirror of the xtensa kernel's lane semantics (keep in lock-step):
/// light/dark = pk > sat_u8(center+b) / pk < max(0, center-b) per cardinal,
/// candidate = any light or dark triple of the four cardinals.
#[cfg(not(target_arch = "xtensa"))]
fn group_words_impl(
    im: &[u8],
    a_c: usize,
    a_p0: usize,
    a_p4: usize,
    a_p8: usize,
    a_p12: usize,
    b: u8,
    _scr: &mut Scratch,
) -> [u32; 4] {
    let b = b as i32;
    let mut out = [0u32; 4];
    for lane in 0..16usize {
        let c = im[a_c + lane] as i32;
        let cb = (c + b).min(255);
        let c_b = (c - b).max(0);
        let pk = [
            im[a_p0 + lane] as i32,
            im[a_p4 + lane] as i32,
            im[a_p8 + lane] as i32,
            im[a_p12 + lane] as i32,
        ];
        let l = [pk[0] > cb, pk[1] > cb, pk[2] > cb, pk[3] > cb];
        let d = [pk[0] < c_b, pk[1] < c_b, pk[2] < c_b, pk[3] < c_b];
        let light = (l[0] && l[1] && l[2])
            || (l[1] && l[2] && l[3])
            || (l[2] && l[3] && l[0])
            || (l[3] && l[0] && l[1]);
        let dark = (d[0] && d[1] && d[2])
            || (d[1] && d[2] && d[3])
            || (d[2] && d[3] && d[0])
            || (d[3] && d[0] && d[1]);
        if light || dark {
            out[lane / 4] |= 0xFF << (8 * (lane % 4));
        }
    }
    out
}

/// EE kernel, one 16-lane group. The a_* args are byte indices into `im`,
/// converted here to addresses (slice base + index) for the asm loads; the
/// host mirror indexes `im` directly, so the shared seam passes indices.
#[cfg(target_arch = "xtensa")]
fn group_words_impl(
    im: &[u8],
    a_c: usize,
    a_p0: usize,
    a_p4: usize,
    a_p8: usize,
    a_p12: usize,
    b: u8,
    scr: &mut Scratch,
) -> [u32; 4] {
    use core::arch::asm;
    let base = im.as_ptr() as usize;
    let sgn_addr = &SIGN_BYTE as *const u8;
    let b_addr = &b as *const u8;
    let s = scr.0.as_mut_ptr(); // 16-aligned (repr(align(16)))
    // Slot pointers: pkx windows 0..3, then the c_bx spill.
    let (s0, s1, s2, s3, s4) =
        (s, unsafe { s.add(16) }, unsafe { s.add(32) }, unsafe { s.add(48) }, unsafe { s.add(64) });
    let (mut w0, mut w1, mut w2, mut w3) = (0u32, 0u32, 0u32, 0u32);
    unsafe {
        asm!(
            // init: constants + center bounds
            "ee.vldbc.8 q0, {sgn}",                  // q0 = {16{0x80}}
            "ee.vldbc.8 q6, {bt}",                   // q6 = {16{b}}
            "ee.ld.128.usar.ip q4, {ac}, 16",        // q4 = aligned block of center; SAR = a_c & 15
            "ee.vld.128.ip      q5, {ac}, -16",      // q5 = next aligned block
            "ee.src.q           q4, q4, q5",         // q4 = center window [a_c, a_c+16)
            "ee.xorq   q5, q4, q0",                  // q5 = center ^ 0x80
            "ee.vadds.s8 q1, q5, q6",                // q1 = cbx  (pattern of (min(255,c+b))^0x80)
            "ee.vsubs.s8 q2, q5, q6",                // q2 = c_bx (pattern of (max(0,c-b))^0x80)
            "ee.vst.128.ip q2, {s4}, 0",             // spill c_bx for the dark pass
            "ee.xorq   q3, q3, q3",                  // q3 = candidate accumulator = 0
            // light pass: flip+store each pk window, build masks
            // pk0 = cardinal (0,+3): row y+3, same x
            "ee.ld.128.usar.ip q4, {p0}, 16",
            "ee.vld.128.ip      q5, {p0}, -16",
            "ee.src.q           q4, q4, q5",
            "ee.xorq   q4, q4, q0",                  // pk0x (saved flipped: dark pass reuses)
            "ee.vst.128.ip q4, {s0}, 0",
            "ee.vcmp.gt.s8 q2, q4, q1",              // l0 = pk0x > cbx
            // pk4 = cardinal (3,0): same row, x+3
            "ee.ld.128.usar.ip q4, {p4}, 16",
            "ee.vld.128.ip      q5, {p4}, -16",
            "ee.src.q           q4, q4, q5",
            "ee.xorq   q4, q4, q0",
            "ee.vst.128.ip q4, {s1}, 0",
            "ee.vcmp.gt.s8 q6, q4, q1",              // l4
            // pk8 = cardinal (0,-3): row y-3, same x
            "ee.ld.128.usar.ip q4, {p8}, 16",
            "ee.vld.128.ip      q5, {p8}, -16",
            "ee.src.q           q4, q4, q5",
            "ee.xorq   q4, q4, q0",
            "ee.vst.128.ip q4, {s2}, 0",
            "ee.vcmp.gt.s8 q7, q4, q1",              // l8
            // T1 = l0&l4&l8 -> cand; partials so l0/l4/l8 can die
            "ee.andq   q4, q2, q6",
            "ee.andq   q4, q4, q7",
            "ee.orq    q3, q3, q4",
            "ee.andq   q4, q6, q7",                  // p48 = l4&l8 (T2 partial)
            "ee.andq   q5, q2, q6",                  // p04 = l0&l4 (T4 partial)
            "ee.andq   q2, q2, q7",                  // p08 = l0&l8 (T3 partial)
            // pk12 = cardinal (-3,0): same row, x-3
            "ee.ld.128.usar.ip q6, {p12}, 16",
            "ee.vld.128.ip      q7, {p12}, -16",
            "ee.src.q           q6, q6, q7",
            "ee.xorq   q6, q6, q0",
            "ee.vst.128.ip q6, {s3}, 0",
            "ee.vcmp.gt.s8 q7, q6, q1",              // l12
            // cand |= p48&l12 | p08&l12 | p04&l12
            "ee.andq   q4, q4, q7",
            "ee.orq    q3, q3, q4",
            "ee.andq   q2, q2, q7",
            "ee.orq    q3, q3, q2",
            "ee.andq   q5, q5, q7",
            "ee.orq    q3, q3, q5",
            // dark pass: reload flipped pk windows from scratch
            "ee.vld.128.ip q1, {s4}, 0",             // q1 = c_bx
            "ee.vld.128.ip q2, {s0}, 0",
            "ee.vcmp.lt.s8 q4, q2, q1",              // d0
            "ee.vld.128.ip q2, {s1}, 0",
            "ee.vcmp.lt.s8 q5, q2, q1",              // d4
            "ee.vld.128.ip q2, {s2}, 0",
            "ee.vcmp.lt.s8 q6, q2, q1",              // d8
            "ee.andq   q7, q4, q5",                  // T1d = d0&d4&d8
            "ee.andq   q7, q7, q6",
            "ee.orq    q3, q3, q7",
            "ee.vld.128.ip q2, {s3}, 0",
            "ee.vcmp.lt.s8 q7, q2, q1",              // d12
            "ee.andq   q0, q5, q6",                  // T2d = d4&d8&d12 (q0 free now)
            "ee.andq   q0, q0, q7",
            "ee.orq    q3, q3, q0",
            "ee.andq   q0, q6, q7",                  // T3d = d8&d12&d0
            "ee.andq   q0, q0, q4",
            "ee.orq    q3, q3, q0",
            "ee.andq   q0, q7, q4",                  // T4d = d12&d0&d4
            "ee.andq   q0, q0, q5",
            "ee.orq    q3, q3, q0",
            // extract 4 x 32-bit candidate words
            "ee.movi.32.a q3, {o0}, 0",
            "ee.movi.32.a q3, {o1}, 1",
            "ee.movi.32.a q3, {o2}, 2",
            "ee.movi.32.a q3, {o3}, 3",
            sgn = in(reg) sgn_addr,
            bt = in(reg) b_addr,
            ac = in(reg) (base + a_c),
            p0 = in(reg) (base + a_p0),
            p4 = in(reg) (base + a_p4),
            p8 = in(reg) (base + a_p8),
            p12 = in(reg) (base + a_p12),
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            s2 = in(reg) s2,
            s3 = in(reg) s3,
            s4 = in(reg) s4,
            o0 = lateout(reg) w0,
            o1 = lateout(reg) w1,
            o2 = lateout(reg) w2,
            o3 = lateout(reg) w3,
        );
    }
    [w0, w1, w2, w3]
}

/// FAST-12 corners via SIMD heuristic + scalar confirm (same contract and
/// corner set as [`super::fast12_detect`]); rows 3..h-4 SIMD, edge strips
/// and last row scalar. Domain: b in 1..=127, stride >= 16, else scalar.
pub fn fast12_detect_ee(
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
    if b < 1 || b > 127 || stride < 16 {
        // Out of domain -> exact scalar detector (identical corner set).
        return super::fast12_detect(im, w, h, stride, b, out);
    }
    debug_assert!(im.len() >= (h - 1) * stride + w);
    let off = circle_offsets(stride);
    let b_byte = b as u8;
    let mut n = 0usize;
    // Raster-order store with cap semantics (count all, store first out.len()).
    macro_rules! push {
        ($x:expr, $y:expr) => {{
            if n < out.len() {
                out[n] = Corner { x: $x, y: $y };
            }
            n += 1;
        }};
    }

    // SIMD rows: y in 3..h-4, full 16-lane groups only.
    let mut scr = Scratch([0u8; 80]);
    for y in 3..h - 4 {
        let row = y * stride;
        let mut x = 0usize;
        while x + 16 <= w {
            let c = row + x; // center window start byte index
            let words = group_words(
                im,
                c,
                c + 3 * stride, // pk0 (0,+3)
                c + 3,          // pk4 (3,0)
                c - 3 * stride, // pk8 (0,-3)
                c - 3,          // pk12 (-3,0)
                b_byte,
                &mut scr,
            );
            for (wi, word) in words.iter().enumerate() {
                if *word == 0 {
                    continue;
                }
                let base_lane = 4 * wi;
                for l in 0..4u32 {
                    if (word >> (8 * l)) & 0xFF != 0 {
                        let xl = x + base_lane + l as usize;
                        if (3..=w - 4).contains(&xl) {
                            let ci = (row + xl) as isize;
                            let v = im[ci as usize] as i32;
                            if cardinal::confirm(im, ci, &off, v + b, v - b) {
                                push!(xl, y);
                            }
                        }
                    }
                }
            }
            x += 16;
        }
        // Scalar right-edge strip past the last full group.
        let x0 = if x < 3 { 3 } else { x };
        if x0 <= w - 4 {
            for xl in x0..=w - 4 {
                let ci = (row + xl) as isize;
                let v = im[ci as usize] as i32;
                if corner_at(im, ci, &off, v, b) {
                    push!(xl, y);
                }
            }
        }
    }

    // Last valid row y = h-4 is fully scalar (see fn doc).
    if h - 4 >= 3 {
        let row = (h - 4) * stride;
        for x in 3..=w - 4 {
            let ci = (row + x) as isize;
            let v = im[ci as usize] as i32;
            if corner_at(im, ci, &off, v, b) {
                push!(x, h - 4);
            }
        }
    }
    n
}

/// SIMD detect -> score -> NMS, the analogue of [`super::fast12_detect_nonmax`]
/// (same scratch contract); corner set bit-identical to the scalar wrapper.
pub fn fast12_detect_nonmax_ee(
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
    let nraw = fast12_detect_ee(im, w, h, stride, b, corners);
    if nraw == 0 {
        return 0;
    }
    let n = super::fast12_score(im, stride, &corners[..nraw.min(corners.len())], b, scores);
    super::nonmax_suppression(&corners[..n], &scores[..n], rowidx, out)
}
