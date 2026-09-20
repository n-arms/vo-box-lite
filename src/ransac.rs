//! Linear-DLT PnP + fixed-iteration RANSAC for the ESP32-S3: f32, no_std,
//! alloc-free, no deps. f32::sqrt etc. live in std, not core — hence the
//! local `sqrt_f32`.

pub const RANSAC_ITERATIONS: u32 = 150;
pub const RANSAC_SAMPLE_SIZE: usize = 8;
pub const RANSAC_REPROJ_THRESHOLD_PX: f32 = 4.0;
pub const MIN_PNP_INLIERS: usize = 8;
pub const MAX_SAMPLE_SIZE: usize = 12; // stack buffers sized for this
const N_ITER_INV: usize = 24; // inverse-power-iteration steps per DLT solve

// ---------------------------------------------------------------- RNG -----

pub trait Rng {
    fn next_u32(&mut self) -> u32;
}

/// xorshift64*; zero seeds are remapped so the register never stalls.
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    pub const fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }
}

impl Rng for Xorshift64 {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }
}

// ------------------------------------------------------------ camera -----

/// COLMAP SIMPLE_RADIAL intrinsics: fx, fy, cx, cy and one k1.
#[derive(Clone, Copy)]
pub struct Camera {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub k1: f32,
}

impl Camera {
    /// Pixel -> undistorted normalized coords: Newton-inverts
    /// `x_d = x_u (1 + k1 r_u^2)` on the radius. Call once per matched
    /// keypoint, before RANSAC.
    pub fn pixel_to_undistorted_normalized(&self, u: f32, v: f32) -> [f32; 2] {
        let xd = (u - self.cx) / self.fx;
        let yd = (v - self.cy) / self.fy;
        let rd2 = xd * xd + yd * yd;
        if rd2 < 1e-20 {
            return [xd, yd];
        }
        let rd = sqrt_f32(rd2);
        let mut r = rd;
        for _ in 0..8 {
            let r2 = r * r;
            let fp = 1.0 + 3.0 * self.k1 * r2;
            if fp.abs() < 1e-12 {
                break;
            }
            r -= (r * (1.0 + self.k1 * r2) - rd) / fp;
            if !(r > 0.0) || !r.is_finite() {
                r = rd;
                break;
            }
        }
        let s = r / rd; // radial direction is preserved
        [xd * s, yd * s]
    }
}

// ------------------------------------------------------------- types ------

/// One match: 3D map point + undistorted normalized image coords.
#[derive(Clone, Copy)]
pub struct Correspondence {
    pub world: [f32; 3],
    pub xn: f32,
    pub yn: f32,
}

/// RANSAC knobs. sample_size is clamped into 6..=12; defaults are the consts.
#[derive(Clone, Copy)]
pub struct PnpOptions {
    pub iterations: u32,
    pub sample_size: usize,
    pub reproj_threshold_px: f32,
    pub min_inliers: usize,
}

impl Default for PnpOptions {
    fn default() -> Self {
        Self {
            iterations: RANSAC_ITERATIONS,
            sample_size: RANSAC_SAMPLE_SIZE,
            reproj_threshold_px: RANSAC_REPROJ_THRESHOLD_PX,
            min_inliers: MIN_PNP_INLIERS,
        }
    }
}

/// World-to-camera pose (`X_cam = R X_world + t`) + inlier statistics.
#[derive(Clone, Copy)]
pub struct PnpResult {
    pub r: [[f32; 3]; 3],
    pub t: [f32; 3],
    pub inlier_count: usize,
    pub mean_reproj_error_px: f32,
}

/// Best RANSAC hypothesis stats, filled by [`pnp_ransac_best`] even when the
/// pose is rejected (0 = no usable hypothesis at all).
#[derive(Clone, Copy, Default)]
pub struct PnpBest {
    pub inlier_count: usize,
    pub mean_reproj_error_px: f32,
}

#[derive(Clone, Copy)]
struct Pose {
    r: [[f32; 3]; 3],
    t: [f32; 3],
}

// ----------------------------------------------------------- helpers ------

/// libm-free f32 sqrt: bit-trick guess + 3 Newton steps (essentially the same as Quake III inverse sqrt).
fn sqrt_f32(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if x.is_infinite() {
        return x;
    }
    let mut r = f32::from_bits((x.to_bits() + 0x3F80_0000) >> 1);
    for _ in 0..3 {
        r = 0.5 * (r + x / r);
    }
    r
}

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn norm3(a: [f32; 3]) -> f32 {
    sqrt_f32(dot3(a, a))
}

#[inline]
fn rot_vec(r: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [dot3(r[0], v), dot3(r[1], v), dot3(r[2], v)]
}

#[inline]
fn det3(m: &[[f32; 3]; 3]) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// m += a a^T (upper triangle only), m is 12x12.
fn add_outer(m: &mut [f32; 144], a: &[f32; 12]) {
    for j in 0..12 {
        for i in 0..=j {
            m[i * 12 + j] += a[i] * a[j];
        }
    }
}

/// In-place Cholesky of m (lower triangle -> L, m = L L^T). False if not PD.
fn cholesky12(m: &mut [f32; 144]) -> bool {
    for i in 0..12 {
        for j in 0..=i {
            let mut s = m[i * 12 + j];
            for k in 0..j {
                s -= m[i * 12 + k] * m[j * 12 + k];
            }
            if i == j {
                if !(s > 0.0) || !s.is_finite() {
                    return false;
                }
                m[i * 12 + i] = sqrt_f32(s);
            } else {
                m[i * 12 + j] = s / m[j * 12 + j];
            }
        }
    }
    true
}

// ---------------------------------------------------------------- DLT -----

/// DLT `A p = 0` solved as the min eigenvector of A^T A (inverse power
/// iteration over a Cholesky factor). Hartley-normalizes per sample: raw f32
/// A^T A squares A's condition number and loses the nullspace of exact data.
fn dlt_pose(corrs: &[Correspondence], idx: &[usize], rng: &mut impl Rng) -> Option<Pose> {
    let k = idx.len() as f32;
    let mut cw = [0f32; 3]; // world centroid
    let mut ci = [0f32; 2]; // image centroid
    for &i in idx {
        let c = &corrs[i];
        cw[0] += c.world[0];
        cw[1] += c.world[1];
        cw[2] += c.world[2];
        ci[0] += c.xn;
        ci[1] += c.yn;
    }
    cw[0] /= k;
    cw[1] /= k;
    cw[2] /= k;
    ci[0] /= k;
    ci[1] /= k;
    let (mut mw, mut mi) = (0f32, 0f32);
    for &i in idx {
        let c = &corrs[i];
        let d = [c.world[0] - cw[0], c.world[1] - cw[1], c.world[2] - cw[2]];
        let e = [c.xn - ci[0], c.yn - ci[1]];
        mw += d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        mi += e[0] * e[0] + e[1] * e[1];
    }
    mw /= k;
    mi /= k;
    if !(mw > 0.0) || !(mi > 0.0) {
        return None;
    }
    let sw = sqrt_f32(3.0 / mw); // scale so mean world dist = sqrt(3)
    let si = sqrt_f32(2.0 / mi); // scale so mean image dist = sqrt(2)

    let mut m = [0f32; 144];
    for &i in idx {
        let c = &corrs[i];
        let x = (c.world[0] - cw[0]) * sw;
        let y = (c.world[1] - cw[1]) * sw;
        let z = (c.world[2] - cw[2]) * sw;
        let xn = (c.xn - ci[0]) * si;
        let yn = (c.yn - ci[1]) * si;
        add_outer(
            &mut m,
            &[
                x,
                y,
                z,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -xn * x,
                -xn * y,
                -xn * z,
                -xn,
            ],
        );
        add_outer(
            &mut m,
            &[
                0.0,
                0.0,
                0.0,
                0.0,
                x,
                y,
                z,
                1.0,
                -yn * x,
                -yn * y,
                -yn * z,
                -yn,
            ],
        );
    }
    for j in 0..12 {
        for i in (j + 1)..12 {
            m[i * 12 + j] = m[j * 12 + i];
        }
    }
    let mut md = 0f32;
    for i in 0..12 {
        md = md.max(m[i * 13]);
    }
    let lam = 1e-6 * md; // shift: M is only PSD, the factor must be SPD
    for i in 0..12 {
        m[i * 13] += lam;
    }
    if !cholesky12(&mut m) {
        return None;
    }

    let mut v = [0f32; 12]; // random unit start vector
    let mut n2 = 0f32;
    for e in v.iter_mut() {
        *e = 2.0 * (rng.next_u32() >> 8) as f32 / (1u32 << 24) as f32 - 1.0;
        n2 += *e * *e;
    }
    if !(n2 > 0.0) {
        return None;
    }
    let inv = 1.0 / sqrt_f32(n2);
    for e in v.iter_mut() {
        *e *= inv;
    }

    let mut y = [0f32; 12];
    for _ in 0..N_ITER_INV {
        // Two triangular solves (L y = v, L^T v' = y) + renormalize.
        for i in 0..12 {
            let mut s = v[i];
            for j in 0..i {
                s -= m[i * 12 + j] * y[j];
            }
            y[i] = s / m[i * 12 + i];
        }
        for i in (0..12).rev() {
            let mut s = y[i];
            for j in (i + 1)..12 {
                s -= m[j * 12 + i] * v[j];
            }
            v[i] = s / m[i * 12 + i];
        }
        let mut n2 = 0f32;
        for e in v.iter() {
            n2 += *e * *e;
        }
        let inv = 1.0 / sqrt_f32(n2);
        for e in v.iter_mut() {
            *e *= inv;
        }
    }
    pose_from_p(&denormalize_p(&v, sw, cw, si, ci))
}

/// Undo the Hartley normalization: P = T_i^-1 P~ T_w (T_i scales about ci by
/// si, T_w scales about cw by sw) for the normalized 3x4 matrix P~.
fn denormalize_p(p: &[f32; 12], sw: f32, cw: [f32; 3], si: f32, ci: [f32; 2]) -> [f32; 12] {
    let sii = 1.0 / si;
    let q = [
        p[0] * sii + ci[0] * p[8],
        p[1] * sii + ci[0] * p[9],
        p[2] * sii + ci[0] * p[10],
        p[3] * sii + ci[0] * p[11],
        p[4] * sii + ci[1] * p[8],
        p[5] * sii + ci[1] * p[9],
        p[6] * sii + ci[1] * p[10],
        p[7] * sii + ci[1] * p[11],
        p[8],
        p[9],
        p[10],
        p[11],
    ];
    let w = [-sw * cw[0], -sw * cw[1], -sw * cw[2]];
    [
        q[0] * sw,
        q[1] * sw,
        q[2] * sw,
        q[0] * w[0] + q[1] * w[1] + q[2] * w[2] + q[3],
        q[4] * sw,
        q[5] * sw,
        q[6] * sw,
        q[4] * w[0] + q[5] * w[1] + q[6] * w[2] + q[7],
        q[8] * sw,
        q[9] * sw,
        q[10] * sw,
        q[8] * w[0] + q[9] * w[1] + q[10] * w[2] + q[11],
    ]
}

/// P ~ [R|t] (3x4, row-major) -> pose. Scale = mean rotation-column norm
/// (sign from det B), t = c/mu; rotation made orthonormal by GS + cross.
/// Malformed candidates (zero/bad scale, non-orthogonal columns) are dropped.
fn pose_from_p(p: &[f32; 12]) -> Option<Pose> {
    let b = [[p[0], p[1], p[2]], [p[4], p[5], p[6]], [p[8], p[9], p[10]]];
    let col = |j: usize| [b[0][j], b[1][j], b[2][j]];
    let n = [norm3(col(0)), norm3(col(1)), norm3(col(2))];
    if n.iter()
        .any(|&x| !(x > f32::MIN_POSITIVE) || !x.is_finite())
    {
        return None;
    }
    let scale = (n[0] + n[1] + n[2]) / 3.0;
    let db = det3(&b);
    if !db.is_finite() || db == 0.0 {
        return None;
    }
    let mu = if db < 0.0 { -scale } else { scale }; // sign(det B) == sign(mu)
    let mut u = [
        col(0).map(|x| x / n[0]),
        col(1).map(|x| x / n[1]),
        col(2).map(|x| x / n[2]),
    ];
    if dot3(u[0], u[1]).abs() > 0.95
        || dot3(u[0], u[2]).abs() > 0.95
        || dot3(u[1], u[2]).abs() > 0.95
    {
        return None;
    }
    if db < 0.0 {
        u[0] = u[0].map(|x| -x);
        u[1] = u[1].map(|x| -x);
    }
    let e0 = u[0];
    let pr = dot3(e0, u[1]);
    let mut e1 = [
        u[1][0] - pr * e0[0],
        u[1][1] - pr * e0[1],
        u[1][2] - pr * e0[2],
    ];
    let ne1 = norm3(e1);
    if !(ne1 > 1e-6) || !ne1.is_finite() {
        return None;
    }
    e1 = [e1[0] / ne1, e1[1] / ne1, e1[2] / ne1];
    let e2 = cross3(e0, e1); // unit: e0, e1 orthonormal
    let r = [
        [e0[0], e1[0], e2[0]],
        [e0[1], e1[1], e2[1]],
        [e0[2], e1[2], e2[2]],
    ];
    let t = [p[3] / mu, p[7] / mu, p[11] / mu];
    if !(r[0][0].is_finite()
        && r[1][1].is_finite()
        && r[2][2].is_finite()
        && t[0].is_finite()
        && t[1].is_finite()
        && t[2].is_finite())
    {
        return None;
    }
    Some(Pose { r, t })
}

// ------------------------------------------------------------ refine -----

#[inline]
fn mat_mul3(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut r = [[0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}

/// Rodrigues `exp([w]x)` with a clamped small-angle Taylor series (no libm):
/// accurate to O(t^6) for the t <= 0.5 steps LM produces.
fn exp_so3(w0: [f32; 3]) -> [[f32; 3]; 3] {
    let mut w = w0;
    let mut t2 = dot3(w, w);
    if t2 > 0.25 {
        let s = 0.5 / sqrt_f32(t2);
        w = [w[0] * s, w[1] * s, w[2] * s];
        t2 = 0.25;
    }
    let sa = 1.0 - t2 / 6.0 + t2 * t2 / 120.0; // sin(t)/t
    let ca = 0.5 - t2 / 24.0 + t2 * t2 / 720.0; // (1 - cos t)/t^2
    let wx = [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]];
    let wx2 = mat_mul3(wx, wx);
    let mut r = [[0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = (if i == j { 1.0 } else { 0.0 }) + sa * wx[i][j] + ca * wx2[i][j];
        }
    }
    r
}

/// Solve the 6x6 SPD system `a x = b` by Cholesky. None if not PD / non-finite.
fn solve6(a: &[[f32; 6]; 6], b: [f32; 6]) -> Option<[f32; 6]> {
    let mut l = [[0f32; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if !(s > 0.0) || !s.is_finite() {
                    return None;
                }
                l[i][i] = sqrt_f32(s);
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    let mut y = [0f32; 6];
    for i in 0..6 {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    let mut x = [0f32; 6];
    for i in (0..6).rev() {
        let mut s = y[i];
        for k in (i + 1)..6 {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Levenberg-Marquardt refinement of `pose` against `corrs` (optionally only
/// the entries where `use_mask[i]`), minimizing normalized reprojection error.
/// Left-multiplied so(3) increment; returns the input if no step improves.
fn refine_pose(mut pose: Pose, corrs: &[Correspondence], use_mask: Option<&[bool]>) -> Pose {
    const MAX_IT: usize = 8;
    let mut used = 0usize;
    let mut cost = 0f32;
    for (i, c) in corrs.iter().enumerate() {
        if use_mask.map_or(false, |m| !m.get(i).copied().unwrap_or(false)) {
            continue;
        }
        used += 1;
        let xc = rot_vec(&pose.r, c.world);
        let z = xc[2] + pose.t[2];
        if z <= 1e-6 {
            cost += 1e6;
            continue;
        }
        let dx = c.xn - (xc[0] + pose.t[0]) / z;
        let dy = c.yn - (xc[1] + pose.t[1]) / z;
        cost += dx * dx + dy * dy;
    }
    if used < 4 {
        return pose;
    }
    let mut lambda = 1e-3f32;
    for _ in 0..MAX_IT {
        // Normal equations of the per-point 2x6 Jacobian.
        let mut a = [[0f32; 6]; 6];
        let mut g = [0f32; 6];
        for (i, c) in corrs.iter().enumerate() {
            if use_mask.map_or(false, |m| !m.get(i).copied().unwrap_or(false)) {
                continue;
            }
            let xc = rot_vec(&pose.r, c.world);
            let x = xc[0] + pose.t[0];
            let y = xc[1] + pose.t[1];
            let z = xc[2] + pose.t[2];
            if z <= 1e-6 {
                continue;
            }
            let iz = 1.0 / z;
            let px = x * iz;
            let py = y * iz;
            // d p/d theta = (d p/d Xc)(-[Xc]x), d p/d t = d p/d Xc.
            let j0 = [
                -px * iz * xc[1],
                iz * z + px * iz * xc[0],
                -iz * xc[1],
                iz,
                0.0,
                -px * iz,
            ];
            let j1 = [
                -iz * z - py * iz * xc[1],
                py * iz * xc[0],
                iz * xc[0],
                0.0,
                iz,
                -py * iz,
            ];
            let r = [c.xn - px, c.yn - py];
            for row in 0..2 {
                let j = if row == 0 { j0 } else { j1 };
                for c0 in 0..6 {
                    g[c0] += j[c0] * r[row];
                    for c1 in 0..6 {
                        a[c0][c1] += j[c0] * j[c1];
                    }
                }
            }
        }
        let mut accepted = false;
        for _ in 0..6 {
            let mut m = a;
            let mut floor = 1e-12f32;
            for i in 0..6 {
                floor = floor.max(a[i][i].abs() * 1e-6);
            }
            for i in 0..6 {
                m[i][i] += lambda * a[i][i].abs().max(floor) + floor;
            }
            if let Some(d) = solve6(&m, g) {
                let cand = Pose {
                    r: mat_mul3(exp_so3([d[0], d[1], d[2]]), pose.r),
                    t: [pose.t[0] + d[3], pose.t[1] + d[4], pose.t[2] + d[5]],
                };
                let mut cc = 0f32;
                for (i, c) in corrs.iter().enumerate() {
                    if use_mask.map_or(false, |mm| !mm.get(i).copied().unwrap_or(false)) {
                        continue;
                    }
                    let xc = rot_vec(&cand.r, c.world);
                    let z = xc[2] + cand.t[2];
                    if z <= 1e-6 {
                        cc += 1e6;
                        continue;
                    }
                    let dx = c.xn - (xc[0] + cand.t[0]) / z;
                    let dy = c.yn - (xc[1] + cand.t[1]) / z;
                    cc += dx * dx + dy * dy;
                }
                if cc < cost {
                    pose = cand;
                    cost = cc;
                    lambda = (lambda * 0.3).max(1e-6);
                    accepted = true;
                    break;
                }
            }
            lambda = (lambda * 3.0).min(1e6);
        }
        if !accepted {
            break;
        }
    }
    pose
}

/// Inlier mask + (count, sum of squared inlier reproj error in px^2) for a pose.
fn mask_and_error(
    pose: &Pose,
    corrs: &[Correspondence],
    cam: &Camera,
    thr2: f32,
    out_mask: &mut [bool],
) -> (usize, f32) {
    let mut fcnt = 0;
    let mut ferr2 = 0f32;
    for (i, c) in corrs.iter().enumerate() {
        let xc = rot_vec(&pose.r, c.world);
        let z = xc[2] + pose.t[2];
        let mut inl = false;
        if z > 0.0 {
            let du = cam.fx * (c.xn - (xc[0] + pose.t[0]) / z);
            let dv = cam.fy * (c.yn - (xc[1] + pose.t[1]) / z);
            let d2 = du * du + dv * dv;
            inl = d2 <= thr2;
            if inl {
                fcnt += 1;
                ferr2 += d2;
            }
        }
        out_mask[i] = inl;
    }
    (fcnt, ferr2)
}

// -------------------------------------------------------------- RANSAC -----

/// Count pose inliers in undistorted pixel space (`Z_c <= 0` never counts).
/// Returns (count, sum of squared pixel errors over the inliers).
fn score_pose(pose: &Pose, corrs: &[Correspondence], cam: &Camera, thr2: f32) -> (usize, f32) {
    let mut cnt = 0;
    let mut err2 = 0f32;
    for c in corrs {
        let xc = rot_vec(&pose.r, c.world);
        let z = xc[2] + pose.t[2];
        if z <= 0.0 {
            continue;
        }
        let du = cam.fx * (c.xn - (xc[0] + pose.t[0]) / z);
        let dv = cam.fy * (c.yn - (xc[1] + pose.t[1]) / z);
        let d2 = du * du + dv * dv;
        if d2 <= thr2 {
            cnt += 1;
            err2 += d2;
        }
    }
    (cnt, err2)
}

/// Draw `k` distinct indices in `0..n` (bounded rejection sampling).
fn sample_distinct(rng: &mut impl Rng, n: usize, k: usize, out: &mut [usize]) -> bool {
    let mut tries = 0;
    for slot in 0..k {
        loop {
            let c = (rng.next_u32() as usize) % n;
            if !out[..slot].contains(&c) {
                out[slot] = c;
                break;
            }
            tries += 1;
            if tries > 200 {
                return false;
            }
        }
    }
    true
}

/// Fixed-iteration RANSAC over the linear DLT PnP. `out_mask.len() >= n`;
/// on success entry `i` flags inlier `i`. Returns `None` on < 6 matches or
/// when no hypothesis reached `opts.min_inliers`.
pub fn pnp_ransac(
    corrs: &[Correspondence],
    cam: &Camera,
    opts: &PnpOptions,
    rng: &mut impl Rng,
    out_mask: &mut [bool],
) -> Option<PnpResult> {
    let mut best = PnpBest::default();
    pnp_ransac_best(corrs, cam, opts, rng, out_mask, &mut best)
}

/// Like [`pnp_ransac`], but also reports the best hypothesis found even when
/// the pose is rejected (< `min_inliers`) — diagnostics only.
pub fn pnp_ransac_best(
    corrs: &[Correspondence],
    cam: &Camera,
    opts: &PnpOptions,
    rng: &mut impl Rng,
    out_mask: &mut [bool],
    out_best: &mut PnpBest,
) -> Option<PnpResult> {
    let n = corrs.len();
    if n < 6 || out_mask.len() < n {
        return None;
    }
    let k = opts.sample_size.clamp(6, MAX_SAMPLE_SIZE).min(n);
    let thr2 = opts.reproj_threshold_px * opts.reproj_threshold_px;

    let mut best: Option<(Pose, usize, f32)> = None; // (pose, inliers, err2)
    let mut sample = [0usize; MAX_SAMPLE_SIZE];
    for _ in 0..opts.iterations {
        if !sample_distinct(rng, n, k, &mut sample[..k]) {
            continue;
        }
        let Some(pose) = dlt_pose(corrs, &sample[..k], rng) else {
            continue;
        };
        // DLT is only a projective fit; LM-refine on the sample to project it
        // onto the rigid-pose manifold before scoring (see refine_pose).
        for b in out_mask[..n].iter_mut() {
            *b = false;
        }
        for &i in &sample[..k] {
            out_mask[i] = true;
        }
        let pose = refine_pose(pose, corrs, Some(out_mask));
        let (cnt, err2) = score_pose(&pose, corrs, cam, thr2);
        let take = match best {
            None => true,
            Some((_, bc, be)) => cnt > bc || (cnt == bc && err2 < be), // ties: mean err
        };
        if take {
            best = Some((pose, cnt, err2));
        }
    }
    if let Some((_, bc, be)) = best {
        out_best.inlier_count = bc;
        out_best.mean_reproj_error_px =
            if bc > 0 { sqrt_f32(be / bc as f32) } else { 0.0 };
    }
    let (pose, cnt, _) = best?;
    if cnt < opts.min_inliers {
        return None;
    }

    // Re-refine on the full inlier set (the RANSAC winner was only refined on
    // its sample), then write the deterministic final inlier mask.
    let (cnt0, _) = mask_and_error(&pose, corrs, cam, thr2, out_mask);
    let final_pose = if cnt0 >= 4 {
        let refined = refine_pose(pose, corrs, Some(out_mask));
        if score_pose(&refined, corrs, cam, thr2).0 >= cnt0 { refined } else { pose }
    } else {
        pose
    };
    let (fcnt, ferr2) = mask_and_error(&final_pose, corrs, cam, thr2, out_mask);
    Some(PnpResult {
        r: final_pose.r,
        t: final_pose.t,
        inlier_count: fcnt,
        mean_reproj_error_px: sqrt_f32(ferr2 / fcnt as f32),
    })
}

// ---------------------------------------------------------------- tests ----

#[cfg(test)]
mod tests {
    use super::*;

    /// Calibration used by the sibling slam-exp COLMAP runs.
    const CAM: Camera = Camera {
        fx: 267.0,
        fy: 267.0,
        cx: 320.0,
        cy: 240.0,
        k1: -0.00208,
    };

    /// Forward distortion model: undistorted normalized -> distorted pixel.
    fn distort_to_pixel(cam: &Camera, xu: f32, yu: f32) -> [f32; 2] {
        let s = 1.0 + cam.k1 * (xu * xu + yu * yu);
        [cam.fx * xu * s + cam.cx, cam.fy * yu * s + cam.cy]
    }

    fn euler(yaw: f32, pitch: f32, roll: f32) -> [[f32; 3]; 3] {
        let (sy, cy) = yaw.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        let (sr, cr) = roll.sin_cos();
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    }

    /// trace(R_a^T R_b) = elementwise dot, so the rotation angle drops out.
    fn rot_err_deg(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> f32 {
        let tr: f32 = a.iter().zip(&b).map(|(ra, rb)| dot3(*ra, *rb)).sum();
        ((tr - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees()
    }

    /// Synthetic scene: points in front of a known pose, projected through the
    /// distortion model; `outlier_frac` are replaced by random pixels.
    fn make_scene(
        rng: &mut Xorshift64,
        n: usize,
        outlier_frac: f32,
        noise_px: f32,
    ) -> (Pose, Vec<Correspondence>, usize) {
        let r_true = euler(0.55, -0.35, 0.2);
        let center = [1.0, -0.35, 0.6];
        let rc = rot_vec(&r_true, center);
        let t_true = [-rc[0], -rc[1], -rc[2]]; // t = -R C

        let u = |rng: &mut Xorshift64| (rng.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
        let mut corrs = Vec::with_capacity(n);
        let mut n_true = 0;
        for _ in 0..n {
            let depth = 2.0 + 6.0 * u(rng);
            let xn = (2.0 * u(rng) - 1.0) * 0.7;
            let yn = (2.0 * u(rng) - 1.0) * 0.7;
            let xc = [xn * depth, yn * depth, depth];
            // World point: R^T x_c + C.
            let rt = [
                [r_true[0][0], r_true[1][0], r_true[2][0]],
                [r_true[0][1], r_true[1][1], r_true[2][1]],
                [r_true[0][2], r_true[1][2], r_true[2][2]],
            ];
            let rtx = rot_vec(&rt, xc);
            let world = [rtx[0] + center[0], rtx[1] + center[1], rtx[2] + center[2]];

            let outlier = u(rng) < outlier_frac;
            let (px, py) = if outlier {
                (u(rng) * 640.0, u(rng) * 480.0)
            } else {
                n_true += 1;
                let [px, py] = distort_to_pixel(&CAM, xn, yn);
                let nz = noise_px * (2.0 * u(rng) - 1.0);
                (px + nz, py + nz)
            };
            let [ux, uy] = CAM.pixel_to_undistorted_normalized(px, py);
            corrs.push(Correspondence {
                world,
                xn: ux,
                yn: uy,
            });
        }
        (
            Pose {
                r: r_true,
                t: t_true,
            },
            corrs,
            n_true,
        )
    }

    #[test]
    fn undistort_roundtrip() {
        for &(u, v) in &[
            (0.0, 0.0),
            (639.0, 479.0),
            (320.0, 240.0),
            (10.0, 470.0),
            (100.0, 50.0),
        ] {
            let [xu, yu] = CAM.pixel_to_undistorted_normalized(u, v);
            let [uu, vv] = distort_to_pixel(&CAM, xu, yu);
            let err = ((uu - u).powi(2) + (vv - v).powi(2)).sqrt();
            assert!(err < 1e-3, "roundtrip err {err} px at ({u},{v})");
        }
    }

    #[test]
    fn dlt_exact_recovers_pose() {
        let mut rng = Xorshift64::new(1);
        let (truth, corrs, _) = make_scene(&mut rng, 30, 0.0, 0.0);
        let sample: Vec<usize> = (0..8).collect();
        let pose = dlt_pose(&corrs, &sample, &mut rng).expect("dlt failed");
        let rerr = rot_err_deg(truth.r, pose.r);
        let terr = norm3([
            pose.t[0] - truth.t[0],
            pose.t[1] - truth.t[1],
            pose.t[2] - truth.t[2],
        ]) / norm3(truth.t);
        assert!(rerr < 0.1, "rotation err {rerr} deg");
        assert!(terr < 1e-2, "translation rel err {terr}");
    }

    #[test]
    fn refine_recovers_from_perturbation() {
        // The DLT+Gram-Schmidt pose is only a rough init; refine_pose must pull
        // it back onto the true rigid pose (the real-data failure mode).
        let mut rng = Xorshift64::new(11);
        let (truth, corrs, _) = make_scene(&mut rng, 80, 0.0, 1.0);
        let perturbed = Pose {
            r: mat_mul3(exp_so3([0.15, -0.12, 0.10]), truth.r),
            t: [truth.t[0] + 0.4, truth.t[1] - 0.3, truth.t[2] + 0.5],
        };
        let refined = refine_pose(perturbed, &corrs, None);
        let rerr = rot_err_deg(truth.r, refined.r);
        let terr = norm3([
            refined.t[0] - truth.t[0],
            refined.t[1] - truth.t[1],
            refined.t[2] - truth.t[2],
        ]) / norm3(truth.t);
        assert!(rerr < 0.5, "rot err {rerr} deg");
        assert!(terr < 0.05, "t rel err {terr}");
    }

    #[test]
    fn ransac_noisy_outliers() {
        let mut worst_rot = 0f32;
        let mut worst_t = 0f32;
        for seed in [1u64, 42, 7, 99, 1234, 555, 2024, 31337] {
            let mut rng = Xorshift64::new(seed);
            let n = 60;
            let (truth, corrs, n_true) = make_scene(&mut rng, n, 0.2, 0.3);
            let mut mask = vec![false; n];
            let opts = PnpOptions {
                iterations: 150,
                sample_size: 8,
                reproj_threshold_px: 4.0,
                min_inliers: 20,
            };
            let res = pnp_ransac(&corrs, &CAM, &opts, &mut rng, &mut mask).expect("pnp failed");
            let rerr = rot_err_deg(truth.r, res.r);
            let terr = norm3([
                res.t[0] - truth.t[0],
                res.t[1] - truth.t[1],
                res.t[2] - truth.t[2],
            ]) / norm3(truth.t);
            worst_rot = worst_rot.max(rerr);
            worst_t = worst_t.max(terr);
            println!(
                "seed {seed}: inliers {}/{} (true {}) rot {rerr:.3}deg t_rel {terr:.4}",
                res.inlier_count, n, n_true
            );
            assert!(res.inlier_count >= n_true - 2, "seed {seed}");
            assert!(rerr < 1.0, "seed {seed}: rot {rerr}");
            assert!(terr < 5e-2, "seed {seed}: t {terr}");
            assert!(res.mean_reproj_error_px < 1.0, "seed {seed}");
            assert_eq!(mask.iter().filter(|&&b| b).count(), res.inlier_count);
        }
        println!("worst over seeds: rot {worst_rot:.3} deg, t_rel {worst_t:.4}");
    }

    #[test]
    fn ransac_heavy_outliers() {
        let mut rng = Xorshift64::new(5);
        let n = 80;
        let (truth, corrs, n_true) = make_scene(&mut rng, n, 0.4, 0.5);
        let mut mask = vec![false; n];
        let opts = PnpOptions {
            iterations: 200,
            sample_size: 6,
            reproj_threshold_px: 4.0,
            min_inliers: 20,
        };
        let res = pnp_ransac(&corrs, &CAM, &opts, &mut rng, &mut mask).expect("pnp failed");
        let rerr = rot_err_deg(truth.r, res.r);
        let terr = norm3([
            res.t[0] - truth.t[0],
            res.t[1] - truth.t[1],
            res.t[2] - truth.t[2],
        ]) / norm3(truth.t);
        println!(
            "heavy: inliers {}/{} (true {}) rot {rerr:.3}deg t_rel {terr:.4}",
            res.inlier_count, n, n_true
        );
        assert!(res.inlier_count >= n_true - 4);
        assert!(rerr < 2.0, "rotation err {rerr} deg");
        assert!(terr < 8e-2, "translation rel err {terr}");
    }

    #[test]
    fn rejects_garbage_and_tiny_input() {
        let mut rng = Xorshift64::new(7);
        let mut corrs = Vec::new();
        for _ in 0..40 {
            let mut u = || (rng.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
            let world = [u() * 4.0 - 2.0, u() * 4.0 - 2.0, u() * 4.0 - 2.0];
            let [xn, yn] = CAM.pixel_to_undistorted_normalized(u() * 640.0, u() * 480.0);
            corrs.push(Correspondence { world, xn, yn });
        }
        let mut mask = vec![false; corrs.len()];
        let opts = PnpOptions {
            min_inliers: 8,
            ..Default::default()
        };
        assert!(pnp_ransac(&corrs, &CAM, &opts, &mut rng, &mut mask).is_none());

        let mut rng2 = Xorshift64::new(8);
        let (_, corrs5, _) = make_scene(&mut rng2, 5, 0.0, 0.0);
        let mut mask = vec![false; 5];
        assert!(pnp_ransac(&corrs5, &CAM, &opts, &mut rng2, &mut mask).is_none());
    }
}
