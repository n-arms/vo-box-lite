//! On-device localization: match pyramid query features against one retrieved
//! map frame's points and recover the pose with PnP RANSAC (no_std, alloc-free).
//!
//! Two matchers (BENCH_VI):
//!   - `localize_frame`: brute force against one frame's points (used by the
//!     embedding fallback and by `main.rs`).
//!   - `select_keyframes` + `localize_prior`: the EKF-prior path — pick the
//!     top-K map frames by attitude gate + prior-frustum projection, then
//!     brute-force match the union of their in-frustum points.

use crate::matcher::{self, DescriptorSource, Match, MatchBuffers};
use crate::pyramid::Feature;
use crate::ransac::{self, Camera, Correspondence, PnpBest, PnpOptions, PnpResult, Rng};
use crate::rbrief::Descriptor;

/// One uploaded map point: COLMAP world xyz + its rBRIEF descriptor.
#[derive(Clone, Copy)]
pub struct MapPoint {
    pub xyz: [f32; 3],
    pub desc: Descriptor,
}

impl DescriptorSource for MapPoint {
    fn descriptor(&self) -> &Descriptor {
        &self.desc
    }
}

impl DescriptorSource for Feature {
    fn descriptor(&self) -> &Descriptor {
        &self.desc
    }
}

/// Caller-owned scratch, reused across frames.
pub struct LocalizeScratch<'a> {
    pub mb: MatchBuffers<'a>,
    pub matches: &'a mut [Match],
    pub corrs: &'a mut [Correspondence],
    pub mask: &'a mut [bool],
}

/// One localization attempt: match count + PnP result (None = PnP failed) +
/// best hypothesis stats + per-phase µs from the caller's clock.
pub struct LocalizeStats {
    pub matches: usize,
    pub pnp: Option<PnpResult>,
    pub pnp_best: PnpBest,
    pub match_us: u64,
    pub pnp_us: u64,
}

/// Match `query` against one map frame's `map`, then solve the pose with PnP
/// RANSAC. Alloc-free: scratch comes from the caller (see `LocalizeScratch`).
pub fn localize_frame(
    query: &[Feature],
    map: &[MapPoint],
    cam: &Camera,
    opts: &PnpOptions,
    rng: &mut impl Rng,
    s: &mut LocalizeScratch,
    now_us: fn() -> u64,
) -> LocalizeStats {
    let t_match = now_us();
    let n = matcher::match_map(query, map, s.matches, &mut s.mb);
    let mut nc = 0;
    for m in &s.matches[..n] {
        if nc >= s.corrs.len() {
            break;
        }
        let q = &query[m.query as usize];
        let [xn, yn] = cam.pixel_to_undistorted_normalized(q.x, q.y);
        s.corrs[nc] = Correspondence { world: map[m.point as usize].xyz, xn, yn };
        nc += 1;
    }
    let match_us = now_us().wrapping_sub(t_match);
    let t_pnp = now_us();
    let mut pnp_best = PnpBest::default();
    let pnp = ransac::pnp_ransac_best(&s.corrs[..nc], cam, opts, rng, s.mask, &mut pnp_best);
    let pnp_us = now_us().wrapping_sub(t_pnp);
    LocalizeStats { matches: n, pnp, pnp_best, match_us, pnp_us }
}

// ---- pose-prior keyframe selection + windowed matching (BENCH_VI) ----

/// Camera pose prior in the map frame: camera center + camera->world
/// quaternion `[w, x, y, z]` (built from the EKF state).
#[derive(Clone, Copy)]
pub struct PosePrior {
    pub pos: [f32; 3],
    pub quat: [f32; 4],
}

/// Stack scratch bound for `select_keyframes` (`top_k` is clamped to this).
pub const MAX_KEYFRAMES: usize = 8;

/// One map frame exposed to keyframe selection / windowed matching
/// (implemented by the firmware's `LocalMapFrame` and the host harness's
/// `MapFrame`).
pub trait PosedFrame {
    fn has_pose(&self) -> bool;
    fn pos(&self) -> [f32; 3];
    /// World->camera quaternion `[w,x,y,z]` exactly as stored in map.txt.
    fn quat(&self) -> [f32; 4];
    fn points(&self) -> &[MapPoint];
}

/// `select_keyframes` result.
pub struct KfSelect {
    /// Ids written to the caller's buffer (top-k, nearest first).
    pub n: usize,
    /// Frames passing the attitude gate.
    pub survivors: usize,
    /// In-frustum points of the best frame.
    pub infrustum: usize,
}

pub fn quat_conj(q: [f32; 4]) -> [f32; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

pub fn quat_to_mat(q: [f32; 4]) -> [[f32; 3]; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

pub fn mat3_vec(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

pub fn transpose3(m: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]]
}

/// Small-angle attitude gate (no libm): true if the relative rotation between
/// two same-convention quats is below `max_angle` radians.
pub fn quat_within(a: [f32; 4], b: [f32; 4], max_angle: f32) -> bool {
    let d = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]).abs().min(1.0);
    let h = 0.5 * max_angle;
    d >= 1.0 - 0.5 * h * h
}

/// Forward pinhole + SIMPLE_RADIAL projection of `p` into a camera with
/// world->camera rotation `r_wc` and center `c`; None if behind the camera.
pub fn project_pixel(
    cam: &Camera,
    r_wc: &[[f32; 3]; 3],
    c: [f32; 3],
    p: [f32; 3],
) -> Option<[f32; 2]> {
    let xc = mat3_vec(r_wc, [p[0] - c[0], p[1] - c[1], p[2] - c[2]]);
    if xc[2] <= 1e-6 {
        return None;
    }
    let x = xc[0] / xc[2];
    let y = xc[1] / xc[2];
    let s = 1.0 + cam.k1 * (x * x + y * y);
    Some([cam.fx * x * s + cam.cx, cam.fy * y * s + cam.cy])
}

/// Select up to `top_k` keyframes for the prior: attitude gate `max_angle`,
/// then rank by camera-center distance (in-frustum count breaks ties).
pub fn select_keyframes<F: PosedFrame>(
    prior: &PosePrior,
    frames: &[F],
    cam: &Camera,
    w: usize,
    h: usize,
    max_angle: f32,
    top_k: usize,
    out_ids: &mut [usize],
) -> KfSelect {
    let r_wc = transpose3(&quat_to_mat(prior.quat));
    let k = top_k.clamp(1, MAX_KEYFRAMES).min(out_ids.len());
    let mut ids = [0usize; MAX_KEYFRAMES];
    let mut dist = [f32::MAX; MAX_KEYFRAMES];
    let mut cnt = [0usize; MAX_KEYFRAMES];
    let (mut n, mut survivors) = (0usize, 0usize);
    for (i, f) in frames.iter().enumerate() {
        if !f.has_pose() {
            continue;
        }
        // map.txt stores world->camera; the prior is camera->world.
        if !quat_within(prior.quat, quat_conj(f.quat()), max_angle) {
            continue;
        }
        survivors += 1;
        let mut count = 0usize;
        for p in f.points() {
            if matcher::is_zero(&p.desc) {
                continue;
            }
            if let Some([u, v]) = project_pixel(cam, &r_wc, prior.pos, p.xyz) {
                if u >= 0.0 && u < w as f32 && v >= 0.0 && v < h as f32 {
                    count += 1;
                }
            }
        }
        if count == 0 {
            continue;
        }
        let d = f.pos();
        let d0 = d[0] - prior.pos[0];
        let d1 = d[1] - prior.pos[1];
        let d2 = d[2] - prior.pos[2];
        let dist2 = d0 * d0 + d1 * d1 + d2 * d2;
        // Insertion position: distance ascending, in-frustum count descending.
        let mut pos = n;
        while pos > 0 {
            let better =
                dist2 < dist[pos - 1] || (dist2 == dist[pos - 1] && count > cnt[pos - 1]);
            if !better {
                break;
            }
            pos -= 1;
        }
        if pos >= k {
            continue;
        }
        let last = if n < k { n } else { k - 1 };
        let mut j = last;
        while j > pos {
            ids[j] = ids[j - 1];
            dist[j] = dist[j - 1];
            cnt[j] = cnt[j - 1];
            j -= 1;
        }
        ids[pos] = i;
        dist[pos] = dist2;
        cnt[pos] = count;
        if n < k {
            n += 1;
        }
    }
    out_ids[..n].copy_from_slice(&ids[..n]);
    KfSelect { n, survivors, infrustum: if n > 0 { cnt[0] } else { 0 } }
}

/// One candidate point: map xyz + descriptor (the prior-projected points that
/// landed in frame).
#[derive(Clone, Copy, Default)]
pub struct Candidate {
    pub xyz: [f32; 3],
    pub desc: Descriptor,
}

/// Caller-owned scratch for `localize_prior` (candidate buffers sized to the
/// max in-frustum points of the selected frames).
pub struct PriorScratch<'a> {
    pub cands: &'a mut [Candidate],
    pub cand_query: &'a mut [u32],
    pub cand_dist: &'a mut [u32],
    pub best_idx: &'a mut [u32],
    pub best_dist: &'a mut [u32],
    pub second_dist: &'a mut [u32],
    pub corrs: &'a mut [Correspondence],
    pub mask: &'a mut [bool],
}

/// Prior-driven localization: project the selected frames' points into the
/// prior camera, keep the in-frame non-empty ones (the "could be visible" set),
/// then brute-force match (Hamming + Lowe ratio + mutual NN) and run PnP
/// RANSAC. `stats.matches` is the accepted correspondence count.
#[allow(clippy::too_many_arguments)]
pub fn localize_prior<F: PosedFrame>(
    query: &[Feature],
    frames: &[F],
    kf_ids: &[usize],
    prior: &PosePrior,
    cam: &Camera,
    w: usize,
    h: usize,
    opts: &PnpOptions,
    rng: &mut impl Rng,
    s: &mut PriorScratch,
    now_us: fn() -> u64,
) -> LocalizeStats {
    let t_match = now_us();
    let r_wc = transpose3(&quat_to_mat(prior.quat));
    let cap = s.cands.len().min(s.cand_query.len()).min(s.cand_dist.len());
    let mut nc = 0usize;
    'build: for &k in kf_ids {
        if k >= frames.len() {
            continue;
        }
        for p in frames[k].points() {
            if matcher::is_zero(&p.desc) {
                continue;
            }
            let Some([u, v]) = project_pixel(cam, &r_wc, prior.pos, p.xyz) else {
                continue;
            };
            if u < 0.0 || u >= w as f32 || v < 0.0 || v >= h as f32 {
                continue;
            }
            if nc >= cap {
                break 'build;
            }
            s.cands[nc] = Candidate { xyz: p.xyz, desc: p.desc };
            nc += 1;
        }
    }
    for d in s.cand_dist[..nc].iter_mut() {
        *d = u32::MAX;
    }
    let nq = query
        .len()
        .min(s.best_idx.len())
        .min(s.best_dist.len())
        .min(s.second_dist.len());
    for qi in 0..nq {
        let q = &query[qi];
        if matcher::is_zero(&q.desc) {
            s.best_idx[qi] = 0;
            s.best_dist[qi] = u32::MAX;
            s.second_dist[qi] = u32::MAX;
            continue;
        }
        let (mut b1, mut b2, mut bi) = (u32::MAX, u32::MAX, 0u32);
        for ci in 0..nc {
            let cd = {
                let c = &s.cands[ci];
                c.desc
            };
            let d = matcher::hamming(&q.desc, &cd);
            if d < s.cand_dist[ci] {
                s.cand_dist[ci] = d;
                s.cand_query[ci] = qi as u32;
            }
            if d < b1 {
                b2 = b1;
                b1 = d;
                bi = ci as u32;
            } else if d < b2 {
                b2 = d;
            }
        }
        s.best_idx[qi] = bi;
        s.best_dist[qi] = b1;
        s.second_dist[qi] = b2;
    }
    let mut ncorr = 0usize;
    for qi in 0..nq {
        let b1 = s.best_dist[qi];
        if b1 > matcher::MATCH_MAX_DISTANCE {
            continue;
        }
        let b2 = s.second_dist[qi];
        if b2 != u32::MAX && b1 as f32 > matcher::LOWE_RATIO * b2 as f32 {
            continue;
        }
        let ci = s.best_idx[qi] as usize;
        if ci >= nc || s.cand_query[ci] != qi as u32 {
            continue; // failed mutual nearest neighbour
        }
        if ncorr >= s.corrs.len() {
            break;
        }
        let q = &query[qi];
        let [xn, yn] = cam.pixel_to_undistorted_normalized(q.x, q.y);
        s.corrs[ncorr] = Correspondence { world: s.cands[ci].xyz, xn, yn };
        ncorr += 1;
    }
    let match_us = now_us().wrapping_sub(t_match);
    let t_pnp = now_us();
    let mut pnp_best = PnpBest::default();
    let pnp = ransac::pnp_ransac_best(&s.corrs[..ncorr], cam, opts, rng, s.mask, &mut pnp_best);
    let pnp_us = now_us().wrapping_sub(t_pnp);
    LocalizeStats { matches: ncorr, pnp, pnp_best, match_us, pnp_us }
}
