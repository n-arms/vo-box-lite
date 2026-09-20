//! On-device localization: match pyramid query features against one retrieved
//! map frame's points and recover the pose with PnP RANSAC (no_std, alloc-free).

use crate::matcher::{self, DescriptorSource, Match, MatchBuffers};
use crate::pyramid::Feature;
use crate::ranac::{self, Camera, Correspondence, PnpBest, PnpOptions, PnpResult, Rng};
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
    let pnp = ranac::pnp_ransac_best(&s.corrs[..nc], cam, opts, rng, s.mask, &mut pnp_best);
    let pnp_us = now_us().wrapping_sub(t_pnp);
    LocalizeStats { matches: n, pnp, pnp_best, match_us, pnp_us }
}
