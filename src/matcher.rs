//! Brute-force rBRIEF descriptor matcher (no_std, alloc-free): 256-bit Hamming
//! with a 100-bit max distance, Lowe's ratio 0.8 and mutual nearest neighbour,
//! mirroring scripts/match_features.py / slam-exp MatchFeaturesFiltered.

/// Descriptor layout shared with rbrief (bit k of word i = pair i*32+k).
pub type Descriptor = [u32; 8];

pub const MATCH_MAX_DISTANCE: u32 = 100;
pub const LOWE_RATIO: f32 = 0.8;

/// Anything that exposes a descriptor (pyramid Feature, localize MapPoint, or a
/// bare descriptor).
pub trait DescriptorSource {
    fn descriptor(&self) -> &Descriptor;
}

impl DescriptorSource for Descriptor {
    fn descriptor(&self) -> &Descriptor {
        self
    }
}

/// True for the all-zero border descriptors, which never match.
pub fn is_zero(d: &Descriptor) -> bool {
    d.iter().all(|w| *w == 0)
}

/// 256-bit Hamming distance between two descriptors.
pub fn hamming(a: &Descriptor, b: &Descriptor) -> u32 {
    let mut d = 0;
    for i in 0..8 {
        d += (a[i] ^ b[i]).count_ones();
    }
    d
}

/// One accepted query -> map match; indices are into the caller's slices.
#[derive(Clone, Copy, Default)]
pub struct Match {
    pub query: u32,
    pub point: u32,
    pub dist: u32,
}

/// Reusable per-query / per-map-point scratch (all caller-owned).
pub struct MatchBuffers<'a> {
    pub best_idx: &'a mut [u32],
    pub best_dist: &'a mut [u32],
    pub second_dist: &'a mut [u32],
    pub point_query: &'a mut [u32],
    pub point_dist: &'a mut [u32],
}

/// Match `query` against `map`, writing accepted matches to `out` and returning
/// the count (capped at `out.len()`). Buffers clamp to the shorter slice lengths.
pub fn match_map<Q: DescriptorSource, M: DescriptorSource>(
    query: &[Q],
    map: &[M],
    out: &mut [Match],
    b: &mut MatchBuffers,
) -> usize {
    let nq = query
        .len()
        .min(b.best_idx.len())
        .min(b.best_dist.len())
        .min(b.second_dist.len());
    let nm = map.len().min(b.point_query.len()).min(b.point_dist.len());
    if nq == 0 || nm == 0 || out.is_empty() {
        return 0;
    }
    // Per-map-point best query (reset every call -> first minimum on ties).
    for d in b.point_dist[..nm].iter_mut() {
        *d = u32::MAX;
    }
    for q in 0..nq {
        let qd = query[q].descriptor();
        if is_zero(qd) {
            b.best_dist[q] = u32::MAX;
            b.second_dist[q] = u32::MAX;
            continue;
        }
        let (mut b1, mut b2, mut bi) = (u32::MAX, u32::MAX, 0u32);
        for m in 0..nm {
            let md = map[m].descriptor();
            if is_zero(md) {
                continue;
            }
            let d = hamming(qd, md);
            if d < b.point_dist[m] {
                b.point_dist[m] = d;
                b.point_query[m] = q as u32;
            }
            if d < b1 {
                b2 = b1;
                b1 = d;
                bi = m as u32;
            } else if d < b2 {
                b2 = d;
            }
        }
        b.best_idx[q] = bi;
        b.best_dist[q] = b1;
        b.second_dist[q] = b2;
    }
    let mut nout = 0;
    for q in 0..nq {
        let b1 = b.best_dist[q];
        if b1 > MATCH_MAX_DISTANCE {
            continue;
        }
        let b2 = b.second_dist[q];
        if b2 != u32::MAX && b1 as f32 > LOWE_RATIO * b2 as f32 {
            continue;
        }
        let m = b.best_idx[q] as usize;
        if b.point_query[m] as usize != q {
            continue; // failed mutual nearest neighbour
        }
        if nout >= out.len() {
            break;
        }
        out[nout] = Match { query: q as u32, point: m as u32, dist: b1 };
        nout += 1;
    }
    nout
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 =
                self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
        fn desc(&mut self) -> Descriptor {
            let mut d = [0u32; 8];
            for w in d.iter_mut() {
                *w = self.next();
            }
            d
        }
    }

    /// Straight mirror of scripts/match_features.py (lower index wins ties).
    fn naive(query: &[Descriptor], map: &[Descriptor]) -> Vec<(usize, usize, u32)> {
        let qk: Vec<usize> = (0..query.len()).filter(|&i| !is_zero(&query[i])).collect();
        let mk: Vec<usize> = (0..map.len()).filter(|&i| !is_zero(&map[i])).collect();
        if qk.is_empty() || mk.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        for &qi in &qk {
            let mut ds: Vec<(u32, usize)> =
                mk.iter().map(|&mi| (hamming(&query[qi], &map[mi]), mi)).collect();
            ds.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
            let (d1, m1) = ds[0];
            if d1 > MATCH_MAX_DISTANCE {
                continue;
            }
            if ds.len() >= 2 && d1 as f32 > LOWE_RATIO * ds[1].0 as f32 {
                continue;
            }
            let (mut bq, mut bd) = (None, u32::MAX);
            for &qj in &qk {
                let d = hamming(&query[qj], &map[m1]);
                if d < bd {
                    bd = d;
                    bq = Some(qj);
                }
            }
            if bq != Some(qi) {
                continue;
            }
            out.push((qi, m1, d1));
        }
        out
    }

    fn run(
        query: &[Descriptor],
        map: &[Descriptor],
        out: &mut [Match],
    ) -> Vec<(usize, usize, u32)> {
        let nq = query.len().max(1);
        let nm = map.len().max(1);
        let mut bi = vec![0u32; nq];
        let mut bd = vec![0u32; nq];
        let mut sd = vec![0u32; nq];
        let mut pq = vec![0u32; nm];
        let mut pd = vec![0u32; nm];
        let mut b = MatchBuffers {
            best_idx: &mut bi,
            best_dist: &mut bd,
            second_dist: &mut sd,
            point_query: &mut pq,
            point_dist: &mut pd,
        };
        let n = match_map(query, map, out, &mut b);
        out[..n].iter().map(|m| (m.query as usize, m.point as usize, m.dist)).collect()
    }

    #[test]
    fn hamming_basics() {
        let z = [0u32; 8];
        let o = [u32::MAX; 8];
        assert_eq!(hamming(&z, &z), 0);
        assert_eq!(hamming(&z, &o), 256);
        assert!(is_zero(&z) && !is_zero(&o));
    }

    #[test]
    fn matches_naive_reference() {
        let mut r = Lcg(0x1234_5678);
        for &(nq, nm) in &[(0usize, 0usize), (1, 0), (0, 1), (1, 1), (5, 3), (40, 60), (120, 90)] {
            let mut query: Vec<Descriptor> = (0..nq).map(|_| r.desc()).collect();
            let mut map: Vec<Descriptor> = (0..nm).map(|_| r.desc()).collect();
            // Force real matches: copy overlapping map descriptors into queries.
            for i in 0..nq.min(nm) {
                query[i] = map[i];
            }
            // Zero (border) descriptors on both sides must never match.
            if nq > 2 {
                query[2] = [0; 8];
            }
            if nm > 3 {
                map[3] = [0; 8];
            }
            let mut out = vec![Match::default(); nq.max(1)];
            let got = run(&query, &map, &mut out);
            let want = naive(&query, &map);
            assert_eq!(got, want, "nq={nq} nm={nm}");
        }
    }

    /// Descriptor with `n` low bits set (words 0..), zero elsewhere.
    fn with_bits(n: u32) -> Descriptor {
        let mut d = [0u32; 8];
        for (i, w) in d.iter_mut().enumerate() {
            let lo = i as u32 * 32;
            if n <= lo {
                break;
            }
            *w = if n >= lo + 32 { u32::MAX } else { (1u32 << (n - lo)) - 1 };
        }
        d
    }

    #[test]
    fn max_distance_and_ratio_reject() {
        // Query has a bit only in the last word, so with_bits() never overlaps it.
        let mut q = [0u32; 8];
        q[7] = 1;
        let mut out = vec![Match::default(); 4];
        // Only candidate is 102 bits away -> over the 100-bit cap.
        assert_eq!(run(&[q], &[with_bits(101)], &mut out).len(), 0);
        // best 51, second 52: 51 > 0.8*52 -> ratio rejects.
        assert_eq!(run(&[q], &[with_bits(50), with_bits(51)], &mut out).len(), 0);
        // best 41, second 61: 41 <= 0.8*61 -> accepted (MNN holds).
        assert_eq!(run(&[q], &[with_bits(40), with_bits(60)], &mut out).len(), 1);
    }
}
