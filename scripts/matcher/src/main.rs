//! Host-side rBRIEF matcher: Rust replacement for `scripts/match_features.py`.
//! Same algorithm as Python (max Hamming distance, Lowe ratio, MNN) over every
//! image pair (or just the pairs in a pair-list file), but AVX2 popcount +
//! rayon across pairs. `VO_MATCHER_SCALAR=1` forces the scalar kernel.
//!
//! usage: vo-matcher <features_dir> <out_matches_csv> <out_stats_json>
//!                   [max_distance] [lowe_ratio] [pairs_file]
//!
//! pairs_file (optional): `stem_a,stem_b` per line — only those pairs are
//! matched (unknown stems are an error). Omitting it matches all pairs.

use rayon::prelude::*;
use std::arch::x86_64::*;
use std::path::Path;
use std::time::Instant;
use std::{env, fs};

const MAX_DISTANCE: u32 = 100;
const LOWE_RATIO: f64 = 0.8;

/// 32-byte descriptor, 32-byte aligned so AVX2 can `load_si256` it directly.
#[repr(C, align(32))]
#[derive(Clone, Copy)]
struct Descriptor([u8; 32]);

impl Descriptor {
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.iter().all(|&b| b == 0)
    }
}

struct ImageEntry {
    stem: String,
    /// Every CSV row's descriptor (feature index == row number).
    descs: Vec<Descriptor>,
    /// Original row indices of the non-zero descriptors (all-zero never match).
    keep: Vec<u32>,
    /// `descs[keep]` — the descriptors actually matched.
    kept: Vec<Descriptor>,
}

#[derive(Default)]
struct PairOut {
    /// `(kept_a_index, kept_b_index, best_hamming)` per accepted match.
    matches: Vec<(u32, u32, u32)>,
    /// Accepted matches whose second candidate is also within the max distance.
    ambiguous: u32,
}

// ---------------------------------------------------------------------------
// Hamming distance kernels
// ---------------------------------------------------------------------------

/// AVX2 nibble-LUT popcount of a full 256-bit vector (sum of both nibble
/// lookups, then `sad_epu8` sums each 8-byte lane into a u64).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn popcount_256(v: __m256i, lut: __m256i, mask: __m256i) -> u32 {
    let lo = _mm256_and_si256(v, mask);
    let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask);
    let pc = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));
    let sum = _mm256_sad_epu8(pc, _mm256_setzero_si256());
    let s128 = _mm_add_epi64(
        _mm256_castsi256_si128(sum),
        _mm256_extracti128_si256::<1>(sum),
    );
    (_mm_extract_epi64::<0>(s128) + _mm_extract_epi64::<1>(s128)) as u32
}

#[inline(always)]
fn hamming_scalar(a: &[u8; 32], b: &[u8; 32]) -> u32 {
    let mut d = 0u32;
    for k in 0..4 {
        let va = u64::from_le_bytes(a[k * 8..k * 8 + 8].try_into().unwrap());
        let vb = u64::from_le_bytes(b[k * 8..k * 8 + 8].try_into().unwrap());
        d += (va ^ vb).count_ones();
    }
    d
}

// ---------------------------------------------------------------------------
// Pair matching
// ---------------------------------------------------------------------------

/// Match image `a`'s descriptors against `b`'s, exactly like Python's
/// `match_features.match_pair`: per-query best/second, ratio + max-distance
/// filter, then the mutual-nearest-neighbour cross-check.
#[target_feature(enable = "avx2")]
unsafe fn match_pair_avx2(
    a: &[Descriptor],
    b: &[Descriptor],
    max_distance: u32,
    lowe_ratio: f64,
) -> PairOut {
    match_pair_core(a, b, true, max_distance, lowe_ratio)
}

unsafe fn match_pair_scalar(
    a: &[Descriptor],
    b: &[Descriptor],
    max_distance: u32,
    lowe_ratio: f64,
) -> PairOut {
    match_pair_core(a, b, false, max_distance, lowe_ratio)
}

#[inline(always)]
unsafe fn match_pair_core(
    a: &[Descriptor],
    b: &[Descriptor],
    use_avx2: bool,
    max_distance: u32,
    lowe_ratio: f64,
) -> PairOut {
    let na = a.len();
    let nb = b.len();
    let mut best_d = vec![u32::MAX; na];
    let mut best_i = vec![0u32; na];
    let mut second_d = vec![u32::MAX; na];
    // Per-target best query, for the MNN cross-check.
    let mut col_best_d = vec![u32::MAX; nb];
    let mut col_best_i = vec![0u32; nb];

    let mut lut = _mm256_setzero_si256();
    let mut mask = _mm256_setzero_si256();
    if use_avx2 {
        // 4-bit popcount LUT, duplicated into both 128-bit halves.
        lut = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        );
        mask = _mm256_set1_epi8(0x0F);
    }

    // Register blocking: 4 queries share each loaded target descriptor.
    let mut i = 0usize;
    while i < na {
        let cnt = core::cmp::min(4, na - i);
        let a0 = _mm256_load_si256(a[i].0.as_ptr() as *const __m256i);
        let a1 = if cnt > 1 {
            _mm256_load_si256(a[i + 1].0.as_ptr() as *const __m256i)
        } else {
            _mm256_setzero_si256()
        };
        let a2 = if cnt > 2 {
            _mm256_load_si256(a[i + 2].0.as_ptr() as *const __m256i)
        } else {
            _mm256_setzero_si256()
        };
        let a3 = if cnt > 3 {
            _mm256_load_si256(a[i + 3].0.as_ptr() as *const __m256i)
        } else {
            _mm256_setzero_si256()
        };

        for j in 0..nb {
            let d = if use_avx2 {
                let bv = _mm256_load_si256(b[j].0.as_ptr() as *const __m256i);
                [
                    popcount_256(_mm256_xor_si256(a0, bv), lut, mask),
                    if cnt > 1 {
                        popcount_256(_mm256_xor_si256(a1, bv), lut, mask)
                    } else {
                        0
                    },
                    if cnt > 2 {
                        popcount_256(_mm256_xor_si256(a2, bv), lut, mask)
                    } else {
                        0
                    },
                    if cnt > 3 {
                        popcount_256(_mm256_xor_si256(a3, bv), lut, mask)
                    } else {
                        0
                    },
                ]
            } else {
                [
                    hamming_scalar(&a[i].0, &b[j].0),
                    if cnt > 1 {
                        hamming_scalar(&a[i + 1].0, &b[j].0)
                    } else {
                        0
                    },
                    if cnt > 2 {
                        hamming_scalar(&a[i + 2].0, &b[j].0)
                    } else {
                        0
                    },
                    if cnt > 3 {
                        hamming_scalar(&a[i + 3].0, &b[j].0)
                    } else {
                        0
                    },
                ]
            };

            for (lane, &di) in d.iter().enumerate().take(cnt) {
                let q = i + lane;
                // Row best/second (strict `<` keeps the lower target index on ties).
                if di < best_d[q] {
                    second_d[q] = best_d[q];
                    best_d[q] = di;
                    best_i[q] = j as u32;
                } else if di < second_d[q] {
                    second_d[q] = di;
                }
                // Column best (queries ascend, so strict `<` keeps the lower query).
                if di < col_best_d[j] {
                    col_best_d[j] = di;
                    col_best_i[j] = q as u32;
                }
            }
        }
        i += cnt;
    }

    // Filter: max distance, Lowe ratio (only when a second candidate exists),
    // then mutual-nearest-neighbour.
    let mut out = PairOut::default();
    for q in 0..na {
        let bd = best_d[q];
        if bd > max_distance {
            continue;
        }
        if nb >= 2 && (bd as f64) > lowe_ratio * (second_d[q] as f64) {
            continue;
        }
        let bj = best_i[q] as usize;
        if col_best_i[bj] as usize != q {
            continue;
        }
        if nb >= 2 && second_d[q] <= max_distance {
            out.ambiguous += 1;
        }
        out.matches.push((q as u32, bj as u32, bd));
    }
    out
}

// ---------------------------------------------------------------------------
// I/O
// ---------------------------------------------------------------------------

fn parse_descriptor(hex: &str) -> Option<[u8; 32]> {
    let b = hex.as_bytes();
    if b.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (k, byte) in out.iter_mut().enumerate() {
        let hi = (b[k * 2] as char).to_digit(16)?;
        let lo = (b[k * 2 + 1] as char).to_digit(16)?;
        *byte = ((hi << 4) | lo) as u8;
    }
    Some(out)
}

fn load_images(features_dir: &Path) -> Result<Vec<ImageEntry>, String> {
    let mut files: Vec<_> = fs::read_dir(features_dir)
        .map_err(|e| format!("cannot read {}: {e}", features_dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "csv").unwrap_or(false))
        .collect();
    files.sort();
    let mut images = Vec::with_capacity(files.len());
    for path in files {
        let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut descs = Vec::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            // slam-exp format: x.xx,y.yy,<64 hex>; tolerate extra columns.
            let hex = line.rsplit(',').next().unwrap_or("").trim();
            descs.push(Descriptor(parse_descriptor(hex).unwrap_or([0u8; 32])));
        }
        let keep: Vec<u32> = (0..descs.len() as u32)
            .filter(|&k| !descs[k as usize].is_zero())
            .collect();
        let kept = keep.iter().map(|&k| descs[k as usize]).collect();
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        images.push(ImageEntry {
            stem,
            descs,
            keep,
            kept,
        });
    }
    Ok(images)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: vo-matcher <features_dir> <out_matches_csv> <out_stats_json> \
             [max_distance] [lowe_ratio] [pairs_file]"
        );
        std::process::exit(2);
    }
    let features_dir = Path::new(&args[1]);
    let out_csv = Path::new(&args[2]);
    let stats_path = args.get(3).map(Path::new);
    let max_distance: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(MAX_DISTANCE);
    let lowe_ratio: f64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(LOWE_RATIO);
    let pairs_file = args.get(6).map(Path::new);

    let avx2 = is_x86_feature_detected!("avx2") && env::var("VO_MATCHER_SCALAR").is_err();
    let images = match load_images(features_dir) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let n = images.len();
    if n < 2 {
        eprintln!(
            "need >= 2 feature CSVs, found {n} in {}",
            features_dir.display()
        );
        std::process::exit(1);
    }
    let total_features: usize = images.iter().map(|im| im.descs.len()).sum();
    // Pair list: everything by default, or exactly the `stem_a,stem_b` rows of
    // the pair file (resolved to image indices, sorted for a stable CSV order).
    let pairs: Vec<(usize, usize)> = match pairs_file {
        None => (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j))).collect(),
        Some(path) => {
            let index_of = |stem: &str| {
                images.iter().position(|im| im.stem == stem).ok_or_else(|| {
                    format!("{}: unknown image stem {stem:?}", path.display())
                })
            };
            let text = fs::read_to_string(path)
                .map_err(|e| format!("{}: {e}", path.display()))
                .unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(1);
                });
            let mut requested = std::collections::BTreeSet::new();
            for line in text.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let cols: Vec<&str> =
                    line.split(',').map(|c| c.trim()).collect();
                if cols.len() < 2 {
                    eprintln!("{}: bad pair line {line:?}", path.display());
                    std::process::exit(1);
                }
                let (a, b) = match (index_of(cols[0]), index_of(cols[1])) {
                    (Ok(a), Ok(b)) => (a.min(b), a.max(b)),
                    (Err(e), _) | (_, Err(e)) => {
                        eprintln!("{e}");
                        std::process::exit(1);
                    }
                };
                if a != b {
                    requested.insert((a, b));
                }
            }
            requested.into_iter().collect()
        }
    };
    eprintln!(
        "matching {n} images, {} pairs, {total_features} features \
         (max_dist {max_distance}, lowe {lowe_ratio}, {})",
        pairs.len(),
        if avx2 { "AVX2" } else { "scalar" }
    );

    let t0 = Instant::now();
    let results: Vec<PairOut> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let a = &images[i];
            let b = &images[j];
            if a.kept.is_empty() || b.kept.is_empty() {
                return PairOut::default();
            }
            let local = unsafe {
                if avx2 {
                    match_pair_avx2(&a.kept, &b.kept, max_distance, lowe_ratio)
                } else {
                    match_pair_scalar(&a.kept, &b.kept, max_distance, lowe_ratio)
                }
            };
            let matches = local
                .matches
                .into_iter()
                .map(|(ia, ib, d)| (a.keep[ia as usize], b.keep[ib as usize], d))
                .collect();
            PairOut {
                matches,
                ambiguous: local.ambiguous,
            }
        })
        .collect();
    let elapsed = t0.elapsed().as_secs_f64();

    // Aggregate + write matches.csv in pair-then-query order (same as Python).
    let mut csv = String::new();
    let mut total = 0u64;
    let mut pairs_with = 0u64;
    let mut ambiguous = 0u64;
    let mut dists: Vec<u32> = Vec::new();
    for (r, &(i, j)) in results.iter().zip(pairs.iter()) {
        if r.matches.is_empty() {
            continue;
        }
        pairs_with += 1;
        ambiguous += r.ambiguous as u64;
        for &(ia, ib, d) in &r.matches {
            csv.push_str(&format!(
                "{},{},{},{}\n",
                images[i].stem, images[j].stem, ia, ib
            ));
            dists.push(d);
            total += 1;
        }
    }
    if let Err(e) = fs::write(out_csv, csv) {
        eprintln!("writing {}: {e}", out_csv.display());
        std::process::exit(1);
    }

    // Stats JSON (mirrors match_features.match_all's `stats` dict).
    let mut stats = format!(
        "{{\"pairs_total\":{},\"pairs_with_matches\":{},\"total_matches\":{},\"total_features\":{},\"ambiguous\":{},\"max_distance\":{},\"lowe_ratio\":{}",
        pairs.len(), pairs_with, total, total_features, ambiguous, max_distance, lowe_ratio
    );
    if total > 0 {
        dists.sort_unstable();
        let m = dists.len();
        let median = if m % 2 == 1 {
            dists[m / 2]
        } else {
            (dists[m / 2 - 1] + dists[m / 2]) / 2
        };
        stats.push_str(&format!(
            ",\"match_dist_min\":{},\"match_dist_median\":{},\"match_dist_max\":{},\"ambiguous_frac\":{}",
            dists[0], median, dists[m - 1], ambiguous as f64 / total as f64
        ));
    }
    stats.push('}');
    if let Some(p) = stats_path {
        if let Err(e) = fs::write(p, stats) {
            eprintln!("writing {}: {e}", p.display());
            std::process::exit(1);
        }
    }
    eprintln!(
        "matching done in {elapsed:.1}s ({total} matches, {pairs_with}/{} pairs)",
        pairs.len()
    );
}
