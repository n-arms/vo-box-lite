#!/usr/bin/env python3
"""Hamming matcher for rBRIEF features (CSV rows `x,y,<64 hex>`): mirrors
slam-exp MatchFeaturesFiltered (100-bit max distance, Lowe 0.8, MNN), numpy
XOR + popcount LUT, chunked. API: match_all(features_dir) -> (rows, stats).
"""

import argparse
import csv as csv_mod
import sys
from pathlib import Path

import numpy as np

MATCH_MAX_DISTANCE = 100
LOWE_RATIO = 0.8
USE_MNN = True

# popcount of every byte, as a u8 LUT (numpy 1.26 has no bitwise_count).
_POPCNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
# distance matrix chunk budget (~16 MB of XOR temps at a time)
_CHUNK_BYTES = 16 * 1024 * 1024


def load_feature_csv(path) -> list:
    """Parse a feature CSV (x,y,<64 hex>) into a list of hex strings."""
    hexes = []
    with open(path, newline="") as f:
        for row in csv_mod.reader(f):
            if len(row) >= 3:
                hexes.append(row[2].strip())
    return hexes


def hexes_to_array(hexes) -> np.ndarray:
    """(N, 32) uint8 descriptor array from a list of 64-char hex strings."""
    n = len(hexes)
    arr = np.zeros((n, 32), dtype=np.uint8)
    if n:
        arr[:] = np.frombuffer(b"".join(bytes.fromhex(h) for h in hexes),
                               dtype=np.uint8).reshape(n, 32)
    return arr


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(len(a), len(b)) uint16 Hamming distances, chunked to bound memory.
    a/b are (N, 32) uint8 descriptor arrays."""
    na, nb = a.shape[0], b.shape[0]
    out = np.empty((na, nb), dtype=np.uint16)
    chunk = max(1, _CHUNK_BYTES // (nb * 32))
    for s in range(0, na, chunk):
        e = min(s + chunk, na)
        x = a[s:e, None, :] ^ b[None, :, :]                 # (rows, nb, 32) u8
        out[s:e] = _POPCNT8[x].sum(axis=2, dtype=np.uint16)
    return out


def _best_and_second(d: np.ndarray):
    """Per-row best + second-best (index, dist); ties keep the lower column
    index. second_d 0 / has_second False when fewer than 2 columns."""
    n, nb = d.shape
    has_second = np.full(n, nb >= 2, dtype=bool)
    if nb == 1:
        return np.zeros(n, dtype=np.intp), d[:, 0].astype(np.uint16), \
            np.zeros(n, dtype=np.uint16), has_second
    idx = np.argpartition(d, 1, axis=1)[:, :2]              # two smallest (unsorted)
    r = np.arange(n)
    i0, i1 = idx[:, 0], idx[:, 1]
    v0, v1 = d[r, i0], d[r, i1]
    swap = v0 > v1
    best_idx = np.where(swap, i1, i0)
    best_d = np.where(swap, v1, v0)
    second_idx = np.where(swap, i0, i1)
    second_d = d[r, second_idx]
    return best_idx, best_d.astype(np.uint16), second_d.astype(np.uint16), has_second


def match_pair(hexes_a, hexes_b, max_distance=MATCH_MAX_DISTANCE,
               lowe_ratio=LOWE_RATIO, use_mnn=USE_MNN):
    """Match hexes_a (queries) against hexes_b. Returns (matches, stats) with
    matches = [(idx_a, idx_b, d_best, d_second), ...] (CSV row indices)."""
    # All-zero (border) descriptors never match; keep true row indices.
    keep_a = [i for i, h in enumerate(hexes_a) if h and h != "0" * 64]
    keep_b = [i for i, h in enumerate(hexes_b) if h and h != "0" * 64]
    a = hexes_to_array([hexes_a[i] for i in keep_a])
    b = hexes_to_array([hexes_b[i] for i in keep_b])
    stats = {"dropped_zero_a": len(hexes_a) - len(keep_a),
             "dropped_zero_b": len(hexes_b) - len(keep_b)}
    if len(keep_a) == 0 or len(keep_b) == 0:
        return [], stats

    d = hamming_matrix(a, b)
    best_idx, best_d, second_d, has_second = _best_and_second(d)
    n = len(keep_a)

    accept = best_d <= max_distance
    # Ratio test needs a second candidate; a lone train feature passes on
    # distance alone.
    ratio_ok = np.ones(n, dtype=bool)
    if has_second.any():
        ratio_ok[has_second] = (best_d[has_second].astype(np.float64)
                                <= lowe_ratio * second_d[has_second].astype(np.float64))
    accept &= ratio_ok

    if use_mnn:
        # Keep only matches whose train feature also picks this query (argmin
        # over rows; first minimum on ties -> deterministic).
        best_over_a = np.argmin(d, axis=0)
        mutual = best_over_a[best_idx[accept]] == np.arange(n, dtype=np.intp)[accept]
        accept_idx = np.nonzero(accept)[0][mutual]
    else:
        accept_idx = np.nonzero(accept)[0]

    matches = []
    n_ambig = 0
    for ai in accept_idx:
        d2 = int(second_d[ai]) if has_second[ai] else 0
        if has_second[ai] and d2 <= max_distance:
            n_ambig += 1  # a second candidate also within acceptance distance
        matches.append((keep_a[ai], keep_b[int(best_idx[ai])],
                        int(best_d[ai]), d2))
    stats.update({
        "raw_candidates": int(accept.sum()),
        "accepted": len(matches),
        "ambiguous": n_ambig,
        "max_distance": max_distance,
        "lowe_ratio": lowe_ratio,
        "mnn": use_mnn,
    })
    return matches, stats


def match_all(features_dir: Path, max_distance=MATCH_MAX_DISTANCE,
              lowe_ratio=LOWE_RATIO, use_mnn=USE_MNN):
    """Match every unordered pair of feature CSVs in `features_dir` (sorted by
    stem). Returns (rows, stats): rows = [(stem_a, stem_b, idx_a, idx_b,
    d_best, d_second)] with stem_a < stem_b; stats aggregates per-pair counts."""
    stems = sorted(p.stem for p in Path(features_dir).glob("*.csv"))
    if len(stems) < 2:
        return [], {"error": f"need >= 2 feature CSVs, found {len(stems)} in {features_dir}"}
    hexes = {s: load_feature_csv(Path(features_dir) / f"{s}.csv") for s in stems}

    rows = []
    stats = {"pairs_total": 0, "pairs_with_matches": 0,
             "total_matches": 0, "ambiguous": 0, "total_features": 0}
    stats["total_features"] = sum(len(h) for h in hexes.values())
    for i in range(len(stems)):
        for j in range(i + 1, len(stems)):
            sa, sb = stems[i], stems[j]
            stats["pairs_total"] += 1
            ms, st = match_pair(hexes[sa], hexes[sb],
                                max_distance=max_distance,
                                lowe_ratio=lowe_ratio, use_mnn=use_mnn)
            if ms:
                stats["pairs_with_matches"] += 1
                stats["ambiguous"] += st["ambiguous"]
            for (ia, ib, db, ds) in ms:
                rows.append((sa, sb, ia, ib, db, ds))
    stats["total_matches"] = len(rows)
    if stats["total_matches"]:
        arr = np.array([r[4] for r in rows])
        stats["match_dist_min"] = int(arr.min())
        stats["match_dist_median"] = int(np.median(arr))
        stats["match_dist_max"] = int(arr.max())
        stats["ambiguous_frac"] = stats["ambiguous"] / stats["total_matches"]
    return rows, stats


def write_matches_csv(path: Path, rows) -> None:
    """matches.csv schema (slam-exp): image_a,image_b,idx_a,idx_b per row."""
    with open(path, "w") as f:
        for (sa, sb, ia, ib, _db, _ds) in rows:
            f.write(f"{sa},{sb},{ia},{ib}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features_dir", help="dir of per-image feature CSVs")
    ap.add_argument("--out", default=None, help="matches.csv path (default: next to features_dir)")
    args = ap.parse_args()
    rows, stats = match_all(Path(args.features_dir))
    out = Path(args.out) if args.out else Path(args.features_dir).parent / "matches.csv"
    write_matches_csv(out, rows)
    print(f"{len(rows)} matches across {stats['pairs_with_matches']}/{stats['pairs_total']} pairs -> {out}")
    sys.exit(0)
