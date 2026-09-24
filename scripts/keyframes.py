#!/usr/bin/env python3
"""KLT keyframe selection + embedding-guided match candidate pairs.

Keyframe rule (each candidate frame vs the last accepted keyframe): track the
keyframe's Shi-Tomasi corners straight into the candidate with OpenCV
Lucas-Kanade flow; accept the candidate as a keyframe when the median drift is
>= `drift_px` or fewer than `min_tracked` points still track. The first frame
is always a keyframe. Non-keyframes are dropped from the map build entirely
(no features, no COLMAP, no map points).

Candidate pairs (per keyframe, causal: prior keyframes only): the previous
`window` keyframes (temporal neighbours) plus the `topk` most similar prior
keyframes by calc8-embedding cosine similarity. COLMAP only sees these edges.
"""

from pathlib import Path

import numpy as np

DEFAULT_DRIFT_PX = 20.0
DEFAULT_MIN_TRACKED = 150
DEFAULT_MAX_CORNERS = 500
DEFAULT_WINDOW = 30
DEFAULT_TOPK = 15


def load_gray(bmps_dir: Path, stem: str) -> np.ndarray:
    from PIL import Image
    with Image.open(bmps_dir / f"{stem}.bmp") as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def _detect(gray: np.ndarray, max_corners: int):
    import cv2
    return cv2.goodFeaturesToTrack(gray, maxCorners=max_corners,
                                   qualityLevel=0.01, minDistance=7,
                                   blockSize=7)


def _track(key_gray: np.ndarray, gray: np.ndarray, p0):
    """(n_tracked, median_drift_px) of p0 into gray, or (0, 0.0) on failure."""
    import cv2
    if p0 is None or len(p0) == 0:
        return 0, 0.0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        key_gray, gray, p0, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if p1 is None or st is None:
        return 0, 0.0
    ok = st.reshape(-1) == 1
    if not ok.any():
        return 0, 0.0
    d = np.linalg.norm((p1[ok] - p0[ok]).reshape(-1, 2), axis=1)
    return int(ok.sum()), float(np.median(d))


def select_keyframes(bmps_dir: Path, stems,
                     drift_px: float = DEFAULT_DRIFT_PX,
                     min_tracked: int = DEFAULT_MIN_TRACKED,
                     max_corners: int = DEFAULT_MAX_CORNERS):
    """-> (keyframes, info): ordered keyframe stems + per-keyframe
    [(stem, reason, tracked, drift)] (reason = first / drift>=X / tracked<N).
    `stems` must be in time order. A keyframe too textureless to detect
    corners degrades to accepting every frame (same as no keyframing)."""
    stems = list(stems)
    if not stems:
        return [], []
    key_gray = load_gray(bmps_dir, stems[0])
    p0 = _detect(key_gray, max_corners)
    keys = [stems[0]]
    info = [(stems[0], "first", 0, 0.0)]
    for s in stems[1:]:
        gray = load_gray(bmps_dir, s)
        n_ok, drift = _track(key_gray, gray, p0)
        if n_ok < min_tracked or drift >= drift_px:
            reason = (f"tracked {n_ok}<{min_tracked}" if n_ok < min_tracked
                      else f"drift {drift:.1f}>={drift_px:g}")
            keys.append(s)
            info.append((s, reason, n_ok, drift))
            key_gray = gray
            p0 = _detect(key_gray, max_corners)
    return keys, info


def candidate_pairs(ordered_stems, embeddings,
                    window: int = DEFAULT_WINDOW, topk: int = DEFAULT_TOPK):
    """-> {(stem_a, stem_b)} with a before b: per keyframe, the previous
    `window` keyframes plus the `topk` most cosine-similar prior keyframes.
    `embeddings`: {stem: 1064 raw uint8 bytes}."""
    stems = list(ordered_stems)
    n = len(stems)
    if n < 2 or (window <= 0 and topk <= 0):
        return set()
    e = np.stack([np.frombuffer(embeddings[s], dtype=np.uint8).astype(np.float32)
                  for s in stems])
    norms = np.linalg.norm(e, axis=1)
    norms[norms == 0] = 1.0
    sim = (e @ e.T) / norms[:, None] / norms[None, :]
    pairs = set()
    for j in range(1, n):
        cand = set(range(max(0, j - window), j))
        if topk > 0 and j > 1:
            cand.update(int(i) for i in np.argsort(sim[j, :j])[-min(topk, j):])
        for i in cand:
            pairs.add((stems[i], stems[j]))
    return pairs
