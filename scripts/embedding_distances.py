#!/usr/bin/env python3
"""All-pairs distance histogram over a semantic run's embeddings: reads
<work>/embeddings/*.npy (raw uint8) and writes a histogram PNG + pairwise CSV.
"""

import argparse
import glob
import os
import sys

import numpy as np


def pairwise(X: np.ndarray, metric: str) -> np.ndarray:
    """n x n ordered-pair distance matrix (diagonal = 0)."""
    Xf = X.astype(np.float32)
    if metric == "l2":
        # ||a-b||^2 = |a|^2 + |b|^2 - 2 a.b  (uint8 values; scale is uniform)
        sq = (Xf * Xf).sum(1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (Xf @ Xf.T)
        return np.sqrt(np.maximum(d2, 0.0))
    if metric == "cosine":
        n = np.linalg.norm(Xf, axis=1, keepdims=True)
        U = Xf / np.maximum(n, 1e-9)
        return 1.0 - U @ U.T
    if metric == "hamming":  # bit-level over the 1064 uint8 bytes
        bits = np.unpackbits(X, axis=1)
        return np.bitwise_xor(bits[:, None, :], bits[None, :, :]).sum(2).astype(np.float32)
    raise ValueError(f"unknown metric {metric!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="semantic_run",
                    help="run dir containing embeddings/ (default semantic_run)")
    ap.add_argument("--metric", default="l2", choices=("l2", "cosine", "hamming"))
    ap.add_argument("--out", default=None, help="output PNG (default <work>/pairwise_<metric>.png)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.work, "embeddings", "*.npy")))
    if not files:
        print(f"no .npy embeddings under {args.work}/embeddings/", file=sys.stderr)
        return 1
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    X = np.stack([np.load(f) for f in files])
    D = pairwise(X, args.metric)
    n = len(names)
    flat = D.ravel()

    print(f"{n} embeddings x {X.shape[1]}-d, metric={args.metric}")
    print(f"all {n * n} ordered pairs (incl. {n} self-pairs): "
          f"min {flat.min():.4g}, mean {flat.mean():.4g}, max {flat.max():.4g}")
    off = D[~np.eye(n, dtype=bool)]
    print(f"excluding self-pairs ({off.size} values): "
          f"min {off.min():.4g}, mean {off.mean():.4g}, max {off.max():.4g}")

    out_png = args.out or os.path.join(args.work, f"pairwise_{args.metric}.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=130)
    ax.hist(flat, bins=30, color="steelblue", edgecolor="white")
    ax.set_xlabel(f"pairwise {args.metric} distance")
    ax.set_ylabel("count")
    ax.set_title(f"All {n}x{n} image-pair distances ({n} embeddings, {args.metric})")
    fig.tight_layout()
    fig.savefig(out_png)

    csv_path = os.path.join(args.work, f"pairwise_{args.metric}.csv")
    np.savetxt(csv_path, D, delimiter=",", fmt="%.4f",
               header=",".join(names), comments="")
    print(f"wrote {out_png} and {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
