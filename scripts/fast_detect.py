#!/usr/bin/env python3
"""Run OpenCV FAST-9/16 corner detection on a grayscale image and label every
feature it finds: numbered markers on a copy of the frame, plus the list of
keypoints printed and saved to <out>.txt.

    python3 fast_detect.py <image> [threshold] [--no-nms] [--out path]

Defaults: threshold 20, nonmax suppression ON (cv2 default), output
`<image_stem>_fast<type>_t<thr>[_nonms].bmp` next to the input.
"""
import argparse
import sys
from pathlib import Path

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
TYPE = cv2.FAST_FEATURE_DETECTOR_TYPE_9_16


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="input image (any OpenCV-readable format)")
    ap.add_argument("threshold", type=int, nargs="?", default=20)
    ap.add_argument("--no-nms", action="store_true", help="disable non-max suppression")
    ap.add_argument("--scale", type=int, default=1, help="downscale factor N before detection (INTER_AREA; default 1 = no downscale)")
    ap.add_argument("--out", help="output image path (default: <stem>_fast9_t<thr>.bmp)")
    args = ap.parse_args()

    src = Path(args.image)
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"cannot read {src}", file=sys.stderr)
        return 1

    if args.scale > 1:
        img = cv2.resize(img, (img.shape[1] // args.scale, img.shape[0] // args.scale),
                         interpolation=cv2.INTER_AREA)

    det = cv2.FastFeatureDetector_create(
        threshold=args.threshold,
        nonmaxSuppression=not args.no_nms,
        type=TYPE,
    )
    kp = det.detect(img, None)

    # 24-bit copy of the frame (like receive_frames.py's _marked.bmp) with
    # every feature numbered at its (x, y).
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, f in enumerate(kp):
        x, y = int(round(f.pt[0])), int(round(f.pt[1]))
        cv2.circle(color, (x, y), 4, (0, 0, 255), 1)   # red ring
        cv2.putText(color, str(i), (x + 5, y - 5), FONT, 0.35, (0, 255, 255), 1)

    ds = f"_ds{args.scale}" if args.scale > 1 else ""
    out = args.out or str(src.with_name(f"{src.stem}_fast9_t{args.threshold}{ds}"
                                        + ("_nonms" if args.no_nms else "") + ".bmp"))
    cv2.imwrite(out, color)

    coords = [(int(round(f.pt[0])), int(round(f.pt[1]))) for f in kp]
    txt = Path(out).with_suffix(".txt")
    txt.write_text("\n".join(f"{x},{y}" for x, y in coords) + "\n")

    print(f"{len(kp)} features (FAST-9/16, t={args.threshold}, "
          f"scale={args.scale}x, NMS={'off' if args.no_nms else 'on'})")
    print("keypoints:", coords)
    print(f"wrote {out} (+ {txt})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
