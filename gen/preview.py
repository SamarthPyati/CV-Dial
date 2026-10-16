"""Contact sheet of generated images with the 7 keypoints (K0..K6) and face polygon drawn.

Usage:
    python -m gen.preview --src data/synth/train --n 16 --out results/preview.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

KP_COLOURS = [(255, 0, 255), (0, 0, 255), (0, 200, 0), (0, 200, 200), (255, 160, 0), (200, 200, 0), (255, 0, 0)]  # BGR


def annotate(img: np.ndarray, rec: dict) -> np.ndarray:
    """Draw the face polygon, keypoints and labels on a BGR image (image frame)."""
    out = img.copy()
    poly = np.asarray(rec["face_polygon"], np.float32)
    cv2.polylines(out, [np.rint(poly).astype(np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)
    x0, y0, x1, y1 = (int(round(v)) for v in rec["bbox"])
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 0), 1)
    kp = np.asarray(rec["keypoints"], float)
    cv2.line(out, tuple(np.rint(kp[0, :2]).astype(int)), tuple(np.rint(kp[1, :2]).astype(int)), (0, 0, 255), 1, cv2.LINE_AA)
    for i, (x, y, v) in enumerate(kp):
        p = (int(round(x)), int(round(y)))
        cv2.circle(out, p, 5, KP_COLOURS[i], 2 if v > 0 else 1, cv2.LINE_AA)
        cv2.putText(out, f"K{i}", (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, f"K{i}", (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, KP_COLOURS[i], 1, cv2.LINE_AA)
    text = f"{rec['id']}  f={rec['f']:.3f}  obl={rec['obliquity_deg']:.1f}"
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def contact_sheet(src: Path, n: int, cols: int = 4, thumb: int = 480) -> np.ndarray:
    recs = sorted((src / "records").glob("*.json"))[:n]
    if not recs:
        raise FileNotFoundError(f"no records under {src / 'records'}")
    tiles = []
    for p in recs:
        rec = json.loads(p.read_text())
        img = annotate(cv2.imread(str(src / rec["image"])), rec)
        h, w = img.shape[:2]
        s = thumb / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        tile = np.full((thumb, thumb, 3), 40, np.uint8)
        tile[: img.shape[0], : img.shape[1]] = img
        tiles.append(tile)
    while len(tiles) % cols:
        tiles.append(np.full((thumb, thumb, 3), 40, np.uint8))
    rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--thumb", type=int, default=480)
    a = ap.parse_args()
    sheet = contact_sheet(Path(a.src), a.n, a.cols, a.thumb)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, sheet)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
