"""Needle reading from keypoints (PLAN §5).

Frame: whatever frame the keypoints are in (image or rectified), x right, y down.
With y down, atan2 angles increase clockwise, which is the scale direction.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2 * np.pi


def apply_h(H: np.ndarray, pts) -> np.ndarray:
    """Apply homography H (3, 3) to points (N, 2); returns (N, 2) float64."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    p = np.column_stack([pts, np.ones(len(pts))]) @ np.asarray(H, float).T
    return p[:, :2] / p[:, 2:3]


def needle_fraction(pivot, tip, p_min, p_max) -> float:
    """Share of full scale, clockwise from the min mark. Image coordinates (y down).

    Angles in the dead zone (outside the sweep) are split at its midpoint: the half next
    to the min mark reads slightly negative, the half next to the max mark reads > 1.
    """
    def ang(p):
        return np.arctan2(p[1] - pivot[1], p[0] - pivot[0])

    sweep = (ang(p_max) - ang(p_min)) % TWO_PI
    rel = (ang(tip) - ang(p_min)) % TWO_PI
    if rel > sweep + (TWO_PI - sweep) / 2:   # needle rests just below the min mark
        rel -= TWO_PI
    return float(rel / sweep)


def read(kp, H: np.ndarray | None = None) -> tuple[float, float]:
    """kp: (7, 2) in K0..K6 order, in the frame H maps from. Returns (f, linearity_error)."""
    r = apply_h(np.eye(3) if H is None else H, kp)
    f = needle_fraction(r[0], r[1], r[2], r[6])
    mids = [needle_fraction(r[0], r[i], r[2], r[6]) for i in (3, 4, 5)]
    return f, float(np.mean(np.abs(np.array(mids) - [0.25, 0.50, 0.75])))
