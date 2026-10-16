"""Square dial crops shared by export (training) and the inference pipeline.

Frames: image px -> crop px, both x right, y down, pixel centres at integer coordinates.
The crop covers the continuous square [cx - side/2, cx + side/2]^2 (image px) mapped onto [-0.5, size - 0.5]^2.
"""

from __future__ import annotations

import cv2
import numpy as np


def square_crop_box(bbox, margin: float = 0.15, scale: float = 1.0, shift=(0.0, 0.0)) -> tuple[float, float, float]:
    """(cx, cy, side) of the square crop around bbox = (x0, y0, x1, y1).

    side = max(w, h) * (1 + margin) * scale; the centre moves by shift * side (jitter in training only).
    """
    x0, y0, x1, y1 = (float(v) for v in bbox)
    side = max(x1 - x0, y1 - y0) * (1 + margin) * scale
    return (x0 + x1) / 2 + shift[0] * side, (y0 + y1) / 2 + shift[1] * side, side


def crop_matrix(cx: float, cy: float, side: float, size: int) -> np.ndarray:
    """Affine A (3, 3): image px -> crop px."""
    s = size / side
    return np.array([[s, 0.0, -s * (cx - side / 2) - 0.5],
                     [0.0, s, -s * (cy - side / 2) - 0.5],
                     [0.0, 0.0, 1.0]])


def warp_crop(img: np.ndarray, A: np.ndarray, size: int) -> np.ndarray:
    """Apply crop matrix A; pixels outside the image are constant black (decisions log).

    When A shrinks by more than 1.5x the image is first area-downscaled, so the crop does not alias.
    """
    s = A[0, 0]
    if s < 1 / 1.5:
        k = min(1.0, 1.5 * s)
        h, w = img.shape[:2]
        sw, sh = max(1, round(w * k)), max(1, round(h * k))
        img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        kx, ky = sw / w, sh / h                   # resize maps x -> (x + 0.5) * k - 0.5
        S = np.array([[kx, 0, 0.5 * kx - 0.5], [0, ky, 0.5 * ky - 0.5], [0, 0, 1.0]])
        A = A @ np.linalg.inv(S)
    return cv2.warpAffine(img, A[:2], (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
