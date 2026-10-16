"""Tilt-correction homographies S0-S2 (PLAN §5).

Frame: all inputs and outputs live in one frame (the 384 px crop in the pipeline,
the full image in tests), x right, y down. Every H maps that frame -> a rectified
frame in which the dial face is a circle (S1/S2) and orientation is preserved.
"""

from __future__ import annotations

import cv2
import numpy as np

from geometry.reading import apply_h, read

STRATEGIES = ("S0", "S1", "S2")
DEFAULT_MAX_LINEARITY = 0.03


def s1_affine(mask: np.ndarray) -> np.ndarray:
    """Affine map (3, 3) turning the fitted face ellipse into a circle (weak perspective).

    mask: (H, W) face mask, non-zero inside the face.
    """
    cnts, _ = cv2.findContours((np.asarray(mask) > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise ValueError("empty mask")
    rim = max(cnts, key=cv2.contourArea).astype(np.float32)
    return ellipse_affine(rim)


def ellipse_affine(rim: np.ndarray) -> np.ndarray:
    """Affine map (3, 3) sending the ellipse fitted to rim points (N, 2) to a circle of radius a."""
    rim = np.asarray(rim, np.float32).reshape(-1, 1, 2)
    if len(rim) < 5:
        raise ValueError("fitEllipse needs at least 5 rim points")
    (cx, cy), (w, h), ang = cv2.fitEllipse(rim)
    if min(w, h) <= 0:
        raise ValueError("degenerate ellipse")
    a, b = max(w, h) / 2, min(w, h) / 2
    # cv2 reports `ang` as the direction of the `w` axis (verified by test_fitellipse_convention).
    t = np.radians(ang if w >= h else ang + 90)   # major-axis direction
    R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    A2 = R.T @ np.diag([1.0, a / b]) @ R
    A = np.eye(3)
    A[:2, :2] = A2
    A[:2, 2] = -A2 @ np.array([cx, cy])
    return A


def hartley_norm(pts: np.ndarray) -> np.ndarray:
    """Similarity (3, 3) moving the centroid to 0 and the mean distance to sqrt(2)."""
    pts = np.asarray(pts, float)
    c = pts.mean(0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(pts - c, axis=1))
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])


def fit_conic(pts: np.ndarray) -> np.ndarray:
    """Symmetric conic matrix (3, 3) through >= 5 points (N, 2), least squares via SVD."""
    pts = np.asarray(pts, float)
    x, y = pts[:, 0], pts[:, 1]
    D = np.column_stack([x * x, x * y, y * y, x, y, np.ones_like(x)])
    a, b, c, d, e, f = np.linalg.svd(D)[2][-1]
    return np.array([[a, b / 2, d / 2], [b / 2, c, e / 2], [d / 2, e / 2, f]])


def ellipse_to_circle(C: np.ndarray) -> np.ndarray:
    """Affine map (3, 3), rotation + axis scaling (never a mirror), sending ellipse C to a circle."""
    Q, q, s = C[:2, :2], C[:2, 2], C[2, 2]
    c0 = -np.linalg.solve(Q, q)
    k = q @ np.linalg.solve(Q, q) - s            # conic is (x - c0)^T Q (x - c0) = k
    if k < 0:
        Q, k = -Q, -k
    lam, V = np.linalg.eigh(Q)
    if lam.min() <= 0:
        raise ValueError("fitted conic is not an ellipse")
    if np.linalg.det(V) < 0:
        V[:, 0] *= -1                             # rotation, never a mirror
    D = np.diag(np.sqrt(lam))
    A = np.eye(3)
    A[:2, :2] = D @ V.T
    A[:2, 2] = -D @ V.T @ c0
    return A


def fix_orientation(H: np.ndarray, p) -> np.ndarray:
    """If H mirrors the image near p the scale would read anticlockwise: compose with a flip."""
    x = np.array([p[0], p[1], 1.0])
    w = H[2] @ x
    J = (H[:2, :2] * w - np.outer(H[:2] @ x, H[2, :2])) / w**2
    return H if np.linalg.det(J) > 0 else np.diag([1.0, -1.0, 1.0]) @ H


def s2_projective(kp: np.ndarray) -> np.ndarray:
    """Rectify from scale points K2..K6 (on a circle) and the pivot K0 (its centre)."""
    kp = np.asarray(kp, float)
    T = hartley_norm(kp[2:7])
    sp, pv = apply_h(T, kp[2:7]), apply_h(T, kp[:1])[0]
    C = fit_conic(sp)
    l = C @ np.array([pv[0], pv[1], 1.0])        # vanishing line = polar of the centre
    if abs(l[2]) < 1e-12:
        raise ValueError("vanishing line through the normalisation origin")
    H1 = np.vstack([[1, 0, 0], [0, 1, 0], l / l[2]])
    H1i = np.linalg.inv(H1)
    H = ellipse_to_circle(H1i.T @ C @ H1i) @ H1 @ T
    return fix_orientation(H, kp[0])


def rectify(strategy: str, kp: np.ndarray, mask: np.ndarray | None = None,
            rim: np.ndarray | None = None) -> tuple[np.ndarray | None, str]:
    """Dispatch S0/S1/S2 -> (H, status). Never raises; on failure H is None and status is 'fail:<reason>'.

    S1 needs either a face `mask` or face `rim` points (N, 2), e.g. the GT polygon in oracle mode.
    """
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            return _rectify(strategy, kp, mask, rim)
    except (ValueError, np.linalg.LinAlgError, cv2.error) as e:
        reason = str(e).splitlines()[0].replace(" ", "_")[:60] if str(e) else type(e).__name__
        return None, f"fail:{reason}"


def _rectify(strategy, kp, mask, rim):
    if strategy == "S0":
        return np.eye(3), "ok"
    if strategy == "S1":
        if rim is not None:
            return ellipse_affine(rim), "ok"
        if mask is None:
            return None, "fail:no_mask"
        return s1_affine(mask), "ok"
    if strategy == "S2":
        if not np.all(np.isfinite(kp)):
            return None, "fail:bad_keypoints"
        H = s2_projective(kp)
        if not np.all(np.isfinite(H)):
            return None, "fail:non_finite"
        return H, "ok"
    return None, f"fail:unknown_strategy_{strategy}"


def read_strategy(strategy: str, kp: np.ndarray, mask: np.ndarray | None = None,
                  rim: np.ndarray | None = None,
                  max_linearity: float = DEFAULT_MAX_LINEARITY) -> dict:
    """Rectify then read. Returns {f, linearity_error, status, H}; failures give f = NaN (no fallback).

    The linearity gate applies to S2 only (PLAN §5): linearity error > max_linearity => failure.
    """
    H, status = rectify(strategy, kp, mask=mask, rim=rim)
    if H is None:
        return {"f": float("nan"), "linearity_error": float("nan"), "status": status, "H": None}
    f, lin = read(kp, H)
    if strategy == "S2" and not lin <= max_linearity:
        return {"f": float("nan"), "linearity_error": lin, "status": "fail:linearity", "H": H}
    return {"f": f, "linearity_error": lin, "status": "ok", "H": H}
