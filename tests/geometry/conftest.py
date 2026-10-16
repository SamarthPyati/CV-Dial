"""Synthetic flat-gauge oracle: canonical keypoints in face pixels, projected with a known pose."""

from __future__ import annotations

import numpy as np
import pytest

from gen.pose import face_to_mm, intrinsics_from_hfov, obliquity_deg, plane_homography, sample_pose
from geometry.reading import apply_h

FACE_SIZE = 1024
IMG_W, IMG_H = 960, 720


def canonical_keypoints(sweep_deg: float, start_deg: float, f: float, radius: float,
                        centre=(FACE_SIZE / 2, FACE_SIZE / 2), needle_frac: float = 0.85) -> np.ndarray:
    """(7, 2) K0..K6 in face pixels (y down). Scale points on a circle of `radius` around the pivot,
    clockwise from `start_deg`; tip at `needle_frac * radius` and angle start + f * sweep."""
    c = np.asarray(centre, float)

    def at(angle_deg, r):
        a = np.radians(angle_deg)
        return c + r * np.array([np.cos(a), np.sin(a)])

    scale = [at(start_deg + q * sweep_deg, radius) for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
    tip = at(start_deg + f * sweep_deg, needle_frac * radius)
    return np.array([c, tip, *scale])


def project_gauge(rng: np.random.Generator, theta_deg: float, *, sweep_deg: float = 270.0,
                  f: float | None = None, delta_frac: float = 0.0, psi_deg: float | None = None,
                  roll_deg: float | None = None) -> dict:
    """Project a flat gauge (needle on plane z = delta) at tilt theta with random psi, roll, f, framing."""
    f = rng.uniform(0, 1) if f is None else f
    psi_deg = rng.uniform(0, 360) if psi_deg is None else psi_deg
    roll_deg = rng.normal(0, 10) if roll_deg is None else roll_deg
    start_deg = -90 - sweep_deg / 2 + rng.uniform(-10, 10)       # centred on top (y down => -90 deg)
    R_px, R_mm = 0.4 * FACE_SIZE, 50.0
    kp_face = canonical_keypoints(sweep_deg, start_deg, f, radius=0.8 * R_px)

    K = intrinsics_from_hfov(rng.uniform(55, 75), IMG_W, IMG_H)
    span = rng.uniform(0.25, 0.7)                                 # dial width / image width
    dist = K[0, 0] * 2 * R_mm / (span * IMG_W)
    off = rng.uniform(-0.15, 0.15, 2) * dist
    R, t = sample_pose(theta_deg, psi_deg, roll_deg, (off[0], off[1], dist))
    M = face_to_mm((FACE_SIZE / 2, FACE_SIZE / 2), R_px, R_mm)
    H_face = plane_homography(K, R, t, M)
    H_needle = plane_homography(K, R, t, M, z=delta_frac * R_mm)

    kp = np.vstack([apply_h(H_needle, kp_face[:2]), apply_h(H_face, kp_face[2:])])
    ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    rim_face = np.column_stack([np.cos(ang), np.sin(ang)]) * R_px + FACE_SIZE / 2
    return {
        "kp": kp, "kp_face": kp_face, "f": f, "rim": apply_h(H_face, rim_face),
        "obliquity_deg": obliquity_deg(R, t), "H_face": H_face, "H_needle": H_needle, "K": K, "R": R, "t": t,
    }


@pytest.fixture
def gauge():
    """Builder: gauge(rng, theta_deg, **kw) -> dict with projected kp (7, 2), f, rim, obliquity_deg, ..."""
    return project_gauge
