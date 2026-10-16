"""Camera pose and plane homographies for the synthetic generator (PLAN §3 step 2).

Frames:
- face pixels: the 1024 px frontal render, x right, y down.
- dial mm: plane z = 0, origin at the face centre, axes aligned with face pixels.
- camera: x right, y down, z forward (OpenCV convention); image pixels via K.
"""

from __future__ import annotations

import cv2
import numpy as np


def rot(axis, angle: float) -> np.ndarray:
    """Rotation matrix (3, 3) about a unit `axis` by `angle` radians (Rodrigues)."""
    return cv2.Rodrigues(np.asarray(axis, float) * angle)[0]


def sample_pose(theta_deg: float, psi_deg: float, roll_deg: float, t) -> tuple[np.ndarray, np.ndarray]:
    """Tilt by theta about an in-plane axis at azimuth psi, then roll the camera.

    t = (tx, ty, dist) in mm, the dial centre in the camera frame. Returns (R, t).
    """
    psi = np.radians(psi_deg)
    R = rot([0, 0, 1], np.radians(roll_deg)) @ rot([np.cos(psi), np.sin(psi), 0], np.radians(theta_deg))
    return R, np.asarray(t, float)


def face_to_mm(c, R_px: float, R_mm: float) -> np.ndarray:
    """M (3, 3): face pixels -> dial mm, centre `c` (face px) -> origin, scale s = R_mm / R_px."""
    s = R_mm / R_px
    cx, cy = float(c[0]), float(c[1])
    return np.array([[s, 0.0, -s * cx], [0.0, s, -s * cy], [0.0, 0.0, 1.0]])


def plane_homography(K: np.ndarray, R: np.ndarray, t: np.ndarray, M: np.ndarray, z: float = 0.0) -> np.ndarray:
    """Homography (3, 3) face pixels -> image pixels for the plane z = const (mm)."""
    return K @ np.column_stack([R[:, 0], R[:, 1], z * R[:, 2] + t]) @ M


def obliquity_deg(R: np.ndarray, t: np.ndarray) -> float:
    """Angle (deg) between the dial-face normal and the camera ray to the dial centre."""
    c = abs(R[:, 2] @ (t / np.linalg.norm(t)))
    return float(np.degrees(np.arccos(min(c, 1.0))))


def intrinsics_from_hfov(hfov_deg: float, width: int, height: int) -> np.ndarray:
    """Pinhole K (3, 3) with square pixels and the principal point at the image centre."""
    fx = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[fx, 0.0, width / 2], [0.0, fx, height / 2], [0.0, 0.0, 1.0]])
