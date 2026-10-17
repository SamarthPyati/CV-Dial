"""Per-phone checkerboard calibration (PLAN §6).

    python -m realdata.calibrate --images 'data/real/calib/pixel7/*.jpg' --out data/real/calib/pixel7.yaml

Writes K (3, 3), distortion coefficients, image size, RMS reprojection error and the images used.
Frame: image px, x right, y down, pixel centres at integer coordinates (OpenCV convention).
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
import yaml

from gen.config import load_config

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)


def board_points(pattern: tuple[int, int], square: float) -> np.ndarray:
    """(N, 3) float32 object points of the inner corners on z = 0, in the units of `square`."""
    cols, rows = pattern
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
    return np.column_stack([grid * square, np.zeros(len(grid), np.float32)])


def find_corners(gray: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    ok, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)


def calibrate(images: list[np.ndarray], pattern: tuple[int, int], square: float) -> dict:
    """Calibrate from grayscale images; raises if fewer than 3 boards are found."""
    obj, img_pts, used = [], [], []
    size = None
    for i, gray in enumerate(images):
        size = size or gray.shape[::-1]
        if gray.shape[::-1] != size:
            raise ValueError("all calibration images must have the same size")
        c = find_corners(gray, pattern)
        if c is not None:
            obj.append(board_points(pattern, square))
            img_pts.append(c)
            used.append(i)
    if len(used) < 3:
        raise ValueError(f"checkerboard found in only {len(used)} images; need >= 3")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img_pts, size, None, None)
    return {"K": K.tolist(), "dist": dist.ravel().tolist(), "image_size": list(size), "rms_px": float(rms),
            "used": used}


def load_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    c = load_config(path)
    return np.asarray(c["K"], float), np.asarray(c["dist"], float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="glob of checkerboard photos from ONE phone")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/real/realdata.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)["calibration"]
    paths = sorted(glob.glob(a.images))
    grays = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]
    res = calibrate(grays, tuple(cfg["pattern"]), cfg["square_mm"])
    res["used"] = [paths[i] for i in res["used"]]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(yaml.safe_dump(res, sort_keys=False))
    print(f"{len(res['used'])}/{len(paths)} boards, RMS {res['rms_px']:.3f} px -> {a.out}")


if __name__ == "__main__":
    main()
