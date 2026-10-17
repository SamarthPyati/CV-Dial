"""Obliquity labels for real photos (PLAN §6).

    python -m realdata.aruco_angle --split data/real --calib data/real/calib   # dir of <phone>.yaml, or one file

Per record: detect the DICT_4X4_50 marker taped coplanar with the dial face, solvePnP (IPPE_SQUARE), and take
obliquity = angle between the marker normal and the camera ray through the annotated pivot (K0).
Fallback when no usable marker: theta ~ arccos(b / a) of the ellipse fitted to the annotated face rim.
Updates records/<id>.json in place (obliquity_deg, obliquity_bin, obliquity_method, obliquity_detail) and writes
obliquity.csv. Frame: image px, x right, y down, pixel centres at integer coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from gen.config import load_config
from realdata.calibrate import load_calibration


def make_detector(dictionary: str = "DICT_4X4_50") -> cv2.aruco.ArucoDetector:
    aruco = cv2.aruco
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return aruco.ArucoDetector(aruco.getPredefinedDictionary(getattr(aruco, dictionary)), params)


def marker_object_points(size_m: float) -> np.ndarray:
    """Corner order of ArucoDetector (TL, TR, BR, BL) on the marker plane z = 0, as IPPE_SQUARE requires."""
    h = size_m / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)


def ray_angle_deg(normal: np.ndarray, pivot_px, K: np.ndarray, dist: np.ndarray) -> float:
    """Angle (deg) between a plane normal (camera frame) and the undistorted camera ray through pivot_px."""
    xy = cv2.undistortPoints(np.asarray(pivot_px, np.float64).reshape(1, 1, 2), K, dist).reshape(2)
    ray = np.array([xy[0], xy[1], 1.0])
    c = abs(normal @ ray) / (np.linalg.norm(normal) * np.linalg.norm(ray))
    return float(np.degrees(np.arccos(min(c, 1.0))))


def aruco_obliquity(gray: np.ndarray, pivot_px, K: np.ndarray, dist: np.ndarray, detector, size_m: float,
                    max_reproj_px: float) -> tuple[float | None, dict]:
    """Obliquity from the largest detected marker, or (None, reason) if no usable marker."""
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None, {"reason": "no_marker"}
    ids = np.asarray(ids).ravel()                          # (N, 1) in OpenCV 4, (N,) in OpenCV 5
    areas = [abs(cv2.contourArea(c.reshape(4, 2))) for c in corners]
    i = int(np.argmax(areas))
    img_pts = corners[i].reshape(4, 2).astype(np.float64)
    obj = marker_object_points(size_m)
    ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, {"reason": "pnp_failed", "marker_id": int(ids[i])}
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - img_pts) ** 2, axis=1))))
    detail = {"marker_id": int(ids[i]), "reproj_px": err, "marker_area_px": float(areas[i])}
    if err > max_reproj_px:
        return None, {**detail, "reason": "reprojection"}
    R = cv2.Rodrigues(rvec)[0]
    return ray_angle_deg(R[:, 2], pivot_px, K, dist), detail


def ellipse_obliquity(rim_px) -> float:
    """theta ~ arccos(b / a) of the ellipse fitted to >= 5 rim points (weak-perspective approximation)."""
    rim = np.asarray(rim_px, np.float32).reshape(-1, 1, 2)
    if len(rim) < 5:
        raise ValueError("need >= 5 rim points")
    _, (w, h), _ = cv2.fitEllipse(rim)
    a, b = max(w, h), min(w, h)
    return float(np.degrees(np.arccos(np.clip(b / a, 0.0, 1.0))))


def label_record(rec: dict, image_bgr: np.ndarray, K: np.ndarray | None, dist: np.ndarray | None,
                 cfg: dict, detector) -> dict:
    """Return the obliquity fields for one record (does not modify it)."""
    pivot = np.asarray(rec["keypoints"], float)[0, :2]
    detail: dict = {}
    if K is not None and image_bgr is not None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        obl, detail = aruco_obliquity(gray, pivot, K, dist, detector, cfg["marker_size_m"], cfg["max_reproj_px"])
        if obl is not None:
            return {"obliquity_deg": obl, "obliquity_method": "aruco", "obliquity_detail": detail}
    elif K is None:
        detail = {"reason": "no_calibration"}
    rim = rec.get("rim_points") or rec.get("face_polygon")
    try:
        obl = ellipse_obliquity(rim)
        return {"obliquity_deg": obl, "obliquity_method": "ellipse", "obliquity_detail": detail}
    except (ValueError, TypeError, cv2.error):
        return {"obliquity_deg": None, "obliquity_method": "none", "obliquity_detail": detail}


def _calibration_for(rec: dict, calib: Path | None):
    if calib is None:
        return None, None
    path = calib / f"{rec.get('phone')}.yaml" if calib.is_dir() else calib
    return load_calibration(path) if path.exists() else (None, None)


def bin_index(obl: float | None, edges) -> int | None:
    """Index of the obliquity bin (last bin closed), as in synthetic records; None if unlabelled/outside."""
    if obl is None or obl < edges[0] or obl > edges[-1]:
        return None
    return int(min(np.searchsorted(edges, obl, side="right") - 1, len(edges) - 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="data/real", help="dir with records/ and images/")
    ap.add_argument("--calib", default=None, help="calibration YAML, or dir of <phone>.yaml")
    ap.add_argument("--config", default="configs/real/realdata.yaml")
    ap.add_argument("--bins", default="0,15,30,45,65")
    a = ap.parse_args()
    cfg = load_config(a.config)["aruco"]
    edges = [float(x) for x in a.bins.split(",")]
    detector = make_detector(cfg["dictionary"])
    split = Path(a.split)
    calib = Path(a.calib) if a.calib else None
    rows = []
    for p in sorted((split / "records").glob("*.json")):
        rec = json.loads(p.read_text())
        K, dist = _calibration_for(rec, calib)
        img = cv2.imread(str(split / rec["image"]), cv2.IMREAD_COLOR)
        rec.update(label_record(rec, img, K, dist, cfg, detector))
        obl = rec["obliquity_deg"]
        rec["obliquity_bin"] = bin_index(obl, edges)
        p.write_text(json.dumps(rec, indent=1))
        rows.append({"id": rec["id"], "obliquity_deg": obl, "method": rec["obliquity_method"],
                     "bin": rec["obliquity_bin"], **{k: rec["obliquity_detail"].get(k) for k in
                                                     ("marker_id", "reproj_px", "reason")}})
    with open(split / "obliquity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "obliquity_deg", "method", "bin", "marker_id", "reproj_px", "reason"])
        w.writeheader()
        w.writerows(rows)
    methods = {m: sum(r["method"] == m for r in rows) for m in ("aruco", "ellipse", "none")}
    print(f"{len(rows)} records: {methods} -> {split / 'obliquity.csv'}")


if __name__ == "__main__":
    main()
