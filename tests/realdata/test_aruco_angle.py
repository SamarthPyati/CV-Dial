import cv2
import numpy as np
import pytest

from gen.config import load_config
from gen.pose import face_to_mm, intrinsics_from_hfov, obliquity_deg, plane_homography, sample_pose
from realdata.aruco_angle import aruco_obliquity, ellipse_obliquity, label_record, make_detector

CFG = load_config("configs/real/realdata.yaml")["aruco"]
W, H = 1600, 1200


def render_scene(theta, psi, roll, rng):
    """Dial plane z = 0 (mm, centre at origin) with a 40 mm DICT_4X4_50 marker centred at (75, 0) mm.
    Returns (gray image, K, pivot px, true obliquity, rim px)."""
    K = intrinsics_from_hfov(65, W, H)
    side_px, border = 400, 100
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 7, side_px)
    plate = np.full((side_px + 2 * border,) * 2, 255, np.uint8)
    plate[border:border + side_px, border:border + side_px] = marker
    c = (plate.shape[0] - 1) / 2
    M = face_to_mm((c, c), side_px / 2, CFG["marker_size_m"] * 1000 / 2)        # plate px -> mm about its centre
    M = np.array([[1, 0, 75.0], [0, 1, 0], [0, 0, 1]]) @ M                      # marker centre at (75, 0) mm
    for _ in range(100):
        dist = rng.uniform(300, 450)
        t = np.array([*rng.uniform(-0.1, 0.1, 2) * dist, dist])
        R, t = sample_pose(theta, psi, roll, t)
        Hm = plane_homography(K, R, t, M)
        corners = cv2.perspectiveTransform(np.array([[[0, 0], [plate.shape[1], 0], [plate.shape[1], plate.shape[0]],
                                                      [0, plate.shape[0]]]], np.float64), Hm)[0]
        if corners.min() > 5 and corners[:, 0].max() < W - 5 and corners[:, 1].max() < H - 5:
            break
    img = cv2.warpPerspective(plate, Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=128)
    pivot = (K @ t)[:2] / (K @ t)[2]
    ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    Hd = plane_homography(K, R, t, np.eye(3))
    rim = cv2.perspectiveTransform((np.column_stack([np.cos(ang), np.sin(ang)]) * 50)[None], Hd)[0]
    return img, K, pivot, obliquity_deg(R, t), rim


@pytest.mark.parametrize("theta", [5, 15, 25, 35, 45, 55, 65])
def test_aruco_recovers_obliquity_within_1deg(theta):
    rng = np.random.default_rng(theta)
    img, K, pivot, truth, _ = render_scene(theta, rng.uniform(0, 360), rng.normal(0, 10), rng)
    obl, detail = aruco_obliquity(img, pivot, K, np.zeros(5), make_detector(), CFG["marker_size_m"],
                                  CFG["max_reproj_px"])
    assert obl is not None, detail
    assert abs(obl - truth) < 1.0, (obl, truth, detail)


def test_fallback_to_ellipse_without_marker():
    rng = np.random.default_rng(1)
    _, K, pivot, truth, rim = render_scene(40, 30, 0, rng)
    rec = {"keypoints": [[*pivot, 2]] + [[0, 0, 2]] * 6, "rim_points": rim.tolist()}
    blank = np.full((H, W, 3), 128, np.uint8)
    out = label_record(rec, blank, K, np.zeros(5), CFG, make_detector())
    assert out["obliquity_method"] == "ellipse" and out["obliquity_detail"]["reason"] == "no_marker"
    assert abs(out["obliquity_deg"] - truth) < 5.0          # weak-perspective approximation


def test_no_calibration_uses_ellipse_and_no_rim_gives_none():
    rim = [[100 + 50 * np.cos(a), 100 + 25 * np.sin(a)] for a in np.linspace(0, 6, 10)]
    rec = {"keypoints": [[100, 100, 2]] * 7, "rim_points": rim}
    out = label_record(rec, None, None, None, CFG, make_detector())
    assert out["obliquity_method"] == "ellipse" and out["obliquity_deg"] == pytest.approx(60, abs=1)
    out = label_record({"keypoints": [[0, 0, 2]] * 7}, None, None, None, CFG, make_detector())
    assert out["obliquity_method"] == "none" and out["obliquity_deg"] is None


def test_ellipse_obliquity_frontal_circle():
    a = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    assert ellipse_obliquity(np.column_stack([np.cos(a), np.sin(a)]) * 100 + 300) == pytest.approx(0, abs=0.5)
