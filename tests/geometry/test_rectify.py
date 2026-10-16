import cv2
import numpy as np
import pytest

from geometry.reading import apply_h, read
from geometry.rectify import fix_orientation, read_strategy, rectify, s1_affine, s2_projective

ANGLES = np.arange(0, 71, 5)


@pytest.mark.parametrize("theta", ANGLES)
def test_s2_noise_free_recovery(gauge, theta):
    rng = np.random.default_rng(1000 + int(theta))
    for sweep in (180.0, 240.0, 270.0, 300.0):
        g = gauge(rng, float(theta), sweep_deg=sweep)
        f, lin = read(g["kp"], s2_projective(g["kp"]))
        assert abs(f - g["f"]) < 1e-6, (theta, sweep, g["obliquity_deg"])
        assert lin < 1e-6


def test_s0_error_grows_with_obliquity(gauge):
    def mean_err(theta):
        rng = np.random.default_rng(7)   # same psi, roll, f, framing at both angles
        errs = [abs(read(g["kp"])[0] - g["f"]) for g in (gauge(rng, theta) for _ in range(200))]
        return np.mean(errs)

    assert mean_err(60.0) > mean_err(15.0)


def test_s1_exact_without_perspective_distortion(gauge):
    # Near-affine view (small dial far away): S1 should beat S0 by a wide margin at 50 deg.
    rng = np.random.default_rng(3)
    e0, e1 = [], []
    for _ in range(50):
        g = gauge(rng, 50.0)
        e0.append(abs(read(g["kp"])[0] - g["f"]))
        H, status = rectify("S1", g["kp"], rim=g["rim"])
        assert status == "ok"
        e1.append(abs(read(g["kp"], H)[0] - g["f"]))
    assert np.mean(e1) < 0.2 * np.mean(e0)


@pytest.mark.parametrize("angle", [0.0, 17.0, 45.0, 90.0, 123.0, 170.0])
@pytest.mark.parametrize("axes", [(220, 110), (110, 220), (200, 150)])
def test_fitellipse_convention(angle, axes):
    canvas = np.zeros((600, 600), np.uint8)
    cv2.ellipse(canvas, (300, 300), axes, angle, 0, 360, 255, -1)
    A = s1_affine(canvas)
    cnts, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    r = np.linalg.norm(apply_h(A, cnts[0].reshape(-1, 2).astype(float)), axis=1)
    assert np.std(r) / np.mean(r) < 0.005
    assert np.linalg.det(A[:2, :2]) > 0


def test_fix_orientation_unmirrors():
    flip = np.diag([-1.0, 1.0, 1.0])
    H = fix_orientation(flip, (10.0, 20.0))
    assert np.linalg.det(H[:2, :2]) > 0


def test_s2_never_mirrors(gauge):
    rng = np.random.default_rng(11)
    for _ in range(100):
        g = gauge(rng, rng.uniform(0, 70))
        H = s2_projective(g["kp"])
        x = np.array([*g["kp"][0], 1.0])
        w = H[2] @ x
        J = (H[:2, :2] * w - np.outer(H[:2] @ x, H[2, :2])) / w**2
        assert np.linalg.det(J) > 0


def test_rectify_never_raises():
    bad = [np.zeros((7, 2)), np.full((7, 2), np.nan), np.arange(14.0).reshape(7, 2)]
    for kp in bad:
        for s in ("S0", "S1", "S2", "S9"):
            H, status = rectify(s, kp, mask=np.zeros((10, 10), np.uint8))
            assert (H is None) == status.startswith("fail:")


def test_read_strategy_failure_is_nan_not_fallback():
    out = read_strategy("S2", np.zeros((7, 2)))
    assert np.isnan(out["f"]) and out["status"].startswith("fail:")


def test_s2_linearity_gate(gauge):
    rng = np.random.default_rng(5)
    g = gauge(rng, 40.0)
    assert read_strategy("S2", g["kp"], max_linearity=0.03)["status"] == "ok"
    kp = g["kp"].copy()
    kp[4] += 50.0          # corrupt the 50 % scale point (~0.05 linearity error)
    out = read_strategy("S2", kp, max_linearity=0.03)
    assert out["status"] == "fail:linearity"
    assert np.isnan(out["f"]) and out["linearity_error"] > 0.03
