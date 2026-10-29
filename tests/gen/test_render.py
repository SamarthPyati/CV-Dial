import numpy as np
import pytest

from gen.config import load_config
from gen.render import render_dial
from geometry.reading import read

CFG = load_config("configs/gen/default.yaml")["render"]


@pytest.mark.parametrize("seed", range(12))
def test_keypoints_match_drawn_features(seed):
    d = render_dial(np.random.default_rng(seed), CFG)
    kp = d.keypoints
    assert kp.shape == (7, 2)
    assert np.allclose(kp[0], d.centre)
    # frontal render: reading the canonical keypoints gives f exactly
    f, lin = read(kp)
    assert f == pytest.approx(d.meta["f"], abs=1e-9) and lin < 1e-9
    # scale points on one ring inside the face
    r = np.linalg.norm(kp[2:] - d.centre, axis=1)
    assert np.ptp(r) < 1e-9 and r[0] < d.face_radius
    # the visible needle ends at K1: opaque just inside the tip, transparent just beyond it
    u = (kp[1] - kp[0]) / np.linalg.norm(kp[1] - kp[0])
    inside = np.rint(kp[1] - 3 * u).astype(int)
    beyond = np.rint(kp[1] + 3 * u).astype(int)
    assert d.needle[inside[1], inside[0], 3] > 128
    assert d.needle[beyond[1], beyond[0], 3] < 128


def test_render_is_deterministic():
    a = render_dial(np.random.default_rng(5), CFG)
    b = render_dial(np.random.default_rng(5), CFG)
    assert np.array_equal(a.face, b.face) and np.array_equal(a.needle, b.needle)


def test_value_matches_f_and_range():
    for s in range(30):
        m = render_dial(np.random.default_rng(100 + s), CFG).meta
        lo, hi = m["range"]
        assert m["value"] == pytest.approx(lo + m["f"] * (hi - lo))
        assert -0.04 <= m["f"] <= 1.04


def test_needle_contrasts_with_face():
    """No invisible needles: black needles never sit on dark faces, white ones never on light faces."""
    from gen.render import _luma
    seen = set()
    for s in range(400):
        d = render_dial(np.random.default_rng(1000 + s), CFG)
        st = d.meta["style"]
        face_luma = _luma(st["face_rgb"])                                       # base colour of the face
        seen.add((st["face"], st["needle_colour"]))
        assert not (st["needle_colour"] == "black" and face_luma <= 0.5), (s, st)
        assert not (st["needle_colour"] == "white" and face_luma > 0.5), (s, st)
    assert ("dark", "red") in seen and ("light", "black") in seen
