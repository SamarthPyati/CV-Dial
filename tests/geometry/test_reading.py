import numpy as np
import pytest

from geometry.reading import apply_h, needle_fraction, read
from tests.geometry.conftest import canonical_keypoints


@pytest.mark.parametrize("sweep", [180.0, 300.0])
@pytest.mark.parametrize("f", [0.0, 1.0, -0.03, 1.03, 0.37])
def test_needle_fraction_edge_cases(sweep, f):
    kp = canonical_keypoints(sweep, -90 - sweep / 2, f, radius=100.0)
    assert needle_fraction(kp[0], kp[1], kp[2], kp[6]) == pytest.approx(f, abs=1e-9)


def test_needle_fraction_is_clockwise_in_image_frame():
    # pivot at origin, min mark pointing left, max pointing right: a 180 deg scale over the top.
    pivot, p_min, p_max = (0, 0), (-1, 0), (1, 0)
    assert needle_fraction(pivot, (0, -1), p_min, p_max) == pytest.approx(0.5)   # straight up (y down)
    assert needle_fraction(pivot, (-1, -1), p_min, p_max) == pytest.approx(0.25)


def test_read_frontal_is_exact_with_zero_linearity():
    kp = canonical_keypoints(270, -225, 0.42, radius=300)
    f, lin = read(kp)
    assert f == pytest.approx(0.42, abs=1e-12)
    assert lin < 1e-12


def test_read_invariant_to_similarity():
    kp = canonical_keypoints(240, -210, 0.8, radius=200)
    a = np.radians(33)
    S = np.array([[2 * np.cos(a), -2 * np.sin(a), 15], [2 * np.sin(a), 2 * np.cos(a), -7], [0, 0, 1]])
    assert read(apply_h(S, kp))[0] == pytest.approx(0.8, abs=1e-12)
