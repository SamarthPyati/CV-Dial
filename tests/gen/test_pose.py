import numpy as np
import pytest

from gen.pose import face_to_mm, intrinsics_from_hfov, obliquity_deg, plane_homography, sample_pose
from geometry.reading import apply_h


def test_face_to_mm_maps_centre_to_origin():
    M = face_to_mm((512, 512), 400, 50)
    assert apply_h(M, [[512, 512]])[0] == pytest.approx([0, 0])
    assert apply_h(M, [[912, 512]])[0] == pytest.approx([50, 0])


@pytest.mark.parametrize("theta", [0.0, 20.0, 65.0])
def test_obliquity_equals_theta_on_optical_axis(theta):
    R, t = sample_pose(theta, 37.0, 5.0, (0, 0, 500))
    assert obliquity_deg(R, t) == pytest.approx(theta, abs=1e-9)


def test_frontal_homography_is_orientation_preserving_similarity():
    K = intrinsics_from_hfov(65, 960, 720)
    R, t = sample_pose(0, 0, 0, (0, 0, 400))
    H = plane_homography(K, R, t, face_to_mm((512, 512), 400, 50))
    H = H / H[2, 2]
    assert apply_h(H, [[512, 512]])[0] == pytest.approx([480, 360])
    assert H[0, 0] == pytest.approx(H[1, 1]) and H[0, 0] > 0
    assert H[0, 1] == pytest.approx(0, abs=1e-12) and H[2, :2] == pytest.approx([0, 0])


def test_needle_plane_parallax():
    K = intrinsics_from_hfov(65, 960, 720)
    R, t = sample_pose(40, 10, 0, (0, 0, 400))
    M = face_to_mm((512, 512), 400, 50)
    p0 = apply_h(plane_homography(K, R, t, M), [[512, 512]])
    p1 = apply_h(plane_homography(K, R, t, M, z=2.0), [[512, 512]])
    assert np.linalg.norm(p1 - p0) > 0.5
