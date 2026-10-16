import numpy as np
import pytest

from geometry.crop import crop_matrix, square_crop_box, warp_crop
from geometry.reading import apply_h


def test_square_box_and_corners():
    cx, cy, side = square_crop_box((100, 50, 300, 150), margin=0.15)
    assert (cx, cy, side) == pytest.approx((200, 100, 230))
    A = crop_matrix(cx, cy, side, 384)
    corners = apply_h(A, [[cx - side / 2, cy - side / 2], [cx + side / 2, cy + side / 2]])
    assert np.allclose(corners, [[-0.5, -0.5], [383.5, 383.5]])


@pytest.mark.parametrize("side", [120.0, 400.0, 900.0])   # upscale, mild and strong downscale
def test_warp_crop_puts_a_dot_where_the_matrix_says(side):
    img = np.zeros((1000, 1200), np.float32)
    y, x = np.mgrid[0:1000, 0:1200]
    p = np.array([611.3, 487.8])
    img += np.exp(-((x - p[0]) ** 2 + (y - p[1]) ** 2) / (2 * (side / 60) ** 2))
    A = crop_matrix(600, 500, side, 384)
    out = warp_crop(img, A, 384)
    yy, xx = np.mgrid[0:384, 0:384]
    w = out / out.sum()
    centroid = np.array([(xx * w).sum(), (yy * w).sum()])
    assert np.linalg.norm(centroid - apply_h(A, [p])[0]) < 0.3


def test_outside_is_black():
    img = np.full((100, 100, 3), 200, np.uint8)
    out = warp_crop(img, crop_matrix(0, 0, 100, 64), 64)
    assert out[5, 5].max() == 0 and out[60, 60].min() == 200
