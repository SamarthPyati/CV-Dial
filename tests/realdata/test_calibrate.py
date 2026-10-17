import cv2
import numpy as np

from gen.pose import intrinsics_from_hfov, sample_pose
from realdata.calibrate import calibrate


def test_recovers_known_intrinsics():
    W, H, pattern, sq_px = 1280, 960, (9, 6), 60
    board = np.full(((pattern[1] + 3) * sq_px, (pattern[0] + 3) * sq_px), 255, np.uint8)
    for r in range(pattern[1] + 1):
        for c in range(pattern[0] + 1):
            if (r + c) % 2 == 0:
                y, x = (r + 1) * sq_px, (c + 1) * sq_px
                board[y:y + sq_px, x:x + sq_px] = 0
    K = intrinsics_from_hfov(62, W, H)
    rng = np.random.default_rng(0)
    imgs = []
    bc = np.array([(board.shape[1] - 1) / 2, (board.shape[0] - 1) / 2])
    S = np.array([[1, 0, -bc[0]], [0, 1, -bc[1]], [0, 0, 1.0]])                       # board px -> plane (px = mm)
    for i in range(12):
        R, t = sample_pose(rng.uniform(10, 45), rng.uniform(0, 360), rng.normal(0, 10),
                           (*rng.uniform(-60, 60, 2), rng.uniform(900, 1300)))
        Hm = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ S
        imgs.append(cv2.warpPerspective(board, Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=200))
    res = calibrate(imgs, pattern, 25.0)
    Kh = np.asarray(res["K"])
    assert len(res["used"]) >= 8 and res["rms_px"] < 0.5
    assert abs(Kh[0, 0] / K[0, 0] - 1) < 0.01 and abs(Kh[1, 1] / K[1, 1] - 1) < 0.01
    assert np.allclose(Kh[:2, 2], K[:2, 2], atol=8)
