import json

import numpy as np
import pytest

from gen.config import load_config
from gen.generate import encode_jpeg, generate, make_sample
from gen.preview import contact_sheet
from geometry.rectify import read_strategy

CFG = load_config("configs/gen/default.yaml")
RECORD_KEYS = {"id", "seed", "image", "size", "f", "value", "range", "units", "sweep_deg", "obliquity_deg",
               "tilt_axis_deg", "roll_deg", "delta_frac", "K", "H_face", "H_needle", "keypoints",
               "face_polygon", "bbox", "nuisance", "background"}


@pytest.fixture(scope="module")
def samples():
    return [make_sample(CFG, "test", i) for i in range(8)]


def test_record_schema(samples):
    for img, rec in samples:
        assert RECORD_KEYS <= rec.keys()
        W, H = rec["size"]
        assert img.shape == (H, W, 3) and img.dtype == np.uint8
        assert len(rec["keypoints"]) == 7 and all(len(k) == 3 for k in rec["keypoints"])
        assert len(rec["face_polygon"]) == 64
        x0, y0, x1, y1 = rec["bbox"]
        assert 0 <= x0 < x1 <= W - 1 and 0 <= y0 < y1 <= H - 1
        assert 0 <= rec["obliquity_deg"] <= CFG["pose"]["max_obliquity_deg"]
        json.dumps(rec)


def test_s2_recovers_f_on_flat_needle(samples):
    for _, rec in samples:
        assert rec["delta_frac"] == 0.0
        out = read_strategy("S2", np.asarray(rec["keypoints"])[:, :2])
        assert out["status"] == "ok"
        assert abs(out["f"] - rec["f"]) < 1e-4


def test_obliquity_label_consistent_with_homography(samples):
    # Rebuild the label from K and H_face alone: K^-1 H_face = [r1 r2 t] M up to scale,
    # so the face normal is r1 x r2 and the dial centre's ray is K^-1 H_face c.
    for _, rec in samples:
        Kinv = np.linalg.inv(np.asarray(rec["K"]))
        B = Kinv @ np.asarray(rec["H_face"])
        n = np.cross(B[:, 0], B[:, 1])
        c = (rec["render"]["render_size"] - 1) / 2
        ray = B @ np.array([c, c, 1.0])
        label = np.degrees(np.arccos(abs(n @ ray) / (np.linalg.norm(n) * np.linalg.norm(ray))))
        assert label == pytest.approx(rec["obliquity_deg"], abs=1e-6)


def test_deterministic_per_index():
    a_img, a_rec = make_sample(CFG, "train", 3)
    b_img, b_rec = make_sample(CFG, "train", 3)
    c_img, _ = make_sample(CFG, "train", 4)
    assert encode_jpeg(a_img, 95) == encode_jpeg(b_img, 95) and a_rec == b_rec
    assert not np.array_equal(a_img, c_img)


def test_generate_writes_and_previews(tmp_path):
    generate(CFG, "val", 3, tmp_path, workers=1)
    assert len(list((tmp_path / "images").glob("*.jpg"))) == 3
    assert len(list((tmp_path / "records").glob("*.json"))) == 3
    assert (tmp_path / "config.yaml").exists()
    sheet = contact_sheet(tmp_path, 3, cols=2, thumb=128)
    assert sheet.shape == (256, 256, 3)
