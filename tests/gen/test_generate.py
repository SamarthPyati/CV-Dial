import copy
import json

import cv2
import numpy as np
import pytest

from gen.config import load_config
from gen.generate import encode_jpeg, generate, make_sample
from gen.preview import contact_sheet
from geometry.rectify import read_strategy

CFG = load_config("configs/gen/default.yaml")
FLAT = copy.deepcopy(CFG)
FLAT["pose"]["delta_frac"] = [0.0, 0.0]
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


def test_s2_recovers_f_on_flat_needle():
    for i in range(6):
        _, rec = make_sample(FLAT, "test", i)
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


def test_needle_parallax_moves_pivot_and_tip(samples):
    for _, rec in samples:
        assert 0.02 <= rec["delta_frac"] <= 0.05
        kp = np.asarray(rec["keypoints"])[:, :2]
        Hf, Hn = np.asarray(rec["H_face"]), np.asarray(rec["H_needle"])
        # K0 is the needle-plane image of the face centre; the face-plane image differs unless frontal
        c = (rec["render"]["render_size"] - 1) / 2
        on_face = Hf @ [c, c, 1.0]
        on_needle = Hn @ [c, c, 1.0]
        assert np.allclose(kp[0], on_needle[:2] / on_needle[2])
        if rec["obliquity_deg"] > 10:
            assert np.linalg.norm(on_face[:2] / on_face[2] - kp[0]) > 0.05


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


def test_byte_identical_across_runs_and_worker_counts(tmp_path):
    generate(CFG, "train", 20, tmp_path / "a", workers=2)
    generate(CFG, "train", 20, tmp_path / "b", workers=1)
    for sub in ("images", "records"):
        fa = sorted((tmp_path / "a" / sub).iterdir())
        fb = sorted((tmp_path / "b" / sub).iterdir())
        assert [p.name for p in fa] == [p.name for p in fb] and len(fa) == 20
        assert all(x.read_bytes() == y.read_bytes() for x, y in zip(fa, fb))


def _with(cfg, **switches):
    c = copy.deepcopy(cfg)
    for name, (enabled, p) in switches.items():
        c["nuisance"][name]["enabled"] = enabled
        if p is not None:
            c["nuisance"][name]["p"] = p
    return c


@pytest.mark.parametrize("name", ["glare", "shadow", "low_light", "blur"])
def test_disabling_one_nuisance_changes_only_that_effect(name):
    on, off = _with(CFG, **{name: (True, 1.0)}), _with(CFG, **{name: (False, 1.0)})
    for i in range(10):
        img_on, rec_on = make_sample(on, "train", i)
        img_off, rec_off = make_sample(off, "train", i)
        n_on, n_off = rec_on.pop("nuisance"), rec_off.pop("nuisance")
        assert rec_on == rec_off                                  # geometry, render, background unchanged
        assert n_on[name] and not n_off[name]
        for other in ("glare", "shadow", "low_light", "blur", "jpeg_quality"):
            if other != name:
                assert n_on[other] == n_off[other]
        assert {k: v for k, v in n_on["params"].items() if k != name} == n_off["params"]
        assert not np.array_equal(img_on, img_off)


def test_glare_is_clipped_to_face():
    quiet = {n: (False, None) for n in ("shadow", "low_light", "blur", "camera")}
    on, off = _with(CFG, glare=(True, 1.0), **quiet), _with(CFG, glare=(False, 1.0), **quiet)
    for i in range(5):
        img_on, rec = make_sample(on, "train", i)
        img_off, _ = make_sample(off, "train", i)
        W, H = rec["size"]
        face = np.zeros((H, W), np.uint8)
        cv2.fillPoly(face, [np.rint(np.asarray(rec["face_polygon"])).astype(np.int32)], 1)
        face = cv2.dilate(face, np.ones((5, 5), np.uint8))
        changed = np.any(img_on != img_off, axis=2)
        assert changed.any() and not (changed & (face == 0)).any()


def test_stratified_bins():
    edges = CFG["pose"]["obliquity_bins"]
    for i in range(8):
        _, rec = make_sample(CFG, "test", i, stratify=True)
        k = i % (len(edges) - 1)
        assert rec["obliquity_bin"] == k
        assert edges[k] <= rec["obliquity_deg"] <= edges[k + 1]


def test_photo_background(tmp_path):
    bg_dir = tmp_path / "bg" / "team"
    bg_dir.mkdir(parents=True)
    cv2.imwrite(str(bg_dir / "wall.png"), np.random.default_rng(0).integers(0, 255, (300, 400, 3), np.uint8))
    cfg = copy.deepcopy(CFG)
    cfg["background"]["dir"] = str(tmp_path / "bg")
    _, rec = make_sample(cfg, "train", 0)
    assert rec["background"] == "team/wall.png"
    _, rec = make_sample(CFG | {"background": {"dir": str(tmp_path / "none"), "zoom": [1, 2]}}, "train", 0)
    assert rec["background"].startswith("procedural:")
