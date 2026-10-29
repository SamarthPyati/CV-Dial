import numpy as np
import pytest

from gen.config import load_config
from pipeline.demo_app import analyse, build_app, demo_reading, synthetic_example
from pipeline.infer import Models, oracle_models

CFG = load_config("configs/eval/pipeline.yaml")
GEN = load_config("configs/gen/default.yaml")


def test_photo_without_dial_is_graceful():
    img = np.full((480, 640, 3), 90, np.uint8)
    over, rect, rows, md = analyse(img, Models(detect=lambda im: None), CFG, 0, 10)
    assert "No dial found" in md and rect is None and over.shape == img.shape
    assert [r[3] for r in rows] == ["fail:no_dial"] * 4


def test_no_models_loaded_is_graceful():
    _, _, rows, md = analyse(np.zeros((300, 300, 3), np.uint8), Models(), CFG)
    assert "No dial found" in md and all(r[3] == "fail:no_dial" for r in rows)


def test_empty_and_grayscale_inputs():
    assert analyse(None, Models(), CFG)[3].startswith("Upload")
    analyse(np.zeros((200, 200), np.uint8), Models(detect=lambda im: None), CFG)


@pytest.mark.parametrize("bin_name", ["0-15", "45-65"])
def test_synthetic_example_reads_close_to_truth(bin_name):
    img, rec = synthetic_example(GEN, 0, bin_name)
    lo, hi = rec["range"]
    over, rect, rows, md = analyse(img, oracle_models(rec, CFG), CFG, lo, hi, rec)
    lo_b, hi_b = (float(v) for v in bin_name.split("-"))
    assert lo_b <= rec["obliquity_deg"] <= hi_b
    s2 = next(r for r in rows if r[0] == "S2")
    assert s2[3] == "ok" and abs(s2[1] - rec["f"]) < 0.02          # needle parallax only
    assert s2[2] == pytest.approx(lo + s2[1] * (hi - lo), abs=1e-3)
    assert "Reading" in md and "Using S2" in md and "true f" in md
    assert rect.shape == (480, 480, 3) and over.shape == img.shape


def test_fallback_is_demo_only_and_s2_first():
    ok = {"f": 0.3, "status": "ok"}
    bad = {"f": float("nan"), "status": "fail:linearity"}
    assert demo_reading({"strategies": {"S1": ok, "S2": {**ok, "f": 0.4}}}) == ("S2", 0.4)
    assert demo_reading({"strategies": {"S1": ok, "S2": bad}}) == ("S1", 0.3)
    assert demo_reading({"strategies": {"S1": bad, "S2": bad}})[0] is None


def test_rectified_view_puts_k4_up_and_keeps_the_reading():
    from geometry.reading import apply_h, read
    from pipeline.demo_app import RECT_SIZE, display_matrix, rectified_view
    from pipeline.infer import infer
    img, rec = synthetic_example(GEN, 1, "30-45")
    res = infer(img, oracle_models(rec, CFG), CFG)
    G = display_matrix(res, "S2")
    r = apply_h(G, res["keypoints"])
    c = (RECT_SIZE - 1) / 2
    assert np.allclose(r[0], [c, c], atol=1e-6)                       # pivot centred
    assert abs(r[4, 0] - c) < 1e-6 and r[4, 1] < c                    # K4 straight up
    assert read(res["keypoints"], G)[0] == pytest.approx(res["strategies"]["S2"]["f"], abs=1e-9)
    assert rectified_view(res, "S2").shape == (RECT_SIZE, RECT_SIZE, 3)


def test_app_builds():
    assert build_app(Models(), CFG, GEN) is not None


def test_scale_range_priority_and_messages():
    from pipeline.scale_ocr import ScaleRange
    img, rec = synthetic_example(GEN, 0, "15-30")
    models = oracle_models(rec, CFG)
    calls = []

    def reader(image, res, strategy):
        calls.append(strategy)
        return ScaleRange(lo=0.0, hi=200.0, read_ends=True, numerals=[(0, 0, 1), (0.5, 100, 1), (1, 200, 1)])

    _, _, rows, md = analyse(img, models, CFG, None, None, scale_reader=reader)          # range read from the dial
    s2 = next(r for r in rows if r[0] == "S2")
    assert calls == ["S2"] and s2[2] == pytest.approx(200 * s2[1], abs=200 * 5e-5) and "read from 3 numerals" in md
    _, _, rows, md = analyse(img, models, CFG, 0, 10, scale_reader=reader)              # typed range wins
    s2 = next(r for r in rows if r[0] == "S2")
    assert calls == ["S2"] and s2[2] == pytest.approx(10 * s2[1], abs=10 * 5e-5) and "entered" in md
    _, _, rows, md = analyse(img, models, CFG, None, None, scale_reader=lambda *a: None)   # nothing readable
    assert all(r[2] is None for r in rows) and "Could not read the scale numbers" in md
    inferred = ScaleRange(lo=0.0, hi=16.0, read_ends=False, numerals=[(0, 0, 1), (0.25, 4, 1), (0.5, 8, 1)])
    _, _, _, md = analyse(img, models, CFG, None, None, scale_reader=lambda *a: inferred)
    assert "inferred" in md and "check" in md
