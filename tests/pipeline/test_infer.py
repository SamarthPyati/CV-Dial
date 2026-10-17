import copy

import numpy as np
import pytest

from gen.config import load_config
from gen.generate import make_sample
from pipeline.infer import STRATEGIES, Models, format_table, infer, oracle_models

CFG = load_config("configs/eval/pipeline.yaml")
GEN = load_config("configs/gen/default.yaml")
FLAT = copy.deepcopy(GEN)
FLAT["pose"]["delta_frac"] = [0.0, 0.0]


@pytest.mark.parametrize("index", range(5))
def test_oracle_pipeline_recovers_f(index):
    img, rec = make_sample(FLAT, "test", index, stratify=True)
    res = infer(img, oracle_models(rec, CFG), CFG)
    s = res["strategies"]
    assert all(s[k]["status"] == "ok" for k in STRATEGIES), {k: v["status"] for k, v in s.items()}
    assert abs(s["S2"]["f"] - rec["f"]) < 1e-6                   # exact: flat needle, true keypoints
    assert abs(s["S1"]["f"] - rec["f"]) < 0.03                   # mask from the face polygon, weak perspective
    assert s["S3"]["f"] == rec["f"]
    assert res["crop"].shape == (384, 384, 3) and res["mask"].shape == (384, 384) and res["mask"].any()
    print(format_table(rec["id"], res))


def test_missed_detection_fails_every_strategy():
    img, _ = make_sample(GEN, "test", 0)
    res = infer(img, Models(detect=lambda img: None, keypoints=lambda c: None, regress=lambda c: 0.5), CFG)
    assert res["status"] == "fail:no_dial"
    assert all(r["status"] == "fail:no_dial" and np.isnan(r["f"]) for r in res["strategies"].values())


@pytest.mark.parametrize("bad", [None, np.zeros((4, 4, 3), np.uint8), np.zeros((100, 100), np.uint8)])
def test_bad_image_never_raises(bad):
    res = infer(bad, Models(detect=lambda img: (np.array([0, 0, 10, 10.0]), None)), CFG)
    assert all(r["status"].startswith("fail:") for r in res["strategies"].values())


def test_adapter_exception_is_contained():
    def boom(img):
        raise RuntimeError("model exploded")
    img, _ = make_sample(GEN, "test", 1)
    res = infer(img, Models(detect=boom), CFG)
    assert all(r["status"] == "fail:error_RuntimeError" for r in res["strategies"].values())


def test_missing_keypoints_and_models():
    img, rec = make_sample(GEN, "test", 2)
    m = oracle_models(rec, CFG)
    res = infer(img, Models(detect=m.detect, keypoints=lambda c: None, regress=None), CFG)
    s = res["strategies"]
    assert {s[k]["status"] for k in ("S0", "S1", "S2")} == {"fail:no_keypoints"}
    assert s["S3"]["status"] == "fail:no_model"
