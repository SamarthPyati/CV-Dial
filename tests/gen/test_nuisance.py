import numpy as np

from gen.config import load_config
from gen.nuisance import ORDER, apply_nuisances

CFG = load_config("configs/gen/default.yaml")["nuisance"]


def _run(cfg, seed=0):
    img = np.full((120, 160, 3), 128, np.float32)
    mask = np.zeros((120, 160), np.float32)
    mask[30:90, 40:120] = 1
    poly = np.array([[40, 30], [119, 30], [119, 89], [40, 89]], float)
    streams = {n: np.random.default_rng([seed, i]) for i, n in enumerate(ORDER)}
    return apply_nuisances(img, cfg, streams, mask, poly, 95)


def test_all_disabled_is_identity_and_default_quality():
    cfg = {n: {**c, "enabled": False} for n, c in CFG.items()}
    out, rec = _run(cfg)
    assert np.array_equal(out, np.full((120, 160, 3), 128, np.float32))
    assert rec["jpeg_quality"] == 95 and rec["params"] == {}


def test_output_range_and_record():
    cfg = {n: {**c, "p": 1.0} if "p" in c else c for n, c in CFG.items()}
    out, rec = _run(cfg, seed=3)
    assert out.dtype == np.float32 and out.min() >= 0 and out.max() <= 255
    assert rec["glare"] and rec["shadow"] and rec["low_light"] and rec["blur"] in ("motion", "defocus")
    assert 40 <= rec["jpeg_quality"] <= 95


def test_low_light_darkens():
    cfg = {n: {**c, "enabled": n == "low_light", "p": 1.0} if "p" in c else {**c, "enabled": False}
           for n, c in CFG.items()}
    out, _ = _run(cfg)
    assert out.mean() < 128 * 0.8
