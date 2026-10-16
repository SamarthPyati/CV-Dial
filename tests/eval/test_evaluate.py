import argparse
import copy
import csv

import numpy as np
import pytest

from eval import plots
from eval.evaluate import bin_label, evaluate, noisy_keypoints, resolve_run_config
from gen.config import load_config
from gen.generate import generate

GEN = load_config("configs/gen/default.yaml")


@pytest.fixture(scope="module")
def splits(tmp_path_factory):
    root = tmp_path_factory.mktemp("e1")
    flat = copy.deepcopy(GEN)
    flat["pose"]["delta_frac"] = [0.0, 0.0]
    generate(flat, "test", 16, root / "flat", workers=1, stratify=True)
    generate(GEN, "test", 16, root / "parallax", workers=1, stratify=True)
    return root


def _args(split, **kw):
    a = dict(config="configs/eval/default.yaml", split=str(split), oracle=True, kp_noise="0", seg_weights=None,
             pose_weights=None, regress_weights=None, limit=None, set=[])
    a.update(kw)
    return argparse.Namespace(**a)


def _metric(m, s, b="all", noise=0.0):
    return next(r for r in m if r["strategy"] == s and r["bin"] == b and r["kp_noise"] == noise)


def test_bin_labels():
    edges = [0, 15, 30, 45, 65]
    assert [bin_label(x, edges) for x in (0, 14.99, 15, 44.9, 45, 65, 65.1)] == \
        ["0-15", "0-15", "15-30", "30-45", "45-65", "45-65", "out"]


def test_oracle_flat_s2_is_exact(splits, tmp_path):
    run = tmp_path / "run"
    m = evaluate(run, resolve_run_config(run, _args(splits / "flat")))
    assert _metric(m, "S2")["mae"] < 0.01 and _metric(m, "S2")["failure_rate"] == 0
    assert _metric(m, "S0")["mae"] > _metric(m, "S2")["mae"]
    assert {r["bin"] for r in m} == {"0-15", "15-30", "30-45", "45-65", "all"}
    for f in ("config.yaml", "predictions.csv", "metrics.csv"):
        assert (run / f).exists()
    with open(run / "predictions.csv") as fh:
        assert len(list(csv.DictReader(fh))) == 16 * 3            # images x (S0, S1, S2)


def test_oracle_parallax_s2_nonzero_and_grows(splits, tmp_path):
    m = evaluate(tmp_path, resolve_run_config(tmp_path, _args(splits / "parallax")))
    assert _metric(m, "S2")["mae"] > 0.01
    assert _metric(m, "S2", "45-65")["mae"] > _metric(m, "S2", "0-15")["mae"]


def test_kp_noise_levels_and_pairing(splits, tmp_path):
    m = evaluate(tmp_path, resolve_run_config(tmp_path, _args(splits / "flat", kp_noise="0,2")))
    assert _metric(m, "S2", noise=2.0)["mae"] > _metric(m, "S2", noise=0.0)["mae"]
    rec = {"id": "x", "keypoints": [[0.0, 0.0, 2]] * 7}
    assert np.array_equal(noisy_keypoints(rec, 2.0, 0), noisy_keypoints(rec, 2.0, 0))


def test_rerun_reuses_saved_config(splits, tmp_path):
    resolve_run_config(tmp_path, _args(splits / "flat", kp_noise="0,4"))
    cfg = resolve_run_config(tmp_path, _args(splits / "flat"))               # CLI noise ignored on re-run
    assert cfg["kp_noise"] == [0.0, 4.0]


def test_plots_render(splits, tmp_path):
    run = tmp_path / "run"
    evaluate(run, resolve_run_config(run, _args(splits / "parallax", kp_noise="0,1")))
    made = plots.run_plots(run)
    assert {p.name for p in made} >= {"mae_vs_obliquity.png", "mae_bins.png", "failure_gallery_S2.png"}
    out = plots.ablation_bars([run, run], ["a", "b"], tmp_path / "e4.png")
    for p in [*made, out]:
        assert p.stat().st_size > 1000 and p.read_bytes()[:4] == b"\x89PNG"
