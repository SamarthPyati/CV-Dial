import copy

import numpy as np
import pytest

from gen.config import load_config
from gen.generate import make_sample
from pipeline.infer import infer, oracle_models
from pipeline.scale_ocr import ScaleReader, available, fit_range, parse_number

CFG = load_config("configs/eval/pipeline.yaml")


def test_parse_number():
    assert parse_number("160") == 160 and parse_number("-0.5") == -0.5 and parse_number("−1") == -1
    assert parse_number("0,25") == 0.25 and parse_number("12.") == 12
    assert parse_number("bar") is None and parse_number("1234567") is None and parse_number("") is None


def test_fit_reads_both_ends_and_rejects_outliers():
    pts = [(0.0, 0.0, 0.9), (0.25, 40.0, 0.8), (0.5, 80.0, 0.9), (0.75, 120.0, 0.9), (1.0, 160.0, 0.95),
           (0.48, 8.0, 0.99), (0.9, 7.0, 0.7)]                       # "bar" read as 8, a tick read as 7
    r = fit_range(pts)
    assert (r.lo, r.hi, r.read_ends, r.how) == (0.0, 160.0, True, "read") and len(r.numerals) == 5


def test_fit_infers_a_missing_end_and_flags_it():
    r = fit_range([(0.0, 0.0, 0.9), (0.25, 4.0, 0.9), (0.5, 8.0, 0.9), (0.75, 12.0, 0.9)])     # max not read
    assert r.lo == 0.0 and r.hi == pytest.approx(16.0) and not r.read_ends and r.how == "inferred"


def test_misread_end_numeral_loses_to_the_fit():
    # "0" at the min mark read as "1"; the other numerals put the line through 0
    pts = [(0.0, 1.0, 0.9), (0.2, 20.0, 0.9), (0.4, 40.0, 0.9), (0.6, 60.0, 0.9), (0.8, 80.0, 0.9), (1.0, 100.0, 0.9)]
    r = fit_range(pts)
    assert (r.lo, r.hi) == (0.0, 100.0) and not r.read_ends


def test_tick_misread_next_to_the_real_zero():
    # both "1" (a tick) and the real "0" read at the min mark; numeral positions put the fit ~0.9 off
    pts = [(-0.004, 1.0, 0.83), (-0.003, 0.0, 0.78), (0.247, 25.0, 1.0), (0.488, 50.0, 1.0), (0.736, 75.0, 1.0),
           (0.999, 100.0, 1.0)]
    r = fit_range(pts)
    assert (r.lo, r.hi, r.read_ends) == (0.0, 100.0, True)


def test_a_real_nonzero_minimum_is_kept():
    r = fit_range([(0.0, 1.0, 0.9), (0.4, 3.0, 0.9), (0.6, 4.0, 0.9), (1.0, 6.0, 0.9)])   # a 1-6 bar gauge
    assert (r.lo, r.hi, r.read_ends) == (1.0, 6.0, True)


def test_inferred_ends_snap_to_the_numeral_step():
    r = fit_range([(0.27, -0.75, 0.9), (0.52, -0.5, 0.9), (0.77, -0.25, 0.9)])       # vacuum gauge, ends unread
    assert (r.lo, r.hi) == (-1.0, 0.0) and str(r.hi) == "0.0" and not r.read_ends


def test_fit_vacuum_gauge():
    r = fit_range([(0.0, -1.0, 0.9), (0.5, -0.5, 0.9), (1.0, 0.0, 0.9)])
    assert (r.lo, r.hi, r.read_ends) == (-1.0, 0.0, True)


@pytest.mark.parametrize("pts", [
    [(0.0, 0.0, 0.9), (1.0, 10.0, 0.9)],                               # only two numerals
    [(0.0, 0.0, 0.9), (0.1, 1.0, 0.9), (0.2, 2.0, 0.9)],              # agreeing numerals cover too little scale
    [(0.0, 0.0, 0.9), (0.5, 50.0, 0.9), (1.0, 7.0, 0.9)],             # no three agree
    [],
])
def test_fit_declines_weak_evidence(pts):
    assert fit_range(pts) is None


@pytest.mark.skipif(not available(), reason="optional `ocr` extra not installed")
def test_reads_range_from_clean_synthetic_dials():
    """End to end with RapidOCR on clean, large, moderately tilted dials (ground-truth keypoints)."""
    gen = load_config("configs/gen/default.yaml")
    gen = copy.deepcopy(gen)
    for n in ("glare", "shadow", "low_light", "blur"):
        gen["nuisance"][n]["enabled"] = False
    gen["pose"]["span_frac"] = [0.6, 0.7]
    gen["pose"]["tilt_deg"] = [0, 40]
    reader = ScaleReader(CFG["scale_ocr"])
    outcomes = []
    for i in range(6):
        img, rec = make_sample(gen, "ocrtest", i)
        res = infer(img, oracle_models(rec, CFG), CFG, ("S2",))
        r = reader(img, res, "S2")
        lo, hi = rec["range"]
        ok = r is not None and abs(r.lo - lo) <= 0.02 * (hi - lo) and abs(r.hi - hi) <= 0.02 * (hi - lo)
        outcomes.append("correct" if ok else ("none" if r is None else f"wrong {r.lo}-{r.hi} vs {lo}-{hi}"))
    assert not [o for o in outcomes if o.startswith("wrong")], outcomes
    assert outcomes.count("correct") >= 4, outcomes
