import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from gen.config import load_config
from geometry.reading import read
from realdata.cvat_to_json import build_records, parse_cvat, read_csv

FIX = Path(__file__).parent / "fixtures"
CFG = load_config("configs/real/realdata.yaml")["cvat"]
SCHEMA = {"id", "seed", "image", "size", "f", "value", "range", "units", "sweep_deg", "obliquity_deg",
          "tilt_axis_deg", "roll_deg", "delta_frac", "K", "H_face", "H_needle", "keypoints", "face_polygon",
          "bbox", "nuisance", "background"}


@pytest.fixture(scope="module")
def converted():
    ann = parse_cvat(FIX / "cvat_sample.xml", CFG)
    return build_records(ann, read_csv(FIX / "states.csv"), read_csv(FIX / "images.csv"), CFG)


def test_schema_matches_synthetic(converted):
    records, _ = converted
    assert len(records) == 2
    for rec in records:
        assert SCHEMA <= rec.keys()
        assert rec["H_face"] is None and rec["K"] is None and rec["seed"] is None
        assert len(rec["keypoints"]) == 7 and len(rec["face_polygon"]) == CFG["polygon_points"]
        json.dumps(rec)


def test_incomplete_image_is_reported_not_converted(converted):
    _, problems = converted
    assert len(problems) == 1 and problems[0].startswith("IMG_0003.jpg") and "rim" in problems[0]


def test_round_trip_coordinates_back_to_cvat(converted):
    """Every coordinate in the record + 0.5 equals the hand-written CVAT value (edge -> centre convention)."""
    records, _ = converted
    root = ET.parse(FIX / "cvat_sample.xml").getroot()
    for rec in records:
        im = next(i for i in root.iter("image") if f"images/{i.get('name')}" == rec["image"])
        assert rec["size"] == [int(im.get("width")), int(im.get("height"))]
        pts = {p.get("label"): p for p in im.find("skeleton").findall("points")}
        for k, (x, y, v) in enumerate(rec["keypoints"]):
            p = pts[f"K{k}"]
            ex, ey = (float(t) for t in p.get("points").split(","))
            assert (x + 0.5, y + 0.5) == pytest.approx((ex, ey))
            assert v == (0 if p.get("outside") == "1" else 2)
        box = im.find("box")
        assert [b + 0.5 for b in rec["bbox"]] == pytest.approx([float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")])
        rim_xml = next(e for e in im if e.get("label") == "rim").get("points")
        rim_xml = np.array([[float(t) for t in q.split(",")] for q in rim_xml.split(";")])
        assert np.allclose(np.asarray(rec["rim_points"]) + 0.5, rim_xml)


def test_reading_labels(converted):
    r1, r2 = sorted(converted[0], key=lambda r: r["id"])
    assert r1["id"] == "real_IMG_0001" and r1["f"] == pytest.approx(0.42) and r1["phone"] == "pixel7"
    assert r2["value"] == pytest.approx(-0.375) and r2["f"] == pytest.approx(0.625)
    assert r2["reading_disagreement"] == pytest.approx(0.05) and r2["range"] == [-1.0, 0.0]


def test_keypoint_order_survives_conversion(converted):
    """The frontal fixture gauge reads 0.42 from the converted keypoints (catches any K-order mix-up)."""
    rec = next(r for r in converted[0] if r["id"] == "real_IMG_0001")
    f, lin = read(np.asarray(rec["keypoints"])[:, :2])
    assert f == pytest.approx(rec["f"], abs=1e-3) and lin < 1e-3


def test_face_polygon_is_fitted_circle(converted):
    rec = next(r for r in converted[0] if r["id"] == "real_IMG_0001")
    r = np.linalg.norm(np.asarray(rec["face_polygon"]) - [500, 500], axis=1)
    assert np.allclose(r, 380, atol=0.5)
