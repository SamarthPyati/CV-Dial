"""CVAT export + reading sheets -> records in the synthetic JSON schema (PLAN §6). Real data is TEST ONLY.

    python -m realdata.cvat_to_json --cvat data/real/annotations/annotations.xml \
        --states data/real/annotations/states.csv --images-csv data/real/annotations/images.csv --out data/real

Inputs
- CVAT for images 1.1 XML: per image a `dial` box, a skeleton with points K0..K6 (outside="1" -> v = 0),
  and >= 5 face-rim points labelled `rim` (points, polyline or polygon). Labels are set in configs/real/realdata.yaml.
- states.csv: state_id, gauge_id, min, max, units, value_a, value_b   (two independent readings of the
  frontal reference photo; the label is their mean, |a - b| is kept as `reading_disagreement`)
- images.csv: image, state_id, phone   (which gauge state each photo shows, and which phone took it)

CVAT coordinates are pixel-edge (image spans [0, W]); records use pixel centres, so 0.5 is subtracted.
Fields that do not apply (seed, K, H_face, H_needle, sweep_deg, pose, nuisance, background) are null;
obliquity is filled in later by realdata.aruco_angle. The original rim points are kept as `rim_points`.
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from gen.config import load_config

EDGE_TO_CENTRE = -0.5


def _pts(s: str) -> np.ndarray:
    return np.array([[float(v) for v in p.split(",")] for p in s.strip().split(";") if p], float)


def parse_cvat(xml_path: str | Path, cfg: dict) -> dict[str, dict]:
    """image name -> {size, box, keypoints (7, 3) edge coords + v, rim (N, 2) edge coords}; missing parts are None."""
    root = ET.parse(xml_path).getroot()
    kp_labels = cfg["keypoint_labels"]
    out = {}
    for im in root.iter("image"):
        entry = {"size": [int(im.get("width")), int(im.get("height"))], "box": None, "keypoints": None, "rim": None}
        for box in im.findall("box"):
            if box.get("label") == cfg["box_label"]:
                entry["box"] = [float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")]
        for sk in im.findall("skeleton"):
            found = {}
            for p in sk.findall("points"):
                if p.get("label") in kp_labels:
                    xy = _pts(p.get("points"))[0]
                    found[p.get("label")] = [xy[0], xy[1], 0 if p.get("outside") == "1" else 2]
            if all(lbl in found for lbl in kp_labels):
                entry["keypoints"] = np.array([found[lbl] for lbl in kp_labels], float)
        for tag in ("points", "polyline", "polygon"):
            for p in im.findall(tag):
                if p.get("label") == cfg["rim_label"]:
                    pts = _pts(p.get("points"))
                    entry["rim"] = pts if entry["rim"] is None else np.vstack([entry["rim"], pts])
        out[im.get("name")] = entry
    return out


def read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="") as fh:
        return [{k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()} for row in csv.DictReader(fh)]


def ellipse_polygon(rim: np.ndarray, n: int) -> np.ndarray:
    """n points on the ellipse fitted to the rim points (same frame as rim)."""
    (cx, cy), (w, h), ang = cv2.fitEllipse(rim.astype(np.float32).reshape(-1, 1, 2))
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    a, b, r = w / 2, h / 2, np.radians(ang)
    x, y = a * np.cos(t), b * np.sin(t)
    return np.column_stack([cx + x * np.cos(r) - y * np.sin(r), cy + x * np.sin(r) + y * np.cos(r)])


def build_records(ann: dict[str, dict], states: list[dict], images: list[dict], cfg: dict) -> tuple[list[dict], list[str]]:
    """Records for every image in images.csv; returns (records, problems). Images with problems are skipped."""
    by_state = {s["state_id"]: s for s in states}
    records, problems = [], []
    for row in images:
        name, sid = row["image"], row["state_id"]
        a = ann.get(name)
        st = by_state.get(sid)
        issues = []
        if a is None:
            issues.append("not in CVAT export")
        else:
            if a["box"] is None:
                issues.append("no dial box")
            if a["keypoints"] is None:
                issues.append("skeleton incomplete")
            if a["rim"] is None or len(a["rim"]) < 5:
                issues.append("fewer than 5 rim points")
        if st is None:
            issues.append(f"unknown state_id {sid!r}")
        if issues:
            problems.append(f"{name}: " + "; ".join(issues))
            continue
        lo, hi = float(st["min"]), float(st["max"])
        va, vb = float(st["value_a"]), float(st["value_b"])
        value = (va + vb) / 2
        kp = a["keypoints"].copy()
        kp[:, :2] += EDGE_TO_CENTRE
        rim = a["rim"] + EDGE_TO_CENTRE
        x0, y0, x1, y1 = (v + EDGE_TO_CENTRE for v in a["box"])
        records.append({
            "id": f"real_{Path(name).stem}", "seed": None, "image": f"images/{name}", "size": a["size"],
            "f": (value - lo) / (hi - lo), "value": value, "range": [lo, hi], "units": st.get("units") or None,
            "sweep_deg": None,
            "obliquity_deg": None, "obliquity_bin": None, "obliquity_method": None,
            "tilt_axis_deg": None, "roll_deg": None, "delta_frac": None,
            "K": None, "H_face": None, "H_needle": None,
            "keypoints": [[float(x), float(y), int(v)] for x, y, v in kp],
            "face_polygon": ellipse_polygon(rim, cfg["polygon_points"]).tolist(),
            "rim_points": rim.tolist(),
            "bbox": [x0, y0, x1, y1],
            "nuisance": None, "background": None,
            "gauge_id": st["gauge_id"], "state_id": sid, "phone": row.get("phone") or None,
            "readings": [va, vb], "reading_disagreement": abs(va - vb),
        })
    return records, problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cvat", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--images-csv", required=True)
    ap.add_argument("--out", default="data/real", help="records go to <out>/records/; images expected in <out>/images/")
    ap.add_argument("--config", default="configs/real/realdata.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)["cvat"]
    records, problems = build_records(parse_cvat(a.cvat, cfg), read_csv(a.states), read_csv(a.images_csv), cfg)
    out = Path(a.out) / "records"
    out.mkdir(parents=True, exist_ok=True)
    for rec in records:
        (out / f"{rec['id']}.json").write_text(json.dumps(rec, indent=1))
    for p in problems:
        print(f"SKIPPED {p}")
    print(f"wrote {len(records)} records to {out} ({len(problems)} skipped)")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
