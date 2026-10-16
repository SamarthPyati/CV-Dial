"""Derive training datasets from generator records (PLAN §3 labels, TASKS M4). Records stay the source of truth.

Usage:
    python -m gen.export --src data/synth/train --formats yolo-seg,yolo-pose,regress [--split train]

Outputs (dirs from the `export` section of the config):
    yolo-seg   <seg_dir>/images/<split>/<id>.jpg (hard link)   <seg_dir>/labels/<split>/<id>.txt   full image, face polygon
    yolo-pose  <crops_dir>/images/<split>/<id>.jpg (384 crop)  <crops_dir>/labels/<split>/<id>.txt  box + 7 keypoints
    regress    <crops_dir>/regress_<split>.csv                 crop path, f, obliquity_deg, id
and the dataset YAMLs configs/train/{seg,pose}_data.yaml.
Frames: seg labels in image px, pose labels in crop px (geometry.crop). Normalised YOLO coordinates use the
pixel-edge convention Ultralytics denormalises with: u = (x + 0.5) / size for our pixel-centre x.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import yaml

from gen.config import load_config
from gen.generate import stream
from geometry.crop import crop_matrix, square_crop_box, warp_crop
from geometry.reading import apply_h

FORMATS = ("yolo-seg", "yolo-pose", "regress")
N_KP = 7


def crop_for_record(rec: dict, cfg_export: dict, jitter: bool) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Crop matrix (image px -> crop px) and (cx, cy, side) for a record; jitter is seeded by the record."""
    scale, shift = 1.0, (0.0, 0.0)
    if jitter:
        rng = stream(rec["seed"], "export_crop")
        j = cfg_export["crop_jitter"]
        scale = float(rng.uniform(1 - j, 1 + j))
        shift = tuple(rng.uniform(-j, j, 2))
    box = square_crop_box(rec["bbox"], cfg_export["crop_margin"], scale, shift)
    return crop_matrix(*box, cfg_export["crop_size"]), box


def seg_label(rec: dict) -> str:
    """YOLO-seg line: class 0 + face polygon normalised by image size."""
    W, H = rec["size"]
    poly = np.clip((np.asarray(rec["face_polygon"], float) + 0.5) / [W, H], 0, 1)
    return "0 " + " ".join(f"{v:.6f}" for v in poly.ravel()) + "\n"


def pose_label(rec: dict, A: np.ndarray, size: int) -> str:
    """YOLO-pose line in crop px: class, box (face bbox) and 7 x (x, y, v); keypoints outside the crop are 0 0 0."""
    poly = apply_h(A, rec["face_polygon"]) + 0.5                       # pixel-edge coordinates
    lo, hi = np.clip(poly.min(0), 0, size), np.clip(poly.max(0), 0, size)
    c, wh = (lo + hi) / 2 / size, (hi - lo) / size
    parts = [f"0 {c[0]:.6f} {c[1]:.6f} {wh[0]:.6f} {wh[1]:.6f}"]
    kp = np.asarray(rec["keypoints"], float)
    pts = apply_h(A, kp[:, :2])
    for (x, y), v in zip(pts, kp[:, 2]):
        inside = v > 0 and -0.5 <= x <= size - 0.5 and -0.5 <= y <= size - 0.5
        parts.append(f"{(x + 0.5) / size:.6f} {(y + 0.5) / size:.6f} 2" if inside else "0 0 0")
    return " ".join(parts) + "\n"


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _export_one(args) -> tuple[str, float, float] | None:
    rec_path, src, split, formats, cfg_e = args
    rec = json.loads(Path(rec_path).read_text())
    img_path = src / rec["image"]
    seg_dir, crops_dir = Path(cfg_e["seg_dir"]), Path(cfg_e["crops_dir"])
    if "yolo-seg" in formats:
        _link_or_copy(img_path, seg_dir / "images" / split / img_path.name)
        (seg_dir / "labels" / split / f"{rec['id']}.txt").write_text(seg_label(rec))
    if "yolo-pose" in formats or "regress" in formats:
        size = cfg_e["crop_size"]
        A, _ = crop_for_record(rec, cfg_e, jitter=split == "train")
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        cv2.imwrite(str(crops_dir / "images" / split / f"{rec['id']}.jpg"), warp_crop(img, A, size),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        (crops_dir / "labels" / split / f"{rec['id']}.txt").write_text(pose_label(rec, A, size))
    return rec["id"], rec["f"], rec["obliquity_deg"]


def write_dataset_yamls(cfg_e: dict, out_dir: Path = Path("configs/train")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seg = {"path": cfg_e["seg_dir"], "train": "images/train", "val": "images/val", "names": {0: "dial"}}
    pose = {"path": cfg_e["crops_dir"], "train": "images/train", "val": "images/val",
            "kpt_shape": [N_KP, 3], "names": {0: "dial"}}
    head = "# Written by gen.export. Relative paths resolve against the repo root (run from there).\n"
    (out_dir / "seg_data.yaml").write_text(head + yaml.safe_dump(seg, sort_keys=False))
    (out_dir / "pose_data.yaml").write_text(
        head + "# keypoints: K0 pivot, K1 tip, K2..K6 scale at 0/25/50/75/100 %. No flip_idx: never flip (CLAUDE.md).\n"
        + yaml.safe_dump(pose, sort_keys=False))


def export(src: Path, formats: tuple[str, ...], cfg: dict, split: str | None = None, workers: int = 0,
           yaml_dir: Path = Path("configs/train")) -> int:
    src = Path(src)
    split = split or src.name
    cfg_e = cfg["export"]
    for f in formats:
        if f not in FORMATS:
            raise ValueError(f"unknown format {f}; choose from {FORMATS}")
    for d, needed in ((cfg_e["seg_dir"], "yolo-seg" in formats),
                      (cfg_e["crops_dir"], "yolo-pose" in formats or "regress" in formats)):
        if needed:
            for sub in ("images", "labels"):
                (Path(d) / sub / split).mkdir(parents=True, exist_ok=True)
    recs = sorted((src / "records").glob("*.json"))
    jobs = [(p, src, split, formats, cfg_e) for p in recs]
    workers = workers or os.cpu_count() or 1
    if workers == 1:
        rows = [_export_one(j) for j in jobs]
    else:
        with Pool(workers) as pool:
            rows = pool.map(_export_one, jobs, chunksize=16)
    if "regress" in formats:
        with open(Path(cfg_e["crops_dir"]) / f"regress_{split}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["crop", "f", "obliquity_deg", "id"])
            for sid, f, obl in rows:
                w.writerow([f"images/{split}/{sid}.jpg", f"{f:.8f}", f"{obl:.4f}", sid])
    write_dataset_yamls(cfg_e, yaml_dir)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="generator output dir (records/, images/)")
    ap.add_argument("--formats", default=",".join(FORMATS))
    ap.add_argument("--split", default=None, help="default: basename of --src (train/val/test)")
    ap.add_argument("--config", default="configs/gen/default.yaml", help="for the `export` section")
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    n = export(Path(a.src), tuple(a.formats.split(",")), load_config(a.config), a.split, a.workers)
    print(f"exported {n} records from {a.src}")


if __name__ == "__main__":
    main()
