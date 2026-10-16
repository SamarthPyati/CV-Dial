import copy
import csv
from pathlib import Path

import numpy as np
import pytest

from gen.config import load_config
from gen.export import crop_for_record, export
from gen.generate import generate
from geometry.reading import apply_h

CFG = load_config("configs/gen/default.yaml")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("m4")
    cfg = copy.deepcopy(CFG)
    cfg["export"]["seg_dir"], cfg["export"]["crops_dir"] = str(root / "seg"), str(root / "crops")
    for split in ("train", "val"):
        generate(cfg, split, 10, root / "synth" / split, workers=1)
        export(root / "synth" / split, ("yolo-seg", "yolo-pose", "regress"), cfg, workers=1, yaml_dir=root / "yamls")
    return root, cfg


def _records(root, split):
    import json
    return [json.loads(p.read_text()) for p in sorted((root / "synth" / split / "records").glob("*.json"))]


def _floats(path: Path) -> np.ndarray:
    return np.array(path.read_text().split(), float)


@pytest.mark.parametrize("split", ["train", "val"])
def test_seg_label_round_trip(exported, split):
    root, _ = exported
    for rec in _records(root, split):
        v = _floats(root / "seg" / "labels" / split / f"{rec['id']}.txt")
        assert v[0] == 0
        poly = v[1:].reshape(-1, 2) * rec["size"] - 0.5
        assert np.abs(poly - np.asarray(rec["face_polygon"])).max() < 0.5
        assert (root / "seg" / "images" / split / f"{rec['id']}.jpg").exists()


@pytest.mark.parametrize("split", ["train", "val"])
def test_pose_label_round_trip(exported, split):
    root, cfg = exported
    size = cfg["export"]["crop_size"]
    for rec in _records(root, split):
        v = _floats(root / "crops" / "labels" / split / f"{rec['id']}.txt")
        assert v[0] == 0 and len(v) == 5 + 7 * 3
        kp = v[5:].reshape(7, 3)
        A, _ = crop_for_record(rec, cfg["export"], jitter=split == "train")
        vis = kp[:, 2] > 0
        assert vis.sum() >= 5
        back = apply_h(np.linalg.inv(A), kp[vis, :2] * size - 0.5)
        assert np.abs(back - np.asarray(rec["keypoints"])[vis, :2]).max() < 0.5


def test_jitter_only_on_train(exported):
    root, cfg = exported
    rec = _records(root, "val")[0]
    _, box = crop_for_record(rec, cfg["export"], jitter=False)
    x0, y0, x1, y1 = rec["bbox"]
    assert box[2] == pytest.approx(max(x1 - x0, y1 - y0) * 1.15)
    rec = _records(root, "train")[0]
    _, a = crop_for_record(rec, cfg["export"], jitter=True)
    _, b = crop_for_record(rec, cfg["export"], jitter=True)
    assert a == b                                        # seeded by the record
    assert a != crop_for_record(rec, cfg["export"], jitter=False)[1]


def test_regress_csv(exported):
    root, _ = exported
    with open(root / "crops" / "regress_train.csv") as fh:
        rows = list(csv.DictReader(fh))
    recs = {r["id"]: r for r in _records(root, "train")}
    assert len(rows) == 10
    for row in rows:
        assert (root / "crops" / row["crop"]).exists()
        assert float(row["f"]) == pytest.approx(recs[row["id"]]["f"], abs=1e-7)


def test_dataset_yamls_never_flip(exported):
    import yaml
    root, _ = exported
    pose = yaml.safe_load((root / "yamls" / "pose_data.yaml").read_text())
    assert pose["kpt_shape"] == [7, 3] and "flip_idx" not in pose


def test_ultralytics_val_reads_pose_dataset(exported):
    from ultralytics import YOLO
    from ultralytics.nn.tasks import PoseModel
    root, _ = exported
    m = YOLO("yolo26n-pose.yaml")
    m.model = PoseModel("yolo26n-pose.yaml", nc=1, data_kpt_shape=(7, 3), verbose=False)
    r = m.val(data=str(root / "yamls" / "pose_data.yaml"), imgsz=384, batch=4, device="cpu", plots=False,
              verbose=False, project=str(root / "runs"))
    assert "fitness" in r.results_dict
