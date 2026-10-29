import numpy as np
import pytest
import torch

import models.common as common
from models import train_pose, train_regress
from models.common import apply_overrides, resolve_config
from models.regress_net import RegressNet, to_tensor


@pytest.mark.parametrize("smoke", [False, True])
def test_pose_config_never_flips(smoke):
    cfg = resolve_config("configs/train/pose.yaml", smoke, [])
    assert cfg["train"]["fliplr"] == 0.0
    train_pose.check_no_flip(cfg)


@pytest.mark.parametrize("smoke", [False, True])
def test_regress_config_never_flips(smoke):
    cfg = resolve_config("configs/train/regress.yaml", smoke, [])
    assert cfg["augment"]["fliplr"] == 0.0
    train_regress.check_no_flip(cfg)


def test_flip_override_is_refused():
    with pytest.raises(SystemExit):
        train_pose.check_no_flip(resolve_config("configs/train/pose.yaml", False, ["train.fliplr=0.5"]))
    with pytest.raises(SystemExit):
        train_regress.check_no_flip(resolve_config("configs/train/regress.yaml", False, ["augment.fliplr=0.5"]))


def test_overrides_are_yaml_typed():
    cfg = apply_overrides({"train": {"epochs": 100}}, ["train.epochs=5", "train.device=cpu", "new.flag=true"])
    assert cfg == {"train": {"epochs": 5, "device": "cpu"}, "new": {"flag": True}}


def test_smoke_section_merges_and_cli_wins():
    cfg = resolve_config("configs/train/regress.yaml", True, ["train.epochs=2"])
    assert cfg["is_smoke"] and cfg["model"]["pretrained"] is False and cfg["train"]["epochs"] == 2
    assert cfg["train"]["lr"] == 3.0e-4                     # untouched keys survive the merge


def test_regress_net_shape():
    net = RegressNet(pretrained=False).eval()
    x = torch.stack([to_tensor(np.zeros((384, 384, 3), np.uint8))] * 2)
    assert net(x).shape == (2,)


def test_regress_smoke_run(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RESULTS", tmp_path)
    cfg = resolve_config("configs/train/regress.yaml", True, ["smoke_data.n_train=8", "smoke_data.n_val=4"])
    run = train_regress.train(cfg, "t")
    assert (run / "weights" / "best.pt").exists() and (run / "config.yaml").exists()
    assert "best_val_mae_fs" in (run / "run.json").read_text()


def test_latest_weights_skips_smoke_and_unfinished(tmp_path):
    import json
    import os
    import time
    from models.common import latest_weights

    def run(name, status="done", with_weights=True, age=0):
        d = tmp_path / name
        (d / "weights").mkdir(parents=True)
        w = d / "weights" / "best.pt"
        if with_weights:
            w.write_bytes(b"x")
        (d / "run.json").write_text(json.dumps({"status": status, "weights": str(w)}))
        t = time.time() - age
        os.utime(d / "run.json", (t, t))
        return w

    old = run("pose_20261020-100000", age=300)
    run("pose_smoke_20261030-100000")                       # newest, but a smoke run
    run("pose_20261029-100000", status="running")           # unfinished
    run("pose_20261029-110000", with_weights=False)         # weights file missing
    assert latest_weights("pose", tmp_path) == old
    newer = run("pose_20261028-100000", age=100)
    assert latest_weights("pose", tmp_path) == newer
    assert latest_weights("seg", tmp_path) is None
