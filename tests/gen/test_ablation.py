import json
import shlex

import pytest
import yaml

from gen.ablation import ablation_configs, build_ablation, training_commands
from gen.config import load_config
from gen.generate import generate

DEFAULT = load_config("configs/gen/default.yaml")
EXPECTED = {
    "no_glare": ("nuisance.glare.enabled", False),
    "no_low_light": ("nuisance.low_light.enabled", False),
    "no_blur": ("nuisance.blur.enabled", False),
    "no_shadow": ("nuisance.shadow.enabled", False),
    "low_tilt_0_10": ("pose.tilt_deg", [0, 10]),
}


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(flatten(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def test_exactly_the_five_ablations_exist():
    assert set(ablation_configs()) == set(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_ablation_differs_from_default_in_exactly_one_key(name):
    base, abl = flatten(DEFAULT), flatten(load_config(ablation_configs()[name]))
    assert set(base) == set(abl)                                   # no keys added or dropped
    changed = {k: abl[k] for k in base if base[k] != abl[k]}
    key, value = EXPECTED[name]
    assert changed == {key: value}


def test_derived_config_rejects_stray_keys(tmp_path):
    (tmp_path / "base.yaml").write_text("a: {b: 1}\nc: 2\n")
    (tmp_path / "ok.yaml").write_text("base: base.yaml\noverrides: {a.b: 5}\n")
    assert load_config(tmp_path / "ok.yaml") == {"a": {"b": 5}, "c": 2}
    (tmp_path / "bad.yaml").write_text("base: base.yaml\nc: 3\n")
    with pytest.raises(ValueError):
        load_config(tmp_path / "bad.yaml")


def test_build_ablation_is_paired_with_default(tmp_path):
    """no_glare's training set equals the default one record-for-record, except the glare flag."""
    val_src = tmp_path / "val"
    generate(DEFAULT, "val", 4, val_src, workers=1)
    generate(DEFAULT, "train", 6, tmp_path / "default_train", workers=1)
    pose_yaml = build_ablation("no_glare", ablation_configs()["no_glare"], 6, tmp_path / "abl", val_src, workers=1)
    pose = yaml.safe_load(pose_yaml.read_text())
    crops = tmp_path / "abl" / "no_glare" / "synth_crops"
    assert pose["path"] == str(crops) and pose["kpt_shape"] == [7, 3]
    assert len(list((crops / "images" / "train").glob("*.jpg"))) == 6
    assert len(list((crops / "images" / "val").glob("*.jpg"))) == 4
    assert (crops / "regress_train.csv").exists() and (crops / "regress_val.csv").exists()
    for p in sorted((tmp_path / "default_train" / "records").glob("*.json")):
        d = json.loads(p.read_text())
        a = json.loads((tmp_path / "abl" / "no_glare" / "synth" / "train" / "records" / p.name).read_text())
        nd, na = d.pop("nuisance"), a.pop("nuisance")
        assert d == a and na["glare"] is False
        assert {k: v for k, v in nd.items() if k not in ("glare", "params")} == \
            {k: v for k, v in na.items() if k not in ("glare", "params")}


def test_training_commands_resolve():
    from models.common import resolve_config
    pose_cmd, reg_cmd = training_commands("no_blur", __import__("pathlib").Path("data/ablation"))
    pose_set = shlex.split(pose_cmd)[shlex.split(pose_cmd).index("--set") + 1:]
    reg_set = shlex.split(reg_cmd)[shlex.split(reg_cmd).index("--set") + 1:]
    assert resolve_config("configs/train/pose.yaml", False, pose_set)["data"] == \
        "data/ablation/no_blur/yamls/pose_data.yaml"
    assert resolve_config("configs/train/regress.yaml", False, reg_set)["data"]["crops_dir"] == \
        "data/ablation/no_blur/synth_crops"
