"""Shared plumbing for the training entry points: config + overrides, seeding, run folders, smoke data."""

from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from gen.config import load_config

RESULTS = Path("results")


def parse_args(description: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny self-generated dataset, 1 epoch, CPU")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="dotted overrides, YAML-typed, e.g. train.epochs=5 train.batch=16")
    return ap.parse_args()


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def apply_overrides(cfg: dict, pairs: list[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if not _:
            raise ValueError(f"override must be KEY=VALUE, got {pair!r}")
        node = cfg
        *path, leaf = key.split(".")
        for k in path:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def resolve_config(path: str, smoke: bool, overrides: list[str]) -> dict:
    """Load YAML, merge the `smoke` section when --smoke, then apply CLI overrides (they win)."""
    cfg = load_config(path)
    smoke_cfg = cfg.pop("smoke", {})
    if smoke:
        cfg = deep_update(cfg, smoke_cfg)
        cfg["is_smoke"] = True
    return apply_overrides(cfg, overrides)


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_run_dir(cfg: dict, run_id: str | None) -> Path:
    run_id = run_id or f"{cfg['name']}{'_smoke' if cfg.get('is_smoke') else ''}_{datetime.now():%Y%m%d-%H%M%S}"
    run = RESULTS / run_id
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return run


def write_run_info(run: Path, **info) -> None:
    path = run / "run.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update({k: str(v) if isinstance(v, Path) else v for k, v in info.items()})
    path.write_text(json.dumps(data, indent=2))


def make_smoke_data(root: Path, formats: tuple[str, ...], n_train: int, n_val: int,
                    gen_config: str = "configs/gen/default.yaml") -> dict:
    """Generate + export a tiny dataset under root. Returns the export section (dirs) and writes root/yamls/*."""
    from gen.export import export
    from gen.generate import generate

    cfg = load_config(gen_config)
    cfg["export"]["seg_dir"] = str(root / "synth_seg")
    cfg["export"]["crops_dir"] = str(root / "synth_crops")
    for split, n in (("train", n_train), ("val", n_val)):
        generate(cfg, split, n, root / "synth" / split, workers=1)
        export(root / "synth" / split, formats, cfg, split=split, workers=1, yaml_dir=root / "yamls")
    return cfg["export"]
