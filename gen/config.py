"""YAML config loading and dotted CLI overrides, shared by every entry point.

A config may derive from another one:

    base: ../default.yaml          # path relative to this file
    overrides:
      nuisance.glare.enabled: false

`load_config` returns the fully resolved dict (base loaded recursively, then the dotted overrides applied).
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def set_dotted(cfg: dict, key: str, value) -> None:
    """cfg['a']['b']['c'] = value for key 'a.b.c', creating intermediate dicts."""
    node = cfg
    *path, leaf = key.split(".")
    for k in path:
        node = node.setdefault(k, {})
    node[leaf] = value


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict) or "base" not in cfg:
        return cfg
    extra = set(cfg) - {"base", "overrides"}
    if extra:
        raise ValueError(f"{path}: a derived config may only contain `base` and `overrides`, found {sorted(extra)}")
    resolved = load_config(path.parent / cfg["base"])
    for key, value in (cfg.get("overrides") or {}).items():
        set_dotted(resolved, key, copy.deepcopy(value))
    return resolved


def apply_overrides(cfg: dict, pairs: list[str]) -> dict:
    """Return a copy of cfg with dotted KEY=VALUE overrides applied; values are parsed as YAML."""
    cfg = copy.deepcopy(cfg)
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise ValueError(f"override must be KEY=VALUE, got {pair!r}")
        set_dotted(cfg, key, yaml.safe_load(raw))
    return cfg
