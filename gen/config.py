"""YAML config loading and dotted CLI overrides, shared by every entry point."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def apply_overrides(cfg: dict, pairs: list[str]) -> dict:
    """Return a copy of cfg with dotted KEY=VALUE overrides applied; values are parsed as YAML."""
    cfg = copy.deepcopy(cfg)
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise ValueError(f"override must be KEY=VALUE, got {pair!r}")
        node = cfg
        *path, leaf = key.split(".")
        for k in path:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg
