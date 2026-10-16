"""YAML config loading shared by the generator scripts."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)
