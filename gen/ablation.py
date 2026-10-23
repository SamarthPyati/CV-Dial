"""E4: generate + export one training set per nuisance ablation (PLAN §7, TASKS M10).

    python -m gen.ablation --n 20000 [--only no_glare,no_blur] [--root data/ablation] [--val-src data/synth/val]

For each configs/gen/ablation/<name>.yaml:
  <root>/<name>/synth/train/            generator output (same seed and split as the default training set, so
                                        each image equals its default counterpart minus one effect)
  <root>/<name>/synth_crops/            YOLO-pose crops + regression CSV for train, and val exported from --val-src
  <root>/<name>/yamls/pose_data.yaml    dataset YAML for models.train_pose
Validation uses the *default* synth-val records for every ablation, so all runs are tuned on the same images.
Only Models B (pose) and C (regression) are retrained in E4; the detector/segmenter is shared.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gen.config import load_config
from gen.export import export
from gen.generate import generate

ABLATION_DIR = Path("configs/gen/ablation")
FORMATS = ("yolo-pose", "regress")


def ablation_configs(directory: Path = ABLATION_DIR) -> dict[str, Path]:
    return {p.stem: p for p in sorted(directory.glob("*.yaml"))}


def build_ablation(name: str, cfg_path: Path, n: int, root: Path, val_src: Path, workers: int = 0) -> Path:
    """Generate + export one ablation; returns its pose dataset YAML."""
    cfg = load_config(cfg_path)
    out = root / name
    cfg["export"]["crops_dir"] = str(out / "synth_crops")
    generate(cfg, "train", n, out / "synth" / "train", workers=workers)
    export(out / "synth" / "train", FORMATS, cfg, split="train", workers=workers, yaml_dir=out / "yamls")
    export(val_src, FORMATS, cfg, split="val", workers=workers, yaml_dir=out / "yamls")
    return out / "yamls" / "pose_data.yaml"


def training_commands(name: str, root: Path) -> list[str]:
    out = root / name
    return [
        f"python -m models.train_pose --config configs/train/pose.yaml --run-id e4_pose_{name} "
        f"--set data={out / 'yamls' / 'pose_data.yaml'}",
        f"python -m models.train_regress --config configs/train/regress.yaml --run-id e4_regress_{name} "
        f"--set data.crops_dir={out / 'synth_crops'}",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--only", default=None, help="comma-separated ablation names (default: all)")
    ap.add_argument("--root", default="data/ablation")
    ap.add_argument("--val-src", default="data/synth/val", help="default synth-val generator output (records/)")
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    configs = ablation_configs()
    names = a.only.split(",") if a.only else list(configs)
    unknown = set(names) - set(configs)
    if unknown:
        raise SystemExit(f"unknown ablation(s) {sorted(unknown)}; available: {sorted(configs)}")
    val_src = Path(a.val_src)
    if not (val_src / "records").is_dir():
        raise SystemExit(f"{val_src}/records not found: generate the default synth-val first")
    root = Path(a.root)
    for name in names:
        print(f"== {name}", flush=True)
        build_ablation(name, configs[name], a.n, root, val_src, a.workers)
    print("\nTrain each ablation (GPU):")
    for name in names:
        print("\n".join(training_commands(name, root)))


if __name__ == "__main__":
    main()
