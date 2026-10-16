"""Model B: YOLO-pose, 7 keypoints on 384 px dial crops (PLAN §4).

    python -m models.train_pose --config configs/train/pose.yaml [--smoke] [--set train.epochs=50]
"""

from models.common import parse_args, resolve_config
from models.yolo_train import train_yolo


def check_no_flip(cfg: dict) -> None:
    """INVARIANT 2 (CLAUDE.md): mirroring swaps min/max and reverses the scale direction."""
    fl = cfg.get("train", {}).get("fliplr")
    if fl != 0.0:
        raise SystemExit(f"pose config must set train.fliplr: 0.0 (got {fl!r}); Ultralytics defaults to 0.5")


def main() -> None:
    a = parse_args(__doc__)
    cfg = resolve_config(a.config, a.smoke, a.set)
    check_no_flip(cfg)
    train_yolo(cfg, a.run_id, ("yolo-pose",), "pose_data.yaml")


if __name__ == "__main__":
    main()
