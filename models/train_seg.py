"""Model A: YOLO-seg dial detector + face segmenter (PLAN §4).

    python -m models.train_seg --config configs/train/seg.yaml [--smoke] [--set train.epochs=30]
"""

from models.common import parse_args, resolve_config
from models.yolo_train import train_yolo


def main() -> None:
    a = parse_args(__doc__)
    cfg = resolve_config(a.config, a.smoke, a.set)
    train_yolo(cfg, a.run_id, ("yolo-seg",), "seg_data.yaml")


if __name__ == "__main__":
    main()
