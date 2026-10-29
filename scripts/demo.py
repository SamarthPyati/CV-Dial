"""Start the gauge-reading demo with the newest trained weights found under results/.

    python -m scripts.demo                 # newest finished seg / pose / regress runs, picked automatically
    python -m scripts.demo --share         # plus a temporary public link
    python -m scripts.demo --pose-weights results/pose_X/ultralytics/weights/best.pt   # pin one model

Takes the same options as `python -m pipeline.demo_app`; any weights given explicitly win over the
automatic choice. Smoke runs and unfinished runs are never picked.
"""

from __future__ import annotations

import argparse

from models.common import latest_weights
from pipeline.demo_app import add_launch_args, launch

MODELS = ("seg", "pose", "regress")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_launch_args(ap)
    a = ap.parse_args()
    for name in MODELS:
        attr = f"{name}_weights"
        if getattr(a, attr) is None:
            found = latest_weights(name)
            setattr(a, attr, str(found) if found else None)
        print(f"{name:8s} {getattr(a, attr) or 'not found - this model is off'}")
    if not (a.seg_weights and a.pose_weights):
        print("No trained detector/keypoint weights under results/: photos will report 'no dial'. "
              "Synthetic examples still work. Unzip the Kaggle/Colab results into the repo root to fix this.")
    launch(a)


if __name__ == "__main__":
    main()
