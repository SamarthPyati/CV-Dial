# cv-dial — Gauge Reader

Perspective-robust reading of analog pointer gauges from a single smartphone photo
(CS4231 Computer Vision, RV University). Output is **f**, the needle position as a fraction
of full scale; we compare four tilt-correction strategies (S0–S3) per camera-obliquity bin.

- Spec: [`docs/PLAN.md`](docs/PLAN.md)
- Work queue: [`docs/TASKS.md`](docs/TASKS.md)
- Conventions and invariants: [`CLAUDE.md`](CLAUDE.md)

## Setup

Python 3.12.

```bash
uv sync                          # local dev (uses uv.lock)
# or, e.g. on Colab/Kaggle:
pip install -r requirements.txt  # pinned export of uv.lock
pytest -q
```

Regenerate the pinned file after changing dependencies:

```bash
uv export --no-hashes --no-emit-project --format requirements-txt -o requirements.txt
```

## Demo

```bash
uv run python -m pipeline.demo_app                    # no weights: synthetic examples only (GT keypoints)
uv run python -m pipeline.demo_app \
    --seg-weights results/<seg_run>/ultralytics/weights/best.pt \
    --pose-weights results/<pose_run>/ultralytics/weights/best.pt \
    --regress-weights results/<regress_run>/weights/best.pt [--share]
```

Opens on http://127.0.0.1:7860. `--share` gives a temporary public link (e.g. to use a phone camera).
The headline reading uses S2 and falls back to S1 only in the demo; the table always shows every strategy.
