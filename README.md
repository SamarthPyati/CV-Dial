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
