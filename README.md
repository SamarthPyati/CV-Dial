# cv-dial — Gauge Reader

Perspective-robust reading of analog pointer gauges from a single smartphone photo
(CS4231 Computer Vision, RV University). The system outputs **f**, the needle position as a fraction of full
scale (0 = min mark, 1 = max mark); the printed reading is `min + f × (max − min)`.

The contribution is the **evaluation**: reading error per camera-obliquity bin, and a controlled comparison of
four ways to handle a tilted camera:

| | Strategy | Idea |
|---|---|---|
| S0 | no correction | measure needle angles directly in the photo (baseline) |
| S1 | ellipse fit | map the dial face's ellipse back to a circle (affine) |
| S2 | conic + pivot | full projective rectification from the scale arc and the pivot |
| S3 | direct regression | a ResNet-18 predicts f from the dial crop, no geometry |

- Spec: [`docs/PLAN.md`](docs/PLAN.md) · Work queue: [`docs/TASKS.md`](docs/TASKS.md) ·
  Conventions, invariants and decisions log: [`CLAUDE.md`](CLAUDE.md)

## Quick start: the demo

```bash
uv sync --extra ocr            # Python 3.12; the `ocr` extra lets the demo read the scale range from the dial
uv run python -m scripts.demo  # newest trained weights in results/ are loaded automatically
```

Open the URL it prints (the first free port from 7860). Then upload a photo, take one with the webcam, or
**paste one anywhere on the page with Ctrl/⌘+V** — it is read straight away. You get:

- the photo with the detected dial and its 7 keypoints, and the dial rectified upright;
- the reading (S2, falling back to S1 if S2 fails — this fallback exists **only** in the demo) and a table with
  every strategy's own result;
- the actual value: type the scale's min/max, or leave them blank and the demo reads the range from the printed
  numerals, saying whether it **read** both end numerals or **inferred** one (check those). If it cannot read
  enough numerals it asks you to type the range; a typed range always wins.

**Where the weights come from.** `scripts.demo` picks, for each model, the newest finished run under `results/`
(smoke runs and unfinished runs are skipped) and prints which ones it loaded. After training on Kaggle/Colab,
unzip `cv-dial-results.zip` into the repo root. With no weights it still runs: uploaded photos then report
"no dial", and the **Synthetic example** panel shows the geometry on generated dials with ground-truth keypoints.

Options (same as `python -m pipeline.demo_app`, which takes explicit weight paths only):

```bash
uv run python -m scripts.demo --share       # temporary public link, e.g. to use a phone camera
uv run python -m scripts.demo --port 7870   # fixed port
uv run python -m scripts.demo --no-ocr      # don't read the scale range from the dial
uv run python -m scripts.demo --pose-weights results/<run>/ultralytics/weights/best.pt   # pin one model
```

> `uv sync` without `--extra ocr` removes the OCR packages again. OCR is demo-only: it is never used in the
> experiments (`eval/` does not import it; CLAUDE.md: no OCR in the experiment path).

## Results so far

Mean absolute reading error, **% of full scale**, on the 2,000-image stratified synthetic test set
(500 per obliquity bin). Every number comes from `eval.evaluate` (`results/<run>/metrics.csv`).

**E1 — geometry with ground-truth keypoints** (`results/e1`; σ = Gaussian keypoint noise in pixels):

| | σ | 0–15° | 15–30° | 30–45° | 45–65° | Overall [95 % CI] |
|---|---|---|---|---|---|---|
| S0 | 0 | 0.27 | 0.68 | 1.87 | 4.01 | 1.71 [1.61, 1.81] |
| S1 | 0 | 0.09 | 0.21 | 0.39 | 0.74 | 0.36 [0.34, 0.37] |
| S2 | 0 | 0.08 | 0.21 | 0.38 | 0.72 | 0.35 [0.33, 0.36] |
| S1 | 2 | 0.59 | 0.68 | 0.71 | 1.07 | 0.76 [0.73, 0.79] |
| S2 | 2 | 0.68 | 0.74 | 0.81 | 1.07 | 0.82 [0.79, 0.86] (1.9 % failures) |

**E2 — full system with trained models** (`results/e2_v2_fixed_needles`):

| | 0–15° | 15–30° | 30–45° | 45–65° | Overall [95 % CI] |
|---|---|---|---|---|---|
| S0 | 0.45 | 0.83 | 1.99 | 4.16 | 1.86 [1.75, 1.97] |
| S1 | 0.31 | 0.44 | 0.58 | 0.96 | 0.57 [0.53, 0.62] |
| S2 | 0.36 | 0.47 | 0.60 | 1.00 | 0.61 [0.57, 0.66] |
| S3 | 0.57 | 0.52 | 0.68 | 1.12 | 0.72 [0.65, 0.81] |

Without correction the error grows ~9× from frontal to steep views; every correction keeps it near 1 % FS.
S1 and S2 are statistically tied: needle parallax (the needle sits above the face) costs S2 about as much as the
weak-perspective approximation costs S1. The E2 models were trained before a generator fix (black needles on
dark faces, see the CLAUDE.md decisions log); retraining on regenerated data is recommended for final numbers.
E3 (real photos, the main result) and E4 (nuisance ablation) are still to run.

## Workflow

### 1. Synthetic data

```bash
uv run python -m gen.generate --config configs/gen/default.yaml --split train --n 20000 --out data/synth/train
uv run python -m gen.generate --config configs/gen/default.yaml --split val   --n 2000  --out data/synth/val
uv run python -m gen.generate --config configs/gen/default.yaml --split test  --n 2000  --out data/synth/test --stratify-obliquity
uv run python -m gen.preview  --src data/synth/train --n 16 --out results/preview.jpg     # keypoint overlay sheet
uv run python -m gen.export   --src data/synth/train && uv run python -m gen.export --src data/synth/val
```

~10 min for 24k images on a 10-core laptop. Backgrounds come from `data/backgrounds/` (DTD textures; procedural
fallback). JSON records in `<out>/records/` are the single source of truth; `gen.export` derives the YOLO and
regression datasets. Config overrides: `--set pose.delta_frac=[0,0]`.

### 2. Training (GPU)

- **Kaggle, 2 × T4** — [`notebooks/train_kaggle.ipynb`](notebooks/train_kaggle.ipynb): segmenter on GPU 0,
  keypoints then regression on GPU 1, then E2. Upload `cv-dial-src.zip` (`git archive -o cv-dial-src.zip HEAD`)
  and `cv-dial-data.zip` (`zip -r -0 cv-dial-data.zip data/synth_seg data/synth_crops data/synth/test`) as a
  Kaggle dataset, set *Accelerator: GPU T4 x2* and *Internet: On*, run all. `FAST = True` (default) ≈ 1.5 h,
  `FAST = False` (spec schedule) ≈ 3.5–5 h. Download `cv-dial-results.zip` from the Output tab.
- **Colab** — [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb): same, one GPU, zips in
  `MyDrive/cv-dial/`.
- **Locally** — `uv run python -m models.train_{seg,pose,regress} --config configs/train/<name>.yaml`;
  add `--smoke` for a 1-epoch CPU check on a self-generated tiny dataset, `--set train.epochs=30` to override.

### 3. Evaluation

```bash
uv run python -m eval.evaluate --run results/e1 --split synth-test --oracle --kp-noise 0,1,2,4      # E1
uv run python -m eval.evaluate --run results/e2 --split synth-test \
    --seg-weights ... --pose-weights ... --regress-weights ...                                       # E2 (E3: --split real)
uv run python -m eval.plots run --run results/e2          # MAE vs obliquity, per-bin bars, failure galleries
```

### 4. Real test set (E3, test only — never train or tune on it)

```bash
uv run python -m realdata.calibrate    --images 'data/real/calib/<phone>/*.jpg' --out data/real/calib/<phone>.yaml
uv run python -m realdata.cvat_to_json --cvat data/real/annotations/annotations.xml \
    --states data/real/annotations/states.csv --images-csv data/real/annotations/images.csv --out data/real
uv run python -m realdata.aruco_angle  --split data/real --calib data/real/calib
```

### 5. Nuisance ablation (E4)

```bash
uv run python -m gen.ablation --n 20000      # one paired training set per configs/gen/ablation/*.yaml
```

It prints the training commands for each ablation; compare runs with `python -m eval.plots ablation`.

## Repository layout

```
configs/    gen/ (generator + ablations), train/ (models, datasets), eval/ (evaluation, pipeline, scale OCR), real/
gen/        synthetic dials: render, pose, nuisances, generate, export, preview, ablation
geometry/   reading (f from keypoints), rectify (S0–S2), crop
models/     training scripts and the S3 regression network
pipeline/   infer (photo -> f per strategy), demo_app (Gradio), scale_ocr (demo-only range reading)
realdata/   phone calibration, ArUco obliquity labels, CVAT -> records
eval/       evaluate, bootstrap CIs, plots
scripts/    demo (one-command demo with automatic weights)
notebooks/  Kaggle and Colab training
tests/      mirrors the packages; run with `uv run pytest -q`
data/, results/   git-ignored
```

## Development

```bash
uv run pytest -q                                   # must pass before every commit
uv export --no-hashes --no-emit-project --format requirements-txt -o requirements.txt   # after dependency changes
```

`requirements.txt` is the pinned export for Kaggle/Colab (the notebooks only add Ultralytics on top of their own
PyTorch). The invariants in [`CLAUDE.md`](CLAUDE.md) — image y-axis down, no horizontal flips for the keypoint and
regression models, real data is test-only, strategies stay pure — must hold for every change.
