# Gauge Reader — CS4231 Computer Vision project

Perspective-robust reading of analog pointer gauges from a single smartphone photo.
Team project at RV University (Pranathi, Samarth, Amit, Rishabh). Final submission: **24 Nov 2026**.

The research contribution is the **evaluation**, not a novel network:
reading error reported per camera-obliquity bin, plus a controlled comparison of
four tilt-correction strategies under one protocol.

- Full technical spec: `docs/PLAN.md` — read the relevant section before starting a task.
- Ordered work queue with acceptance criteria: `docs/TASKS.md` — work top-down, tick items when done.

## Core definitions (do not change without asking)

- **f** — needle position as a fraction of full scale, clockwise from the min mark.
  0 = min mark, 1 = max mark. Values slightly outside [0, 1] are legal (needle at rest below zero, or pegged).
  Printed reading = min + f × (max − min).
- **Error metric** — |f̂ − f| × 100, i.e. % of full scale. No OCR anywhere in the experiment path.
- **Keypoints** — 7, in this fixed order:
  K0 pivot, K1 needle tip, K2–K6 points on the scale arc at 0 / 25 / 50 / 75 / 100 % of full scale.
- **Obliquity** — angle in degrees between the dial-face normal and the camera ray to the dial centre.
  Bins: [0, 15), [15, 30), [30, 45), [45, 65].
- **Strategies**
  - S0: no correction — angles measured directly on image keypoints (baseline).
  - S1: affine ellipse fit — `cv2.fitEllipse` on the dial-face mask, affine map ellipse → circle.
  - S2: conic + pivot — conic through K2–K6, vanishing line = polar of pivot, full projective rectification.
  - S3: direct regression — ResNet-18 predicts f from the dial crop; no geometry.
- **Experiments** — E1 oracle-keypoint geometry, E2 full system on synth-test, E3 full system on real test
  (main result), E4 nuisance ablation. Details in `docs/PLAN.md` §7.

## Invariants — breaking any of these silently corrupts results

1. Image coordinates, y pointing down. `atan2` angles therefore increase clockwise. Never flip y.
2. **No horizontal-flip augmentation** for the keypoint model or the regression model (`fliplr=0.0`).
   Mirroring swaps min/max and reverses the scale direction. Ultralytics defaults to `fliplr=0.5` — always override.
   The detector/segmenter may flip.
3. The real test set (`data/real/`) is **test only**. Never train, fine-tune or tune a threshold on it. Tune on synth-val.
4. Every rectifying homography must be orientation-preserving at the pivot (`geometry.rectify.fix_orientation`).
5. Strategies stay **pure** in experiments: a failure is recorded as `f = NaN` plus a `status` reason, never
   silently replaced by another strategy. The S2 → S1 fallback exists only in `pipeline/demo_app.py`.
6. Synthetic JSON records are the single source of truth. YOLO label files are derived by `gen/export.py` and disposable.
7. Every reported number comes from `eval/evaluate.py` writing `results/<run_id>/metrics.csv` next to the run config.
8. Determinism: the generator output depends only on (config, seed, index). Seed numpy, `random` and torch from config.

## Stack

Python 3.12 · PyTorch 2.x · Ultralytics YOLO (seg + pose) · OpenCV ≥ 4.7 (new `cv2.aruco.ArucoDetector` API)
· NumPy · Pillow · Albumentations · Gradio · pytest. Dependencies are managed with uv (`pyproject.toml` + `uv.lock`);
`requirements.txt` is the pinned export for Colab/Kaggle — regenerate it with
`uv export --no-hashes --no-emit-project --format requirements-txt -o requirements.txt` after any dependency change.
Run things locally with `uv run …` (e.g. `uv run pytest -q`).

GPU training runs on Kaggle/Colab, run by the team. Keep every training entry point a plain script with
CLI args + a YAML config (no notebook-only logic), and make sure it also runs a tiny smoke config on CPU.

## Repository layout

```
cv-dial/
├── CLAUDE.md
├── docs/            PLAN.md (spec), TASKS.md (work queue)
├── configs/         gen/*.yaml, train/*.yaml, eval/*.yaml
├── gen/             render.py, pose.py, nuisance.py, generate.py, export.py, preview.py
├── geometry/        reading.py (needle_fraction, read), rectify.py (S0–S2)
├── models/          train_seg.py, train_pose.py, train_regress.py, regress_net.py
├── pipeline/        infer.py (photo → f per strategy), demo_app.py (Gradio)
├── realdata/        calibrate.py, aruco_angle.py, cvat_to_json.py
├── eval/            evaluate.py, bootstrap.py, plots.py
├── tests/           mirrors the package layout
├── data/            git-ignored: synth/{train,val,test}, real/, backgrounds/
└── results/         one folder per run: config.yaml, metrics.csv, plots/
```

## Target commands (create these interfaces as modules are built; keep names stable)

```bash
pip install -r requirements.txt
pytest -q                                                    # must pass before every commit
python -m gen.generate --config configs/gen/default.yaml --split train --n 20000 --out data/synth/train
python -m gen.preview  --src data/synth/train --n 16 --out results/preview.jpg   # keypoint overlay sheet
python -m gen.export   --src data/synth/train --formats yolo-seg,yolo-pose,regress
python -m models.train_pose    --config configs/train/pose.yaml   [--smoke]
python -m pipeline.infer       --image photo.jpg --strategies S0,S1,S2,S3
python -m eval.evaluate        --run results/<run_id> --split synth-test|real [--oracle --kp-noise 0,1,2,4]
```

## Conventions

- `geometry/` holds small pure functions: NumPy in, NumPy out, no file I/O. Every docstring states the coordinate frame.
- Keypoints: float64 arrays of shape (7, 2). Homographies: float64 (3, 3). Angles in radians internally, degrees in configs and reports.
- Everything the spec randomises lives in YAML under `configs/`, not as literals in code.
- Type hints on public functions. Tests in `tests/` mirror the package layout.
- `data/`, model weights and large outputs are git-ignored.

## How to work in this repo

- Read the matching section of `docs/PLAN.md` before coding a task, then follow `docs/TASKS.md` in order.
- For geometry, write the unit test first; projected synthetic points with a known answer are the oracle.
- Spec numbers (ranges, probabilities, epochs) are tunable defaults. The invariants above are not.
  If a spec detail looks wrong or ambiguous, ask instead of guessing.
- Ask before adding dependencies outside the stack, downloading datasets (e.g. DTD textures),
  or committing anything large.
- You can't run GPU training here: verify training scripts with `--smoke` on a few CPU images,
  and leave full runs to the team.

## Known gotchas

- `cv2.fitEllipse`'s angle convention has varied across OpenCV versions;
  `tests/geometry/test_rectify.py::test_fitellipse_convention` is the arbiter.
- S2 fits a conic through exactly five points, so it has no redundancy against one bad keypoint.
  Failures at small sweeps are expected: record them, don't hide them.
- The needle layer sits at z = δ above the face plane: pivot and tip are projected with `H_needle`,
  scale points and face outline with `H_face`. The resulting parallax is intentional.
- Crops and masks share one frame (the 384 px crop). Keep keypoints, mask and homographies in that frame.

## Decisions log (resolved spec ambiguities)

- 2026-10-16 — Python 3.12 (not 3.11), uv-managed; spec moved from `spec/` to the repo root.
- 2026-10-16 — Obliquity labels > 65° (possible when the dial is off-centre, since the label uses the ray to the
  dial centre): **resample the pose** until `obliquity_deg ≤ 65`; the bins stay [0,15) [15,30) [30,45) [45,65].
- 2026-10-16 — Crops that extend past the image border are padded with **constant black** (0).
- 2026-10-16 — Fonts: no font downloads; use existing fonts already installed with our dependencies
  (matplotlib's bundled DejaVu family), located at runtime rather than copied into `gen/assets/fonts/`.
- 2026-10-16 — S2 linearity gate default 0.03 lives in `geometry.rectify.DEFAULT_MAX_LINEARITY`; experiment
  configs override it. One scale point off by ~25 px at 40° still passes the gate (~1.3 % FS error) — expected,
  five points give no redundancy.
- 2026-10-16 — "±10° jitter" in the scale row is applied to the **centring** of the sweep (rotation away from top),
  not to the sweep size (`render.scale.centre_jitter_deg`).
- 2026-10-16 — Generator output layout: `<out>/images/<id>.jpg`, `<out>/records/<id>.json`, `<out>/config.yaml`.
  (`records/`, not `labels/`, so it never collides with YOLO label dirs.)
- 2026-10-16 — Needle tips are blunt (≥ 3 face px wide) so the visible end of the needle is exactly K1.
- 2026-10-16 — With θ ~ U(0, 65°) the obliquity *label* skews high (off-axis framing adds to it); synth-test
  balance comes from `--stratify-obliquity` (M3), not from changing the θ prior.
- 2026-10-16 — Every random factor of an image (frame, render, pose, background, each nuisance) has its own
  stream (`gen.generate.stream(seed, name)`); each nuisance draws its on/off decision even when disabled.
  This is what makes an E4 ablation change only its own effect — keep it that way when adding factors.
- 2026-10-16 — `--stratify-obliquity` assigns bin = index % n_bins and rejection-samples the pose until the label
  falls in that bin; `--start`/`--n` must be multiples of n_bins. Records carry `obliquity_bin`.
- 2026-10-16 — `nuisance` record field: {glare, shadow, low_light, blur, jpeg_quality, params}; `make_sample`
  returns the pre-JPEG image, JPEG quality is applied when writing.
- 2026-10-16 — Crop maths lives in `geometry/crop.py` (`square_crop_box`, `crop_matrix`, `warp_crop`) and is shared by
  export and the pipeline. Export applies the ±8 % crop jitter only to the `train` split, seeded by the record.
- 2026-10-16 — Normalised YOLO coordinates use the pixel-edge convention: u = (x + 0.5) / size for our
  pixel-centre x. Keypoints that fall outside a crop are written as `0 0 0`.
- 2026-10-16 — Ultralytics 8.4 defaults to the YOLO26 family; "latest pose checkpoint" = `yolo26n-pose`.
  Seg images are hard-linked (copied if linking fails) into `data/synth_seg/`.
