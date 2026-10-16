# Work queue

Work top-down. A task is done only when its acceptance criteria (AC) pass, `pytest -q` is green,
and no invariant in `CLAUDE.md` is broken. Tick the box and add a one-line note (date, what changed).
Section references point to `docs/PLAN.md`.

## M0 — Scaffold

- [x] Create the layout from `CLAUDE.md`, `requirements.txt` with pinned versions, `.gitignore`
      (data/, results/, weights, `*.pt`), `pyproject.toml` or `setup.cfg` so `python -m gen.generate` etc. resolve,
      empty `configs/{gen,train,eval}/`, a README stub.
      **AC:** `pip install -r requirements.txt` succeeds in a fresh venv; `pytest -q` runs (0 tests is fine);
      every package imports.
      _2026-10-16: scaffold + uv/pyproject (Py 3.12), pinned requirements.txt export; verified fresh-venv install, imports, pytest._

## M1 — Geometry core (§5) — tests first

- [x] `gen/pose.py`: `rot`, `sample_pose`, `plane_homography`, `obliquity_deg`, plus `face_to_mm(c, R_px, R_mm)` returning M.
- [x] `tests/geometry/conftest.py`: a helper that builds canonical keypoints for a flat gauge
      (given sweep, start angle, f, radius) and projects them with a random pose.
- [x] `geometry/reading.py`: `apply_h`, `needle_fraction`, `read`.
- [x] `geometry/rectify.py`: `s1_affine`, `hartley_norm`, `fit_conic`, `ellipse_to_circle`, `fix_orientation`,
      `s2_projective`, and a `rectify(strategy, kp, mask=None) -> (H, status)` dispatcher that never raises
      (returns `status="fail:<reason>"`).
      **AC:** all three unit tests in §5 pass — noise-free S2 recovery |Δf| < 1e-6 for 0–70°;
      `needle_fraction` edge cases; `test_fitellipse_convention` (radius spread < 0.5 %).
      Also: S0 mean error at 60° > S0 mean error at 15° on the same random poses.
      _2026-10-16: all ACs pass (59 tests). Added `read_strategy` (rectify + read + S2 linearity gate → NaN on fail)
      and `intrinsics_from_hfov`. Oracle preview at 60°: S0 5.1 %, S1 0.29 %, S2 0.0 % FS MAE._

## M2 — Generator v1 (§3 steps 1–2)

- [x] `gen/render.py`: frontal face + separate needle RGBA layer + canonical keypoints + face circle,
      covering the parameter table in §3 (fonts bundled under `gen/assets/fonts/`, open licences only).
- [x] `gen/generate.py`: pose sampling, warp face (`H_face`) and needle (`H_needle`, δ = 0 in v1),
      composite on a plain/procedural background, write image + JSON record (schema in §3).
- [x] `gen/preview.py`: contact sheet of N images with the 7 projected keypoints drawn and labelled K0–K6.
      **AC:** 100 images generate without error; preview shows keypoints on the right features;
      on these δ = 0 images, `read(gt_kp, s2)` recovers the JSON `f` to < 1e-4; obliquity labels span 0–65°.
      _2026-10-16: 100 images OK (~0.33 CPU-s/image); S2 max |Δf| 1e-15; obliquity 1.6–64.7°; preview checked at
      full res. Multiprocessing + per-image seed already in place. Records live in `<out>/records/<id>.json`._

## M3 — Generator v2 (§3 step 3)

- [x] Needle parallax: δ ~ U(2 %, 5 %) of radius; pivot/tip projected with `H_needle`.
- [x] `gen/nuisance.py`: glare (clipped to face region), low light + sensor noise, blur, shadow, camera effects;
      each with an on/off switch and probability in `configs/gen/default.yaml`.
- [x] Backgrounds from `data/backgrounds/` with procedural fallback (ask before downloading DTD).
- [x] Multiprocessing; per-image seed = hash(config seed, split, index).
- [x] `--stratify-obliquity` option producing exactly 500 images per bin for synth-test.
      **AC:** same config + seed ⇒ byte-identical output for 20 sample images; turning one nuisance off
      changes only that effect (spot-check 10 images); synth-test has 500 per bin.
      _2026-10-16: all ACs pass (tests for 20-image byte identity across worker counts, per-nuisance isolation on
      10 images, glare confined to face). synth-test 2000 = 500/bin. ~0.46 CPU-s/image (20k ≈ 25 min on 10 cores;
      Poisson noise is the largest single cost, fonts are not). Oracle S2 MAE with δ>0: 0.09 → 0.72 % FS by bin.
      Backgrounds: DTD r1.0.1 downloaded (approved 2026-10-16) → `data/backgrounds/dtd/` (5,640 images, 47 classes);
      team photos can go in other sub-folders of `data/backgrounds/`._

## M4 — Export (§3 labels)

- [x] `gen/export.py --formats yolo-seg,yolo-pose,regress`:
      YOLO-seg on full images (face polygon); crops at 384 px from the GT box with ±8 % jitter
      + YOLO-pose labels (`kpt_shape: [7, 3]`) in K0..K6 order; regression CSV (crop path, f, obliquity, id).
      Write the dataset YAMLs into `configs/train/`.
      **AC:** round trip label → denormalised coordinates matches JSON within 0.5 px;
      an Ultralytics `val` call on 10 exported images runs without a dataset error.
      _2026-10-16: both ACs pass (round trip < 0.5 px for seg + pose; `val` on 10 images with untrained
      yolo26n-pose/-seg built from YAML, no weights download). Shared crop maths in `geometry/crop.py`._

## M5 — Training scripts (§4)

- [x] `models/train_seg.py`, `models/train_pose.py`, `models/regress_net.py` + `models/train_regress.py`.
      YAML config + CLI overrides; `--smoke` runs on ≤ 32 images for 1 epoch on CPU.
      Pose and regression configs hard-code `fliplr: 0.0`; the pose script asserts it at start-up.
      Each run writes `results/<run_id>/` with the resolved config and weights path.
      **AC:** all three `--smoke` runs complete on CPU; a test asserts the pose config has `fliplr == 0`.
      _2026-10-16: all three smokes pass on CPU (Ultralytics args.yaml confirms fliplr 0 / mosaic 0 for pose).
      `--smoke` self-generates 16 train + 8 val images and trains from scratch (no weight downloads).
      Overrides: `--set train.epochs=50`; flip overrides are refused for pose and regression._

## M6 — Inference pipeline

- [x] `pipeline/infer.py`: photo → Model A (box, mask) → crop 384 (+15 %) → Model B keypoints →
      S0 / S1 / S2 via `geometry.rectify` → f; Model C → f for S3. Returns, per strategy,
      `{f, linearity_error, status, H}` plus crop, mask and keypoints for visualisation. Never raises on a bad image.
      **AC:** with smoke weights, runs end to end on 5 synthetic images and prints a per-strategy table;
      a missed detection yields `status="fail:no_dial"` for every strategy.
      _2026-10-16: CLI ran end to end on 5 synth-test images with smoke weights (thresholds lowered via `--set`).
      Models are injected as adapters (`detect`, `keypoints`, `regress`); oracle adapters recover f exactly (S2,
      δ = 0) through all frame conversions. no_dial / bad image / adapter exceptions all tested._

## M7 — Evaluation (§7)

- [ ] `eval/evaluate.py`: predictions.csv + metrics.csv per strategy × bin (MAE, median, ≤ 1 %, ≤ 2.5 %,
      failure rate); `--oracle` uses JSON keypoints (and the GT face polygon for S1); `--kp-noise 0,1,2,4`.
- [ ] `eval/bootstrap.py`: 95 % CI with 1,000 resamples, seeded.
- [ ] `eval/plots.py`: MAE vs obliquity (5° bins), per-bin bars with CIs, E4 ablation bars, failure gallery.
      **AC:** `--oracle --kp-noise 0` on a δ = 0 synth-test sample gives S2 MAE < 0.01 % FS;
      with δ > 0 the S2 error is non-zero and grows with obliquity; plots render to PNG.

## M8 — Real-data tooling (§6)

- [ ] `realdata/calibrate.py`, `realdata/aruco_angle.py` (with the arccos(b/a) fallback and a `method` column),
      `realdata/cvat_to_json.py` producing records in the synthetic schema.
      **AC:** on a synthetic render with a rendered ArUco marker coplanar to the dial,
      `aruco_angle.py` recovers obliquity within 1°; `cvat_to_json.py` round-trips a hand-written sample export.

## M9 — Demo

- [ ] `pipeline/demo_app.py` (Gradio): upload photo → overlay keypoints, rectified dial, readings for
      S0–S3, with the S2 → S1 fallback **only here**; shows printed value if the user enters the scale range.
      **AC:** launches locally and handles a photo with no dial gracefully.

## M10 — E4 ablation configs

- [ ] Five configs in `configs/gen/ablation/` (no_glare, no_low_light, no_blur, no_shadow, low_tilt_0_10),
      each differing from default in exactly one switch, plus a script that generates each training set.
      **AC:** a test diffs each ablation config against default and finds exactly one changed key.

## Later / only if asked

- Heatmap keypoint model (§4 upgrade path).
- Nine scale keypoints instead of five (changes the keypoint schema — needs team sign-off).
- OCR of scale numerals for the demo.
