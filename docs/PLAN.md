# Gauge Reader — Technical Specification

Source: the team's implementation plan (16 Oct 2026), adapted for implementation.
Section numbers match the original plan.

## 1. System architecture

The system takes one phone photo and outputs **f**, the needle position as a fraction of full scale
(0 at the min mark, 1 at the max). Printed reading = min + f × (max − min). The evaluation metric is
|f̂ − f| × 100, so no experiment needs the scale numbers read (no OCR).

### Pipeline

```
photo ──► Model A: dial detector + segmenter (YOLO-seg, full image, 640 px)
            │  box → square crop (+15 % margin) resized to 384 px
            │  face mask (in crop frame)
            ├──► Model B: keypoints (YOLO-pose, 384 px crop, 7 kpts)
            │       └──► tilt correction S0 | S1 | S2  ──► angle ratio ──► f
            └──► Model C: direct regression (ResNet-18 on crop)  ─────────► f   (S3)
```

### Seven keypoints (fixed order)

| Index | Name | Meaning |
|---|---|---|
| K0 | pivot | needle pivot / centre cap |
| K1 | tip | needle tip |
| K2 | s000 | scale arc at 0 % (min mark) |
| K3 | s025 | scale arc at 25 % |
| K4 | s050 | scale arc at 50 % |
| K5 | s075 | scale arc at 75 % |
| K6 | s100 | scale arc at 100 % (max mark) |

K3–K5 give S2 extra points on the scale circle and act as a self-check: after a correct
rectification on a linear scale they read 0.25 / 0.50 / 0.75 (the "linearity error").

### Four strategies

| ID | Strategy | How it handles tilt | Exact when |
|---|---|---|---|
| S0 | No correction (baseline) | angles measured directly on image keypoints | camera is frontal |
| S1 | Affine ellipse fit | fit ellipse to face mask; affine map ellipse → circle | weak perspective |
| S2 | Conic + pivot | conic through K2–K6; pivot's polar = vanishing line; full projective rectification | always, given an accurate pivot on the dial plane |
| S3 | Direct regression | CNN predicts f from the crop; invariance learned from tilted training data | test angles covered by training |

## 2. Stack

Python 3.12, PyTorch 2.x, Ultralytics YOLO (seg + pose, nano or small), OpenCV ≥ 4.7, NumPy, Pillow,
Albumentations, Gradio, pytest. CVAT for annotation (outside the repo). Pin versions in `requirements.txt`.

## 3. Synthetic data generator

Every training image comes from the generator; reading and camera angle are known exactly.

### Step 1 — render a frontal dial (`gen/render.py`)

Render at 1024 × 1024. Output: face RGB layer, needle RGBA layer (separate), canonical keypoints
(pivot, tip, 5 scale points) in face pixels, face circle (centre, radius).

| Element | Randomised over |
|---|---|
| Face | white/cream 80 %, black with light print 15 %, coloured 5 %; bezel chrome/black/brass, 3–10 % of radius |
| Scale | sweep ∈ {180°, 240°, 270° (p = 0.5), 300°}, centred on top, ±10° jitter. Range ∈ {0–4, 0–10, 0–16, 0–100, 0–160, −1–0} so 25/50/75 % fall on ticks |
| Ticks | 4–10 major ticks with numerals; 2–10 minor divisions per major; tick ring at 70–88 % of radius |
| Text | numerals in one of 4–6 TTF fonts (bundled, open licence); unit label (bar, psi, kg/cm², °C); optional brand text; green/red zone on 30 % of faces |
| Needle | tapered / arrow / counterweighted; black, red or white; length 75–95 % of tick ring; pivot cap 4–10 % of radius |
| Value | f ~ U(0, 1); 5 % of samples slightly outside (−0.04…0 or 1…1.04) |

Scale direction is clockwise from min to max (image frame, y down).

### Step 2 — 3D pose and projection (`gen/pose.py`)

- Face = plane z = 0 in millimetres. M maps face pixels → mm: with face centre c and s = R_mm / R_px,
  `M = [[s, 0, -s*c], [0, s, -s*c], [0, 0, 1]]`.
- Camera K from horizontal FOV ~ U(55°, 75°) for the chosen output size.
- Pose: obliquity θ ~ U(0°, 65°) about an in-plane axis at azimuth ψ ~ U(0°, 360°); camera roll ~ N(0°, 10°).
- Framing: distance so the dial spans 25–70 % of image width; random in-frame offset (tx, ty).
- Needle parallax: needle layer on plane z = δ, δ ~ U(2 %, 5 %) of radius.
- Obliquity **label** is computed from the pose (`obliquity_deg`), not copied from θ.

```python
import numpy as np, cv2

def rot(axis, angle):
    return cv2.Rodrigues(np.asarray(axis, float) * angle)[0]

def sample_pose(theta_deg, psi_deg, roll_deg, t):
    """Tilt by theta about an in-plane axis at azimuth psi, then roll the camera. t = (tx, ty, dist) in mm."""
    psi = np.radians(psi_deg)
    R = rot([0, 0, 1], np.radians(roll_deg)) @ rot([np.cos(psi), np.sin(psi), 0], np.radians(theta_deg))
    return R, np.asarray(t, float)

def plane_homography(K, R, t, M, z=0.0):
    """Image of the plane z = const. M maps face pixels to dial millimetres (3x3)."""
    return K @ np.column_stack([R[:, 0], R[:, 1], z * R[:, 2] + t]) @ M

def obliquity_deg(R, t):
    return np.degrees(np.arccos(abs(R[:, 2] @ (t / np.linalg.norm(t)))))

# H_face   = plane_homography(K, R, t, M)          -> scale points K2..K6, face outline
# H_needle = plane_homography(K, R, t, M, z=delta) -> pivot K0, tip K1, needle layer
```

Warp face with `cv2.warpPerspective(face, H_face, (W, H))`, needle RGBA with `H_needle`,
alpha-composite needle over face, then composite onto the background using the warped face/bezel mask.

### Step 3 — nuisances (`gen/nuisance.py`)

Each factor has its own config switch so E4 is a config change.

| Factor | Implementation | Default p |
|---|---|---|
| glare | additive elliptical highlight, Gaussian falloff, peak 0.3–0.9, clipped to the glass (face) region; sometimes a thin streak | 0.5 |
| low_light | gamma 1.5–3.0, gain 0.3–0.7, then Poisson + Gaussian sensor noise | 0.3 |
| blur | motion blur 3–15 px, or defocus σ 0.5–2.5 | 0.4 |
| shadow | random polygon darkened 20–60 % | 0.3 |
| background | random crop from `data/backgrounds/` (DTD textures + team photos); fallback: procedural noise | always |
| camera | colour jitter, JPEG quality 40–95, mild vignetting | always |
| tilt | θ range 0–65° (ablation setting: 0–10°) | always |

### Labels — one JSON record per image (single source of truth)

```json
{
  "id": "train_000123",
  "seed": 1234567,
  "image": "images/train_000123.jpg",
  "size": [960, 720],
  "f": 0.42, "value": 4.2, "range": [0, 10], "units": "bar", "sweep_deg": 270.0,
  "obliquity_deg": 31.7, "tilt_axis_deg": 112.0, "roll_deg": -4.1, "delta_frac": 0.03,
  "K": [[...], [...], [...]],
  "H_face": [[...], [...], [...]],
  "H_needle": [[...], [...], [...]],
  "keypoints": [[x, y, v], "... 7 rows in K0..K6 order"],
  "face_polygon": [[x, y], "... 64 points"],
  "bbox": [x0, y0, x1, y1],
  "nuisance": {"glare": true, "low_light": false, "blur": "motion", "shadow": false},
  "background": "dtd/banded_0004.jpg"
}
```

Visibility v: 2 = in image, 0 = outside the image. `gen/export.py` derives YOLO-seg labels
(full image, face polygon), YOLO-pose labels (crops), and a regression CSV (crop path, f).

### Splits

| Split | Images | Use |
|---|---|---|
| synth-train | 20,000 | training all models |
| synth-val | 2,000 | early stopping, threshold tuning |
| synth-test | 2,000, stratified 500 per obliquity bin | E1, E2 |

Use multiprocessing; 20k images should take minutes. If not, profile font rendering first.

## 4. Models and training

All models train on synth-train only. Before full runs, a smoke run (5,000 images, 10 epochs on GPU;
a few images, 1 epoch on CPU via `--smoke`).

### Model A — detector + segmenter (`models/train_seg.py`)

YOLO-seg, 1 class `dial`, full image at 640 px. Label polygon = outer edge of the dial face inside the bezel.
Defaults: 60 epochs, batch 32, default mosaic, `close_mosaic=10`. Horizontal flips allowed here.
Output: box → square crop (+15 %) resized to 384 px; face mask mapped into the crop frame (used by S1).
Optional: fine-tune the detector only on public real gauge photos if licences permit — never on our real test set.

### Model B — keypoints (`models/train_pose.py`)

YOLO-pose on 384 px crops from the ground-truth box with ±8 % scale/position jitter.

```yaml
# configs/train/pose_data.yaml
path: data/synth_crops
train: images/train
val: images/val
kpt_shape: [7, 3]   # pivot, tip, scale 0/25/50/75/100 %
names: {0: dial}
```

```python
from ultralytics import YOLO
YOLO("yolo11n-pose.pt").train(          # use the latest Ultralytics pose checkpoint available
    data="configs/train/pose_data.yaml", imgsz=384, epochs=100, batch=64,
    fliplr=0.0,   # INVARIANT: mirroring swaps min/max and reverses the scale
    mosaic=0.0,   # one dial per crop
    degrees=10,
)
```

Watch tip/pivot error ÷ dial radius. Budget: tip error e px on needle length L px ≈ e/L rad;
on a 270° scale 1° = 0.37 % FS, so 1.5 px on a 150 px needle ≈ 0.2 % FS.
Upgrade path if tip precision limits results: heatmap model (ResNet-18 + 3 deconv layers,
96 × 96 heatmaps, soft-argmax), ~150 lines of PyTorch, same crops and labels.

### Model C — direct regression, S3 (`models/train_regress.py`)

ResNet-18 (ImageNet weights) on the 384 px crop → global pool → FC → 1 linear output f (unbounded).
Smooth L1 loss; AdamW lr 3e-4, weight decay 1e-4, cosine schedule, 30 epochs, batch 64.
Augmentation: crop jitter only (nuisances are already in the images). No horizontal flips.

## 5. Rectification and reading (`geometry/`)

Image coordinates, y down; `atan2` increases clockwise. Keypoints, mask and homographies share the crop frame.

```python
import numpy as np, cv2

TWO_PI = 2 * np.pi

def apply_h(H, pts):
    p = np.column_stack([np.asarray(pts, float), np.ones(len(pts))]) @ H.T
    return p[:, :2] / p[:, 2:3]

def needle_fraction(pivot, tip, p_min, p_max):
    """Share of full scale, clockwise from the min mark. Image coordinates (y down)."""
    ang = lambda p: np.arctan2(p[1] - pivot[1], p[0] - pivot[0])
    sweep = (ang(p_max) - ang(p_min)) % TWO_PI
    rel = (ang(tip) - ang(p_min)) % TWO_PI
    if rel > sweep + (TWO_PI - sweep) / 2:   # needle rests just below the min mark
        rel -= TWO_PI
    return rel / sweep

def read(kp, H=np.eye(3)):
    """kp: (7, 2) array in K0..K6 order. Returns (f, linearity_error)."""
    r = apply_h(H, kp)
    f = needle_fraction(r[0], r[1], r[2], r[6])
    mids = [needle_fraction(r[0], r[i], r[2], r[6]) for i in (3, 4, 5)]
    return f, float(np.mean(np.abs(np.array(mids) - [0.25, 0.50, 0.75])))
```

**S0** = `read(kp)`.

### S1 — affine ellipse fit

```python
def s1_affine(mask):
    """Affine map turning the fitted face ellipse into a circle (weak perspective)."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rim = max(cnts, key=cv2.contourArea).astype(np.float32)
    (cx, cy), (w, h), ang = cv2.fitEllipse(rim)
    a, b = max(w, h) / 2, min(w, h) / 2
    t = np.radians(ang if w >= h else ang + 90)   # major-axis direction: confirm with the unit test
    R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    A2 = R.T @ np.diag([1.0, a / b]) @ R
    A = np.eye(3); A[:2, :2] = A2; A[:2, 2] = -A2 @ np.array([cx, cy])
    return A
```

Why it works: under weak perspective the dial→image map is affine; mapping the observed ellipse back to a
circle makes the composite a similarity, which preserves angles. Under full perspective the ellipse centre
is not the image of the dial centre; E1 measures the resulting error.

### S2 — conic + pivot (full projective)

The scale points lie on a circle whose centre is the pivot. In the image, the vanishing line of the dial
plane is the polar of the pivot w.r.t. the fitted conic: l = C·p. Send l to infinity, then map the
remaining ellipse to a circle (metric rectification from a circle and its centre; Hartley & Zisserman).

```python
def hartley_norm(pts):
    c = pts.mean(0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(pts - c, axis=1))
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])

def fit_conic(pts):
    x, y = pts[:, 0], pts[:, 1]
    D = np.column_stack([x * x, x * y, y * y, x, y, np.ones_like(x)])
    a, b, c, d, e, f = np.linalg.svd(D)[2][-1]
    return np.array([[a, b / 2, d / 2], [b / 2, c, e / 2], [d / 2, e / 2, f]])

def ellipse_to_circle(C):
    Q, q, s = C[:2, :2], C[:2, 2], C[2, 2]
    c0 = -np.linalg.solve(Q, q)
    k = q @ np.linalg.solve(Q, q) - s            # conic is (x - c0)^T Q (x - c0) = k
    if k < 0:
        Q, k = -Q, -k
    lam, V = np.linalg.eigh(Q)
    if lam.min() <= 0:
        raise ValueError("fitted conic is not an ellipse")
    if np.linalg.det(V) < 0:
        V[:, 0] *= -1                             # rotation, never a mirror
    D = np.diag(np.sqrt(lam))
    A = np.eye(3); A[:2, :2] = D @ V.T; A[:2, 2] = -D @ V.T @ c0
    return A

def fix_orientation(H, p):
    """If H mirrors the image near p the scale would read anticlockwise: compose with a flip."""
    x = np.array([p[0], p[1], 1.0]); w = H[2] @ x
    J = (H[:2, :2] * w - np.outer(H[:2] @ x, H[2, :2])) / w**2
    return H if np.linalg.det(J) > 0 else np.diag([1.0, -1.0, 1.0]) @ H

def s2_projective(kp):
    """Rectify from scale points K2..K6 (on a circle) and the pivot K0 (its centre)."""
    T = hartley_norm(kp[2:7])
    sp, pv = apply_h(T, kp[2:7]), apply_h(T, kp[:1])[0]
    C = fit_conic(sp)
    l = C @ np.array([pv[0], pv[1], 1.0])        # vanishing line = polar of the centre
    H1 = np.vstack([[1, 0, 0], [0, 1, 0], l / l[2]])
    H1i = np.linalg.inv(H1)
    H = ellipse_to_circle(H1i.T @ C @ H1i) @ H1 @ T
    return fix_orientation(H, kp[0])
```

- The orientation guard is mandatory: sending l to infinity can mirror the dial depending on which side
  of l the pivot lies, which would silently invert the reading.
- Five points fit a conic exactly (no redundancy). If S2 is noisy, the keypoint set may grow to nine scale
  points (every 12.5 %); `fit_conic` accepts any number of points. Ask before changing the keypoint count.
- Experiments: S2 raising, or linearity error > 0.03 (configurable), ⇒ S2 failure (`f = NaN`, status recorded).
  Demo only: fall back to S1.

### Unit tests (write before model code)

1. **Noise-free recovery** — project canonical keypoints of a flat gauge (δ = 0) at obliquities 0–70° in 5° steps
   with random ψ, roll, f. S2: |Δf| < 1e-6 and linearity error < 1e-6 at every angle. S0 error grows with angle.
2. **`needle_fraction` edge cases** — f ∈ {0, 1, −0.03, 1.03} on sweeps of 180° and 300°.
3. **`fitEllipse` convention** — draw an ellipse with `cv2.ellipse` at known angle/axes, fit it, rectify the
   contour points; radius spread < 0.5 %.

## 6. Real-world test set (`realdata/`)

Test only (see CLAUDE.md invariant 3). Captured and annotated by the team; code needed:

- `calibrate.py` — checkerboard calibration per phone (`cv2.calibrateCamera`), saves K and distortion to YAML.
- `aruco_angle.py` — per image: detect a 4 cm `DICT_4X4_50` marker taped coplanar with the dial face,
  `solvePnP` (IPPE_SQUARE), obliquity = angle between marker normal and the ray through the annotated pivot.
  Fallback when the marker is unusable: θ ≈ arccos(b / a) from the annotated face-rim ellipse.
  Record which method produced each label.

```python
aruco = cv2.aruco
detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_50))
corners, ids, _ = detector.detectMarkers(gray)
h = 0.04 / 2                                        # half marker side, metres
obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)
ok, rvec, tvec = cv2.solvePnP(obj, corners[0][0], K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
R = cv2.Rodrigues(rvec)[0]
ray = np.linalg.inv(K) @ np.array([pivot_u, pivot_v, 1.0])
obliquity = np.degrees(np.arccos(abs(R[:, 2] @ ray / np.linalg.norm(ray))))
```

- `cvat_to_json.py` — convert CVAT exports (7-point skeleton in K0..K6 order, dial box, ≥ 5 face-rim points)
  plus the reading spreadsheet (value, range per gauge state) into records with the **same schema** as the
  synthetic JSON (fields that don't apply, e.g. `H_face`, set to null).

Ground-truth reading: two people read each frontal reference photo independently to half a minor division;
f = (value − min) / (max − min). Capture target: ~350 images, ≥ 50 per obliquity bin, ≥ 5 gauge faces.

## 7. Experiments and evaluation (`eval/`)

### Metrics

| Metric | Definition |
|---|---|
| Reading error (primary) | \|f̂ − f\| × 100; mean (MAE) and median per obliquity bin |
| Within tolerance | share with error ≤ 1 % and ≤ 2.5 % FS |
| Failure rate | share per bin with missed detection or rectification failure; excluded from MAE, reported separately |
| Keypoint error | pixel distance ÷ dial radius, per keypoint |
| Speed | ms per image, GPU and CPU |

Bins [0, 15), [15, 30), [30, 45), [45, 65]. 95 % bootstrap CI (1,000 resamples) on every MAE.

### Experiments

| ID | Question | Setup | Data |
|---|---|---|---|
| E1 | Geometry error with perfect keypoints | S0, S1, S2 on ground-truth keypoints (S1 uses GT face polygon), then Gaussian keypoint noise σ = 1, 2, 4 px | synth-test |
| E2 | Full system, synthetic | S0–S3 with predicted boxes, masks, keypoints | synth-test |
| E3 | Full system, real (main result) | S0–S3 | real test |
| E4 | Which nuisance causes the sim-to-real drop | retrain Models B and C five times, each with one nuisance off | real test |

### Outputs

`eval/evaluate.py` writes `results/<run_id>/metrics.csv` (one row per strategy × bin) and
`predictions.csv` (one row per image × strategy: id, bin, f_true, f_pred, error, status).
`eval/plots.py` produces: MAE vs obliquity (5° bins, synth-test, one line per strategy — the headline figure);
real-set bars per bin with CI error bars; E4 ablation bars; a failure gallery (6–8 images with keypoints drawn).

Results table template (E2 and E3):

| Strategy | 0–15° | 15–30° | 30–45° | 45–65° | Overall | Failure rate |
|---|---|---|---|---|---|---|
| S0 | | | | | | |
| S1 | | | | | | |
| S2 | | | | | | |
| S3 | | | | | | |

## 8. Plan context

Owners: Pranathi — generator; Samarth — model training, pipeline, demo; Amit — geometry, E1;
Rishabh — real data tooling, evaluation. Timeline: week 1 (16–22 Oct) generator v1 + geometry tests;
week 2 (23–29 Oct) generator v2, first models — **checkpoint: one photo → f through S0 end to end**;
week 3 (30 Oct–5 Nov) full pipeline, E1; week 4 (6–12 Nov) E2 + E3; week 5 (13–19 Nov) E4, plots, demo;
20–24 Nov report and buffer. Cut order if late: E4, then S3, then the second phone model.
