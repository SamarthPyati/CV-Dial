"""Synthetic gauge images + JSON records (PLAN §3).

Usage:
    python -m gen.generate --config configs/gen/default.yaml --split train --n 20000 --out data/synth/train
    python -m gen.generate --config configs/gen/default.yaml --split test --n 2000 --out data/synth/test \
        --stratify-obliquity          # equal count per obliquity bin (index % n_bins)

Output: <out>/images/<id>.jpg, <out>/records/<id>.json (single source of truth), <out>/config.yaml.
Frames: render = face px (gen.render); image = output pixels, x right, y down.
Each image depends only on (config, split, index) via `image_seed`; every random factor (frame size,
render, pose, background, each nuisance) draws from its own stream so toggling one leaves the rest unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zlib
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import yaml

from gen.config import load_config
from gen.nuisance import ORDER as NUISANCES
from gen.nuisance import apply_nuisances
from gen.pose import face_to_mm, intrinsics_from_hfov, obliquity_deg, plane_homography, sample_pose
from gen.render import Dial, render_dial
from geometry.reading import apply_h

N_POLY = 64
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_seed(seed: int, split: str, index: int) -> int:
    """Per-image seed from (config seed, split, index); independent of worker count and order."""
    h = hashlib.sha256(f"{seed}:{split}:{index}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def stream(seed: int, name: str) -> np.random.Generator:
    """Independent generator for one random factor of one image."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(zlib.crc32(name.encode()),)))


def obliquity_bin_range(cfg: dict, index: int) -> tuple[float, float, bool]:
    """(lo, hi, closed_hi) of the bin assigned to `index` under stratification: bin = index % n_bins."""
    edges = cfg["pose"]["obliquity_bins"]
    k = index % (len(edges) - 1)
    return float(edges[k]), float(edges[k + 1]), k == len(edges) - 2


def circle(c, r, n=N_POLY) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.asarray(c, float) + r * np.column_stack([np.cos(a), np.sin(a)])


def sample_camera(rng: np.random.Generator, cfg: dict, dial: Dial, W: int, H: int,
                  obl_range: tuple[float, float, bool] | None = None, max_tries: int = 5000) -> dict:
    """Sample K, pose and framing so the whole dial is in frame and obliquity <= max (and inside
    `obl_range` = (lo, hi, closed_hi) when stratifying). Returns homographies and pose labels."""
    p = cfg["pose"]
    R_mm = p["face_radius_mm"]
    M = face_to_mm(dial.centre, dial.face_radius, R_mm)
    outline = circle(dial.centre, dial.dial_radius)
    margin = p["frame_margin_frac"] * min(W, H)
    for _ in range(max_tries):
        K = intrinsics_from_hfov(rng.uniform(*p["hfov_deg"]), W, H)
        theta, psi, roll = rng.uniform(*p["tilt_deg"]), rng.uniform(0, 360), rng.normal(0, p["roll_sigma_deg"])
        span = rng.uniform(*p["span_frac"])
        dist = K[0, 0] * 2 * R_mm * dial.dial_radius / dial.face_radius / (span * W)
        u, v = rng.uniform(0, W), rng.uniform(0, H)
        x = np.linalg.solve(K, [u, v, 1.0])
        R, t = sample_pose(theta, psi, roll, x * dist)
        obl = obliquity_deg(R, t)
        if obl > p["max_obliquity_deg"]:
            continue
        if obl_range is not None:
            lo, hi, closed = obl_range
            if obl < lo or obl > hi or (obl == hi and not closed):
                continue
        H_face = plane_homography(K, R, t, M)
        pts = apply_h(H_face, outline)
        if pts.min() < margin or pts[:, 0].max() > W - 1 - margin or pts[:, 1].max() > H - 1 - margin:
            continue
        delta = rng.uniform(*p["delta_frac"])
        H_needle = plane_homography(K, R, t, M, z=delta * R_mm)
        return {
            "K": K, "R": R, "t": t, "H_face": H_face / H_face[2, 2], "H_needle": H_needle / H_needle[2, 2],
            "obliquity_deg": obl, "tilt_axis_deg": psi, "roll_deg": roll, "delta_frac": delta,
        }
    raise RuntimeError("could not sample an in-frame pose; check pose.span_frac / frame_margin_frac")


def procedural_background(rng: np.random.Generator, W: int, H: int) -> tuple[np.ndarray, str]:
    """Float32 (H, W, 3) background in [0, 255] and a short description."""
    kind = ["blobs", "gradient", "stripes", "flat"][rng.integers(4)]
    if kind == "blobs":
        gh, gw = int(rng.integers(2, 12)), int(rng.integers(2, 16))
        img = cv2.resize(rng.uniform(0, 255, (gh, gw, 3)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    elif kind == "gradient":
        c0, c1 = rng.uniform(0, 255, 3), rng.uniform(0, 255, 3)
        a = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        s = xx * np.cos(a) + yy * np.sin(a)
        s = (s - s.min()) / (np.ptp(s) + 1e-6)
        img = c0 * (1 - s[..., None]) + c1 * s[..., None]
    elif kind == "stripes":
        c0, c1 = rng.uniform(0, 255, 3), rng.uniform(0, 255, 3)
        a, period = rng.uniform(0, np.pi), rng.uniform(8, 120)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        s = 0.5 + 0.5 * np.sin(2 * np.pi * (xx * np.cos(a) + yy * np.sin(a)) / period + rng.uniform(0, 6))
        img = c0 * (1 - s[..., None]) + c1 * s[..., None]
    else:
        img = np.broadcast_to(rng.uniform(0, 255, 3), (H, W, 3)).copy()
    img = img + rng.normal(0, rng.uniform(1, 12), (H, W, 3))
    return np.clip(img, 0, 255).astype(np.float32), f"procedural:{kind}"


@lru_cache(maxsize=4)
def background_files(directory: str) -> tuple[str, ...]:
    """Sorted image paths under `directory` (recursive), relative to it; empty if it does not exist."""
    root = Path(directory)
    if not root.is_dir():
        return ()
    return tuple(sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS))


def photo_background(rng: np.random.Generator, cfg_bg: dict, W: int, H: int) -> tuple[np.ndarray, str] | None:
    """Random zoomed crop (with random flip / 90° turn) of a background photo; None if none are available."""
    files = background_files(cfg_bg["dir"])
    if not files:
        return None
    name = files[rng.integers(len(files))]
    img = cv2.imread(str(Path(cfg_bg["dir"]) / name), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.rot90(img, int(rng.integers(4)))
    if rng.uniform() < 0.5:
        img = img[:, ::-1]
    h, w = img.shape[:2]
    scale = max(W / w, H / h) * rng.uniform(*cfg_bg["zoom"])
    img = cv2.resize(np.ascontiguousarray(img), (max(W, round(w * scale)), max(H, round(h * scale))),
                     interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    y0 = int(rng.integers(img.shape[0] - H + 1))
    x0 = int(rng.integers(img.shape[1] - W + 1))
    return img[y0:y0 + H, x0:x0 + W].astype(np.float32), name


def _prescale(dial: Dial, H: np.ndarray) -> tuple[float, np.ndarray]:
    """Pick a downscale factor for the render so the warp never minifies by more than ~1.5x.
    Returns (s, S) with S mapping face px -> downscaled px (cv2.resize INTER_AREA pixel-centre convention)."""
    c = dial.centre
    x = np.array([c[0], c[1], 1.0])
    w = H[2] @ x
    J = (H[:2, :2] * w - np.outer(H[:2] @ x, H[2, :2])) / w**2
    s = float(min(1.0, 1.5 * np.linalg.svd(J, compute_uv=False)[0]))
    off = 0.5 * s - 0.5
    return s, np.array([[s, 0, off], [0, s, off], [0, 0, 1.0]])


def composite(dial: Dial, cam: dict, bg: np.ndarray) -> np.ndarray:
    """Warp the face (H_face) and needle (H_needle) layers and alpha-composite onto bg. Returns float32 RGB."""
    Himg, Wimg = bg.shape[:2]
    s, S = _prescale(dial, cam["H_face"])
    size = max(8, round(dial.face.shape[0] * s))
    Si = np.linalg.inv(S)

    def prep(img):
        return img if s == 1.0 else cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    a = dial.alpha[..., None]
    face_pm = prep(np.concatenate([dial.face.astype(np.float32) * a, a], axis=2))
    na = dial.needle[..., 3:4].astype(np.float32) / 255.0
    needle_pm = prep(np.concatenate([dial.needle[..., :3].astype(np.float32) * na, na], axis=2))

    def warp(img, H):
        return cv2.warpPerspective(img, H @ Si, (Wimg, Himg), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    fw, nw = warp(face_pm, cam["H_face"]), warp(needle_pm, cam["H_needle"])
    out = bg * (1 - fw[..., 3:4]) + fw[..., :3]
    out = out * (1 - nw[..., 3:4]) + nw[..., :3]
    return np.clip(out, 0, 255).astype(np.float32)


def face_mask(poly: np.ndarray, W: int, H: int) -> np.ndarray:
    """Anti-aliased (H, W) float32 mask of the face polygon (image frame)."""
    m = np.zeros((H, W), np.float32)
    cv2.fillPoly(m, [np.rint(np.asarray(poly) * 16).astype(np.int32)], 1.0, lineType=cv2.LINE_AA, shift=4)
    return m


def make_sample(cfg: dict, split: str, index: int, stratify: bool = False) -> tuple[np.ndarray, dict]:
    """Render, pose, composite and degrade one sample. Returns (uint8 RGB image before JPEG, JSON record).

    With `stratify`, the obliquity label is constrained to bin index % n_bins of `pose.obliquity_bins`.
    """
    seed = image_seed(cfg["seed"], split, index)
    sizes = cfg["image"]["sizes"]
    W, H = (int(v) for v in sizes[stream(seed, "frame").integers(len(sizes))])

    dial = render_dial(stream(seed, "render"), cfg["render"])
    obl_range = obliquity_bin_range(cfg, index) if stratify else None
    cam = sample_camera(stream(seed, "pose"), cfg, dial, W, H, obl_range)
    rng_bg = stream(seed, "background")
    photo = photo_background(rng_bg, cfg["background"], W, H)
    bg, bg_name = photo if photo is not None else procedural_background(rng_bg, W, H)
    img = composite(dial, cam, bg)

    kp_face = dial.keypoints
    kp = np.vstack([apply_h(cam["H_needle"], kp_face[:2]), apply_h(cam["H_face"], kp_face[2:])])
    vis = np.where((kp[:, 0] >= 0) & (kp[:, 0] <= W - 1) & (kp[:, 1] >= 0) & (kp[:, 1] <= H - 1), 2, 0)
    poly = apply_h(cam["H_face"], circle(dial.centre, dial.face_radius))
    x0, y0 = np.clip(poly.min(0), 0, [W - 1, H - 1])
    x1, y1 = np.clip(poly.max(0), 0, [W - 1, H - 1])

    mask = face_mask(apply_h(cam["H_face"], circle(dial.centre, dial.face_radius, 256)), W, H)
    img, nuis = apply_nuisances(img, cfg["nuisance"], {n: stream(seed, n) for n in NUISANCES}, mask, poly,
                                cfg["image"]["jpeg_quality"])
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)

    m = dial.meta
    sid = f"{split}_{index:06d}"
    record = {
        "id": sid, "seed": seed, "image": f"images/{sid}.jpg", "size": [W, H],
        "f": m["f"], "value": round(m["value"], 6), "range": m["range"], "units": m["units"],
        "sweep_deg": m["sweep_deg"], "start_deg": m["start_deg"],
        "obliquity_deg": cam["obliquity_deg"],
        "obliquity_bin": int(np.clip(np.searchsorted(cfg["pose"]["obliquity_bins"], cam["obliquity_deg"], "right") - 1,
                                     0, len(cfg["pose"]["obliquity_bins"]) - 2)),
        "tilt_axis_deg": cam["tilt_axis_deg"],
        "roll_deg": cam["roll_deg"], "delta_frac": cam["delta_frac"],
        "K": cam["K"].tolist(), "H_face": cam["H_face"].tolist(), "H_needle": cam["H_needle"].tolist(),
        "keypoints": [[float(x), float(y), int(v)] for (x, y), v in zip(kp, vis)],
        "face_polygon": poly.tolist(),
        "bbox": [float(x0), float(y0), float(x1), float(y1)],
        "nuisance": nuis,
        "background": bg_name,
        "render": {**m["style"], "face_radius_px": dial.face_radius, "dial_radius_px": dial.dial_radius,
                   "render_size": int(dial.face.shape[0])},
    }
    return img, record


def encode_jpeg(img: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def _write_one(args) -> str:
    cfg, split, index, out, stratify = args
    img, rec = make_sample(cfg, split, index, stratify)
    (out / rec["image"]).write_bytes(encode_jpeg(img, rec["nuisance"]["jpeg_quality"]))
    (out / "records" / f"{rec['id']}.json").write_text(json.dumps(rec))
    return rec["id"]


def generate(cfg: dict, split: str, n: int, out: Path, start: int = 0, workers: int = 0,
             stratify: bool = False) -> None:
    n_bins = len(cfg["pose"]["obliquity_bins"]) - 1
    if stratify and (start % n_bins or n % n_bins):
        raise ValueError(f"--stratify-obliquity needs --start and --n divisible by {n_bins}")
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "records").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    jobs = [(cfg, split, i, out, stratify) for i in range(start, start + n)]
    workers = workers or os.cpu_count() or 1
    if workers == 1:
        for j in jobs:
            _write_one(j)
        return
    with Pool(workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(_write_one, jobs, chunksize=8), 1):
            if k % 1000 == 0 or k == n:
                print(f"{k}/{n}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0, help="first index (for resuming / sharding)")
    ap.add_argument("--workers", type=int, default=0, help="0 = all CPUs")
    ap.add_argument("--stratify-obliquity", action="store_true", help="equal images per pose.obliquity_bins bin")
    a = ap.parse_args()
    generate(load_config(a.config), a.split, a.n, Path(a.out), a.start, a.workers, a.stratify_obliquity)


if __name__ == "__main__":
    main()
