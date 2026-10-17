"""Photo -> f for each strategy (PLAN §1 pipeline).

    python -m pipeline.infer --image photo.jpg [more.jpg ...] --strategies S0,S1,S2,S3 \
        [--seg-weights W] [--pose-weights W] [--regress-weights W] [--config configs/eval/pipeline.yaml]

Frames: the photo (image px) -> square crop (crop px, geometry.crop). Box, mask, keypoints and every H live in
the crop frame. Ultralytics returns pixel-edge coordinates; they are shifted by -0.5 to our pixel-centre frame.
Strategies stay pure: a failure is f = NaN with a status, never a fallback (CLAUDE.md invariant 5).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from gen.config import apply_overrides, load_config
from geometry.crop import crop_matrix, square_crop_box, warp_crop
from geometry.reading import apply_h
from geometry.rectify import read_strategy

STRATEGIES = ("S0", "S1", "S2", "S3")

# Adapters (all RGB uint8 in):
#   detect(image)     -> (bbox xyxy, face polygon (N, 2) or None) in image px, or None if no dial
#   keypoints(crop)   -> (kp (7, 2), conf (7,)) in crop px, or None
#   regress(crop)     -> f
Detect = Callable[[np.ndarray], "tuple[np.ndarray, np.ndarray | None] | None"]
Keypoints = Callable[[np.ndarray], "tuple[np.ndarray, np.ndarray] | None"]
Regress = Callable[[np.ndarray], float]


@dataclass
class Models:
    detect: Detect | None = None
    keypoints: Keypoints | None = None
    regress: Regress | None = None


def _fail(status: str) -> dict:
    return {"f": float("nan"), "linearity_error": float("nan"), "status": status, "H": None}


def infer(image: np.ndarray, models: Models, cfg: dict, strategies=STRATEGIES) -> dict:
    """Run the pipeline on one RGB image. Never raises: errors become per-strategy fail statuses."""
    out = {"strategies": {s: _fail("fail:not_run") for s in strategies}, "box": None, "crop": None,
           "crop_matrix": None, "mask": None, "keypoints": None, "kp_conf": None, "status": "ok"}
    try:
        _infer(image, models, cfg, strategies, out)
    except Exception as e:                                    # noqa: BLE001 - must never raise on a bad image
        status = f"fail:error_{type(e).__name__}"
        out["status"] = status
        for s in strategies:
            if out["strategies"][s]["status"] == "fail:not_run":
                out["strategies"][s] = _fail(status)
    return out


def _infer(image, models, cfg, strategies, out) -> None:
    if image is None or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 8:
        raise ValueError("not an RGB image")
    det = models.detect(image) if models.detect else None
    if det is None:
        out["status"] = "fail:no_dial"
        for s in strategies:
            out["strategies"][s] = _fail("fail:no_dial")
        return
    box, poly = det
    size = cfg["crop"]["size"]
    A = crop_matrix(*square_crop_box(box, cfg["crop"]["margin"]), size)
    crop = warp_crop(image, A, size)
    out.update(box=np.asarray(box, float), crop=crop, crop_matrix=A)
    if poly is not None and len(poly) >= 3:
        mask = np.zeros((size, size), np.uint8)
        cv2.fillPoly(mask, [np.rint(apply_h(A, poly) * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
        out["mask"] = mask

    geo = [s for s in strategies if s in ("S0", "S1", "S2")]
    if geo:
        kp = models.keypoints(crop) if models.keypoints else None
        if kp is None:
            for s in geo:
                out["strategies"][s] = _fail("fail:no_keypoints")
        else:
            out["keypoints"], out["kp_conf"] = np.asarray(kp[0], float), np.asarray(kp[1], float)
            for s in geo:
                out["strategies"][s] = read_strategy(s, out["keypoints"], mask=out["mask"],
                                                     max_linearity=cfg["max_linearity"])
    if "S3" in strategies:
        if models.regress is None:
            out["strategies"]["S3"] = _fail("fail:no_model")
        else:
            out["strategies"]["S3"] = {"f": float(models.regress(crop)), "linearity_error": float("nan"),
                                       "status": "ok", "H": None}


def oracle_models(rec: dict, cfg: dict) -> Models:
    """Adapters that return a record's ground truth (box, face polygon, keypoints, f).

    Used by the demo's synthetic examples and by tests: every frame conversion in `infer` still runs.
    """
    A = crop_matrix(*square_crop_box(rec["bbox"], cfg["crop"]["margin"]), cfg["crop"]["size"])
    kp = apply_h(A, np.asarray(rec["keypoints"], float)[:, :2])
    box, poly = np.asarray(rec["bbox"], float), np.asarray(rec["face_polygon"], float)
    return Models(detect=lambda img: (box, poly), keypoints=lambda crop: (kp, np.ones(len(kp))),
                  regress=lambda crop: float(rec["f"]))


# ---- real model adapters -----------------------------------------------------------------------

def load_models(cfg: dict, seg: str | None = None, pose: str | None = None, regress: str | None = None) -> Models:
    """Wrap trained weights as adapters. Arguments override cfg['weights']; missing weights -> adapter None."""
    w = {**cfg["weights"], **{k: v for k, v in (("seg", seg), ("pose", pose), ("regress", regress)) if v}}
    device = cfg["device"]
    m = Models()
    if w.get("seg"):
        from ultralytics import YOLO
        yseg = YOLO(w["seg"])

        def detect(img):
            r = yseg.predict(img[..., ::-1], imgsz=cfg["seg"]["imgsz"], conf=cfg["seg"]["conf"], device=device,
                             verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                return None
            i = int(r.boxes.conf.argmax())
            box = r.boxes.xyxy[i].cpu().numpy().astype(float) - 0.5
            poly = None
            if r.masks is not None and len(r.masks.xy) > i and len(r.masks.xy[i]) >= 3:
                poly = np.asarray(r.masks.xy[i], float) - 0.5
            return box, poly
        m.detect = detect
    if w.get("pose"):
        from ultralytics import YOLO
        ypose = YOLO(w["pose"])

        def keypoints(crop):
            r = ypose.predict(crop[..., ::-1], imgsz=cfg["pose"]["imgsz"], conf=cfg["pose"]["conf"], device=device,
                              verbose=False)[0]
            if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
                return None
            i = int(r.boxes.conf.argmax())
            kp = r.keypoints.xy[i].cpu().numpy().astype(float) - 0.5
            conf = r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(kp))
            return (kp, conf) if kp.shape == (7, 2) else None
        m.keypoints = keypoints
    if w.get("regress"):
        from models.regress_net import load_regressor, predict
        net = load_regressor(w["regress"], device)
        m.regress = lambda crop: float(predict(net, [crop], device)[0])
    return m


def format_table(name: str, res: dict) -> str:
    lines = [f"{name}  ({res['status']})", f"  {'strategy':<8} {'f':>8} {'lin_err':>8}  status"]
    for s, r in res["strategies"].items():
        lines.append(f"  {s:<8} {r['f']:>8.4f} {r['linearity_error']:>8.4f}  {r['status']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", nargs="+", required=True)
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--config", default="configs/eval/pipeline.yaml")
    ap.add_argument("--seg-weights")
    ap.add_argument("--pose-weights")
    ap.add_argument("--regress-weights")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. seg.conf=0.1")
    a = ap.parse_args()
    cfg = apply_overrides(load_config(a.config), a.set)
    models = load_models(cfg, a.seg_weights, a.pose_weights, a.regress_weights)
    strategies = tuple(a.strategies.split(","))
    for path in a.image:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        res = infer(None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB), models, cfg, strategies)
        print(format_table(path, res))


if __name__ == "__main__":
    main()
