"""Gradio demo: photo -> keypoints, rectified dial, readings for S0-S3 (PLAN §1, TASKS M9).

    python -m pipeline.demo_app [--seg-weights W --pose-weights W --regress-weights W] [--port 7860] [--share]

Upload or take a photo; optionally enter the printed scale range to see the value as well as f.
Without trained weights the app still runs "synthetic examples": generator images read with their
ground-truth keypoints (oracle adapters), so the geometry can be shown before models are trained.

The S2 -> S1 fallback lives ONLY here (CLAUDE.md invariant 5): the headline "demo reading" uses S2, or S1 if
S2 failed. Every strategy's own result is still shown unmodified in the table.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from gen.config import load_config
from geometry.reading import apply_h
from pipeline.infer import STRATEGIES, Models, infer, load_models, oracle_models

RECT_SIZE = 480
KP_NAMES = ["K0 pivot", "K1 tip", "K2 0%", "K3 25%", "K4 50%", "K5 75%", "K6 100%"]
KP_COLOURS = [(255, 0, 255), (255, 40, 40), (0, 170, 0), (0, 170, 170), (0, 120, 255), (200, 160, 0), (120, 0, 200)]
BIN_EDGES = [0, 15, 30, 45, 65]


def demo_reading(res: dict) -> tuple[str | None, float]:
    """(strategy used, f): S2, falling back to S1. Demo only - experiments never fall back."""
    for s in ("S2", "S1"):
        r = res["strategies"].get(s)
        if r is not None and r["status"] == "ok":
            return s, r["f"]
    return None, float("nan")


def overlay(image: np.ndarray, res: dict) -> np.ndarray:
    """Photo with the detected box, face outline and keypoints (image frame)."""
    out = image.copy()
    if res["box"] is None:
        return out
    t = max(2, round(max(out.shape[:2]) / 400))
    x0, y0, x1, y1 = (int(round(v)) for v in res["box"])
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 200, 0), t)
    Ainv = np.linalg.inv(res["crop_matrix"])
    if res["mask"] is not None:
        cnts, _ = cv2.findContours(res["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            pts = apply_h(Ainv, c.reshape(-1, 2).astype(float))
            cv2.polylines(out, [np.rint(pts).astype(np.int32)], True, (0, 255, 0), t, cv2.LINE_AA)
    if res["keypoints"] is not None:
        kp = apply_h(Ainv, res["keypoints"])
        p = [tuple(int(round(v)) for v in q) for q in kp]
        cv2.line(out, p[0], p[1], (255, 40, 40), t, cv2.LINE_AA)
        for i, q in enumerate(p):
            cv2.circle(out, q, 3 * t, KP_COLOURS[i], -1, cv2.LINE_AA)
            cv2.putText(out, f"K{i}", (q[0] + 3 * t, q[1] - 3 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * t,
                        KP_COLOURS[i], max(1, t // 2), cv2.LINE_AA)
    return out


def display_matrix(res: dict, strategy: str | None) -> np.ndarray | None:
    """Crop px -> display px: the strategy's H followed by a similarity (angles unchanged) that centres the pivot,
    fixes the scale-ring radius and turns the 50 % mark (K4) straight up, as the dial would be read."""
    if strategy is None or res["crop"] is None or res["keypoints"] is None:
        return None
    H = res["strategies"][strategy]["H"]
    r = apply_h(H, res["keypoints"])
    radius = float(np.mean(np.linalg.norm(r[2:] - r[0], axis=1)))
    if not np.isfinite(radius) or radius <= 0:
        return None
    s = 0.36 * RECT_SIZE / radius
    d = r[4] - r[0]
    rot = -np.pi / 2 - np.arctan2(d[1], d[0])                                          # K4 -> up (y down)
    S = np.array([[s * np.cos(rot), -s * np.sin(rot), 0.0], [s * np.sin(rot), s * np.cos(rot), 0.0], [0, 0, 1.0]])
    S[:2, 2] = (RECT_SIZE - 1) / 2 - S[:2, :2] @ r[0]
    return S @ H


def rectified_view(res: dict, strategy: str | None) -> np.ndarray | None:
    """The crop in the display frame of `display_matrix`, with the rectified keypoints drawn."""
    G = display_matrix(res, strategy)
    if G is None:
        return None
    out = cv2.warpPerspective(res["crop"], G, (RECT_SIZE, RECT_SIZE), flags=cv2.INTER_LINEAR, borderValue=(30, 30, 30))
    q = [tuple(int(round(v)) for v in p) for p in apply_h(G, res["keypoints"])]
    cv2.line(out, q[0], q[1], (255, 40, 40), 2, cv2.LINE_AA)
    for i in range(2, 7):
        cv2.line(out, q[0], q[i], (0, 170, 0), 1, cv2.LINE_AA)
    for i, p in enumerate(q):
        cv2.circle(out, p, 5, KP_COLOURS[i], -1, cv2.LINE_AA)
    return out


def _value(f: float, vmin, vmax) -> float | None:
    if vmin is None or vmax is None or vmin == vmax or not np.isfinite(f):
        return None
    return float(vmin) + f * (float(vmax) - float(vmin))


def analyse(image: np.ndarray | None, models: Models, cfg: dict, vmin=None, vmax=None,
            truth: dict | None = None) -> tuple[np.ndarray | None, np.ndarray | None, list[list], str]:
    """Run the pipeline and build the demo outputs: (overlay, rectified, table rows, markdown summary)."""
    if image is None:
        return None, None, [], "Upload or take a photo of an analog gauge."
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = image[..., :3]
    res = infer(image, models, cfg)
    used, f = demo_reading(res)
    rows = []
    for s in STRATEGIES:
        r = res["strategies"][s]
        v = _value(r["f"], vmin, vmax)
        rows.append([s, None if not np.isfinite(r["f"]) else round(r["f"], 4),
                     None if v is None else round(v, 4), r["status"]])
    if res["status"] == "fail:no_dial":
        md = "### No dial found\nTry a closer, sharper photo with the whole gauge face in view."
    elif used is None:
        md = f"### Could not read the needle\nPipeline status: `{res['status']}`; S1 and S2 both failed."
    else:
        v = _value(f, vmin, vmax)
        reading = f"**{v:.3g}** (f = {f:.3f})" if v is not None else f"f = **{f:.3f}** of full scale"
        note = "" if used == "S2" else " — S2 failed, showing the S1 fallback (demo only)"
        md = f"### Reading: {reading}\nUsing {used}{note}."
    if truth is not None:
        tv = _value(truth["f"], vmin, vmax)
        md += (f"\n\nSynthetic example, ground-truth keypoints · true f = {truth['f']:.3f}"
               f"{'' if tv is None else f' (value {tv:.3g})'} · obliquity {truth['obliquity_deg']:.1f}°")
    return overlay(image, res), rectified_view(res, used), rows, md


def synthetic_example(gen_cfg: dict, index: int, bin_name: str) -> tuple[np.ndarray, dict]:
    """A generator image whose obliquity lies in `bin_name` (e.g. '30-45')."""
    from gen.generate import make_sample

    k = [f"{a}-{b}" for a, b in zip(BIN_EDGES[:-1], BIN_EDGES[1:])].index(bin_name)
    return make_sample(gen_cfg, "demo", int(index) * (len(BIN_EDGES) - 1) + k, stratify=True)


def build_app(models: Models, cfg: dict, gen_cfg: dict):
    import gradio as gr

    has_models = models.detect is not None
    bins = [f"{a}-{b}" for a, b in zip(BIN_EDGES[:-1], BIN_EDGES[1:])]
    with gr.Blocks(title="Gauge Reader") as app:
        gr.Markdown("# Gauge Reader\nReads an analog pointer gauge from one photo and corrects for camera tilt. "
                    "S0 = no correction, S1 = ellipse fit, S2 = conic + pivot, S3 = direct regression.")
        if not has_models:
            gr.Markdown("> No trained weights loaded: uploaded photos will report *no dial*. "
                        "Use **Synthetic example** to see the geometry with ground-truth keypoints.")
        truth = gr.State(None)
        with gr.Row():
            with gr.Column():
                photo = gr.Image(type="numpy", label="Photo", sources=["upload", "webcam", "clipboard"])
                with gr.Row():
                    vmin = gr.Number(label="Scale min (optional)", value=None)
                    vmax = gr.Number(label="Scale max (optional)", value=None)
                read_btn = gr.Button("Read gauge", variant="primary")
                with gr.Accordion("Synthetic example (ground-truth keypoints)", open=not has_models):
                    with gr.Row():
                        bin_dd = gr.Dropdown(bins, value="30-45", label="Obliquity bin (deg)")
                        idx = gr.Number(value=0, precision=0, label="Example #")
                    ex_btn = gr.Button("Load and read example")
            with gr.Column():
                summary = gr.Markdown()
                over = gr.Image(label="Detected dial and keypoints", interactive=False)
                rect = gr.Image(label="Rectified dial (S2, or S1 fallback)", interactive=False)
                table = gr.Dataframe(headers=["strategy", "f", "value", "status"], label="All strategies",
                                     interactive=False)

        def example_models(rec):
            # Ground-truth box/mask/keypoints, but S3 only from a real model: its "oracle" would just echo f.
            m = oracle_models(rec, cfg)
            m.regress = models.regress
            return m

        def on_read(img, lo, hi, rec):
            return analyse(img, example_models(rec) if rec is not None else models, cfg, lo, hi, rec)

        def on_example(b, i, lo, hi):
            img, rec = synthetic_example(gen_cfg, i, b)
            return (img, rec, *analyse(img, example_models(rec), cfg, lo, hi, rec))

        read_btn.click(on_read, [photo, vmin, vmax, truth], [over, rect, table, summary])
        photo.input(lambda: None, None, truth)                  # a user-changed photo is never read with oracle keypoints
        ex_btn.click(on_example, [bin_dd, idx, vmin, vmax], [photo, truth, over, rect, table, summary])
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/eval/pipeline.yaml")
    ap.add_argument("--gen-config", default="configs/gen/default.yaml")
    ap.add_argument("--seg-weights")
    ap.add_argument("--pose-weights")
    ap.add_argument("--regress-weights")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="public Gradio link (for presenting from another device)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    models = load_models(cfg, a.seg_weights, a.pose_weights, a.regress_weights)
    build_app(models, cfg, load_config(a.gen_config)).launch(server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
