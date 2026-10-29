"""Gradio demo: photo -> keypoints, rectified dial, readings for S0-S3 (PLAN §1, TASKS M9).

    python -m pipeline.demo_app [--seg-weights W --pose-weights W --regress-weights W] [--port 7860] [--share] [--no-ocr]

Upload, paste or take a photo. The value is computed from the scale range: typed min/max win; otherwise (with the
optional `ocr` extra) the range is read from the dial's numerals by pipeline.scale_ocr - demo only, never in eval.
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
from pipeline.infer import STRATEGIES, Models, infer, load_models, oracle_models, upright_matrix

RECT_SIZE = 480
KP_NAMES = ["K0 pivot", "K1 tip", "K2 0%", "K3 25%", "K4 50%", "K5 75%", "K6 100%"]
KP_COLOURS = [(255, 0, 255), (255, 40, 40), (0, 170, 0), (0, 170, 170), (0, 120, 255), (200, 160, 0), (120, 0, 200)]
BIN_EDGES = [0, 15, 30, 45, 65]

# Ctrl/Cmd+V anywhere on the page: an image on the clipboard is handed to the Photo component's own file input,
# so Gradio treats it exactly like an upload (and fires `upload`). Text pastes are left alone.
PASTE_JS = """
<script>
(() => {
  const waitFor = (sel, ms) => new Promise((resolve) => {
    const t0 = Date.now();
    const tick = () => {
      const el = document.querySelector(sel);
      if (el || Date.now() - t0 > ms) resolve(el); else setTimeout(tick, 50);
    };
    tick();
  });
  document.addEventListener("paste", async (event) => {
    const items = Array.from((event.clipboardData && event.clipboardData.items) || []);
    const item = items.find((i) => i.kind === "file" && i.type.startsWith("image/"));
    if (!item) return;
    event.preventDefault();
    const blob = item.getAsFile();
    const ext = (blob.type.split("/")[1] || "png").replace("jpeg", "jpg");
    const file = new File([blob], `pasted-${Date.now()}.${ext}`, { type: blob.type });
    let input = document.querySelector("#photo input[type=file]");
    if (!input) {                                   // a photo is already shown: clear it to get the upload input back
      const clear = document.querySelector("#photo button[aria-label='Clear'], #photo button[title='Clear']");
      if (clear) clear.click();
      input = await waitFor("#photo input[type=file]", 2000);
    }
    if (!input) { console.warn("gauge demo: photo upload input not found"); return; }
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
})();
</script>
"""


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


def display_matrix(res: dict, strategy: str | None, size: int = RECT_SIZE) -> np.ndarray | None:
    """Crop px -> display px (see pipeline.infer.upright_matrix)."""
    return upright_matrix(res, strategy, size)


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


def _fmt(v: float) -> str:
    return f"{v:.6g}"


def scale_note(entered: bool, scale, reader_on: bool) -> str:
    """One line saying where the min/max used for the value came from."""
    if entered:
        return "Scale range: entered."
    if scale is not None and scale.read_ends:
        return (f"Scale range **{_fmt(scale.lo)} – {_fmt(scale.hi)}**, read from {len(scale.numerals)} numerals "
                "on the dial. Type min/max to override.")
    if scale is not None:
        return (f"Scale range **≈ {_fmt(scale.lo)} – {_fmt(scale.hi)}**, *inferred* from {len(scale.numerals)} "
                "numerals (an end value was not readable) — please check it, or type min/max.")
    if reader_on:
        return "Could not read the scale numbers: type min and max to get the value."
    return "Type the scale's min and max to get the value (automatic reading needs `uv sync --extra ocr`)."


def analyse(image: np.ndarray | None, models: Models, cfg: dict, vmin=None, vmax=None,
            truth: dict | None = None, scale_reader=None) -> tuple[np.ndarray | None, np.ndarray | None, list[list], str]:
    """Run the pipeline and build the demo outputs: (overlay, rectified, table rows, markdown summary).

    The value uses the entered min/max; otherwise, if a scale_reader is given (demo-only OCR), the range read
    from the dial's numerals.
    """
    if image is None:
        return None, None, [], "Upload or take a photo of an analog gauge."
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = image[..., :3]
    res = infer(image, models, cfg)
    used, f = demo_reading(res)
    entered = vmin is not None and vmax is not None and vmin != vmax
    scale = None
    if not entered and scale_reader is not None and used is not None:
        scale = scale_reader(image, res, used)
        if scale is not None:
            vmin, vmax = scale.lo, scale.hi
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
        md = f"### Reading: {reading}\nUsing {used}{note}. {scale_note(entered, scale, scale_reader is not None)}"
    if truth is not None:
        tv = _value(truth["f"], *truth["range"])                     # the dial's true range, not the detected one
        md += (f"\n\nSynthetic example, ground-truth keypoints · true f = {truth['f']:.3f}"
               f"{'' if tv is None else f' (value {tv:.3g})'} · obliquity {truth['obliquity_deg']:.1f}°")
    return overlay(image, res), rectified_view(res, used), rows, md


def synthetic_example(gen_cfg: dict, index: int, bin_name: str) -> tuple[np.ndarray, dict]:
    """A generator image whose obliquity lies in `bin_name` (e.g. '30-45')."""
    from gen.generate import make_sample

    k = [f"{a}-{b}" for a, b in zip(BIN_EDGES[:-1], BIN_EDGES[1:])].index(bin_name)
    return make_sample(gen_cfg, "demo", int(index) * (len(BIN_EDGES) - 1) + k, stratify=True)


def build_app(models: Models, cfg: dict, gen_cfg: dict, scale_reader=None):
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
                photo = gr.Image(type="numpy", label="Photo  (upload, webcam, or paste with Ctrl/⌘+V)",
                                 sources=["upload", "webcam", "clipboard"], elem_id="photo")
                with gr.Row():
                    vmin = gr.Number(label="Scale min (blank: read from dial)", value=None)
                    vmax = gr.Number(label="Scale max (blank: read from dial)", value=None)
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
            return analyse(img, example_models(rec) if rec is not None else models, cfg, lo, hi, rec, scale_reader)

        def on_example(b, i, lo, hi):
            img, rec = synthetic_example(gen_cfg, i, b)
            return (img, rec, *analyse(img, example_models(rec), cfg, lo, hi, rec, scale_reader))

        def on_upload(img, lo, hi):
            # An uploaded or pasted photo is read straight away, always with the real models.
            return (None, *analyse(img, models, cfg, lo, hi, scale_reader=scale_reader))

        read_btn.click(on_read, [photo, vmin, vmax, truth], [over, rect, table, summary])
        photo.input(lambda: None, None, truth)                  # a user-changed photo is never read with oracle keypoints
        photo.upload(on_upload, [photo, vmin, vmax], [truth, over, rect, table, summary])
        ex_btn.click(on_example, [bin_dd, idx, vmin, vmax], [photo, truth, over, rect, table, summary])
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/eval/pipeline.yaml")
    ap.add_argument("--gen-config", default="configs/gen/default.yaml")
    ap.add_argument("--seg-weights")
    ap.add_argument("--pose-weights")
    ap.add_argument("--regress-weights")
    ap.add_argument("--port", type=int, default=None, help="default: first free port from 7860")
    ap.add_argument("--no-ocr", action="store_true", help="do not read the scale range from the dial numerals")
    ap.add_argument("--share", action="store_true", help="public Gradio link (for presenting from another device)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    models = load_models(cfg, a.seg_weights, a.pose_weights, a.regress_weights)
    reader = None
    if not a.no_ocr:
        from pipeline import scale_ocr
        if scale_ocr.available():
            reader = scale_ocr.ScaleReader(cfg["scale_ocr"])
            reader._ocr(np.full((32, 32, 3), 255, np.uint8))              # load the OCR models now, not on first photo
        else:
            print("Scale OCR off: install it with `uv sync --extra ocr` to read the range from the dial.")
    app = build_app(models, cfg, load_config(a.gen_config), reader)
    app.launch(server_port=a.port, share=a.share, head=PASTE_JS)


if __name__ == "__main__":
    main()
