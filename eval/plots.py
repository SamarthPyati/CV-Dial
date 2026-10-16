"""Figures for the report (PLAN §7). Reads only evaluate.py outputs (and the split's images for the gallery).

    python -m eval.plots run --run results/<run_id>                 # all per-run figures -> <run>/plots/
    python -m eval.plots ablation --runs results/a results/b ... --labels base no_glare ... --out results/e4.png

Per-run figures: mae_vs_obliquity.png (5 deg bins, one line per strategy - the headline figure),
mae_bins.png (per-bin bars with 95 % CI), failure_gallery.png (worst cases with keypoints drawn).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gen.config import load_config  # noqa: E402

COLOURS = {"S0": "#7f7f7f", "S1": "#1f77b4", "S2": "#d62728", "S3": "#2ca02c"}


def read_csv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _f(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def mae_vs_obliquity(preds: list[dict], out: Path, step: float = 5.0, max_deg: float = 65.0) -> Path:
    """MAE (% FS) of successful predictions in `step`-degree obliquity bins; one line per strategy x noise."""
    edges = np.arange(0, max_deg + step, step)
    centres = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(7, 4.2))
    groups = sorted({(r["strategy"], _f(r["kp_noise"])) for r in preds})
    noises = sorted({n for _, n in groups})
    styles = ["-", "--", ":", "-."]
    for s, n in groups:
        sel = [r for r in preds if r["strategy"] == s and _f(r["kp_noise"]) == n and r["status"] == "ok"]
        obl = np.array([_f(r["obliquity_deg"]) for r in sel])
        err = np.array([_f(r["error"]) for r in sel])
        idx = np.clip(np.digitize(obl, edges) - 1, 0, len(centres) - 1)
        mae = [err[idx == k].mean() if np.any(idx == k) else np.nan for k in range(len(centres))]
        label = s if len(noises) == 1 else f"{s}, σ={n:g}px"
        ax.plot(centres, mae, styles[noises.index(n) % 4], marker="o", ms=3, color=COLOURS.get(s), label=label)
    ax.set_xlabel("obliquity (deg)")
    ax.set_ylabel("MAE (% of full scale)")
    ax.set_xlim(0, max_deg)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2 if len(groups) > 4 else 1)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def mae_bins(metrics: list[dict], out: Path, kp_noise: float | None = None) -> Path:
    """Grouped bars: MAE per obliquity bin for each strategy, 95 % bootstrap CI error bars."""
    noise = kp_noise if kp_noise is not None else min(_f(m["kp_noise"]) for m in metrics)
    rows = [m for m in metrics if _f(m["kp_noise"]) == noise]
    strategies = sorted({m["strategy"] for m in rows})
    bins = list(dict.fromkeys(m["bin"] for m in rows if m["bin"] not in ("all", "out"))) + ["all"]
    width = 0.8 / max(len(strategies), 1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(bins))
    for i, s in enumerate(strategies):
        by_bin = {m["bin"]: m for m in rows if m["strategy"] == s}
        mae = np.array([_f(by_bin[b]["mae"]) if b in by_bin else np.nan for b in bins])
        lo = np.array([_f(by_bin[b]["mae_ci_lo"]) if b in by_bin else np.nan for b in bins])
        hi = np.array([_f(by_bin[b]["mae_ci_hi"]) if b in by_bin else np.nan for b in bins])
        yerr = np.nan_to_num(np.vstack([mae - lo, hi - mae]))
        ax.bar(x + (i - (len(strategies) - 1) / 2) * width, mae, width, yerr=yerr, capsize=2,
               color=COLOURS.get(s), label=s)
    ax.set_xticks(x, [f"{b}°" if b != "all" else "overall" for b in bins])
    ax.set_ylabel("MAE (% of full scale)")
    ax.set_title(f"MAE per obliquity bin (95 % CI){'' if noise == 0 else f', σ={noise:g}px'}", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def ablation_bars(runs: list[Path], labels: list[str], out: Path, kp_noise: float = 0.0) -> Path:
    """E4: overall MAE per strategy for each ablation run (each run has its own metrics.csv)."""
    data = {lab: {m["strategy"]: m for m in read_csv(r / "metrics.csv")
                  if m["bin"] == "all" and _f(m["kp_noise"]) == kp_noise} for r, lab in zip(runs, labels)}
    strategies = sorted({s for d in data.values() for s in d})
    width = 0.8 / max(len(strategies), 1)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 4.2))
    x = np.arange(len(labels))
    for i, s in enumerate(strategies):
        ms = [data[lab].get(s) for lab in labels]
        mae = np.array([_f(m["mae"]) if m else np.nan for m in ms])
        lo = np.array([_f(m["mae_ci_lo"]) if m else np.nan for m in ms])
        hi = np.array([_f(m["mae_ci_hi"]) if m else np.nan for m in ms])
        ax.bar(x + (i - (len(strategies) - 1) / 2) * width, mae, width,
               yerr=np.nan_to_num(np.vstack([mae - lo, hi - mae])), capsize=2, color=COLOURS.get(s), label=s)
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylabel("overall MAE (% of full scale)")
    ax.set_title("E4: nuisance ablation", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def failure_gallery(run: Path, out: Path, strategy: str = "S2", n: int = 8, thumb: int = 320) -> Path | None:
    """Worst cases (failures first, then largest error) for `strategy`: crop around the dial with GT keypoints
    (green) and, in pipeline runs, predicted keypoints (red)."""
    import cv2

    cfg = load_config(run / "config.yaml")
    split_dir = Path(cfg["splits"].get(cfg["split"], cfg["split"]))
    preds = [r for r in read_csv(run / "predictions.csv") if r["strategy"] == strategy]
    if not preds:
        return None
    noise0 = min(_f(r["kp_noise"]) for r in preds)
    preds = [r for r in preds if _f(r["kp_noise"]) == noise0]
    preds.sort(key=lambda r: (r["status"] == "ok", -np.nan_to_num(_f(r["error"]), nan=1e9)))
    kp_pred = {}
    if (run / "keypoints.csv").exists():
        kp_pred = {r["id"]: np.array(json.loads(r["kp_pred"])) for r in read_csv(run / "keypoints.csv")}
    tiles = []
    for r in preds[:n]:
        rec = json.loads((split_dir / "records" / f"{r['id']}.json").read_text())
        img = cv2.imread(str(split_dir / rec["image"]))
        if img is None:
            continue
        x0, y0, x1, y1 = rec["bbox"]
        pad = 0.15 * max(x1 - x0, y1 - y0)
        X0, Y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
        crop = img[Y0:int(y1 + pad) + 1, X0:int(x1 + pad) + 1]
        s = thumb / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

        def draw(kp, colour):
            for i, (x, y) in enumerate(kp):
                p = (int(round((x - X0) * s)), int(round((y - Y0) * s)))
                cv2.circle(crop, p, 4 if i < 2 else 3, colour, -1 if i < 2 else 1, cv2.LINE_AA)
            a = (int(round((kp[0][0] - X0) * s)), int(round((kp[0][1] - Y0) * s)))
            b = (int(round((kp[1][0] - X0) * s)), int(round((kp[1][1] - Y0) * s)))
            cv2.line(crop, a, b, colour, 1, cv2.LINE_AA)

        draw(np.asarray(rec["keypoints"])[:, :2], (0, 200, 0))
        if r["id"] in kp_pred:
            draw(kp_pred[r["id"]], (0, 0, 255))
        tile = np.full((thumb + 36, thumb, 3), 255, np.uint8)
        tile[:crop.shape[0], :crop.shape[1]] = crop
        txt = f"{r['id']} obl={_f(r['obliquity_deg']):.0f}"
        res = r["status"] if r["status"] != "ok" else f"err={_f(r['error']):.2f}%FS"
        cv2.putText(tile, txt, (4, thumb + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(tile, res[:40], (4, thumb + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 180), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return None
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(out), sheet)
    return out


def run_plots(run: Path) -> list[Path]:
    plots = run / "plots"
    plots.mkdir(exist_ok=True)
    preds, metrics = read_csv(run / "predictions.csv"), read_csv(run / "metrics.csv")
    made = [mae_vs_obliquity(preds, plots / "mae_vs_obliquity.png"), mae_bins(metrics, plots / "mae_bins.png")]
    for s in sorted({r["strategy"] for r in preds}):
        g = failure_gallery(run, plots / f"failure_gallery_{s}.png", strategy=s)
        if g:
            made.append(g)
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run", required=True)
    e = sub.add_parser("ablation")
    e.add_argument("--runs", nargs="+", required=True)
    e.add_argument("--labels", nargs="+", required=True)
    e.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        for p in run_plots(Path(a.run)):
            print(f"wrote {p}")
    else:
        if len(a.runs) != len(a.labels):
            raise SystemExit("--runs and --labels must have the same length")
        print(f"wrote {ablation_bars([Path(p) for p in a.runs], a.labels, Path(a.out))}")


if __name__ == "__main__":
    main()
