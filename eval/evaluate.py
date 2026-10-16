"""Evaluate strategies per obliquity bin (PLAN §7). The only source of reported numbers (CLAUDE.md invariant 7).

    python -m eval.evaluate --run results/<run_id> --split synth-test|real|<dir> [--oracle --kp-noise 0,1,2,4]
        [--config configs/eval/default.yaml] [--seg-weights W --pose-weights W --regress-weights W] [--set k=v]

Writes into results/<run_id>/: config.yaml (resolved; reused if present), predictions.csv
(one row per image x strategy x noise level), metrics.csv (one row per strategy x noise x bin, plus 'all'),
and keypoints.csv in pipeline mode (per-keypoint error / dial radius, predicted keypoints in image px).

Oracle mode (E1): S0-S2 on ground-truth keypoints (image px) with seeded Gaussian noise sigma (px) on all 7
keypoints; S1 uses the ground-truth face polygon. Pipeline mode (E2/E3): pipeline.infer with trained weights.
Errors are |f_pred - f| x 100 (% FS); failures (f = NaN) are excluded from MAE and reported as failure_rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
import yaml

from eval.bootstrap import bootstrap_ci
from gen.config import apply_overrides, load_config
from geometry.reading import apply_h
from geometry.rectify import read_strategy

PRED_FIELDS = ["id", "strategy", "kp_noise", "bin", "obliquity_deg", "f_true", "f_pred", "error", "status",
               "linearity_error", "ms"]
METRIC_FIELDS = ["strategy", "kp_noise", "bin", "n", "n_ok", "failure_rate", "mae", "mae_ci_lo", "mae_ci_hi",
                 "median", "ms"]


def bin_label(obl: float, edges) -> str:
    """'lo-hi' for the bin containing obl; the last bin is closed. Outside the edges -> 'out'."""
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        if lo <= obl < hi or (k == len(edges) - 2 and obl == hi):
            return f"{lo:g}-{hi:g}"
    return "out"


def load_records(split_dir: Path, limit: int | None = None) -> list[dict]:
    paths = sorted((split_dir / "records").glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no records in {split_dir / 'records'}")
    return [json.loads(p.read_text()) for p in paths[:limit]]


def noisy_keypoints(rec: dict, sigma: float, seed: int) -> np.ndarray:
    """GT keypoints (7, 2) in image px plus N(0, sigma^2) noise; the same noise for every strategy (paired)."""
    kp = np.asarray(rec["keypoints"], float)[:, :2]
    if sigma <= 0:
        return kp
    rng = np.random.default_rng([seed, zlib.crc32(rec["id"].encode()), int(round(sigma * 1000))])
    return kp + rng.normal(0, sigma, kp.shape)


def _row(rec, strategy, sigma, res, edges, ms) -> dict:
    f_pred = res["f"]
    err = abs(f_pred - rec["f"]) * 100 if np.isfinite(f_pred) else float("nan")
    return {"id": rec["id"], "strategy": strategy, "kp_noise": sigma, "bin": bin_label(rec["obliquity_deg"], edges),
            "obliquity_deg": rec["obliquity_deg"], "f_true": rec["f"], "f_pred": f_pred, "error": err,
            "status": res["status"], "linearity_error": res["linearity_error"], "ms": ms}


def oracle_rows(recs: list[dict], cfg: dict, sigmas: list[float]) -> list[dict]:
    strategies = [s for s in cfg["strategies"] if s in ("S0", "S1", "S2")]
    rows = []
    for sigma in sigmas:
        for rec in recs:
            kp = noisy_keypoints(rec, sigma, cfg["seed"])
            rim = np.asarray(rec["face_polygon"], float)
            for s in strategies:
                t0 = time.perf_counter()
                res = read_strategy(s, kp, rim=rim, max_linearity=cfg["max_linearity"])
                rows.append(_row(rec, s, sigma, res, cfg["obliquity_bins"], 1000 * (time.perf_counter() - t0)))
    return rows


def pipeline_rows(recs: list[dict], split_dir: Path, cfg: dict) -> tuple[list[dict], list[dict]]:
    from pipeline.infer import infer, load_models

    pcfg = load_config(cfg["pipeline"])
    pcfg["max_linearity"] = cfg["max_linearity"]
    models = load_models(pcfg, **{k: v for k, v in cfg["weights"].items() if v})
    rows, kp_rows = [], []
    for rec in recs:
        img = cv2.imread(str(split_dir / rec["image"]), cv2.IMREAD_COLOR)
        img = None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        res = infer(img, models, pcfg, tuple(cfg["strategies"]))
        ms = 1000 * (time.perf_counter() - t0)
        for s, r in res["strategies"].items():
            rows.append(_row(rec, s, 0.0, r, cfg["obliquity_bins"], ms))
        if res["keypoints"] is not None:
            kp_img = apply_h(np.linalg.inv(res["crop_matrix"]), res["keypoints"])
            gt = np.asarray(rec["keypoints"], float)[:, :2]
            poly = np.asarray(rec["face_polygon"], float)
            radius = np.sqrt(abs(cv2.contourArea(poly.astype(np.float32))) / np.pi)   # equal-area radius
            err = np.linalg.norm(kp_img - gt, axis=1) / radius
            kp_rows.append({"id": rec["id"], "bin": bin_label(rec["obliquity_deg"], cfg["obliquity_bins"]),
                            **{f"err_k{i}": float(e) for i, e in enumerate(err)},
                            "kp_pred": json.dumps(np.round(kp_img, 2).tolist())})
    return rows, kp_rows


def metrics(rows: list[dict], cfg: dict) -> list[dict]:
    """One row per (strategy, kp_noise, bin) plus bin='all'."""
    tols = cfg["tolerances_fs"]
    bins = [bin_label(0.5 * (a + b), cfg["obliquity_bins"]) for a, b in
            zip(cfg["obliquity_bins"][:-1], cfg["obliquity_bins"][1:])]
    out = []
    keys = sorted({(r["strategy"], r["kp_noise"]) for r in rows}, key=lambda k: (k[0], k[1]))
    for s, sigma in keys:
        sel = [r for r in rows if r["strategy"] == s and r["kp_noise"] == sigma]
        present = bins + sorted({r["bin"] for r in sel} - set(bins))
        for b in present + ["all"]:
            group = sel if b == "all" else [r for r in sel if r["bin"] == b]
            if not group:
                continue
            ok = np.array([r["error"] for r in group if r["status"] == "ok"], float)
            lo, hi = bootstrap_ci(ok, resamples=cfg["bootstrap"]["resamples"], ci=cfg["bootstrap"]["ci"],
                                  seed=cfg["seed"], key=f"{s}|{sigma}|{b}")
            m = {"strategy": s, "kp_noise": sigma, "bin": b, "n": len(group), "n_ok": len(ok),
                 "failure_rate": 1 - len(ok) / len(group),
                 "mae": float(ok.mean()) if len(ok) else float("nan"), "mae_ci_lo": lo, "mae_ci_hi": hi,
                 "median": float(np.median(ok)) if len(ok) else float("nan"),
                 "ms": float(np.mean([r["ms"] for r in group]))}     # pipeline: whole image; oracle: geometry only
            for t in tols:
                m[f"within_{t:g}"] = float((ok <= t).mean()) if len(ok) else float("nan")
            out.append(m)
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def resolve_run_config(run: Path, args) -> dict:
    """Reuse run/config.yaml if it exists (re-evaluation), else build it from --config + CLI and save it."""
    path = run / "config.yaml"
    if path.exists():
        cfg = load_config(path)
        cfg = apply_overrides(cfg, args.set)
    else:
        cfg = apply_overrides(load_config(args.config), args.set)
        cfg["split"] = args.split
        cfg["oracle"] = bool(args.oracle)
        cfg["kp_noise"] = [float(s) for s in args.kp_noise.split(",")]
        for k in ("seg", "pose", "regress"):
            v = getattr(args, f"{k}_weights")
            if v:
                cfg["weights"][k] = v
        if args.limit is not None:
            cfg["limit"] = args.limit
    run.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg


def evaluate(run: Path, cfg: dict) -> list[dict]:
    split = cfg["split"]
    split_dir = Path(cfg["splits"].get(split, split))
    recs = load_records(split_dir, cfg.get("limit"))
    if cfg["oracle"]:
        rows, kp_rows = oracle_rows(recs, cfg, cfg["kp_noise"]), []
    else:
        rows, kp_rows = pipeline_rows(recs, split_dir, cfg)
    write_csv(run / "predictions.csv", rows, PRED_FIELDS)
    m = metrics(rows, cfg)
    write_csv(run / "metrics.csv", m, METRIC_FIELDS + [f"within_{t:g}" for t in cfg["tolerances_fs"]])
    if kp_rows:
        write_csv(run / "keypoints.csv", kp_rows, ["id", "bin"] + [f"err_k{i}" for i in range(7)] + ["kp_pred"])
    return m


def format_metrics(m: list[dict]) -> str:
    lines = [f"{'strategy':<8} {'noise':>5} {'bin':>6} {'n':>5} {'fail':>6} {'MAE':>7} {'95% CI':>17} {'median':>7}"]
    for r in m:
        lines.append(f"{r['strategy']:<8} {r['kp_noise']:>5g} {r['bin']:>6} {r['n']:>5} {r['failure_rate']:>6.3f} "
                     f"{r['mae']:>7.3f} [{r['mae_ci_lo']:>7.3f},{r['mae_ci_hi']:>7.3f}] {r['median']:>7.3f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="results/<run_id>")
    ap.add_argument("--split", default="synth-test", help="name from config `splits`, or a directory")
    ap.add_argument("--oracle", action="store_true", help="E1: ground-truth keypoints / face polygon")
    ap.add_argument("--kp-noise", default="0", help="comma-separated Gaussian sigmas in image px (oracle)")
    ap.add_argument("--config", default="configs/eval/default.yaml")
    ap.add_argument("--seg-weights")
    ap.add_argument("--pose-weights")
    ap.add_argument("--regress-weights")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    run = Path(a.run)
    cfg = resolve_run_config(run, a)
    m = evaluate(run, cfg)
    print(format_metrics(m))
    print(f"wrote {run / 'metrics.csv'}")


if __name__ == "__main__":
    main()
