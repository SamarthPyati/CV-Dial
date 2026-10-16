"""Ultralytics training shared by Model A (seg) and Model B (pose)."""

from __future__ import annotations

from pathlib import Path

from models.common import make_run_dir, make_smoke_data, seed_everything, write_run_info


def train_yolo(cfg: dict, run_id: str | None, smoke_formats: tuple[str, ...], data_yaml_name: str) -> Path:
    """Train with cfg['train'] passed to YOLO.train(); returns the run dir. Weights path goes to run.json."""
    from ultralytics import YOLO

    seed_everything(cfg["seed"])
    run = make_run_dir(cfg, run_id)
    data = cfg["data"]
    if cfg.get("is_smoke"):
        sd = cfg["smoke_data"]
        make_smoke_data(run / "smoke_data", smoke_formats, sd["n_train"], sd["n_val"])
        data = str(run / "smoke_data" / "yamls" / data_yaml_name)
    write_run_info(run, data=data, model=cfg["model"], status="running")

    args = {k: v for k, v in cfg["train"].items() if v is not None}
    model = YOLO(cfg["model"])
    model.train(data=data, project=str(run.resolve()), name="ultralytics", exist_ok=True,
                seed=cfg["seed"], deterministic=True, **args)
    weights = run / "ultralytics" / "weights" / "best.pt"
    write_run_info(run, weights=weights, status="done" if weights.exists() else "no_weights")
    print(f"run: {run}\nweights: {weights}")
    return run
