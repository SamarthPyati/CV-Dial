"""Model C (S3): train the ResNet-18 regressor on exported crops (PLAN §4).

    python -m models.train_regress --config configs/train/regress.yaml [--smoke] [--set train.epochs=10]

Reads <crops_dir>/regress_<split>.csv (crop, f, obliquity_deg, id). Writes results/<run_id>/
{config.yaml, run.json, metrics.csv, weights/{best,last}.pt}. Augmentation: crop jitter only, never a flip.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.common import make_run_dir, make_smoke_data, parse_args, resolve_config, seed_everything, write_run_info
from models.regress_net import RegressNet, save_checkpoint, to_tensor


def check_no_flip(cfg: dict) -> None:
    """INVARIANT 2 (CLAUDE.md): the regression model must never see mirrored dials."""
    fl = cfg.get("augment", {}).get("fliplr")
    if fl != 0.0:
        raise SystemExit(f"regress config must set augment.fliplr: 0.0 (got {fl!r})")


class CropDataset(Dataset):
    """Rows of regress_<split>.csv; optional deterministic jitter keyed by (seed, epoch, index)."""

    def __init__(self, crops_dir: Path, split: str, jitter: float = 0.0, seed: int = 0):
        self.root = Path(crops_dir)
        with open(self.root / f"regress_{split}.csv") as fh:
            self.rows = list(csv.DictReader(fh))
        self.jitter, self.seed, self.epoch = jitter, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        img = cv2.cvtColor(cv2.imread(str(self.root / row["crop"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if self.jitter > 0:
            rng = np.random.default_rng([self.seed, self.epoch, i])
            n = img.shape[0]
            s = rng.uniform(1 - self.jitter, 1 + self.jitter)
            t = rng.uniform(-self.jitter, self.jitter, 2) * n
            c = (n - 1) / 2
            A = np.array([[s, 0, c - s * c + t[0]], [0, s, c - s * c + t[1]]], np.float32)   # no reflection
            img = cv2.warpAffine(img, A, (n, n), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return to_tensor(img), torch.tensor(float(row["f"]), dtype=torch.float32)


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    err, n = 0.0, 0
    for x, y in loader:
        p = model(x.to(device)).float().cpu()
        err += (p - y).abs().sum().item()
        n += len(y)
    return 100.0 * err / max(n, 1)                  # MAE in % of full scale


def train(cfg: dict, run_id: str | None) -> Path:
    check_no_flip(cfg)
    seed_everything(cfg["seed"])
    run = make_run_dir(cfg, run_id)
    crops_dir = Path(cfg["data"]["crops_dir"])
    if cfg.get("is_smoke"):
        sd = cfg["smoke_data"]
        crops_dir = Path(make_smoke_data(run / "smoke_data", ("regress",), sd["n_train"], sd["n_val"])["crops_dir"])
    t, d = cfg["train"], cfg["data"]
    device = pick_device(t["device"])
    write_run_info(run, crops_dir=crops_dir, device=device, status="running")

    train_ds = CropDataset(crops_dir, d["train_split"], cfg["augment"]["jitter"], cfg["seed"])
    val_ds = CropDataset(crops_dir, d["val_split"])
    g = torch.Generator().manual_seed(cfg["seed"])
    train_dl = DataLoader(train_ds, batch_size=t["batch"], shuffle=True, num_workers=t["workers"],
                          generator=g, drop_last=len(train_ds) > t["batch"], persistent_workers=False)
    val_dl = DataLoader(val_ds, batch_size=t["batch"], shuffle=False, num_workers=t["workers"])

    model = RegressNet(pretrained=cfg["model"]["pretrained"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    steps = max(1, t["epochs"] * math.ceil(len(train_dl)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    loss_fn = torch.nn.SmoothL1Loss(beta=t["smooth_l1_beta"])

    (run / "weights").mkdir()
    best = math.inf
    with open(run / "metrics.csv", "w", newline="") as fh:
        log = csv.writer(fh)
        log.writerow(["epoch", "train_loss", "val_mae_fs", "lr"])
        for epoch in range(t["epochs"]):
            model.train()
            train_ds.set_epoch(epoch)
            total, n = 0.0, 0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                loss = loss_fn(model(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item() * len(y)
                n += len(y)
            mae = evaluate(model, val_dl, device)
            log.writerow([epoch, f"{total / max(n, 1):.6f}", f"{mae:.4f}", f"{sched.get_last_lr()[0]:.3e}"])
            fh.flush()
            print(f"epoch {epoch}: train_loss {total / max(n, 1):.5f}  val MAE {mae:.3f} %FS")
            save_checkpoint(run / "weights" / "last.pt", model, cfg, epoch=epoch, val_mae_fs=mae)
            if mae < best:
                best = mae
                save_checkpoint(run / "weights" / "best.pt", model, cfg, epoch=epoch, val_mae_fs=mae)
    write_run_info(run, weights=run / "weights" / "best.pt", best_val_mae_fs=best, status="done")
    print(f"run: {run}\nweights: {run / 'weights' / 'best.pt'}")
    return run


def main() -> None:
    a = parse_args(__doc__)
    train(resolve_config(a.config, a.smoke, a.set), a.run_id)


if __name__ == "__main__":
    main()
