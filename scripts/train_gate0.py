#!/usr/bin/env python3
"""Full train 50ep per model on RTX 4060 (train-only, no Grad-CAM/ONNX/bench).

Saves: models/fp32/<name>_gate0.pth + logs/<name>_gate0_history.json
Eval/ONNX/INT8 terpisah besok pagi setelah train kelar.

Usage (LAB, dalam tmux):
  uv run scripts/train_gate0.py --models mobilenetv2 shufflenetv2 efficientnet_b0
  uv run scripts/train_gate0.py --models mobilenetv2   # satu aja

Resumable per model: kalau ckpt sudah ada, skip (pakai --force untuk ulang).
"""
import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import train_pipeline as P

CKPT_DIR = ROOT / "models" / "fp32"
LOG_DIR = ROOT / "logs"


def train_full(name, train_loader, val_loader, device, epochs=50):
    ckpt = CKPT_DIR / f"{name}_gate0.pth"
    hist_path = LOG_DIR / f"{name}_gate0_history.json"
    if ckpt.exists() and not ARGS.force:
        print(f"  {name}: ckpt exists, skip (use --force to retrain)")
        return None
    model = P.get_model(name).to(device)
    print(f"  {name}: params={sum(p.numel() for p in model.parameters()):,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc, best_wts = 0.0, copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "train_acc": [], "val_loss": [],
               "val_acc": [], "lr": [], "epoch_time_s": []}
    t_start = time.perf_counter()
    for epoch in range(epochs):
        t0 = time.perf_counter()
        tr_loss, tr_acc = P.train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, _, _ = P.validate(model, val_loader, criterion, device)
        scheduler.step()
        dt = time.perf_counter() - t0
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(scheduler.get_last_lr()[0])
        history["epoch_time_s"].append(round(dt, 1))
        print(f"  [{name} {epoch+1}/{epochs}] train {tr_loss:.4f}/{tr_acc:.1f}% | "
              f"val {va_loss:.4f}/{va_acc:.1f}% | {dt:.0f}s", flush=True)
        if va_acc > best_acc:
            best_acc = va_acc
            best_wts = copy.deepcopy(model.state_dict())
    total = time.perf_counter() - t_start
    model.load_state_dict(best_wts)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(hist_path, "w") as f:
        json.dump({"model": name, "seed": 42, "best_val_acc": best_acc,
                   "total_time_s": round(total, 1),
                   "finished": datetime.now().isoformat(), **history}, f, indent=2)
    print(f"  {name}: DONE best_val={best_acc:.2f}% total={total/60:.1f}min -> {ckpt}")
    del model
    torch.cuda.empty_cache()
    return best_acc


def main():
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)} | models: {ARGS.models}")
    P.prepare_data()
    train_loader = DataLoader(
        P.PoultryDataset(P.DATA_DIR, "train", P.get_transforms(train=True)),
        batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        P.PoultryDataset(P.DATA_DIR, "val", P.get_transforms(train=False)),
        batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)}")
    for name in ARGS.models:
        train_full(name, train_loader, val_loader, device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["mobilenetv2", "shufflenetv2", "efficientnet_b0"])
    ap.add_argument("--force", action="store_true")
    ARGS = ap.parse_args()
    main()
