#!/usr/bin/env python3
"""Retrain EfficientNet-B0 on the rebuilt sterile split (5100/1325/1728).

Run LOCALLY on pc-putih (option 2):
  cd the project dir
  run: python -u train_eb0_box.py

Uses train_pipeline.py functions (same hyperparams as paper):
  Adam 1e-4, wd 1e-4, cosine, 50 epoch, batch 32, seed 42,
  ImageNet pretrain, augments from get_transforms().
Saves:
  models/fp32/efficientnet_b0_sterile5100.pth
  onnx/efficientnet_b0_fp32_sterile5100.onnx
  results/train_eb0_box.json (val best, test acc, per-class)
"""
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["POULTRY_DATA_DIR"] = str(ROOT / "data" / "images_sterile_paper")

sys.path.insert(0, str(ROOT))
assert (ROOT / "train_pipeline.py").exists(), ROOT
os.chdir(ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import train_pipeline as tp

tp.NUM_WORKERS = 0  # Windows: avoid dataloader spawn issues
tp.EPOCHS = 50

print(f"device: {tp.DEVICE}", flush=True)
print(f"data: {tp.DATA_DIR}", flush=True)

tp.prepare_data()

train_ds = tp.PoultryDataset(tp.DATA_DIR, "train", tp.get_transforms(train=True))
val_ds = tp.PoultryDataset(tp.DATA_DIR, "val", tp.get_transforms(train=False))
test_ds = tp.PoultryDataset(tp.DATA_DIR, "test", tp.get_transforms(train=False))
print(f"dataset: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)
assert len(train_ds) == 5100, len(train_ds)
assert len(val_ds) == 1325, len(val_ds)
assert len(test_ds) == 1728, len(test_ds)

train_loader = DataLoader(train_ds, batch_size=tp.BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=tp.BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=tp.BATCH_SIZE, shuffle=False,
                         num_workers=0, pin_memory=True)

model = tp.get_model("efficientnet_b0")
model, history, best_val = tp.train_model("efficientnet_b0_sterile5100",
                                          model, train_loader, val_loader)

crit = nn.CrossEntropyLoss()
test_loss, test_acc, preds, labels = tp.validate(model, test_loader, crit, tp.DEVICE)
print(f"TEST: loss={test_loss:.4f} acc={test_acc:.2f}%", flush=True)

onnx_path = tp.export_onnx(model, "efficientnet_b0_fp32_sterile5100")

out = {
    "train_n": len(train_ds), "val_n": len(val_ds), "test_n": len(test_ds),
    "best_val_acc": round(float(best_val), 2),
    "test_acc": round(float(test_acc), 2),
    "onnx": str(onnx_path),
    "onnx_mb": round(onnx_path.stat().st_size / 1024 / 1024, 2),
}
(ROOT / "results" / "train_eb0_box.json").write_text(json.dumps(out, indent=2))
print("SAVED results/train_eb0_box.json: " + json.dumps(out), flush=True)
