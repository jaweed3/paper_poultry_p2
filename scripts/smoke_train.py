#!/usr/bin/env python3
"""Smoke test: validate data + model pipeline on RTX 4060 before full train.

- Runs prepare_data() (carves val from publisher train, seed 42).
- Builds loaders, forward 1 batch + 1 optimizer step per model on CUDA.
- Prints params (sum p.numel), batch shapes, loss, timing.
- Exit 0 = pipeline sehat, boleh lanjut full train.

Run on LAB: uv run scripts/smoke_train.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import train_pipeline as P


def smoke_model(name, train_loader, device):
    t0 = time.perf_counter()
    model = P.get_model(name).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    model.train()
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    opt.zero_grad()
    out = model(images)
    assert out.shape == (images.shape[0], 4), out.shape
    loss = nn.CrossEntropyLoss()(out, labels)
    loss.backward()
    opt.step()
    dt = time.perf_counter() - t0
    print(f"  {name}: params={n_params:,} batch={tuple(images.shape)} "
          f"loss={loss.item():.4f} fwd+bwd={dt:.2f}s [OK]")
    del model
    torch.cuda.empty_cache()
    return n_params


def main():
    assert torch.cuda.is_available(), "CUDA required for smoke"
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}")

    P.prepare_data()
    train_ds = P.PoultryDataset(P.DATA_DIR, "train", P.get_transforms(train=True))
    val_ds = P.PoultryDataset(P.DATA_DIR, "val", P.get_transforms(train=False))
    test_ds = P.PoultryDataset(P.DATA_DIR, "test", P.get_transforms(train=False))
    print(f"counts: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    assert len(train_ds) == 5088, len(train_ds)
    assert len(val_ds) == 1337, len(val_ds)
    assert len(test_ds) == 1728, len(test_ds)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=True)
    for name in ["mobilenetv2", "shufflenetv2", "efficientnet_b0"]:
        smoke_model(name, train_loader, device)
    print("SMOKE OK — pipeline sehat, gas full train.")


if __name__ == "__main__":
    main()
