#!/usr/bin/env python3
"""
scripts/train_pruning_controls.py
Systematic evaluation of pruning training controls:
Tests whether minority class (Newcastle Disease, 8.16% prevalence) recall collapse
is an artifact of training duration (12 epochs unweighted vs 30 epochs weighted+KD)
or a structural characteristic of unweighted L1 pruning.

Protocol:
- Base: MobileNetV2 FP32 baseline (models/fp32/mobilenetv2_gate0.pth)
- Architecture: Torch-Pruning DepGraph L1, sparsity = 0.30
- Dataset: Sterile benchmark (5,088 train, 1,337 val, 1,728 test)
- Training: Exactly 30 epochs, AdamW (lr=1e-4, weight_decay=1e-4), cosine annealing
- Experimental Grid:
  1. Unweighted CE, No KD (30 epochs) — Seeds: [42, 123]
  2. Class-Weighted CE, No KD (30 epochs) — Seeds: [42, 123]
  3. Class-Weighted CE + KD (T=3.0, alpha=0.5, 30 epochs) — Seeds: [42, 123]
- In-memory dataset caching for accelerated GPU throughput.
- Output: results/pruning_controls_comparison.json
"""

import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import torch_pruning as tp

CLASSES = ["cocci", "healthy", "ncd", "salmo"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class InMemDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list  # [(PIL.Image, int_label)]
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img, label = self.data_list[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def load_in_mem(data_dir, split):
    split_dir = Path(data_dir) / split
    items = []
    print(f"Loading {split} images into RAM from {split_dir}...")
    for c in CLASSES:
        cls_dir = split_dir / c
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.glob("*.jpg")):
            with Image.open(p) as img:
                # Resize to 256 to speed up transform while preserving full quality
                img_resized = img.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
                items.append((img_resized, CLASS_TO_IDX[c]))
    print(f"  Loaded {len(items)} {split} images into memory.")
    return items


def get_base_model(num_classes=4):
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m


def prune_mobilenetv2_s30(model, device):
    example_input = torch.randn(1, 3, 224, 224).to(device)
    imp = tp.importance.MagnitudeImportance(p=1)
    ignored_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_input,
        importance=imp,
        pruning_ratio=0.30,
        ignored_layers=ignored_layers,
    )
    pruner.step()
    return model


def wilson_score_interval(successes, total, confidence=0.95):
    if total == 0:
        return 0.0, 0.0
    z = 1.959964  # 95%
    p = successes / total
    denominator = 1 + z**2 / total
    centre_adjusted_probability = (p + z**2 / (2 * total)) / denominator
    adjusted_std_error = (z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)) / denominator
    lower = max(0.0, centre_adjusted_probability - adjusted_std_error)
    upper = min(1.0, centre_adjusted_probability + adjusted_std_error)
    return round(lower * 100, 2), round(upper * 100, 2)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labs.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = 100.0 * float((all_preds == all_labels).sum()) / len(all_labels)
    per_class = {}
    for i, c in enumerate(CLASSES):
        m = (all_labels == i)
        total = int(m.sum())
        correct = int((all_preds[m] == i).sum()) if total > 0 else 0
        rec = round(100.0 * correct / total, 2) if total > 0 else 0.0
        ci_low, ci_high = wilson_score_interval(correct, total)
        per_class[c] = {
            "recall": rec,
            "correct": correct,
            "total": total,
            "ci_95": [ci_low, ci_high]
        }

    return {
        "acc": round(acc, 2),
        "correct": int((all_preds == all_labels).sum()),
        "total": len(all_labels),
        "per_class": per_class
    }


def train_one_configuration(name, seed, use_weights, use_kd, train_items, val_items, test_items, device):
    set_seed(seed)
    print(f"\n=======================================================", flush=True)
    print(f"Condition: {name} | Seed: {seed} | Weights: {use_weights} | KD: {use_kd}", flush=True)
    print(f"=======================================================", flush=True)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(InMemDataset(train_items, train_tf), batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(InMemDataset(val_items, val_tf), batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(InMemDataset(test_items, val_tf), batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    # Load baseline teacher checkpoint
    teacher_ckpt = ROOT / "models" / "fp32" / "mobilenetv2_gate0.pth"
    assert teacher_ckpt.exists(), f"Missing teacher checkpoint {teacher_ckpt}"

    teacher = get_base_model().to(device)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Initialize student from teacher weights, then prune
    student = get_base_model().to(device)
    student.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    student = prune_mobilenetv2_s30(student, device)

    # Class weights: total / (4 * class_count)
    if use_weights:
        class_counts = torch.tensor([1555.0, 1537.0, 417.0, 1579.0], device=device)
        total_s = class_counts.sum()
        cw = total_s / (4.0 * class_counts)
        ce_loss_fn = nn.CrossEntropyLoss(weight=cw)
    else:
        ce_loss_fn = nn.CrossEntropyLoss()

    epochs = 30
    lr = 1e-4
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    temp = 3.0
    alpha = 0.5

    best_val_acc = 0.0
    best_weights = copy.deepcopy(student.state_dict())

    t0 = time.time()
    for ep in range(1, epochs + 1):
        student.train()
        train_loss, correct, total = 0.0, 0, 0
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad()

            logits = student(imgs)
            ce = ce_loss_fn(logits, labs)

            if use_kd:
                with torch.no_grad():
                    t_logits = teacher(imgs)
                kd = F.kl_div(
                    F.log_softmax(logits / temp, dim=1),
                    F.softmax(t_logits / temp, dim=1),
                    reduction="batchmean"
                ) * (temp ** 2)
                loss = (1.0 - alpha) * ce + alpha * kd
            else:
                loss = ce

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labs)
            preds = logits.argmax(dim=1)
            correct += (preds == labs).sum().item()
            total += len(labs)

        scheduler.step()
        train_acc = 100.0 * correct / total

        # Validation
        val_eval = evaluate(student, val_loader, device)
        val_acc = val_eval["acc"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(student.state_dict())

        if ep % 5 == 0 or ep == epochs:
            print(f"  Epoch {ep:2d}/{epochs}: Train Loss={train_loss/total:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}% (Best={best_val_acc:.2f}%)", flush=True)

    train_time = round(time.time() - t0, 1)
    print(f"  Training finished in {train_time}s. Best Val Acc: {best_val_acc:.2f}%", flush=True)

    # Evaluate best checkpoint on test set
    student.load_state_dict(best_weights)
    test_eval = evaluate(student, test_loader, device)
    print(f"  Test Acc: {test_eval['acc']}% | NCD Recall: {test_eval['per_class']['ncd']['recall']}% ({test_eval['per_class']['ncd']['correct']}/{test_eval['per_class']['ncd']['total']}) [CI: {test_eval['per_class']['ncd']['ci_95']}]", flush=True)

    return {
        "condition": name,
        "seed": seed,
        "epochs": epochs,
        "use_weights": use_weights,
        "use_kd": use_kd,
        "train_time_s": train_time,
        "best_val_acc": best_val_acc,
        "test_eval": test_eval
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = ROOT / "data" / "images_sterile"
    assert data_dir.exists(), f"Sterile dataset not found at {data_dir}"

    train_items = load_in_mem(data_dir, "train")
    val_items = load_in_mem(data_dir, "val")
    test_items = load_in_mem(data_dir, "test")

    conditions = [
        ("unweighted_ce_no_kd", False, False),
        ("weighted_ce_no_kd", True, False),
        ("weighted_ce_with_kd", True, True)
    ]
    seeds = [42, 123]

    all_results = []
    summary_by_condition = {}

    out_file = ROOT / "results" / "pruning_controls_comparison.json"
    out_file.parent.mkdir(exist_ok=True)

    # Load existing runs if resuming
    existing_runs = {}
    if out_file.exists():
        try:
            with open(out_file) as f:
                prev_data = json.load(f)
                for r in prev_data.get("all_runs", []):
                    key = (r["condition"], r["seed"])
                    existing_runs[key] = r
            print(f"Resuming: found {len(existing_runs)} existing runs in {out_file.name}")
        except Exception as e:
            print(f"Could not load previous runs: {e}")

    for name, use_w, use_kd in conditions:
        runs = []
        for s in seeds:
            key = (name, s)
            if key in existing_runs:
                print(f"\nSkipping already completed run: {name} (seed {s})")
                res = existing_runs[key]
            else:
                res = train_one_configuration(name, s, use_w, use_kd, train_items, val_items, test_items, device)
                existing_runs[key] = res
            runs.append(res)
            if res not in all_results:
                all_results.append(res)

            # Save incrementally after each run
            accs = [r["test_eval"]["acc"] for r in runs]
            ncd_recs = [r["test_eval"]["per_class"]["ncd"]["recall"] for r in runs]
            summary_by_condition[name] = {
                "mean_acc": round(float(np.mean(accs)), 2),
                "std_acc": round(float(np.std(accs)), 2),
                "mean_ncd_recall": round(float(np.mean(ncd_recs)), 2),
                "std_ncd_recall": round(float(np.std(ncd_recs)), 2),
                "runs": runs
            }
            output_data = {
                "metadata": {
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "epochs": 30,
                    "architecture": "MobileNetV2 (s=0.30)",
                    "benchmark": "Sterile test split (n=1,728)",
                    "classes": CLASSES
                },
                "summary": summary_by_condition,
                "all_runs": list(existing_runs.values())
            }
            with open(out_file, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"  [Checkpoint saved to {out_file.name}]")

    print(f"\n=======================================================")
    print(f"PRUNING CONTROLS EXPERIMENT SUMMARY (30 EPOCHS):")
    print(f"=======================================================")
    for k, v in summary_by_condition.items():
        print(f"  {k:22s} -> Top-1: {v['mean_acc']} ± {v['std_acc']}% | NCD Recall: {v['mean_ncd_recall']} ± {v['std_ncd_recall']}%")
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    main()
