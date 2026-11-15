#!/usr/bin/env python3
"""
scripts/train_pruned_weighted_distill.py
Tests the hypothesis: Does class-weighted loss + knowledge distillation
recover the minority class (NCD, 8.2%) recall after structured pruning?

Protocol:
- Base: MobileNetV2 FP32 baseline (97.92% top-1, 94.33% NCD recall)
- Pruning: Torch-Pruning DepGraph L1, sparsity = 0.30
- Loss: Class-Weighted Cross-Entropy + FP32 Teacher Distillation (T=3.0, alpha=0.5)
- Training: 30 epochs, AdamW (lr=1e-4, weight_decay=1e-4), cosine annealing
- Evaluation: Sterile test set (n=1,728)

Outputs:
- models/pruned/mobilenetv2_s30_weighted_distill.pth
- results/finetune_mobilenetv2_s30_weighted_distill.json
"""

import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["POULTRY_DATA_DIR"] = "data/images_sterile"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import train_pipeline as P
import torch_pruning as tp

CLASSES = ["cocci", "healthy", "ncd", "salmo"]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Prepare Data
    P.prepare_data()
    train_tf = P.get_transforms(train=True)
    val_tf = P.get_transforms(train=False)

    train_ds = P.PoultryDataset(P.DATA_DIR, "train", train_tf)
    val_ds = P.PoultryDataset(P.DATA_DIR, "val", val_tf)
    test_ds = P.PoultryDataset(P.DATA_DIR, "test", val_tf)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    assert len(test_ds) == 1728

    # 2. Compute Class Weights based on sterile train distribution
    # train distribution: cocci: 1555, healthy: 1537, ncd: 417, salmo: 1579
    class_counts = torch.tensor([1555.0, 1537.0, 417.0, 1579.0], device=device)
    total_samples = class_counts.sum()
    class_weights = total_samples / (4.0 * class_counts)
    print(f"Class counts: {class_counts.tolist()}")
    print(f"Computed inverse-frequency class weights: {class_weights.tolist()}")
    print(f"NCD weight boost ratio: {class_weights[2] / class_weights[0]:.2f}x")

    # 3. Load Teacher (FP32 baseline)
    teacher = P.get_model("mobilenetv2")
    teacher_ckpt = ROOT / "models" / "fp32" / "mobilenetv2_gate0.pth"
    assert teacher_ckpt.exists(), f"Missing teacher checkpoint {teacher_ckpt}"
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # 4. Create Student and Prune to Sparsity 0.30
    student = P.get_model("mobilenetv2")
    student.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    student = student.to(device)

    # Apply DepGraph L1 Pruning
    example_input = torch.randn(1, 3, 224, 224).to(device)
    imp = tp.importance.MagnitudeImportance(p=1)
    ignored_layers = []
    for m in student.modules():
        if isinstance(m, nn.Linear):
            ignored_layers.append(m)

    pruner = tp.pruner.MagnitudePruner(
        student,
        example_inputs=example_input,
        importance=imp,
        pruning_ratio=0.30,
        ignored_layers=ignored_layers,
    )
    pruner.step()
    orig_params = sum(p.numel() for p in teacher.parameters())
    pruned_params = sum(p.numel() for p in student.parameters())
    print(f"Pruned MobileNetV2 (s=0.30): {orig_params:,} -> {pruned_params:,} params ({100.0 * (1 - pruned_params/orig_params):.1f}% reduction)")

    # 5. Training Setup
    epochs = 30
    lr = 1e-4
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    temp = 3.0
    alpha = 0.5

    best_val_acc = 0.0
    best_weights = None
    history = []

    print("\nStarting 30-epoch fine-tuning with Class-Weighted Cross-Entropy + KD...")
    t0 = time.time()
    for epoch in range(epochs):
        student.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            student_logits = student(imgs)
            ce_loss = ce_loss_fn(student_logits, labels)

            with torch.no_grad():
                teacher_logits = teacher(imgs)

            kd_loss = F.kl_div(
                F.log_softmax(student_logits / temp, dim=1),
                F.softmax(teacher_logits / temp, dim=1),
                reduction="batchmean"
            ) * (temp ** 2)

            loss = (1.0 - alpha) * ce_loss + alpha * kd_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = student_logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += imgs.size(0)

        scheduler.step()
        train_acc = 100.0 * train_correct / train_total

        # Validation
        student.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = student(imgs).argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
        val_acc = 100.0 * val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in student.state_dict().items()}

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss / train_total, 4),
            "train_acc": round(train_acc, 2),
            "val_acc": round(val_acc, 2),
            "lr": optimizer.param_groups[0]["lr"]
        })
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:02d}/{epochs}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%, Best Val = {best_val_acc:.2f}%")

    total_time = time.time() - t0
    print(f"\nTraining completed in {total_time:.1f}s")

    # 6. Final Evaluation on Sterile Test Split (n=1,728)
    student.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    student.eval()

    test_preds, test_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = student(imgs).argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(labels.numpy())

    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)

    overall_acc = 100.0 * float((test_preds == test_labels).sum()) / len(test_labels)
    per_class = {}
    for i, c in enumerate(CLASSES):
        m = test_labels == i
        rec = 100.0 * float((test_preds[m] == i).sum()) / max(int(m.sum()), 1)
        per_class[c] = {
            "recall": round(rec, 2),
            "correct": int((test_preds[m] == i).sum()),
            "total": int(m.sum())
        }

    print("\n=======================================================")
    print(f"PRUNED MOBILENETV2 (s=0.30) + CLASS-WEIGHTED + KD RESULTS:")
    print(f"Sterile Test Accuracy: {overall_acc:.2f}%")
    print(f"Per-Class Recall:")
    for c in CLASSES:
        print(f"  - {c.upper()}: {per_class[c]['recall']}% ({per_class[c]['correct']}/{per_class[c]['total']})")
    print("=======================================================")

    # Save checkpoint & results
    out_dir = ROOT / "models" / "pruned"
    out_dir.mkdir(exist_ok=True, parents=True)
    ckpt_path = out_dir / "mobilenetv2_s30_weighted_distill.pth"
    torch.save(best_weights, ckpt_path)

    results_out = {
        "model": "mobilenetv2",
        "sparsity": 0.30,
        "epochs": epochs,
        "training_time_s": round(total_time, 1),
        "class_weights": class_weights.cpu().tolist(),
        "temperature": temp,
        "alpha": alpha,
        "params": pruned_params,
        "param_reduction_pct": round(100.0 * (1 - pruned_params / orig_params), 2),
        "test_acc": round(overall_acc, 2),
        "per_class": per_class,
        "ncd_recall": per_class["ncd"]["recall"],
        "history": history
    }

    results_file = ROOT / "results" / "finetune_mobilenetv2_s30_weighted_distill.json"
    with open(results_file, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"Saved results to {results_file}")

if __name__ == "__main__":
    main()
