#!/usr/bin/env python3
"""
scripts/train_and_evaluate_raw_leakage.py
Genuinely measures the effect of training-test data leakage by:
1. Training MobileNetV2 on the ORIGINAL uncurated training partition (5,614 images),
   which CONTAINS the twins of the 156 cross-partition duplicate test images.
2. Evaluating this raw-trained model on:
   - Leaked test subset (156 images, whose twins were seen in train)
   - Clean non-leaked test subset (1,598 images)
   - Full original test partition (1,754 images)
   - Sterile test partition (1,728 images)
3. Evaluating the sterile-trained MobileNetV2 (trained on 5,088 images where twins were purged)
   on the same test sets for rigorous, unconfounded side-by-side comparison.
4. Outputs: results/leakage_raw_comparison.json
"""

import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

CLASSES = ["cocci", "healthy", "ncd", "salmo"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

class SimplePoultryDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items  # list of (Path, label_idx)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, str(path)


def get_model(num_classes=4):
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labs, _ in loader:
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
        mask = (all_labels == i)
        total = int(mask.sum())
        correct = int((all_preds[mask] == i).sum()) if total > 0 else 0
        rec = round(100.0 * correct / total, 2) if total > 0 else 0.0
        per_class[c] = {"recall": rec, "correct": correct, "total": total}

    return {
        "acc": round(acc, 2),
        "correct": int((all_preds == all_labels).sum()),
        "total": len(all_labels),
        "per_class": per_class
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 1. Load duplicate pairs from pHash dedup audit
    with open(ROOT / "results" / "phash_dedup.json") as f:
        dedup = json.load(f)
    near_groups = dedup["near_groups"]

    def split_of(p):
        return p.split("/")[2]

    leaked_test_filenames = set()
    for a, b in near_groups:
        sa, sb = split_of(a), split_of(b)
        if (sa == "test" and sb == "train"):
            leaked_test_filenames.add(Path(a).name)
        elif (sb == "test" and sa == "train"):
            leaked_test_filenames.add(Path(b).name)

    print(f"Identified {len(leaked_test_filenames)} cross-partition duplicate test filenames.")

    # 2. Gather datasets
    orig_train_dir = ROOT / "data" / "images" / "train"
    orig_val_dir = ROOT / "data" / "images" / "val"
    orig_test_dir = ROOT / "data" / "images" / "test"
    sterile_test_dir = ROOT / "data" / "images_sterile" / "test"

    raw_train_items = []
    for c in CLASSES:
        for p in (orig_train_dir / c).rglob("*.jpg"):
            raw_train_items.append((p, CLASS_TO_IDX[c]))

    raw_val_items = []
    for c in CLASSES:
        for p in (orig_val_dir / c).rglob("*.jpg"):
            raw_val_items.append((p, CLASS_TO_IDX[c]))

    orig_test_items = []
    for c in CLASSES:
        for p in (orig_test_dir / c).rglob("*.jpg"):
            is_leaked = (p.name in leaked_test_filenames)
            orig_test_items.append((p, CLASS_TO_IDX[c], is_leaked))

    leaked_test_items = [(p, l) for p, l, is_l in orig_test_items if is_l]
    clean_test_items = [(p, l) for p, l, is_l in orig_test_items if not is_l]
    all_orig_test_items = [(p, l) for p, l, is_l in orig_test_items]

    sterile_test_items = []
    for c in CLASSES:
        for p in (sterile_test_dir / c).rglob("*.jpg"):
            sterile_test_items.append((p, CLASS_TO_IDX[c]))

    print(f"Dataset summary:")
    print(f"  - Raw Train: {len(raw_train_items)}")
    print(f"  - Raw Val: {len(raw_val_items)}")
    print(f"  - Orig Test: {len(all_orig_test_items)} (Leaked: {len(leaked_test_items)}, Clean: {len(clean_test_items)})")
    print(f"  - Sterile Test: {len(sterile_test_items)}")

    # Loaders
    raw_train_loader = DataLoader(SimplePoultryDataset(raw_train_items, train_tf), batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    raw_val_loader = DataLoader(SimplePoultryDataset(raw_val_items, val_tf), batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    leaked_test_loader = DataLoader(SimplePoultryDataset(leaked_test_items, val_tf), batch_size=32, shuffle=False)
    clean_test_loader = DataLoader(SimplePoultryDataset(clean_test_items, val_tf), batch_size=32, shuffle=False)
    orig_test_loader = DataLoader(SimplePoultryDataset(all_orig_test_items, val_tf), batch_size=32, shuffle=False)
    sterile_test_loader = DataLoader(SimplePoultryDataset(sterile_test_items, val_tf), batch_size=32, shuffle=False)

    # 3. Train MobileNetV2 on RAW uncurated training set
    raw_model_path = ROOT / "models" / "fp32" / "mobilenetv2_raw_uncleaned.pth"
    model_raw = get_model(num_classes=4).to(device)

    epochs = 30
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_raw.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model_raw.state_dict())

    print(f"\n--- Training MobileNetV2 on Raw Uncurated Training Set (5,614 images) for {epochs} epochs ---")
    start_time = time.time()
    for ep in range(1, epochs + 1):
        model_raw.train()
        running_loss = 0.0
        correct, total = 0, 0
        for imgs, labs, _ in raw_train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad()
            out = model_raw(imgs)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labs)
            preds = out.argmax(dim=1)
            correct += (preds == labs).sum().item()
            total += len(labs)
        scheduler.step()

        train_acc = 100.0 * correct / total
        train_loss = running_loss / total

        # Validation
        val_eval = evaluate(model_raw, raw_val_loader, device)
        val_acc = val_eval["acc"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model_raw.state_dict())

        if ep % 5 == 0 or ep == epochs:
            print(f"Epoch {ep:2d}/{epochs:2d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}% (Best={best_val_acc:.2f}%)")

    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.1f}s. Best Val Acc: {best_val_acc:.2f}%")

    model_raw.load_state_dict(best_weights)
    torch.save(model_raw.state_dict(), raw_model_path)
    print(f"Saved raw model to {raw_model_path}")

    # 4. Evaluate RAW trained model
    print("\n--- Evaluating Model Trained on RAW (Uncleaned) Train Set ---")
    raw_eval_leaked = evaluate(model_raw, leaked_test_loader, device)
    raw_eval_clean = evaluate(model_raw, clean_test_loader, device)
    raw_eval_orig = evaluate(model_raw, orig_test_loader, device)
    raw_eval_sterile = evaluate(model_raw, sterile_test_loader, device)

    print(f"RAW Model -> Leaked Test (n={raw_eval_leaked['total']}): {raw_eval_leaked['acc']}% ({raw_eval_leaked['correct']}/{raw_eval_leaked['total']}) [NCD: {raw_eval_leaked['per_class']['ncd']['recall']}%]")
    print(f"RAW Model -> Clean Test  (n={raw_eval_clean['total']}): {raw_eval_clean['acc']}% ({raw_eval_clean['correct']}/{raw_eval_clean['total']}) [NCD: {raw_eval_clean['per_class']['ncd']['recall']}%]")
    print(f"RAW Model -> Full Orig   (n={raw_eval_orig['total']}): {raw_eval_orig['acc']}% ({raw_eval_orig['correct']}/{raw_eval_orig['total']})")
    print(f"RAW Model -> Sterile Test (n={raw_eval_sterile['total']}): {raw_eval_sterile['acc']}%")

    raw_leakage_diff = round(raw_eval_leaked["acc"] - raw_eval_clean["acc"], 2)
    print(f"Empirical Leakage Bias on Raw Model: {raw_eval_leaked['acc']}% - {raw_eval_clean['acc']}% = {raw_leakage_diff:+0.2f} pp")

    # 5. Evaluate STERILE trained model (mobilenetv2_gate0.pth)
    sterile_model_path = ROOT / "models" / "fp32" / "mobilenetv2_gate0.pth"
    model_sterile = get_model(num_classes=4).to(device)
    model_sterile.load_state_dict(torch.load(sterile_model_path, map_location=device))

    print("\n--- Evaluating Model Trained on STERILE (Decontaminated) Train Set ---")
    sterile_eval_leaked = evaluate(model_sterile, leaked_test_loader, device)
    sterile_eval_clean = evaluate(model_sterile, clean_test_loader, device)
    sterile_eval_orig = evaluate(model_sterile, orig_test_loader, device)
    sterile_eval_sterile = evaluate(model_sterile, sterile_test_loader, device)

    print(f"STERILE Model -> Leaked Test (n={sterile_eval_leaked['total']}): {sterile_eval_leaked['acc']}% ({sterile_eval_leaked['correct']}/{sterile_eval_leaked['total']})")
    print(f"STERILE Model -> Clean Test  (n={sterile_eval_clean['total']}): {sterile_eval_clean['acc']}% ({sterile_eval_clean['correct']}/{sterile_eval_clean['total']})")
    print(f"STERILE Model -> Full Orig   (n={sterile_eval_orig['total']}): {sterile_eval_orig['acc']}%")
    print(f"STERILE Model -> Sterile Test (n={sterile_eval_sterile['total']}): {sterile_eval_sterile['acc']}%")

    results = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": epochs,
            "train_time_sec": round(train_time, 2),
            "sample_sizes": {
                "raw_train": len(raw_train_items),
                "leaked_test": len(leaked_test_items),
                "clean_test": len(clean_test_items),
                "orig_test": len(all_orig_test_items),
                "sterile_test": len(sterile_test_items)
            }
        },
        "raw_trained_model": {
            "leaked_test_subset": raw_eval_leaked,
            "clean_test_subset": raw_eval_clean,
            "full_orig_test": raw_eval_orig,
            "sterile_test": raw_eval_sterile,
            "leakage_bias_pp": raw_leakage_diff
        },
        "sterile_trained_model": {
            "leaked_test_subset": sterile_eval_leaked,
            "clean_test_subset": sterile_eval_clean,
            "full_orig_test": sterile_eval_orig,
            "sterile_test": sterile_eval_sterile,
            "difference_pp": round(sterile_eval_leaked["acc"] - sterile_eval_clean["acc"], 2)
        }
    }

    out_file = ROOT / "results" / "leakage_raw_comparison.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison to {out_file}")

if __name__ == "__main__":
    main()
