#!/usr/bin/env python3
"""
scripts/measure_leakage_effect.py
Directly measures the effect of training-test data leakage on model performance.

Evaluates trained models on:
1. Original benchmark test set (1,754 images):
   - Subset of 156 leaked images (near-duplicates of training frames).
   - Subset of 1,598 non-leaked images.
   - Overall original test set.
2. Sanitized sterile test set (1,728 images).

Outputs: results/leakage_effect_size.json
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import train_pipeline as P

CLASSES = ["cocci", "healthy", "ncd", "salmo"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

class ImagePathDataset(Dataset):
    def __init__(self, items, transform=None):
        # items: list of (Path, label_idx)
        self.items = items
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


def evaluate_torch(model, loader, device):
    model.eval()
    all_preds, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for imgs, labs, paths in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labs.numpy())
            all_paths.extend(paths)
    return np.array(all_preds), np.array(all_labels), all_paths


def evaluate_onnx(onnx_path, loader):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    all_preds, all_labels, all_paths = [], [], []
    for imgs, labs, paths in loader:
        x = imgs.numpy()
        outputs = sess.run(None, {inp_name: x})[0]
        preds = outputs.argmax(axis=1)
        all_preds.extend(preds)
        all_labels.extend(labs.numpy())
        all_paths.extend(paths)
    return np.array(all_preds), np.array(all_labels), all_paths


def compute_metrics(preds, labels):
    if len(labels) == 0:
        return {"acc": 0.0, "count": 0, "per_class": {}}
    acc = 100.0 * float((preds == labels).sum()) / len(labels)
    per_class = {}
    for i, c in enumerate(CLASSES):
        mask = labels == i
        if mask.sum() > 0:
            rec = 100.0 * float((preds[mask] == i).sum()) / mask.sum()
            per_class[c] = {
                "recall": round(rec, 2),
                "correct": int((preds[mask] == i).sum()),
                "total": int(mask.sum())
            }
        else:
            per_class[c] = {"recall": 0.0, "correct": 0, "total": 0}
    return {
        "acc": round(acc, 2),
        "correct": int((preds == labels).sum()),
        "total": len(labels),
        "per_class": per_class
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 1. Load duplicate pairs
    with open(ROOT / "results" / "phash_dedup.json") as f:
        dedup = json.load(f)
    near_groups = dedup["near_groups"]

    def split_of(p):
        return p.split("/")[2]

    # Find test paths that leak into train
    leaked_test_paths = set()
    for a, b in near_groups:
        sa, sb = split_of(a), split_of(b)
        if (sa == "test" and sb == "train"):
            leaked_test_paths.add(a.split("/", 2)[-1]) # e.g. test/cocci/file.jpg
        elif (sb == "test" and sa == "train"):
            leaked_test_paths.add(b.split("/", 2)[-1])

    print(f"Identified {len(leaked_test_paths)} test images that duplicate training images.")

    # 2. Gather original test images
    orig_test_dir = ROOT / "data" / "images" / "test"
    orig_items = []
    for c in CLASSES:
        for p in (orig_test_dir / c).rglob("*.jpg"):
            rel = f"test/{c}/{p.name}"
            orig_items.append((p, CLASS_TO_IDX[c], rel in leaked_test_paths))

    print(f"Original test set: {len(orig_items)} images")
    leaked_items = [(p, l) for p, l, is_l in orig_items if is_l]
    clean_orig_items = [(p, l) for p, l, is_l in orig_items if not is_l]
    all_orig_items = [(p, l) for p, l, is_l in orig_items]
    print(f"  - Leaked subset: {len(leaked_items)}")
    print(f"  - Non-leaked subset: {len(clean_orig_items)}")

    # 3. Gather sterile test images
    sterile_test_dir = ROOT / "data" / "images_sterile" / "test"
    sterile_items = []
    for c in CLASSES:
        for p in (sterile_test_dir / c).rglob("*.jpg"):
            sterile_items.append((p, CLASS_TO_IDX[c]))
    print(f"Sterile test set: {len(sterile_items)} images")

    # Loaders
    leaked_loader = DataLoader(ImagePathDataset(leaked_items, val_tf), batch_size=32, shuffle=False)
    clean_orig_loader = DataLoader(ImagePathDataset(clean_orig_items, val_tf), batch_size=32, shuffle=False)
    all_orig_loader = DataLoader(ImagePathDataset(all_orig_items, val_tf), batch_size=32, shuffle=False)
    sterile_loader = DataLoader(ImagePathDataset(sterile_items, val_tf), batch_size=32, shuffle=False)

    models_to_test = [
        ("mobilenetv2", "torch", ROOT / "models" / "fp32" / "mobilenetv2_gate0.pth"),
        ("shufflenetv2", "torch", ROOT / "models" / "fp32" / "shufflenetv2_gate0.pth"),
        ("efficientnet_b0", "torch", ROOT / "models" / "fp32" / "efficientnet_b0_gate0.pth"),
        ("efficientnet_b0_static_qdq", "onnx", ROOT / "models" / "ptq" / "efficientnet_b0_gate0_static_per_channel_minmax.onnx")
    ]

    out_data = {
        "metadata": {
            "num_leaked_test": len(leaked_items),
            "num_clean_orig_test": len(clean_orig_items),
            "num_all_orig_test": len(all_orig_items),
            "num_sterile_test": len(sterile_items)
        },
        "evaluations": {}
    }

    for name, mtype, path in models_to_test:
        print(f"\nEvaluating {name} ({mtype})...")
        if mtype == "torch":
            model = P.get_model(name)
            model.load_state_dict(torch.load(path, map_location=device))
            model = model.to(device)

            p_leak, l_leak, _ = evaluate_torch(model, leaked_loader, device)
            p_clean_orig, l_clean_orig, _ = evaluate_torch(model, clean_orig_loader, device)
            p_all_orig, l_all_orig, _ = evaluate_torch(model, all_orig_loader, device)
            p_sterile, l_sterile, _ = evaluate_torch(model, sterile_loader, device)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            p_leak, l_leak, _ = evaluate_onnx(path, leaked_loader)
            p_clean_orig, l_clean_orig, _ = evaluate_onnx(path, clean_orig_loader)
            p_all_orig, l_all_orig, _ = evaluate_onnx(path, all_orig_loader)
            p_sterile, l_sterile, _ = evaluate_onnx(path, sterile_loader)

        m_leak = compute_metrics(p_leak, l_leak)
        m_clean_orig = compute_metrics(p_clean_orig, l_clean_orig)
        m_all_orig = compute_metrics(p_all_orig, l_all_orig)
        m_sterile = compute_metrics(p_sterile, l_sterile)

        out_data["evaluations"][name] = {
            "leaked_subset": m_leak,
            "non_leaked_subset": m_clean_orig,
            "original_full_test": m_all_orig,
            "sterile_test": m_sterile,
            "leakage_bias_pp": round(m_leak["acc"] - m_clean_orig["acc"], 2),
            "ncd_leakage_bias_pp": round(m_leak["per_class"]["ncd"]["recall"] - m_clean_orig["per_class"]["ncd"]["recall"], 2)
        }

        print(f"  Accuracy on Leaked subset ({m_leak['total']}): {m_leak['acc']}% (NCD recall: {m_leak['per_class']['ncd']['recall']}%)")
        print(f"  Accuracy on Non-leaked subset ({m_clean_orig['total']}): {m_clean_orig['acc']}% (NCD recall: {m_clean_orig['per_class']['ncd']['recall']}%)")
        print(f"  Accuracy on Full Original Test ({m_all_orig['total']}): {m_all_orig['acc']}%")
        print(f"  Accuracy on Sterile Test ({m_sterile['total']}): {m_sterile['acc']}% (NCD recall: {m_sterile['per_class']['ncd']['recall']}%)")
        print(f"  Leakage Bias: +{out_data['evaluations'][name]['leakage_bias_pp']} pp overall, +{out_data['evaluations'][name]['ncd_leakage_bias_pp']} pp NCD recall!")

    out_file = ROOT / "results" / "leakage_effect_size.json"
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved leakage effect size results to {out_file}")

if __name__ == "__main__":
    main()
