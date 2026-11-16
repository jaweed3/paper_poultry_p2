#!/usr/bin/env python3
"""
scripts/compute_mcnemar_and_bootstrap.py
Evaluates all baseline models on the sterile test set (n=1,728) to compute:
1. Exact paired predictions for McNemar tests (b, c, chi2, p-value with continuity correction).
2. Wilson 95% Confidence Intervals & 10,000-iteration Bootstrap CIs for overall accuracy and NCD recall.
3. Outputs: results/mcnemar_and_bootstrap_stats.json
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort
from tqdm import tqdm
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data/images_sterile/test"
CLASSES = ["cocci", "healthy", "ncd", "salmo"]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_test_data():
    samples = []
    for idx, c in enumerate(CLASSES):
        p_list = sorted(list((TEST_DIR / c).glob("*.*")))
        for p in p_list:
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                samples.append((str(p), idx))
    return samples

def evaluate_onnx(onnx_path, samples):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    preds = []
    labels = []
    for p, label in samples:
        img = Image.open(p).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = ((arr - MEAN) / STD).transpose(2, 0, 1)[None, ...].astype(np.float32)
        out = sess.run(None, {inp_name: arr})[0]
        pred = int(np.argmax(out[0]))
        preds.append(pred)
        labels.append(label)
    return np.array(preds), np.array(labels)

def compute_mcnemar(y_true, pred_a, pred_b):
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    
    n00 = int(np.sum((~correct_a) & (~correct_b)))
    n01 = int(np.sum((~correct_a) & correct_b))  # A wrong, B right (c)
    n10 = int(np.sum(correct_a & (~correct_b)))  # A right, B wrong (b)
    n11 = int(np.sum(correct_a & correct_b))
    
    b, c = n10, n01
    total_discordant = b + c
    if total_discordant == 0:
        chi2 = 0.0
        p_val = 1.0
    else:
        # Edwards continuity correction
        chi2 = ((abs(b - c) - 1.0) ** 2) / (b + c)
        p_val = stats.chi2.sf(chi2, df=1)
        
    return {
        "n00": n00,
        "n01_c": c,
        "n10_b": b,
        "n11": n11,
        "chi2": float(round(chi2, 4)),
        "p_value": float(p_val)
    }

def wilson_ci(k, n, z=1.95996):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return round(max(0.0, (centre - half) * 100), 2), round(min(1.0, (centre + half) * 100), 2)

def bootstrap_ci(correct_mask, B=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(correct_mask)
    # Vectorized bootstrap
    boot_means = rng.choice(correct_mask.astype(float), size=(B, n), replace=True).mean(axis=1) * 100
    low = np.percentile(boot_means, 2.5)
    high = np.percentile(boot_means, 97.5)
    return round(float(low), 2), round(float(high), 2)

def main():
    samples = load_test_data()
    print(f"Loaded {len(samples)} sterile test samples.")
    y_true = np.array([s[1] for s in samples])
    
    models = {
        "mobilenetv2": {
            "fp32": ROOT / "models/onnx/mobilenetv2_gate0_fp32.onnx",
            "dyn": ROOT / "models/ptq/mobilenetv2_gate0_int8.onnx",
            "static": ROOT / "models/ptq/mobilenetv2_gate0_static_per_channel_minmax.onnx"
        },
        "shufflenetv2": {
            "fp32": ROOT / "models/onnx/shufflenetv2_gate0_fp32.onnx",
            "dyn": ROOT / "models/ptq/shufflenetv2_gate0_int8.onnx",
            "static": ROOT / "models/ptq/shufflenetv2_gate0_static_per_channel_minmax.onnx"
        },
        "efficientnet_b0": {
            "fp32": ROOT / "models/onnx/efficientnet_b0_gate0_fp32.onnx",
            "dyn": ROOT / "models/ptq/efficientnet_b0_gate0_int8.onnx",
            "static": ROOT / "models/ptq/efficientnet_b0_gate0_static_per_channel_minmax.onnx"
        }
    }
    
    results = {}
    
    for family, paths in models.items():
        print(f"\n=== Evaluating {family} ===")
        preds = {}
        for variant, path in paths.items():
            print(f"Running inference: {variant} ({path.name})...")
            p, _ = evaluate_onnx(path, samples)
            preds[variant] = p
            
            correct = (p == y_true)
            acc = float(np.mean(correct) * 100)
            b_ci = bootstrap_ci(correct)
            w_ci = wilson_ci(int(np.sum(correct)), len(y_true))
            
            # Per class recall
            per_class = {}
            for c_idx, c_name in enumerate(CLASSES):
                mask = (y_true == c_idx)
                c_correct = (p[mask] == c_idx)
                c_k = int(np.sum(c_correct))
                c_n = int(np.sum(mask))
                c_rec = float(c_k / c_n * 100) if c_n > 0 else 0.0
                c_w_ci = wilson_ci(c_k, c_n)
                c_b_ci = bootstrap_ci(c_correct)
                per_class[c_name] = {
                    "correct": c_k,
                    "total": c_n,
                    "recall": round(c_rec, 2),
                    "wilson_ci": c_w_ci,
                    "bootstrap_ci": c_b_ci
                }
                
            results[f"{family}_{variant}"] = {
                "acc": round(acc, 2),
                "wilson_ci": w_ci,
                "bootstrap_ci": b_ci,
                "per_class": per_class
            }
            print(f"  {variant}: {acc:.2f}% (Wilson 95%: {w_ci}, Boot: {b_ci}) | NCD: {per_class['ncd']['recall']:.2f}% {per_class['ncd']['wilson_ci']}")
            
        # McNemar tests
        # 1. Static QDQ vs FP32
        mcn_static_vs_fp32 = compute_mcnemar(y_true, preds["fp32"], preds["static"])
        # 2. Dynamic INT8 vs FP32
        mcn_dyn_vs_fp32 = compute_mcnemar(y_true, preds["fp32"], preds["dyn"])
        # 3. Static QDQ vs Dynamic INT8
        mcn_static_vs_dyn = compute_mcnemar(y_true, preds["dyn"], preds["static"])
        
        results[f"{family}_mcnemar"] = {
            "static_vs_fp32": mcn_static_vs_fp32,
            "dyn_vs_fp32": mcn_dyn_vs_fp32,
            "static_vs_dyn": mcn_static_vs_dyn
        }
        print(f"  McNemar Static vs FP32: b={mcn_static_vs_fp32['n10_b']}, c={mcn_static_vs_fp32['n01_c']}, chi2={mcn_static_vs_fp32['chi2']}, p={mcn_static_vs_fp32['p_value']:.4e}")
        print(f"  McNemar Static vs Dyn:  b={mcn_static_vs_dyn['n10_b']}, c={mcn_static_vs_dyn['n01_c']}, chi2={mcn_static_vs_dyn['chi2']}, p={mcn_static_vs_dyn['p_value']:.4e}")

    out_file = ROOT / "results/mcnemar_and_bootstrap_stats.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {out_file}")

if __name__ == "__main__":
    main()
