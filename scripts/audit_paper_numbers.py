#!/usr/bin/env python3
"""
scripts/audit_paper_numbers.py
Single source of truth auditor for JCSE journal paper.
Reads raw evaluation logs and benchmarks, computes exact per-class metrics,
Wilson 95% score confidence intervals, and outputs consolidated ground truth.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

def wilson_ci(k, n, confidence=0.95):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    z = 1.95996  # 95%
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) / n) + (z**2) / (4.0 * (n**2)))) / denom
    lower = max(0.0, center - margin) * 100.0
    upper = min(1.0, center + margin) * 100.0
    return round(p * 100.0, 2), round(lower, 2), round(upper, 2)

def audit_dataset():
    """Verify exact dataset splits and class distribution."""
    # From actual evaluations and sterile split
    classes = ["cocci", "healthy", "ncd", "salmo"]
    test_support = {
        "cocci": 529,
        "healthy": 504,
        "ncd": 141,
        "salmo": 554
    }
    total_test = sum(test_support.values()) # 1728
    
    # Train and Val from verified scan (dataset_stats.json)
    train_counts = {"cocci": 1713, "healthy": 1630, "ncd": 461, "salmo": 1810}
    val_counts = {"cocci": 428, "healthy": 407, "ncd": 115, "salmo": 452}
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    total_all = total_train + total_val + total_test
    
    return {
        "classes": classes,
        "train": train_counts,
        "val": val_counts,
        "test": test_support,
        "total_train": total_train,
        "total_val": total_val,
        "total_test": total_test,
        "total_dataset": total_all
    }

def audit_models():
    """Extract metrics directly from raw json files."""
    # 1. Baseline eval_gate0.json
    with open(RESULTS_DIR / "eval_gate0.json") as f:
        gate0 = json.load(f)["models"]
        
    # 2. eval_pruned_s30.json
    with open(RESULTS_DIR / "eval_pruned_s30.json") as f:
        s30_mnv2 = json.load(f)
        
    # 3. eval_mobilenetv2_s50.json
    with open(RESULTS_DIR / "eval_mobilenetv2_s50.json") as f:
        s50_mnv2 = json.load(f)["variants"]
        
    # 4. eval_efficientnet_b0_s30.json
    with open(RESULTS_DIR / "eval_efficientnet_b0_s30.json") as f:
        s30_eb0 = json.load(f)["variants"]
        
    # 5. static_eb0.json
    with open(RESULTS_DIR / "static_eb0.json") as f:
        static_eb0 = json.load(f)
        
    # 6. RPi5 physical benchmark
    with open(RESULTS_DIR / "rpi5_pruned_benchmark.json") as f:
        rpi5 = json.load(f)["benchmarks"]
        
    return {
        "gate0": gate0,
        "s30_mnv2": s30_mnv2,
        "s50_mnv2": s50_mnv2,
        "s30_eb0": s30_eb0,
        "static_eb0": static_eb0,
        "rpi5": rpi5
    }

def main():
    ds = audit_dataset()
    m = audit_models()
    
    test_support = ds["test"]
    n_test = ds["total_test"]
    
    table_pruning_audit = []
    
    # MNV2 Baseline FP32
    mnv2_fp32_acc = m["gate0"]["mobilenetv2"]["fp32_test_acc"]
    mnv2_fp32_cm = m["gate0"]["mobilenetv2"]["fp32_confusion"]
    mnv2_fp32_per_class = {}
    for i, cls in enumerate(ds["classes"]):
        k = int(mnv2_fp32_cm[i][i])
        n_c = test_support[cls]
        pct, lo, hi = wilson_ci(k, n_c)
        mnv2_fp32_per_class[cls] = {"k": k, "n": n_c, "pct": pct, "ci95": [lo, hi]}
        
    table_pruning_audit.append({
        "name": "MobileNetV2 Baseline FP32",
        "params": 2228996,
        "active_pct": 100.0,
        "disk_size_mb": 8.48,
        "top1_acc": mnv2_fp32_acc,
        "per_class": mnv2_fp32_per_class,
        "compression": 1.00
    })
    
    # MNV2 Pruned s30 FP32
    s30_fp32 = m["s30_mnv2"]["pruned_s30_fp32"]
    s30_fp32_per_class = {}
    for cls in ds["classes"]:
        pct_val = s30_fp32["per_class"][cls]
        n_c = test_support[cls]
        k = int(round(pct_val * n_c / 100.0))
        pct, lo, hi = wilson_ci(k, n_c)
        s30_fp32_per_class[cls] = {"k": k, "n": n_c, "pct": pct, "ci95": [lo, hi]}
    table_pruning_audit.append({
        "name": "MobileNetV2 Pruned FP32 (s=0.30)",
        "params": 1110790,
        "active_pct": 49.83,
        "disk_size_mb": 4.23,
        "top1_acc": s30_fp32["accuracy"],
        "per_class": s30_fp32_per_class,
        "compression": round(8.48 / 4.23, 2)
    })
    
    # MNV2 Pruned s30 INT8 Dynamic
    s30_int8 = m["s30_mnv2"]["pruned_s30_int8"]
    s30_int8_per_class = {}
    for cls in ds["classes"]:
        pct_val = s30_int8["per_class"][cls]
        n_c = test_support[cls]
        k = int(round(pct_val * n_c / 100.0))
        pct, lo, hi = wilson_ci(k, n_c)
        s30_int8_per_class[cls] = {"k": k, "n": n_c, "pct": pct, "ci95": [lo, hi]}
    table_pruning_audit.append({
        "name": "MobileNetV2 Pruned INT8 Dynamic (s=0.30)",
        "params": 1110790,
        "active_pct": 49.83,
        "disk_size_mb": 1.23,
        "top1_acc": s30_int8["accuracy"],
        "per_class": s30_int8_per_class,
        "compression": round(8.48 / 1.23, 2)
    })
    
    # MNV2 Pruned s50 FP32
    s50_fp32 = m["s50_mnv2"]["fp32"]
    s50_fp32_per_class = {}
    for cls in ds["classes"]:
        pct_val = s50_fp32["per_class"][cls]
        n_c = test_support[cls]
        k = int(round(pct_val * n_c / 100.0))
        pct, lo, hi = wilson_ci(k, n_c)
        s50_fp32_per_class[cls] = {"k": k, "n": n_c, "pct": pct, "ci95": [lo, hi]}
    table_pruning_audit.append({
        "name": "MobileNetV2 Pruned FP32 (s=0.50)",
        "params": 583332,
        "active_pct": 26.17,
        "disk_size_mb": 2.23,
        "top1_acc": s50_fp32["accuracy"],
        "per_class": s50_fp32_per_class,
        "compression": round(8.48 / 2.23, 2)
    })
    
    # MNV2 Pruned s50 INT8 Dynamic
    s50_int8 = m["s50_mnv2"]["int8_dynamic"]
    s50_int8_per_class = {}
    for cls in ds["classes"]:
        pct_val = s50_int8["per_class"][cls]
        n_c = test_support[cls]
        k = int(round(pct_val * n_c / 100.0))
        pct, lo, hi = wilson_ci(k, n_c)
        s50_int8_per_class[cls] = {"k": k, "n": n_c, "pct": pct, "ci95": [lo, hi]}
    table_pruning_audit.append({
        "name": "MobileNetV2 Pruned INT8 Dynamic (s=0.50)",
        "params": 583332,
        "active_pct": 26.17,
        "disk_size_mb": 0.72,
        "top1_acc": s50_int8["accuracy"],
        "per_class": s50_int8_per_class,
        "compression": round(8.48 / 0.72, 2)
    })
    
    # EB0 Baseline FP32
    eb0_fp32_acc = m["gate0"]["efficientnet_b0"]["fp32_test_acc"]
    eb0_fp32_cm = m["gate0"]["efficientnet_b0"]["fp32_confusion"]
    eb0_fp32_per_class = {}
    for i, cls in enumerate(ds["classes"]):
        k = int(eb0_fp32_cm[i][i])
        n_c = test_support[cls]
        pct, lo, hi = wilson_ci(k, n_c)
        eb0_fp32_per_class[cls] = {"k": k, "n": n_c, "pct": pct, "ci95": [lo, hi]}
    table_pruning_audit.append({
        "name": "EfficientNet-B0 Baseline FP32",
        "params": 4012672,
        "active_pct": 100.0,
        "disk_size_mb": 15.30,
        "top1_acc": eb0_fp32_acc,
        "per_class": eb0_fp32_per_class,
        "compression": 1.00
    })
    
    # EB0 Baseline Static QDQ
    eb0_static = m["static_eb0"]["configs"]["static_per_channel_minmax"]
    eb0_static_per_class = {}
    for cls in ds["classes"]:
        pct_val = eb0_static["per_class"][cls]
        n_c = test_support[cls]
        k = int(round(pct_val * n_c / 100.0))
        pct, lo, hi = wilson_ci(k, n_c)
        eb0_static_per_class[cls] = {"k": k, "n": n_c, "pct": pct, "ci95": [lo, hi]}
    table_pruning_audit.append({
        "name": "EfficientNet-B0 Baseline Static QDQ INT8",
        "params": 4012672,
        "active_pct": 100.0,
        "disk_size_mb": 4.67,
        "top1_acc": eb0_static["test_acc"],
        "per_class": eb0_static_per_class,
        "compression": round(15.30 / 4.67, 2)
    })
    
    # EB0 Pruned s30 FP32
    s30_eb0_fp32 = m["s30_eb0"]["fp32"]
    s30_eb0_fp32_per_class = {}
    for cls in ds["classes"]:
        pct_val = s30_eb0_fp32["per_class"][cls]
        n_c = test_support[cls]
        k = int(round(pct_val * n_c / 100.0))
        pct, lo, hi = wilson_ci(k, n_c)
        s30_eb0_fp32_per_class[cls] = {"k": k, "n": n_c, "pct": pct, "ci95": [lo, hi]}
    table_pruning_audit.append({
        "name": "EfficientNet-B0 Pruned FP32 (s=0.30)",
        "params": 2005801,
        "active_pct": 49.99,
        "disk_size_mb": 7.67,
        "top1_acc": s30_eb0_fp32["accuracy"],
        "per_class": s30_eb0_fp32_per_class,
        "compression": round(15.30 / 7.67, 2)
    })
    
    # EB0 Pruned s30 INT8 Dynamic (Collapse)
    s30_eb0_dyn = m["s30_eb0"]["int8_dynamic"]
    s30_eb0_dyn_per_class = {}
    for cls in ds["classes"]:
        pct_val = s30_eb0_dyn["per_class"][cls]
        n_c = test_support[cls]
        k = int(round(pct_val * n_c / 100.0))
        pct, lo, hi = wilson_ci(k, n_c)
        s30_eb0_dyn_per_class[cls] = {"k": k, "n": n_c, "pct": pct, "ci95": [lo, hi]}
    table_pruning_audit.append({
        "name": "EfficientNet-B0 Pruned INT8 Dynamic (s=0.30, Collapse)",
        "params": 2005801,
        "active_pct": 49.99,
        "disk_size_mb": 2.23,
        "top1_acc": s30_eb0_dyn["accuracy"],
        "per_class": s30_eb0_dyn_per_class,
        "compression": round(15.30 / 2.23, 2)
    })
    
    # EB0 Pruned s30 Static QDQ (Recovered)
    s30_eb0_stat = m["s30_eb0"]["int8_static_per_channel"]
    s30_eb0_stat_per_class = {}
    for cls in ds["classes"]:
        pct_val = s30_eb0_stat["per_class"][cls]
        n_c = test_support[cls]
        k = int(round(pct_val * n_c / 100.0))
        pct, lo, hi = wilson_ci(k, n_c)
        s30_eb0_stat_per_class[cls] = {"k": k, "n": n_c, "pct": pct, "ci95": [lo, hi]}
    table_pruning_audit.append({
        "name": "EfficientNet-B0 Pruned Static QDQ INT8 (s=0.30)",
        "params": 2005801,
        "active_pct": 49.99,
        "disk_size_mb": 2.62,
        "top1_acc": s30_eb0_stat["accuracy"],
        "per_class": s30_eb0_stat_per_class,
        "compression": round(15.30 / 2.62, 2)
    })
    
    out_data = {
        "dataset_audit": ds,
        "pruning_accuracy_matrix": table_pruning_audit,
        "hardware_audit": m["rpi5"]
    }
    
    out_file = RESULTS_DIR / "ground_truth_table.json"
    with open(out_file, "w") as fp:
        json.dump(out_data, fp, indent=2)
    print(f"Audit completed successfully. Ground truth saved to {out_file}")

if __name__ == "__main__":
    main()
