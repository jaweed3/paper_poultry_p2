#!/usr/bin/env python3
"""
scripts/quantize_and_eval_weighted_distill.py
Exports mobilenetv2_s30_weighted_distill.pth to ONNX,
quantizes it using static per-channel MinMax QDQ,
and evaluates both FP32 and Static QDQ on the sterile test set (n=1,728).
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["POULTRY_DATA_DIR"] = "data/images_sterile"

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torch_pruning as tp
import onnxruntime as ort
import onnx
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat
)

import train_pipeline as P

CLASSES = ["cocci", "healthy", "ncd", "salmo"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class PoultryCalibDataReader(CalibrationDataReader):
    def __init__(self, calib_dir, num_samples=100):
        self.samples = []
        for c in CLASSES:
            for p in sorted(list((calib_dir / c).glob("*.*"))):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append(str(p))
        np.random.default_rng(42).shuffle(self.samples)
        self.samples = self.samples[:num_samples]
        self.idx = 0
        print(f"CalibrationDataReader initialized with {len(self.samples)} samples.")

    def get_next(self):
        if self.idx >= len(self.samples):
            return None
        p = self.samples[self.idx]
        self.idx += 1
        im = Image.open(p).convert("RGB").resize((224, 224))
        arr = ((np.array(im, dtype=np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
        return {"input": arr[None, ...].astype(np.float32)}

    def rewind(self):
        self.idx = 0

def wilson_ci(k, n, z=1.95996):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return round(max(0.0, (centre - half) * 100), 2), round(min(1.0, (centre + half) * 100), 2)

def evaluate_onnx(onnx_path, test_samples):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    
    y_true = np.array([s[1] for s in test_samples])
    imgs = []
    for p, _ in test_samples:
        im = Image.open(p).convert("RGB").resize((224, 224))
        arr = (np.array(im, dtype=np.float32) / 255.0 - MEAN) / STD
        imgs.append(arr.transpose(2, 0, 1))
    batch = np.stack(imgs, axis=0).astype(np.float32)
    
    preds = []
    for i in range(0, len(batch), 64):
        out = sess.run(None, {inp_name: batch[i:i+64]})[0]
        preds.append(np.argmax(out, axis=1))
    preds = np.concatenate(preds)
    
    correct = (preds == y_true)
    top1_acc = float(np.mean(correct) * 100)
    
    per_class = {}
    for idx, c in enumerate(CLASSES):
        m = (y_true == idx)
        k = int(np.sum(preds[m] == idx))
        n = int(np.sum(m))
        rec = float(k / n * 100) if n > 0 else 0.0
        w_ci = wilson_ci(k, n)
        per_class[c] = {
            "recall": round(rec, 2),
            "correct": k,
            "total": n,
            "wilson_ci": w_ci
        }
        
    return {
        "top1_acc": round(top1_acc, 2),
        "total_correct": int(np.sum(correct)),
        "total_samples": len(y_true),
        "per_class": per_class
    }

def main():
    device = torch.device("cpu")
    model_pth = ROOT / "models/pruned/mobilenetv2_s30_weighted_distill.pth"
    assert model_pth.exists(), f"Missing {model_pth}"
    
    # Reconstruct student model
    student = P.get_model("mobilenetv2")
    example_input = torch.randn(1, 3, 224, 224)
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
    student.load_state_dict(torch.load(model_pth, map_location=device))
    student.eval()
    
    onnx_dir = ROOT / "onnx_pruned/s30"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    fp32_onnx = onnx_dir / "mobilenetv2_s30_weighted_distill_fp32.onnx"
    static_onnx = onnx_dir / "mobilenetv2_s30_weighted_distill_static_int8.onnx"
    
    # 1. Export FP32 ONNX
    print(f"Exporting FP32 ONNX to {fp32_onnx}...")
    torch.onnx.export(
        student,
        example_input,
        str(fp32_onnx),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13
    )
    fp32_size_mb = round(os.path.getsize(fp32_onnx) / (1024 * 1024), 2)
    print(f"FP32 ONNX exported ({fp32_size_mb} MB).")
    
    # 2. Static Per-Channel Quantization
    print("Performing static per-channel QDQ calibration...")
    dr = PoultryCalibDataReader(ROOT / "data/images_sterile/train", num_samples=100)
    quantize_static(
        model_input=str(fp32_onnx),
        model_output=str(static_onnx),
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8
    )
    static_size_mb = round(os.path.getsize(static_onnx) / (1024 * 1024), 2)
    print(f"Static QDQ ONNX exported to {static_onnx} ({static_size_mb} MB).")
    
    # 3. Evaluate on sterile test set
    test_dir = ROOT / "data/images_sterile/test"
    test_samples = []
    for idx, c in enumerate(CLASSES):
        for p in sorted(list((test_dir / c).glob("*.*"))):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                test_samples.append((str(p), idx))
    print(f"Evaluating on {len(test_samples)} sterile test samples...")
    
    res_fp32 = evaluate_onnx(fp32_onnx, test_samples)
    res_static = evaluate_onnx(static_onnx, test_samples)
    
    print("\n================ RESULTS ================")
    print("FP32 (Weighted + KD):", json.dumps(res_fp32, indent=2))
    print("\nStatic QDQ (Weighted + KD):", json.dumps(res_static, indent=2))
    
    out_json = ROOT / "results/eval_mobilenetv2_s30_weighted_distill_static.json"
    with open(out_json, "w") as f:
        json.dump({
            "fp32": res_fp32,
            "static_qdq": res_static,
            "fp32_size_mb": fp32_size_mb,
            "static_size_mb": static_size_mb
        }, f, indent=2)
    print(f"Saved results to {out_json}")

if __name__ == "__main__":
    main()
