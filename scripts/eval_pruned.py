#!/usr/bin/env python3
"""
scripts/eval_pruned.py — Comprehensive evaluation for pruned models on sterile test split (n=1728)
Evaluates:
  1. Pruned FP32 ONNX
  2. Pruned Dynamic INT8 ONNX
  3. Pruned Static Per-Channel INT8 QDQ ONNX (automatically calibrated if needed)
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort
from tqdm import tqdm
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data/images_sterile/test"
TRAIN_DIR = ROOT / "data/images_sterile/train"
CLASSES = ["cocci", "healthy", "ncd", "salmo"]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class CalibrationReader:
    def __init__(self, calib_dir, input_name, n=100):
        self.samples = []
        for c in CLASSES:
            for p in (calib_dir / c).glob("*.*"):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append(str(p))
                    if len(self.samples) >= n:
                        break
            if len(self.samples) >= n:
                break
        self.input_name = input_name
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.samples):
            return None
        p = self.samples[self.idx]
        self.idx += 1
        img = Image.open(p).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = ((arr - MEAN) / STD).transpose(2, 0, 1)[None, ...].astype(np.float32)
        return {self.input_name: arr}

def calibrate_static_qdq(fp32_onnx, out_onnx):
    from onnxruntime.quantization import quantize_static, CalibrationMethod, QuantFormat, QuantType
    import onnx
    m = onnx.load(str(fp32_onnx))
    input_name = m.graph.input[0].name
    reader = CalibrationReader(TRAIN_DIR, input_name, n=100)
    print(f"[calib] Generating static per-channel QDQ calibration: {out_onnx}")
    quantize_static(
        model_input=str(fp32_onnx),
        model_output=str(out_onnx),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )

def load_data():
    samples = []
    for idx, c in enumerate(CLASSES):
        for p in (TEST_DIR / c).glob("*.*"):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                samples.append((str(p), idx))
    return samples

def evaluate_session(sess, input_name, samples, desc=""):
    correct = 0
    per_class_correct = {c: 0 for c in CLASSES}
    per_class_total = {c: 0 for c in CLASSES}

    for p, label in tqdm(samples, desc=desc):
        img = Image.open(p).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = ((arr - MEAN) / STD).transpose(2, 0, 1)[None, ...].astype(np.float32)
        out = sess.run(None, {input_name: arr})[0]
        pred = np.argmax(out[0])
        c_name = CLASSES[label]
        per_class_total[c_name] += 1
        if pred == label:
            correct += 1
            per_class_correct[c_name] += 1

    acc = 100.0 * correct / len(samples)
    per_class_acc = {c: round(100.0 * per_class_correct[c] / max(per_class_total[c], 1), 2) for c in CLASSES}
    return acc, per_class_acc, correct

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mobilenetv2")
    parser.add_argument("--sparsity", type=float, default=0.3)
    parser.add_argument("--static", action="store_true", help="Also generate and evaluate static per-channel QDQ")
    args = parser.parse_args()

    s_tag = f"s{int(args.sparsity * 100)}"
    onnx_dir = ROOT / f"onnx_pruned/{s_tag}"
    fp32_path = onnx_dir / f"{args.model}.onnx"
    int8_dyn_path = onnx_dir / f"{args.model}_int8.onnx"
    int8_static_path = onnx_dir / f"{args.model}_static_int8.onnx"

    samples = load_data()
    print(f"=== Evaluating {args.model} ({s_tag}) on {len(samples)} sterile test images ===")

    models_to_eval = {}
    if fp32_path.exists():
        models_to_eval["fp32"] = fp32_path
    if int8_dyn_path.exists():
        models_to_eval["int8_dynamic"] = int8_dyn_path

    # If static is requested or model is efficientnet, ensure static QDQ exists
    if args.static or args.model == "efficientnet_b0":
        if not int8_static_path.exists() and fp32_path.exists():
            try:
                calibrate_static_qdq(fp32_path, int8_static_path)
            except Exception as e:
                print(f"[WARN] Failed to calibrate static QDQ: {e}")
        if int8_static_path.exists():
            models_to_eval["int8_static_per_channel"] = int8_static_path

    results = {
        "model": args.model,
        "sparsity": args.sparsity,
        "test_n": len(samples),
        "variants": {}
    }

    for name, mpath in models_to_eval.items():
        size_mb = mpath.stat().st_size / (1024 * 1024)
        sess = ort.InferenceSession(str(mpath), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        acc, per_class_acc, correct = evaluate_session(sess, input_name, samples, desc=f"{name} ({size_mb:.2f} MB)")
        
        print(f"\n[{name}] Accuracy: {acc:.2f}% | Size: {size_mb:.2f} MB ({correct}/{len(samples)})")
        for c, c_acc in per_class_acc.items():
            print(f"  - {c}: {c_acc:.2f}%")

        results["variants"][name] = {
            "accuracy": round(acc, 2),
            "size_mb": round(size_mb, 2),
            "per_class": per_class_acc
        }

    out_file = ROOT / f"results/eval_{args.model}_{s_tag}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved evaluation results to {out_file}")

if __name__ == "__main__":
    main()
