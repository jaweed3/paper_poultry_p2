#!/usr/bin/env python3
"""Evaluate FP32 vs INT8 ONNX accuracy on poultry test set (lab Windows)."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort

PROJECT = Path(__file__).resolve().parent  # repo root (absolute build path redacted for blind review)
DATA_TEST = PROJECT / "data" / "images" / "test"
MODELS = {
    "mobilenetv2": ("mobilenetv2_fp32.onnx", "mobilenetv2_int8.onnx"),
    "shufflenetv2": ("shufflenetv2_fp32.onnx", "shufflenetv2_int8.onnx"),
    "efficientnet_b0": ("efficientnet_b0_fp32.onnx", "efficientnet_b0_int8.onnx"),
}
CLASS_NAMES = ["cocci", "healthy", "ncd", "salmo"]
CLS2IDX = {c:i for i,c in enumerate(CLASS_NAMES)}

IMG_SIZE = 224
MEAN = np.array([0.485,0.456,0.406], dtype=np.float32)
STD = np.array([0.229,0.224,0.225], dtype=np.float32)

def preprocess(p):
    img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2,0,1)[None, ...]  # 1x3xHxW
    return arr.astype(np.float32)

def gather_test():
    samples = []
    for cls in CLASS_NAMES:
        d = DATA_TEST / cls
        if not d.exists():
            print(f"MISSING {d}")
            continue
        for f in d.iterdir():
            if f.suffix.lower() in (".jpg",".jpeg",".png",".bmp"):
                samples.append((str(f), CLS2IDX[cls]))
    print(f"Test samples: {len(samples)}")
    # distribution
    from collections import Counter
    c = Counter(s[1] for s in samples)
    print(dict(c))
    return samples

def evaluate_onnx(onnx_path, samples):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    correct = 0
    # per-class
    from collections import Counter, defaultdict
    per_correct = Counter()
    per_total = Counter()
    for img_path, label in samples:
        x = preprocess(img_path)
        out = sess.run(None, {inp: x})[0]  # Nx4
        pred = int(np.argmax(out, axis=1)[0])
        if pred == label:
            correct += 1
            per_correct[label] += 1
        per_total[label] += 1
    acc = correct / len(samples) * 100
    per_acc = {CLASS_NAMES[k]: round(per_correct[k]/per_total[k]*100,2) if per_total[k] else 0 for k in range(len(CLASS_NAMES))}
    return acc, per_acc

def main():
    samples = gather_test()
    result = {}
    for name, (fp32_file, int8_file) in MODELS.items():
        fp32_path = PROJECT / "models" / "onnx" / fp32_file
        int8_path = PROJECT / "models" / "ptq" / int8_file
        print(f"\n=== {name} ===")
        print(f" FP32: {fp32_path} exists={fp32_path.exists()}")
        print(f" INT8: {int8_path} exists={int8_path.exists()}")
        if not fp32_path.exists() or not int8_path.exists():
            print(" SKIP missing")
            continue
        fp32_acc, fp32_per = evaluate_onnx(fp32_path, samples)
        print(f" FP32 acc: {fp32_acc:.2f}% {fp32_per}")
        int8_acc, int8_per = evaluate_onnx(int8_path, samples)
        print(f" INT8 acc: {int8_acc:.2f}% {int8_per}")
        result[name] = {"fp32_acc": round(fp32_acc,4), "fp32_per_class": fp32_per, "int8_acc": round(int8_acc,4), "int8_per_class": int8_per, "drop": round(fp32_acc - int8_acc,4)}
    out = PROJECT / "results" / "int8_real_accuracy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out}")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
