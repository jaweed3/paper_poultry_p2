import json
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data/images_sterile/test"
CLASSES = ["cocci", "healthy", "ncd", "salmo"]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_data():
    samples = []
    for idx, c in enumerate(CLASSES):
        for p in (TEST_DIR / c).glob("*.*"):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                samples.append((str(p), idx))
    return samples

def main():
    samples = load_data()
    print(f"Total test samples: {len(samples)}")

    models_to_test = {
        "pruned_s30_fp32": ROOT / "onnx_pruned/s30/mobilenetv2.onnx",
        "pruned_s30_int8": ROOT / "onnx_pruned/s30/mobilenetv2_int8.onnx"
    }

    results = {}
    for name, mpath in models_to_test.items():
        if not mpath.exists():
            print(f"{name} not found at {mpath}")
            continue
        sess = ort.InferenceSession(str(mpath), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        correct = 0
        per_class_correct = {c: 0 for c in CLASSES}
        per_class_total = {c: 0 for c in CLASSES}

        for p, label in tqdm(samples, desc=f"Evaluating {name}"):
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
        size_mb = mpath.stat().st_size / (1024 * 1024)
        print(f"\n[{name}] Test Accuracy: {acc:.2f}% | Size: {size_mb:.2f} MB ({correct}/{len(samples)})")
        for c in CLASSES:
            c_acc = 100.0 * per_class_correct[c] / max(per_class_total[c], 1)
            print(f"  - {c}: {c_acc:.2f}% ({per_class_correct[c]}/{per_class_total[c]})")
        
        results[name] = {
            "accuracy": round(acc, 2),
            "size_mb": round(size_mb, 2),
            "per_class": {c: round(100.0 * per_class_correct[c] / max(per_class_total[c], 1), 2) for c in CLASSES}
        }

    out_file = ROOT / "results/eval_pruned_s30.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
