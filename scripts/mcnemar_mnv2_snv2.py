#!/usr/bin/env python3
"""P0-4 McNemar MNv2/SNv2 — lightweight ONNX-only (no torch).

Uses ONNX FP32 + INT8 dynamic on sterile test 1728. No training, no HF download
if data/images_sterile already exists. Falls back to copying from Pi via scp.

McNemar with continuity correction, p via math.erfc (no scipy).

Run:
  python scripts/mcnemar_mnv2_snv2.py
  POULTRY_DATA_DIR=data/images_sterile python scripts/mcnemar_mnv2_snv2.py
"""
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__import__("os").environ.get("POULTRY_DATA_DIR", str(ROOT / "data" / "images_sterile")))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

CLASS_NAMES = ["cocci", "healthy", "ncd", "salmo"]
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MODELS = {
    "mobilenetv2": ("models/onnx/mobilenetv2_gate0_fp32.onnx", "models/ptq/mobilenetv2_gate0_int8.onnx"),
    "shufflenetv2": ("models/onnx/shufflenetv2_gate0_fp32.onnx", "models/ptq/shufflenetv2_gate0_int8.onnx"),
}


def mcnemar(a_correct, b_correct):
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    assert len(a) == len(b)
    b_cnt = int(((a) & (~b)).sum())  # a right, b wrong
    c_cnt = int(((~a) & (b)).sum())  # a wrong, b right
    n = b_cnt + c_cnt
    if n == 0:
        return {"b": b_cnt, "c": c_cnt, "chi2": 0.0, "p": 1.0, "n_discord": 0}
    chi2 = (abs(b_cnt - c_cnt) - 1.0) ** 2 / n
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return {"b": b_cnt, "c": c_cnt, "chi2": round(float(chi2), 3), "p": float(p), "n_discord": n}


def preprocess(pil_img):
    img = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC 0..1
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)  # CHW
    return arr.astype(np.float32)


def collect_samples():
    samples = []
    for cls_idx, cls in enumerate(CLASS_NAMES):
        d = DATA_DIR / "test" / cls
        if not d.exists():
            continue
        for p in sorted(d.glob("*.*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                samples.append((str(p), cls_idx))
    # deterministic order: sort by path so pairing is stable across runs
    samples.sort(key=lambda x: x[0])
    print(f"DATA_DIR={DATA_DIR} test samples={len(samples)}")
    if len(samples) != 1728:
        print(f"WARNING expected 1728, got {len(samples)} — counts per class:")
        from collections import Counter
        print(Counter(s[1] for s in samples))
    if len(samples) == 0:
        print(f"ERROR: no test images in {DATA_DIR}/test")
        print(f"  On Mac sterile split is not materialized (expected).")
        print(f"  On myserver/Pi: ensure data/images_sterile/test exists (scp from Pi or rebuild via scripts/rebuild_sterile.py).")
        print(f"  This script is expected to SKIP gracefully on Mac; on server it should have 1728.")
        return []
    return samples


def run_ort(onnx_path, samples):
    import onnxruntime as ort
    assert Path(onnx_path).exists(), f"missing {onnx_path} — run ./download_models.sh or scp from Pi"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    preds = []
    # batch 1, no torch DataLoader, simple loop with tqdm if available
    try:
        from tqdm import tqdm
        iterator = tqdm(samples, desc=Path(onnx_path).name)
    except ImportError:
        iterator = samples
    for p, _ in iterator:
        x = preprocess(Image.open(p).convert("RGB"))[None, ...]  # 1x3x224x224
        out = sess.run(None, {inp: x})[0]
        preds.append(int(np.argmax(out, axis=1)[0]))
    return np.array(preds, dtype=np.int64)


def main():
    samples = collect_samples()
    if not samples:
        out = {"data_dir": str(DATA_DIR), "test_n": 0, "models": {}, "note": "no test images — skip on Mac, run on myserver/Pi"}
        out_path = RESULTS / "mcnemar_mnv2_snv2.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved stub -> {out_path} (no data on this host, run on myserver/Pi)")
        return
    y_true = np.array([c for _, c in samples], dtype=np.int64)
    # verify ONNX files
    for name, (fp32_rel, int8_rel) in MODELS.items():
        fp32 = ROOT / fp32_rel
        int8 = ROOT / int8_rel
        print(f"\n[{name}] fp32={fp32} exists={fp32.exists()} | int8={int8} exists={int8.exists()}")
        if not fp32.exists() or not int8.exists():
            print(f"  -> missing model for {name}. Run: ./download_models.sh --archive onnx  -- or scp from pi:~/p2bench/models/onnx/")
    out = {"data_dir": str(DATA_DIR), "test_n": len(samples), "models": {}}
    for name, (fp32_rel, int8_rel) in MODELS.items():
        fp32 = ROOT / fp32_rel
        int8 = ROOT / int8_rel
        if not fp32.exists() or not int8.exists():
            print(f"SKIP {name}: missing ONNX")
            continue
        print(f"\n=== {name} ===")
        fp32_pred = run_ort(fp32, samples)
        int8_pred = run_ort(int8, samples)
        fp32_acc = 100.0 * (fp32_pred == y_true).mean()
        int8_acc = 100.0 * (int8_pred == y_true).mean()
        drop = fp32_acc - int8_acc
        # McNemar on correctness
        fp32_ok = (fp32_pred == y_true)
        int8_ok = (int8_pred == y_true)
        mc = mcnemar(fp32_ok, int8_ok)
        # also save discord vs paper
        print(f"  FP32 {fp32_acc:.2f}% | INT8 {int8_acc:.2f}% | drop {drop:.2f}pp")
        print(f"  McNemar b={mc['b']} c={mc['c']} chi2={mc['chi2']} p={mc['p']:.3e} n={mc['n_discord']}")
        # interpretation helper
        if mc["p"] < 0.05:
            print(f"  -> significant at alpha=0.05")
        else:
            print(f"  -> NOT significant at alpha=0.05 (do not claim significance)")
        out["models"][name] = {
            "fp32_acc": round(float(fp32_acc), 2),
            "int8_acc": round(float(int8_acc), 2),
            "drop_pp": round(float(drop), 2),
            "mcnemar": mc,
        }
    out_path = RESULTS / "mcnemar_mnv2_snv2.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_path}")
    # also print a Table-I-ready snippet
    for name, r in out["models"].items():
        mc = r["mcnemar"]
        print(f"{name}: {r['fp32_acc']:.2f} -> {r['int8_acc']:.2f} (drop {r['drop_pp']:.2f}pp) McNemar b={mc['b']} c={mc['c']} p={mc['p']:.2e}")


if __name__ == "__main__":
    main()
