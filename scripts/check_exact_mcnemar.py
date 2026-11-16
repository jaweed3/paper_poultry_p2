#!/usr/bin/env python3
import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
from PIL import Image
from scipy import stats

ROOT = Path("/home/master_core_ti/paper_poultry_p2")
TEST_DIR = ROOT / "data/images_sterile/test"
CLASSES = ["cocci", "healthy", "ncd", "salmo"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

samples = []
for idx, c in enumerate(CLASSES):
    for p in sorted(list((TEST_DIR / c).glob("*.*"))):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            samples.append((str(p), idx))

y_true = np.array([s[1] for s in samples])
print(f"Loaded {len(samples)} test samples.")

imgs = []
for p, _ in samples:
    im = Image.open(p).convert("RGB").resize((224, 224))
    arr = (np.array(im, dtype=np.float32) / 255.0 - MEAN) / STD
    imgs.append(arr.transpose(2, 0, 1))
batch = np.stack(imgs, axis=0).astype(np.float32)

def run_model(path):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    preds = []
    for i in range(0, len(batch), 64):
        out = sess.run(None, {inp_name: batch[i:i+64]})[0]
        preds.append(np.argmax(out, axis=1))
    return np.concatenate(preds)

models = {
    "mobilenetv2": (
        ROOT / "models/onnx/mobilenetv2_gate0_fp32.onnx",
        ROOT / "models/ptq/mobilenetv2_gate0_static_per_channel_minmax.onnx"
    ),
    "shufflenetv2": (
        ROOT / "models/onnx/shufflenetv2_gate0_fp32.onnx",
        ROOT / "models/ptq/shufflenetv2_gate0_static_per_channel_minmax.onnx"
    ),
    "efficientnet_b0": (
        ROOT / "models/onnx/efficientnet_b0_gate0_fp32.onnx",
        ROOT / "models/ptq/efficientnet_b0_gate0_static_per_channel_minmax.onnx"
    )
}

for name, (p_fp32_path, p_static_path) in models.items():
    p_fp32 = run_model(p_fp32_path)
    p_static = run_model(p_static_path)
    
    c_fp32 = (p_fp32 == y_true)
    c_static = (p_static == y_true)
    
    b = int(np.sum(c_fp32 & (~c_static)))
    c = int(np.sum((~c_fp32) & c_static))
    
    n_fp32 = int(np.sum(c_fp32))
    n_static = int(np.sum(c_static))
    
    chi2 = ((abs(b - c) - 1.0) ** 2) / (b + c) if (b + c) > 0 else 0.0
    p_val = stats.chi2.sf(chi2, df=1) if (b + c) > 0 else 1.0
    
    # Paired difference CI (Wald interval for paired difference p_diff = (b - c) / n)
    n = len(y_true)
    p_diff = (b - c) / n
    var_diff = ((b + c) - ((b - c) ** 2) / n) / (n ** 2)
    se_diff = np.sqrt(max(0, var_diff))
    ci_diff_low = (p_diff - 1.95996 * se_diff) * 100
    ci_diff_high = (p_diff + 1.95996 * se_diff) * 100
    
    print(f"\n=== {name} ===")
    print(f"  FP32 correct:   {n_fp32}/1728 ({n_fp32/1728*100:.2f}%)")
    print(f"  Static correct: {n_static}/1728 ({n_static/1728*100:.2f}%)")
    print(f"  b={b}, c={c}, b-c={b-c}")
    print(f"  Edwards chi2 = {chi2:.4f}, p = {p_val:.4e}")
    print(f"  Paired Acc Diff = {p_diff*100:+.2f}% [95% CI: {ci_diff_low:+.2f}%, {ci_diff_high:+.2f}%]")
