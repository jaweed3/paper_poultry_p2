#!/usr/bin/env python3
"""Gate 0 eval: test-sterile accuracy (FP32 torch + INT8 ORT dynamic) + latency.

Loads models/fp32/<name>_gate0.pth (best weights), evaluates on
data/images_sterile/test (1728), exports ONNX (CPU), applies ORT dynamic
INT8, evaluates INT8 on the same test set via onnxruntime, benchmarks
FP32/INT8 latency (CPU, 200 runs), saves results/eval_gate0.json.

Run on LAB: bash /tmp/run_eval.sh  (sets POULTRY_DATA_DIR + LD_PRELOAD)
~10-15 min for 3 models.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import train_pipeline as P

CKPT_DIR = ROOT / "models" / "fp32"
OUT = ROOT / "results" / "eval_gate0.json"


def per_class_acc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {}
    for i, c in enumerate(P.CLASS_NAMES):
        m = y_true == i
        out[c] = round(100.0 * float((y_pred[m] == i).sum()) / max(int(m.sum()), 1), 2)
    return out


def confusion(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(P.CLASS_NAMES)
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1
    return cm


def eval_ort(onnx_path, loader):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    preds, labels = [], []
    for images, lab in loader:
        x = images.numpy()
        out = sess.run(None, {inp: x})[0]
        preds.extend(out.argmax(1).tolist())
        labels.extend(lab.tolist())
    return np.array(preds), np.array(labels)


def bench_ort(onnx_path, num_runs=200):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    for _ in range(10):
        sess.run(None, {inp: dummy})
    ts = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        sess.run(None, {inp: dummy})
        ts.append((time.perf_counter() - t0) * 1000)
    return round(float(np.mean(ts)), 2), round(float(np.percentile(ts, 50)), 2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device(torch eval): {device}")
    P.prepare_data()
    val_tf = P.get_transforms(train=False)
    test_loader = DataLoader(P.PoultryDataset(P.DATA_DIR, "test", val_tf),
                             batch_size=32, shuffle=False, num_workers=4)
    print(f"test={len(test_loader.dataset)}")
    assert len(test_loader.dataset) == 1728, len(test_loader.dataset)

    results = {"split": "sterile-8153", "test_n": 1728, "seed": 42, "models": {}}
    for name in ["mobilenetv2", "shufflenetv2", "efficientnet_b0"]:
        print(f"\n[MODEL] {name}")
        ckpt = CKPT_DIR / f"{name}_gate0.pth"
        assert ckpt.exists(), f"missing {ckpt}"
        model = P.get_model(name)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model = model.to(device)
        crit = nn.CrossEntropyLoss()
        _, test_acc, preds, labels = P.validate(model, test_loader, crit, device)
        print(f"  FP32 test acc: {test_acc:.2f}%")
        r = {
            "fp32_test_acc": round(float(test_acc), 2),
            "fp32_per_class": per_class_acc(labels, preds),
            "fp32_confusion": confusion(labels, preds),
            "params": sum(p.numel() for p in model.parameters()),
        }
        # ONNX export (CPU) + dynamic INT8
        onnx_fp32 = P.export_onnx(model, f"{name}_gate0")
        onnx_int8 = P.quantize_onnx(onnx_fp32, f"{name}_gate0")
        # INT8 accuracy via ORT on the same test set
        ipred, ilab = eval_ort(onnx_int8, test_loader)
        int8_acc = 100.0 * float((ipred == ilab).sum()) / len(ilab)
        print(f"  INT8 test acc: {int8_acc:.2f}%")
        r["int8_test_acc"] = round(int8_acc, 2)
        r["int8_per_class"] = per_class_acc(ilab, ipred)
        r["int8_confusion"] = confusion(ilab, ipred)
        r["acc_drop_pp"] = round(r["fp32_test_acc"] - int8_acc, 2)
        # latency (WSL x86 CPU reference)
        f_mean, f_p50 = bench_ort(onnx_fp32)
        i_mean, i_p50 = bench_ort(onnx_int8)
        print(f"  FP32: {f_mean} ms | INT8: {i_mean} ms (x86 WSL ref)")
        r["fp32_lat_ms"] = f_mean
        r["int8_lat_ms"] = i_mean
        r["fp32_size_mb"] = round(onnx_fp32.stat().st_size / 1024 / 1024, 2)
        r["int8_size_mb"] = round(onnx_int8.stat().st_size / 1024 / 1024, 2)
        results["models"][name] = r
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {OUT}")
    for name, r in results["models"].items():
        print(f"  {name}: FP32 {r['fp32_test_acc']}% -> INT8 {r['int8_test_acc']}% "
              f"(drop {r['acc_drop_pp']}pp) | {r['fp32_lat_ms']}ms -> {r['int8_lat_ms']}ms")


if __name__ == "__main__":
    main()
