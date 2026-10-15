#!/usr/bin/env python3
"""Jalur A3: EB0 static QDQ — per-tensor vs per-channel (MinMax, QDQ).

Hipotesis: kolaps EB0 dynamic (98.55% -> 10.42%, mode collapse ke ncd)
disebabkan depthwise conv yang di-quant per-tensor. Kalau per-channel
sembuh (atau jauh lebih baik), biang keroknya depthwise per-tensor.

2 config:
  A: static QDQ, per_channel=False, MinMax (calib 100 train images)
  B: static QDQ, per_channel=True,  MinMax (calib 100 train images)

Eval: test steril 1728 via ORT CPU. Simpan prediksi -> McNemar FP32 vs
tiap config + vs dynamic INT8. Output: results/static_eb0.json.

Run on LAB: bash /tmp/run_static.sh  (~10-20 min di CPU lab)
"""
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import train_pipeline as P

FP32_ONNX = ROOT / "models" / "onnx" / "efficientnet_b0_gate0_fp32.onnx"
OUT = ROOT / "results" / "static_eb0.json"
CALIB_N = 100


from onnxruntime.quantization import CalibrationDataReader


class SterileReader(CalibrationDataReader):
    """CalibrationDataReader over CALIB_N train-sterile images."""
    def __init__(self, loader, input_name, n=CALIB_N):
        self.loader = loader
        self.input_name = input_name
        self.n = n
        self._it = None
        self._count = 0

    def get_next(self):
        if self._it is None:
            self._it = iter(self.loader)
        if self._count >= self.n:
            return None
        try:
            images, _ = next(self._it)
        except StopIteration:
            return None
        self._count += 1
        return {self.input_name: images.numpy()}

    def rewind(self):
        self._it = None
        self._count = 0


def run_ort(onnx_path, loader):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    preds, labels = [], []
    for images, lab in loader:
        out = sess.run(None, {inp: images.numpy()})[0]
        preds.extend(np.argmax(out, axis=1).tolist())
        labels.extend(lab.tolist())
    return np.array(preds), np.array(labels)


def mcnemar(a_correct, b_correct):
    """McNemar with continuity correction. a=reference, b=challenger."""
    a_correct = np.asarray(a_correct, dtype=bool)
    b_correct = np.asarray(b_correct, dtype=bool)
    b = int(((a_correct) & (~b_correct)).sum())   # a right, b wrong
    c = int(((~a_correct) & (b_correct)).sum())   # a wrong, b right
    n_discord = b + c
    if n_discord == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(b - c) - 1.0) ** 2 / n_discord
    p = math.erfc(math.sqrt(chi2 / 2.0))  # chi2(1df) survival
    return {"b": b, "c": c, "chi2": round(float(chi2), 3), "p": p}


def main():
    from onnxruntime.quantization import (
        quantize_static, CalibrationMethod, QuantFormat, QuantType,
        quant_pre_process,
    )
    import onnxruntime as ort

    assert FP32_ONNX.exists(), f"missing {FP32_ONNX}, run eval_gate0 first"
    P.prepare_data()
    tf = P.get_transforms(train=False)
    train_ds = P.PoultryDataset(P.DATA_DIR, "train", tf)
    test_loader = DataLoader(P.PoultryDataset(P.DATA_DIR, "test", tf),
                             batch_size=32, shuffle=False, num_workers=4)
    calib_loader = DataLoader(Subset(train_ds, list(range(CALIB_N))),
                              batch_size=8, shuffle=False, num_workers=2)
    print(f"calib={CALIB_N} train images, test={len(test_loader.dataset)}")
    assert len(test_loader.dataset) == 1728

    # FP32 reference predictions (for McNemar pairing)
    fp32_pred, y = run_ort(FP32_ONNX, test_loader)
    fp32_acc = 100.0 * (fp32_pred == y).mean()
    fp32_ok = (fp32_pred == y)
    print(f"FP32 ref: {fp32_acc:.2f}%")

    # dynamic INT8 predictions (existing file, for McNemar baseline)
    dyn_onnx = ROOT / "models" / "ptq" / "efficientnet_b0_gate0_int8.onnx"
    dyn_pred, _ = run_ort(dyn_onnx, test_loader)
    dyn_ok = (dyn_pred == y)
    print(f"dynamic INT8: {100.0 * dyn_ok.mean():.2f}%")

    # shape-infer the FP32 model once for static quant
    prepped = FP32_ONNX.with_name("efficientnet_b0_gate0_prepped.onnx")
    quant_pre_process(str(FP32_ONNX), str(prepped))

    probe = ort.InferenceSession(str(prepped), providers=["CPUExecutionProvider"])
    input_name = probe.get_inputs()[0].name

    configs = {
        "static_per_tensor_minmax": dict(per_channel=False,
                                         calibrate_method=CalibrationMethod.MinMax),
        "static_per_channel_minmax": dict(per_channel=True,
                                          calibrate_method=CalibrationMethod.MinMax),
    }
    results = {
        "fp32_test_acc": round(float(fp32_acc), 2),
        "dynamic_int8_test_acc": round(float(100.0 * dyn_ok.mean()), 2),
        "calib_n": CALIB_N,
        "configs": {},
    }
    for cfg_name, kw in configs.items():
        print(f"\n[CONFIG] {cfg_name}")
        t0 = time.perf_counter()
        out_path = ROOT / "models" / "ptq" / f"efficientnet_b0_gate0_{cfg_name}.onnx"
        reader = SterileReader(calib_loader, input_name)
        quantize_static(
            model_input=str(prepped),
            model_output=str(out_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            **kw,
        )
        dt = time.perf_counter() - t0
        pred, _ = run_ort(out_path, test_loader)
        acc = 100.0 * (pred == y).mean()
        ok = (pred == y)
        print(f"  acc={acc:.2f}% quant_time={dt:.0f}s "
              f"size={out_path.stat().st_size/1024/1024:.2f}MB")
        per_class = {}
        for i, c in enumerate(P.CLASS_NAMES):
            m = y == i
            per_class[c] = round(100.0 * float((pred[m] == i).sum()) / max(int(m.sum()), 1), 2)
        print(f"  per-class: {per_class}")
        results["configs"][cfg_name] = {
            "test_acc": round(float(acc), 2),
            "per_class": per_class,
            "size_mb": round(out_path.stat().st_size / 1024 / 1024, 2),
            "quant_time_s": round(dt, 1),
            "mcnemar_vs_fp32": mcnemar(fp32_ok, ok),
            "mcnemar_vs_dynamic": mcnemar(dyn_ok, ok),
        }

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {OUT}")
    for cfg_name, r in results["configs"].items():
        m1 = r["mcnemar_vs_fp32"]
        print(f"  {cfg_name}: {r['test_acc']}% "
              f"| McNemar vs FP32: b={m1['b']} c={m1['c']} p={m1['p']:.2e}")


if __name__ == "__main__":
    main()
