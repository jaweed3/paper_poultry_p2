#!/usr/bin/env python3
"""
scripts/quantize_static_all.py
Calibrate and generate static per-channel QDQ INT8 models for:
- mobilenetv2_gate0
- shufflenetv2_gate0
- efficientnet_b0_gate0

Evaluates accuracy, NCD recall, and confusion matrix on sterile test split (1728).
Saves output ONNX to models/ptq/<name>_static_per_channel_minmax.onnx
and results to results/static_quant_all.json.
"""

import os
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["POULTRY_DATA_DIR"] = "data/images_sterile"

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image

import train_pipeline as P

CLASSES = ["cocci", "healthy", "ncd", "salmo"]
CALIB_N = 256  # standard calibration sample size

from onnxruntime.quantization import (
    quantize_static, CalibrationMethod, QuantFormat, QuantType,
    quant_pre_process, CalibrationDataReader
)
import onnxruntime as ort

class CalibrationReader(CalibrationDataReader):
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
        self._count += images.shape[0]
        return {self.input_name: images.numpy()}

    def rewind(self):
        self._it = None
        self._count = 0


def eval_ort(onnx_path, loader):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    preds, labels = [], []
    for imgs, labs in loader:
        x = imgs.numpy()
        out = sess.run(None, {inp_name: x})[0]
        preds.extend(out.argmax(axis=1).tolist())
        labels.extend(labs.tolist())
    preds = np.array(preds)
    labels = np.array(labels)
    acc = 100.0 * float((preds == labels).sum()) / len(labels)
    per_class = {}
    for i, c in enumerate(CLASSES):
        m = labels == i
        rec = 100.0 * float((preds[m] == i).sum()) / max(int(m.sum()), 1)
        per_class[c] = round(rec, 2)
    return round(acc, 2), per_class, preds, labels


def main():
    P.prepare_data()
    tf = P.get_transforms(train=False)
    train_ds = P.PoultryDataset(P.DATA_DIR, "train", tf)
    test_ds = P.PoultryDataset(P.DATA_DIR, "test", tf)

    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
    calib_loader = DataLoader(Subset(train_ds, list(range(CALIB_N))),
                              batch_size=16, shuffle=False, num_workers=2)

    print(f"Calibration size: {CALIB_N}, Test size: {len(test_ds)}")
    assert len(test_ds) == 1728

    models = ["mobilenetv2", "shufflenetv2", "efficientnet_b0"]
    results = {}

    for name in models:
        print(f"\n==========================================")
        print(f"Processing Static QDQ for: {name}")
        print(f"==========================================")
        fp32_onnx = ROOT / "models" / "onnx" / f"{name}_gate0_fp32.onnx"
        assert fp32_onnx.exists(), f"Missing {fp32_onnx}"

        # 1. Preprocess with shape inference
        prepped_onnx = ROOT / "models" / "onnx" / f"{name}_gate0_prepped.onnx"
        print(f"Quant pre-process -> {prepped_onnx.name}")
        quant_pre_process(str(fp32_onnx), str(prepped_onnx))

        probe = ort.InferenceSession(str(prepped_onnx), providers=["CPUExecutionProvider"])
        inp_name = probe.get_inputs()[0].name

        # 2. Static QDQ Quantize
        out_onnx = ROOT / "models" / "ptq" / f"{name}_gate0_static_per_channel_minmax.onnx"
        reader = CalibrationReader(calib_loader, inp_name, n=CALIB_N)
        print(f"Quantizing static per-channel QDQ -> {out_onnx.name}")
        t0 = time.perf_counter()
        quantize_static(
            model_input=str(prepped_onnx),
            model_output=str(out_onnx),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            calibrate_method=CalibrationMethod.MinMax,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
        )
        quant_time = time.perf_counter() - t0
        print(f"Calibration & quantization completed in {quant_time:.1f}s")

        # 3. Evaluate accuracy
        acc, per_class, preds, labels = eval_ort(out_onnx, test_loader)
        file_size_mb = out_onnx.stat().st_size / (1024 * 1024)
        print(f"Static QDQ {name}: Top-1 Acc = {acc}%, NCD Recall = {per_class['ncd']}%, Size = {file_size_mb:.2f} MB")
        print(f"Per-class: {per_class}")

        results[name] = {
            "top1_acc": acc,
            "per_class": per_class,
            "ncd_recall": per_class["ncd"],
            "size_mb": round(file_size_mb, 2),
            "size_bytes": out_onnx.stat().st_size,
            "quant_time_s": round(quant_time, 1)
        }

    out_json = ROOT / "results" / "static_quant_all.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll static models calibrated and saved to {out_json}")

if __name__ == "__main__":
    main()
