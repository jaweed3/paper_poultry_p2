#!/usr/bin/env python3
"""Selective quantization ablation: WHICH layer type causes the collapse?

Jawaban untuk R1.1 dan R3.1a:
  R1.1: "Support the depthwise-quantization diagnosis with a layer-wise or
         ablation experiment (e.g., keeping only depthwise layers per-tensor vs.
         per-channel, or excluding SE blocks)"
  R3.1a: "Please provide selective quantization ablations for depthwise, pointwise,
          and squeeze-and-excitation components, supported by layerwise weight or
          activation error analysis."

Strategi: exported EfficientNet-B0 dipecah per tipe layer (depthwise / pointwise /
SE / classic), lalu tiap grup di-quant dengan granularity berbeda. Hasilnya
mengisolasi layer mana yang sensitif.

  A. baseline          : seluruh model per-tensor          (expect ~9%, collapse)
  B. baseline_pc       : seluruh model per-channel         (expect ~94.6%, recovered)
  C. dw_pt_pc          : depthwise saja per-channel, sisanya per-tensor
  D. dw_per_tensor     : depthwise saja per-tensor, sisanya per-channel
     -> kalau D collapse & C safe, depthwise = culprit
  E. se_excluded_pt    : SE block TIDAK di-quant sama sekali, sisanya per-tensor
     -> kalau E safe, SE = contributor
  F. se_excluded_dw    : depthwise TIDAK di-quant, sisanya per-tensor
     -> kalau F safe, depthwise = culprit (konfirmasi independen dari D)

Plus: layerwise weight-error analysis (SQNR per layer) via `weight_error_report()`.

Cara pakai (butuh lab CPU ~10-20 menit, TIDAK butuh GPU, TIDAK retrain):
  python scripts/ablate_layers.py --model efficientnet_b0
  python scripts/ablate_layers.py --model mobilenetv2   # kontrol
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "models" / "onnx"
OUT_DIR = ROOT / "results"


# ── layer classification ────────────────────────────────────────────────────

def classify_node(node_name: str, node_op: str) -> str:
    """Klasifikasikan ONNX node ke dalam: dw / pw / se / other."""
    n = node_name.lower()
    # EfficientNet MBConv: DepthwiseConv node / Conv dengan group=depth
    if "depthwise" in n:
        return "dw"
    # Squeeze-and-Excitation: GlobalAveragePool + ReduceMean + 1x1 Conv pair
    if any(k in n for k in ("se_", "squeeze", "excitation", "se_reduce", "se_expand")):
        return "se"
    # pointwise / 1x1 / Gemm / MatMul = channel mixing
    if any(k in n for k in ("conv2d_1x1", "1x1", "pointwise", "pw")):
        return "pw"
    return "other"


def group_nodes(onnx_path: Path):
    """Return {group: [node names]} plus a function to detect depthwise via group attr."""
    import onnx

    model = onnx.load(str(onnx_path))
    groups = {"dw": [], "pw": [], "se": [], "other": []}

    # Kumpulkan SE block ranges dari Pool/Reduce node
    se_markers = []
    for node in model.graph.node:
        if node.op_type in ("GlobalAveragePool", "ReduceMean"):
            se_markers.append(node)

    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        g = classify_node(node.name, node.op_type)
        # fallback: cek attr group > 1 = depthwise
        attrs = {a.name: a for a in node.attribute}
        if "group" in attrs and attrs["group"].i > 1:
            g = "dw"
        groups[g].append(node.name)

    return groups, se_markers


def make_exclusion(onnx_path: Path, exclude_group: str):
    """Return list node names to EXCLUDE from quantization."""
    groups, _ = group_nodes(onnx_path)
    return set(groups[exclude_group])


def make_per_channel_override(onnx_path: Path, per_channel_group: str):
    """Return set of node names that should get per-channel treatment."""
    groups, _ = group_nodes(onnx_path)
    return set(groups[per_channel_group])


# ── quantization with node exclusion ───────────────────────────────────────

def quantize_selective(fp32_onnx: Path, out_path: Path, *,
                      nodes_to_exclude=None,
                      per_channel_nodes=None,
                      calib_loader=None,
                      input_name=None,
                      calib_n=100):
    """Static QDQ per-tensor by default; nodes in per_channel_nodes get per-channel.

    nodes_to_exclude : set of node names -> left in FP32 (no QDQ inserted)
    per_channel_nodes : set of node names -> QDQ with per_channel=True
    """
    from onnxruntime.quantization import (
        quantize_static, CalibrationMethod, QuantFormat, QuantType,
        quant_pre_process, quantize_dynamic,
    )

    nodes_to_exclude = nodes_to_exclude or set()
    per_channel_nodes = per_channel_nodes or set()

    # Strategy: opset-level first, then refine per-node via ONNX NodeQuantConfig
    from onnxruntime.quantization import QuantType as QT
    from onnxruntime.quantization.node_quant_config import NodeQuantConfig

    # Bangun extra_options untuk exclude + per-channel override
    extra = {
        "nodes_to_exclude": list(nodes_to_exclude),
    }

    if not per_channel_nodes:
        # single-shot: semua per-tensor (atau semua per-channel)
        pc = len(per_channel_nodes) == 0 and "per_channel_all" in str(out_path)
        prepped = fp32_onnx.with_name(fp32_onnx.stem + "_prepped.onnx")
        quant_pre_process(str(fp32_onnx), str(prepped))
        reader = calib_reader(calib_loader, input_name, calib_n)
        quantize_static(
            model_input=str(prepped),
            model_output=str(out_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            per_channel=pc,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            extra_options=extra,
        )
        return out_path

    # Mixed granularity: split approach.
    # Praktisnya: exclude non-target, quantize target per-channel, selesai.
    all_names = set()
    import onnx
    m = onnx.load(str(fp32_onnx))
    for node in m.graph.node:
        all_names.add(node.name)
    to_exclude = (all_names - per_channel_nodes) | nodes_to_exclude
    extra["nodes_to_exclude"] = list(to_exclude)

    prepped = fp32_onnx.with_name(fp32_onnx.stem + "_prepped.onnx")
    quant_pre_process(str(fp32_onnx), str(prepped))
    reader = calib_reader(calib_loader, input_name, calib_n)
    quantize_static(
        model_input=str(prepped),
        model_output=str(out_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options=extra,
    )
    return out_path


def calib_reader(loader, input_name, n):
    from onnxruntime.quantization import CalibrationDataReader

    class R(CalibrationDataReader):
        def __init__(self):
            self.it = None
            self.c = 0

        def get_next(self):
            if self.it is None:
                self.it = iter(loader)
            if self.c >= n:
                return None
            try:
                imgs, _ = next(self.it)
            except StopIteration:
                return None
            self.c += 1
            return {input_name: imgs.numpy()}

        def rewind(self):
            self.it = None
            self.c = 0

    return R()


# ── evaluation ─────────────────────────────────────────────────────────────

def run_ort_acc(onnx_path: Path, loader):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    preds, labels = [], []
    for imgs, lab in loader:
        out = sess.run(None, {inp: imgs.numpy()})[0]
        preds.extend(np.argmax(out, axis=1).tolist())
        labels.extend(lab.tolist())
    p, y = np.array(preds), np.array(labels)
    return 100.0 * (p == y).mean(), p, y


def weight_error_report(fp32_onnx: Path, group_names: dict):
    """SQNR per weight tensor (FP32 vs simulated per-tensor INT8) per group."""
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(fp32_onnx))
    init = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}

    rows = []
    for group, names in group_names.items():
        if not names:
            continue
        errs = []
        for name in names:
            arr = init.get(name)
            if arr is None or arr.size == 0:
                continue
            arr = arr.astype(np.float32)
            # simulate per-tensor int8 round
            amax = np.abs(arr).max()
            if amax == 0:
                continue
            scale = amax / 127.0
            q = np.round(arr / scale)
            deq = q * scale
            num = float((arr ** 2).sum())
            den = max(float(((arr - deq) ** 2).sum()), 1e-30)
            errs.append(10 * np.log10(num / den))
        if errs:
            rows.append({
                "group": group,
                "n_tensors": len(errs),
                "sqnr_mean_db": round(float(np.mean(errs)), 2),
                "sqnr_min_db": round(float(np.min(errs)), 2),
                "sqnr_median_db": round(float(np.median(errs)), 2),
            })
    return rows


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0",
                    choices=["efficientnet_b0", "mobilenetv2", "shufflenetv2"])
    ap.add_argument("--calib-n", type=int, default=100)
    a = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, Subset
    import train_pipeline as P

    fp32 = MODEL_DIR / f"{a.model}_gate0_fp32.onnx"
    if not fp32.exists():
        print(f"[ERROR] missing {fp32}")
        print("  run: python scripts/eval_gate0.py  (atau train dulu)")
        sys.exit(1)

    P.prepare_data()
    tf = P.get_transforms(train=False)
    train_ds = P.PoultryDataset(P.DATA_DIR, "train", tf)
    test_loader = DataLoader(P.PoultryDataset(P.DATA_DIR, "test", tf),
                             batch_size=32, shuffle=False, num_workers=4)
    calib_loader = DataLoader(Subset(train_ds, list(range(a.calib_n))),
                              batch_size=8, shuffle=False, num_workers=2)
    print(f"test={len(test_loader.dataset)} calib={a.calib_n}")

    out_dir = ROOT / "results" / f"ablate_{a.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    groups, _ = group_nodes(fp32)
    print("\n=== node groups ===")
    for g, names in groups.items():
        print(f"  {g:<8} {len(names):>4} nodes")
    (out_dir / "node_groups.json").write_text(
        json.dumps({g: list(n) for g, n in groups.items()}, indent=2))

    print("\n=== layerwise weight SQNR (per-tensor INT8 simulation) ===")
    sqnr = weight_error_report(fp32, groups)
    for r in sqnr:
        print(f"  {r['group']:<8} n={r['n_tensors']:>3}  "
              f"SQNR mean={r['sqnr_mean_db']:>6.2f}dB  min={r['sqnr_min_db']:>6.2f}dB")
    (out_dir / "weight_sqnr.json").write_text(json.dumps(sqnr, indent=2))

    import onnxruntime as ort
    import onnx
    prepped = fp32.with_name(fp32.stem + "_probe.onnx")
    from onnxruntime.quantization import quant_pre_process
    quant_pre_process(str(fp32), str(prepped))
    sess = ort.InferenceSession(str(prepped), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # base FP32 accuracy
    fp32_acc, fp32_pred, y = run_ort_acc(fp32, test_loader)
    print(f"\nFP32 reference: {fp32_acc:.2f}%")

    experiments = [
        # (name, nodes_to_exclude, per_channel_nodes)
        ("A_all_pertensor", set(), set()),
        ("B_all_perchannel", set(), set()),   # special-cased
        ("C_dw_perchannel_only", None, "dw"),
        ("D_dw_pertensor_only", None, "pw+se+other"),
        ("E_se_excluded", "se", set()),
        ("F_dw_excluded", "dw", set()),
    ]

    results = {
        "model": a.model,
        "fp32_acc": round(float(fp32_acc), 2),
        "calib_n": a.calib_n,
        "groups": {g: len(n) for g, n in groups.items()},
        "weight_sqnr": sqnr,
        "experiments": {},
    }

    ptq_dir = ROOT / "models" / "ptq"
    ptq_dir.mkdir(parents=True, exist_ok=True)

    for name, excl, pcn in experiments:
        print(f"\n[{name}]")
        out_onnx = ptq_dir / f"{a.model}_ablate_{name}.onnx"

        if name == "B_all_perchannel":
            from onnxruntime.quantization import (
                quantize_static, CalibrationMethod, QuantFormat, QuantType)
            quantize_static(
                model_input=str(prepped), model_output=str(out_onnx),
                calibration_data_reader=calib_reader(calib_loader, input_name, a.calib_n),
                quant_format=QuantFormat.QDQ, per_channel=True,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax)
        elif name == "A_all_pertensor":
            from onnxruntime.quantization import (
                quantize_static, CalibrationMethod, QuantFormat, QuantType)
            quantize_static(
                model_input=str(prepped), model_output=str(out_onnx),
                calibration_data_reader=calib_reader(calib_loader, input_name, a.calib_n),
                quant_format=QuantFormat.QDQ, per_channel=False,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax)
        else:
            if isinstance(excl, str):
                excl = set(groups[excl])
            if isinstance(pcn, str) and "+" in pcn:
                pcn = set()
                for part in pcn.split("+"):
                    pcn |= set(groups[part])
            elif pcn == "dw":
                pcn = set(groups["dw"])
            quantize_selective(fp32, out_onnx,
                               nodes_to_exclude=excl,
                               per_channel_nodes=pcn,
                               calib_loader=calib_loader,
                               input_name=input_name,
                               calib_n=a.calib_n)

        if not out_onnx.exists():
            print(f"  [WARN] quantization produced no file, skip")
            continue

        acc, pred, _ = run_ort_acc(out_onnx, test_loader)
        size_mb = out_onnx.stat().st_size / 1024 / 1024
        print(f"  acc={acc:.2f}%  size={size_mb:.2f}MB")

        # mode collapse check
        ncd_frac = (pred == 3).mean() * 100 if len(P.CLASS_NAMES) > 3 else 0
        results["experiments"][name] = {
            "test_acc": round(float(acc), 2),
            "size_mb": round(size_mb, 2),
            "ncd_pred_frac_pct": round(float(ncd_frac), 1),
            "collapsed": bool(acc < 50),
            "excluded": sorted(excl)[:20] if isinstance(excl, set) else None,
            "excluded_count": len(excl) if isinstance(excl, set) else 0,
            "per_channel_count": len(pcn) if isinstance(pcn, set) else 0,
        }
        print(f"  collapsed={acc < 50}  ncd_pred={ncd_frac:.1f}%")

    (out_dir / "ablation.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out_dir / 'ablation.json'}")

    print("\n=== RINGKASAN ===")
    print(f"{'experiment':<28}{'acc':>8}{'collapsed':>12}")
    for k, v in results["experiments"].items():
        print(f"{k:<28}{v['test_acc']:>8.2f}{str(v['collapsed']):>12}")


if __name__ == "__main__":
    main()
