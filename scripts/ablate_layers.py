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

def _weight_dims(g, node):
    """Weight tensor dims for a Conv node, or None."""
    init = {t.name: t for t in g.initializer}
    if len(node.input) < 2:
        return None
    w = node.input[1]
    if w not in init:
        return None
    return list(init[w].dims)


def _group_attr(node):
    for a in node.attribute:
        if a.name == "group":
            return a.i
    return 1


def classify_layers(g):
    """Classify every Conv in the graph into dw / se_reduce / se_expand / pw / stem.

    Structure-derived, not name-matched, because EfficientNet-B0 exports SE
    blocks as plain `fc1/Conv` + `fc2/Conv` with no "SE" substring anywhere:

      - depthwise      : group == channels (Conv with group>1, 1x1 spatial, in_ch==1)
      - SE reduce      : 1x1 Conv whose out_channels < in_channels  (e.g. 144 -> 6)
      - SE expand      : 1x1 Conv whose out_channels > in_channels  (e.g. 6 -> 144)
      - stem           : first Conv, in_channels == 3 (RGB)
      - pointwise      : everything else that is 1x1

    The SE reduce/expand pair is what makes per-tensor scaling dangerous: their
    channel counts are tiny (4-48), so a single shared scale over 4 channels has
    far less room for outlier cancellation than a 1152-channel pointwise.
    """
    convs = [n for n in g.node if n.op_type == "Conv"]
    out = {"dw": [], "se_reduce": [], "se_expand": [], "pw": [], "stem": [], "other": []}

    for n in convs:
        dims = _weight_dims(g, n)
        if dims is None or len(dims) != 4:
            out["other"].append(n.name)
            continue
        out_c, in_c_per_g, kh, kw = dims
        grp = _group_attr(n)

        if grp > 1:
            out["dw"].append(n.name)
        elif in_c_per_g == 3:
            out["stem"].append(n.name)
        elif kh == 1 and kw == 1:
            # pointwise: compare against the block width via the sibling conv
            out["pw"].append(n.name)   # refined below
        else:
            out["other"].append(n.name)

    # Resolve SE pairs: fc1 (reduce) sits directly before fc2 (expand) inside
    # the same MBConv block. Match on the shared block path prefix.
    for n in convs:
        if n.name not in out["pw"]:
            continue
        if "/fc1/" not in n.name and "/fc2/" not in n.name:
            continue
        block = n.name.split("/block/block.")[0] if "/block/block." in n.name \
            else n.name.rsplit("/", 2)[0]
        if "/fc1/" in n.name:
            out["pw"].remove(n.name)
            out["se_reduce"].append(n.name)
        elif "/fc2/" in n.name:
            out["pw"].remove(n.name)
            out["se_expand"].append(n.name)

    return out


def group_nodes(onnx_path: Path):
    """Return ({group: [conv node names]}, raw graph) for an ONNX file."""
    import onnx
    m = onnx.load(str(onnx_path))
    return classify_layers(m.graph), m.graph


def make_exclusion(onnx_path: Path, exclude_group: str):
    """Return list node names to EXCLUDE from quantization."""
    groups, _ = group_nodes(onnx_path)
    return set(groups[exclude_group])


def make_per_channel_override(onnx_path: Path, per_channel_group: str):
    """Return set of node names that should get per-channel treatment."""
    groups, _ = group_nodes(onnx_path)
    return set(groups[per_channel_group])


# ── quantization with node exclusion ───────────────────────────────────────

def _resolve_groups(spec, groups: dict) -> set:
    """Turn a group spec into a set of node names.

    Accepts: None/empty set, a single group name ("dw"), or a "+"-joined union
    ("pw+se_reduce+se_expand+stem+other").
    """
    if not spec:
        return set()
    if isinstance(spec, set):
        return set(spec)
    out = set()
    for part in str(spec).split("+"):
        part = part.strip()
        if part not in groups:
            raise KeyError(f"unknown layer group {part!r}; have {sorted(groups)}")
        out |= set(groups[part])
    return out


def quantize_selective(fp32_onnx: Path, out_path: Path, *,
                      nodes_to_exclude=None,
                      per_channel_nodes=None,
                      calib_loader=None,
                      input_name=None,
                      calib_n=100):
    """Selective static QDQ with mixed granularity.

    ORT 1.29 gotcha (verified empirically, see scripts/probe_selective_api.py):
    `extra_options["nodes_to_exclude"]` does NOT prevent weight quantization
    for Conv nodes. Excluding every node still produced 644 Q/DQ pairs, byte
    identical to the no-exclusion baseline. The working control is the
    `nodes_to_quantize` allowlist, which reduced Q/DQ to 96 when restricted to
    the 16 depthwise convs.

    So both knobs are expressed through `nodes_to_quantize`:
      - per-tensor family : included in the allowlist
      - per-channel family: included in the allowlist, quantize per_channel=True
      - left in FP32       : simply omitted from the allowlist
    """
    from onnxruntime.quantization import (
        quantize_static, CalibrationMethod, QuantFormat, QuantType,
        quant_pre_process,
    )

    if not per_channel_nodes:
        # Uniform per-tensor over everything except the excluded set.
        prepped = fp32_onnx.with_name(fp32_onnx.stem + "_prepped.onnx")
        quant_pre_process(str(fp32_onnx), str(prepped))
        reader = calib_reader(calib_loader, input_name, calib_n)

        import onnx as _onnx
        g = _onnx.load(str(prepped)).graph
        conv_names = [n.name for n in g.node if n.op_type == "Conv"]
        allow = [n for n in conv_names if n not in set(nodes_to_exclude or set())]

        quantize_static(
            model_input=str(prepped),
            model_output=str(out_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            per_channel=False,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            nodes_to_quantize=allow,
        )
        return out_path

    # Mixed granularity: per-channel for the target family, per-tensor for the
    # remaining Conv nodes, FP32 for anything explicitly excluded.
    prepped = fp32_onnx.with_name(fp32_onnx.stem + "_prepped.onnx")
    quant_pre_process(str(fp32_onnx), str(prepped))

    import onnx as _onnx
    g = _onnx.load(str(prepped)).graph
    conv_names = [n.name for n in g.node if n.op_type == "Conv"]
    exclude = set(nodes_to_exclude or set())
    target = set(per_channel_nodes)

    # Per-channel pass over the target family only. Each quantize_static call
    # consumes its reader, so a fresh one is required per pass.
    quantize_static(
        model_input=str(prepped),
        model_output=str(out_path),
        calibration_data_reader=calib_reader(calib_loader, input_name, calib_n),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_quantize=[n for n in conv_names if n in target],
    )

    # Per-tensor pass over the remaining Conv nodes, chaining onto the result.
    rest = [n for n in conv_names if n not in target and n not in exclude]
    if rest:
        quantize_static(
            model_input=str(out_path),
            model_output=str(out_path),
            calibration_data_reader=calib_reader(calib_loader, input_name, calib_n),
            quant_format=QuantFormat.QDQ,
            per_channel=False,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            nodes_to_quantize=rest,
        )
    return out_path


def calib_reader(loader, input_name, n_images):
    """CalibrationDataReader yielding exactly n_images samples.

    n_images counts IMAGES, not batches, to match scripts/static_eb0.py which
    calibrates on the first 100 images. Counting batches would silently give
    ceil(n/batch_size)*batch_size images and make this ablation
    non-comparable to the headline static numbers.
    """
    from onnxruntime.quantization import CalibrationDataReader

    class R(CalibrationDataReader):
        def __init__(self):
            self.it = iter(loader)
            self.n = 0
            self.done = False

        def get_next(self):
            if self.done or self.n >= n_images:
                self.done = True
                return None
            try:
                imgs, _ = next(self.it)
            except StopIteration:
                self.done = True
                return None
            self.n += imgs.shape[0]
            return {input_name: imgs.numpy()}

        def rewind(self):
            self.it = iter(loader)
            self.n = 0
            self.done = False

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
    """SQNR per weight tensor (FP32 vs simulated INT8) per layer family.

    Computes BOTH per-tensor and per-channel SQNR so the comparison shows how
    much error per-tensor scaling injects into each family relative to what
    per-channel scaling could recover. A family with a large per-tensor/per-
    channel gap is a family that per-channel quantization will rescue.

    Node names are mapped to their weight initializer via node.input[1]; using
    node.name directly silently matches nothing, so results must come back
    non-empty (main() asserts this).
    """
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(fp32_onnx))
    init = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    node_by_name = {n.name: n for n in m.graph.node}

    def sqnr_db(arr: np.ndarray, axis=None) -> float:
        a = arr.astype(np.float32)
        amax = np.abs(a).max(axis=axis, keepdims=True)
        if amax.size == 0 or float(np.max(amax)) == 0.0:
            return float("nan")
        scale = amax / 127.0
        scale = np.where(scale == 0, 1.0, scale)
        deq = np.round(a / scale) * scale
        num = float((a ** 2).sum())
        den = max(float(((a - deq) ** 2).sum()), 1e-30)
        return 10 * np.log10(num / den)

    rows = []
    for group, names in group_names.items():
        if not names:
            continue
        pt, pc, shapes = [], [], []
        for name in names:
            node = node_by_name.get(name)
            if node is None or len(node.input) < 2:
                continue
            arr = init.get(node.input[1])
            if arr is None or arr.size == 0:
                continue
            shapes.append(tuple(arr.shape))
            pt.append(sqnr_db(arr, axis=None))
            # per-channel: scale each output channel independently
            pc.append(sqnr_db(arr.reshape(arr.shape[0], -1), axis=1))

        pt = [v for v in pt if v == v]      # drop NaN
        pc = [v for v in pc if v == v]
        if not pt:
            continue
        n_par = sum(int(np.prod(s)) for s in shapes)
        rows.append({
            "group": group,
            "n_tensors": len(pt),
            "n_params": n_par,
            "sqnr_pertensor_mean_db": round(float(np.mean(pt)), 2),
            "sqnr_pertensor_min_db": round(float(np.min(pt)), 2),
            "sqnr_perchannel_mean_db": round(float(np.mean(pc)), 2) if pc else None,
            "perchannel_gain_db": (round(float(np.mean(pc) - np.mean(pt)), 2)
                                   if pc else None),
        })
    return rows


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0",
                    choices=["efficientnet_b0", "mobilenetv2", "shufflenetv2"])
    ap.add_argument("--calib-n", type=int, default=100)
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these experiment names (default: all)")
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

    print("\n=== layerwise weight SQNR: per-tensor vs per-channel INT8 ===")
    sqnr = weight_error_report(fp32, groups)
    assert sqnr, "SQNR report empty — node->weight mapping is broken"
    print(f"  {'group':<11}{'n':>4}{'params':>10}{'per-tensor':>12}{'per-ch':>10}{'gain':>9}")
    for r in sqnr:
        print(f"  {r['group']:<11}{r['n_tensors']:>4}{r['n_params']:>10}"
              f"{r['sqnr_pertensor_mean_db']:>10.2f}dB"
              f"{r['sqnr_perchannel_mean_db']:>8.2f}dB"
              f"{r['perchannel_gain_db']:>8.2f}dB")
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

    # Each experiment isolates one layer family. Design logic:
    #
    #   A/B are the two extremes and must reproduce the paper's headline
    #      numbers (9.32% / 94.56%) or the ablation is not comparable.
    #
    #   C tests the paper's central claim directly: if ONLY depthwise gets
    #      per-channel treatment and everything else stays per-tensor, does it
    #      recover? If yes, depthwise is necessary and sufficient.
    #
    #   D is the complement of C (everything per-channel EXCEPT depthwise).
    #      If D collapses while C survives, depthwise is proven necessary.
    #      If D also survives, granularity is not the whole story.
    #
    #   E/F leave one family in FP32 entirely (no QDQ at all), isolating whether
    #      quantizing that family at all is harmful vs merely quantizing it
    #      per-tensor.
    experiments = [
        # (name, nodes_to_exclude, per_channel_nodes)
        ("A_all_pertensor", set(), set()),
        ("B_all_perchannel", set(), set()),      # special-cased below
        ("C_dw_perchannel", set(), "dw"),
        ("D_allpc_except_dw", set(), "pw+se_reduce+se_expand+stem+other"),
        ("E_dw_left_fp32", "dw", set()),
        ("F_se_left_fp32", "se_reduce+se_expand", set()),
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
        if a.only and name not in a.only:
            continue
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
            excl = _resolve_groups(excl, groups)
            pcn = _resolve_groups(pcn, groups)
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
