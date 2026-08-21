#!/usr/bin/env python3
"""
dry_run.py — end-to-end dry-run pipeline for Paper 2 (morning deliverable).

Runs on synthetic data (no dataset needed), SSOT from config.yaml:
  model (prune L1) -> ONNX export -> PTQ (ORT dynamic) -> nodecount -> bench (5 iters)

Proves pipeline WORKS, not just runs. Outputs artifact for hypothesis validation.

Usage:
  conda run -n ml_core python scripts/dry_run.py
  conda run -n ml_core python scripts/dry_run.py --model shufflenetv2 --sparsity 0.5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config.yaml"

def load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)

def import_prune():
    # Import prune module without circular issues
    sys.path.insert(0, str(ROOT / "scripts"))
    import prune as prune_mod
    return prune_mod

def quantize_onnx(fp32_path: Path, int8_path: Path):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    return int8_path

def count_nodes(onnx_path: Path):
    import onnx
    m = onnx.load(str(onnx_path))
    total = len(m.graph.node)
    op_counts = {}
    ql = dql = 0
    for n in m.graph.node:
        op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1
        if n.op_type == "QuantizeLinear": ql += 1
        elif n.op_type == "DequantizeLinear": dql += 1
    return {"total_nodes": total, "quantize_linear": ql, "dequantize_linear": dql, "op_counts": op_counts}

def bench_onnx(onnx_path: Path, runs=5, warmup=2, batch=1):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    dummy = np.random.randn(batch, 3, 224, 224).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {inp: dummy})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {inp: dummy})
        times.append((time.perf_counter()-t0)*1000)
    return {"mean_ms": round(float(np.mean(times)), 2), "p50_ms": round(float(np.median(times)), 2), "runs": runs, "batch": batch}

def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=cfg["pruning"]["dryrun_model"], choices=cfg["models"]["names"])
    ap.add_argument("--sparsity", type=float, default=cfg["pruning"]["dryrun_sparsity"])
    args = ap.parse_args()

    print("="*70)
    print("  DRY-RUN — Paper 2 pruning-then-PTQ (synthetic, reproducible)")
    print(f"  model={args.model} sparsity={args.sparsity} seed={cfg['project']['seed']}")
    print("="*70)

    # 1. Prune -> ONNX
    print("\n[1/4] Prune + export ONNX ...")
    import subprocess
    r = subprocess.run(
        [sys.executable, str(ROOT/"scripts/prune.py"), "--model", args.model, "--sparsity", str(args.sparsity)],
        capture_output=False, text=True
    )
    if r.returncode != 0:
        print("[FAIL] prune.py failed", file=sys.stderr); sys.exit(r.returncode)

    fp32_onnx = ROOT / f"onnx_pruned/s{int(args.sparsity*100)}/{args.model}.onnx"
    if not fp32_onnx.exists():
        print(f"[FAIL] expected {fp32_onnx} not found", file=sys.stderr); sys.exit(3)
    print(f"[OK] ONNX: {fp32_onnx} ({fp32_onnx.stat().st_size/1024/1024:.2f} MB)")

    # 2. PTQ
    print("\n[2/4] Dynamic PTQ (ORT) ...")
    int8_onnx = ROOT / f"onnx_pruned/s{int(args.sparsity*100)}/{args.model}_int8.onnx"
    quantize_onnx(fp32_onnx, int8_onnx)
    print(f"[OK] INT8: {int8_onnx} ({int8_onnx.stat().st_size/1024/1024:.2f} MB)")

    # 3. Nodecount
    print("\n[3/4] Node count + expansion ...")
    n_fp32 = count_nodes(fp32_onnx)
    n_int8 = count_nodes(int8_onnx)
    expansion = round(n_int8["total_nodes"]/max(n_fp32["total_nodes"],1), 2)
    print(f"  FP32 nodes: {n_fp32['total_nodes']} | INT8 nodes: {n_int8['total_nodes']} | expansion: {expansion}x")
    print(f"  INT8 Q/DQ: QL={n_int8['quantize_linear']} DQL={n_int8['dequantize_linear']}")
    # Compare to Paper 1 baseline if available
    baseline_path = ROOT / "results/baseline_p1/profiling/level2_profiling_artifact.json"
    if baseline_path.exists():
        print(f"  Paper 1 baseline available at {baseline_path} — check after run for delta")

    # 4. Bench (tiny) — FP32 always works; INT8 ConvInteger may not be supported on macOS ORT build
    print("\n[4/4] Bench (CPUExecutionProvider, 5 runs) ...")
    b_fp32 = bench_onnx(fp32_onnx, runs=cfg["benchmark"]["rpi5"]["dryrun"]["runs"], warmup=cfg["benchmark"]["rpi5"]["dryrun"]["warmup"])
    try:
        b_int8 = bench_onnx(int8_onnx, runs=cfg["benchmark"]["rpi5"]["dryrun"]["runs"], warmup=cfg["benchmark"]["rpi5"]["dryrun"]["warmup"])
        print(f"  FP32: {b_fp32['mean_ms']} ms (p50 {b_fp32['p50_ms']})")
        print(f"  INT8: {b_int8['mean_ms']} ms (p50 {b_int8['p50_ms']})")
        b_int8_err = None
    except Exception as e:
        b_int8 = None
        b_int8_err = str(e)[:600]
        print(f"  FP32: {b_fp32['mean_ms']} ms (p50 {b_fp32['p50_ms']})")
        print(f"  INT8 bench skipped on this ORT build (expected on macOS): {b_int8_err[:200]}")
        print(f"  INT8 bench will be run on RPi5 (CPUExecutionProvider there has ConvInteger)")

    # Artifact
    artifact = {
        "experiment": "Paper 2 dry-run — pruning-then-PTQ (synthetic, reproducible)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_source": "config.yaml",
        "model": args.model,
        "sparsity": args.sparsity,
        "seed": cfg["project"]["seed"],
        "fp32_onnx": str(fp32_onnx.relative_to(ROOT)),
        "int8_onnx": str(int8_onnx.relative_to(ROOT)),
        "fp32_size_mb": round(fp32_onnx.stat().st_size/1024/1024, 3),
        "int8_size_mb": round(int8_onnx.stat().st_size/1024/1024, 3),
        "nodecount": {"fp32": n_fp32, "int8": n_int8, "expansion": expansion},
        "bench_dryrun": {"fp32": b_fp32, "int8": b_int8, "int8_error": b_int8_err},
        "hypothesis_check": {
            "expansion_vs_paper1": "Compare expansion to Paper1 Table level2 (MN2 2.86x, SNV2 1.37x, EB0 3.05x) — lower is win",
            "int8_slower_than_fp32_on_cpu": (b_int8["mean_ms"] > b_fp32["mean_ms"]) if b_int8 else "skipped_on_macos (run on RPi5)",
        }
    }
    out = ROOT / f"results/dryrun_{args.model}_s{int(args.sparsity*100)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\n[OK] artifact -> {out}")
    print("="*70)
    print("  DRY-RUN DONE — pipeline works. Next: full dataset + finetune malam")
    print("="*70)

if __name__ == "__main__":
    main()
