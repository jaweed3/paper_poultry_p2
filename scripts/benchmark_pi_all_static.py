#!/usr/bin/env python3
"""Benchmark Static QDQ INT8 models on Raspberry Pi 5 Cortex-A76.
Measures latency (mean, std, p50, p95, p99), FPS, and thermal state
for MobileNetV2, ShuffleNetV2, and EfficientNet-B0 static per-channel QDQ models.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

MODELS = {
    "mbv2_fp32": "models/onnx/mobilenetv2_gate0_fp32.onnx",
    "mbv2_int8_dynamic": "models/ptq/mobilenetv2_gate0_int8.onnx",
    "mbv2_static_per_channel": "models/ptq/mobilenetv2_gate0_static_per_channel_minmax.onnx",
    
    "snv2_fp32": "models/onnx/shufflenetv2_gate0_fp32.onnx",
    "snv2_int8_dynamic": "models/ptq/shufflenetv2_gate0_int8.onnx",
    "snv2_static_per_channel": "models/ptq/shufflenetv2_gate0_static_per_channel_minmax.onnx",
    
    "eb0_fp32": "models/onnx/efficientnet_b0_gate0_fp32.onnx",
    "eb0_int8_dynamic": "models/ptq/efficientnet_b0_gate0_int8.onnx",
    "eb0_static_per_channel": "models/ptq/efficientnet_b0_gate0_static_per_channel_minmax.onnx",
}

THREADS = [1, 4]
BATCH = 1
OUT_DIR = ROOT / "results" / "rpi5_static_all"
OUT_JSON = OUT_DIR / "rpi5_static_all.json"


def board_state():
    try:
        temp = subprocess.getoutput("vcgencmd measure_temp 2>/dev/null")
        throttled = subprocess.getoutput("vcgencmd get_throttled 2>/dev/null")
        return {"temp_raw": temp.strip(), "throttled": throttled.strip()}
    except Exception:
        return {}


def ort_version():
    import onnxruntime as ort
    return getattr(ort, "__version__", "unknown")


def cpu_features():
    import re
    txt = Path("/proc/cpuinfo").read_text()
    m = re.search(r"^Features\s*:\s*(.+)$", txt, re.M)
    feats = set(m.group(1).split()) if m else set()
    interesting = ["asimd", "sve", "i8mm", "bf16", "asimddp", "asimdfhm", "sme"]
    return {f: (f in feats) for f in interesting}


def run_one(path, threads, runs, warmup, batch=1):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    shape = [batch] + list(sess.get_inputs()[0].shape[1:])
    x = np.random.rand(*shape).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {inp: x})
    lat = []
    t_start = time.perf_counter()
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {inp: x})
        lat.append((time.perf_counter() - t0) * 1000.0)
    wall = time.perf_counter() - t_start

    a = np.array(lat)
    return {
        "n": runs,
        "mean_ms": round(float(a.mean()), 2),
        "std_ms": round(float(a.std()), 2),
        "p50_ms": round(float(np.percentile(a, 50)), 2),
        "p95_ms": round(float(np.percentile(a, 95)), 2),
        "p99_ms": round(float(np.percentile(a, 99)), 2),
        "fps": round(1000.0 / float(a.mean()), 1),
        "wall_s": round(wall, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.runs, a.warmup = 15, 3

    missing = [p for p in MODELS.values() if not (ROOT / p).exists()]
    if missing:
        print("[ERROR] missing models:")
        for m in missing:
            print("  -", m)
        sys.exit(1)

    print(f"ORT {ort_version()} | runs={a.runs} warmup={a.warmup} threads={a.threads}")
    print("CPU features:", json.dumps(cpu_features()))
    print("board start:", board_state())

    results = {
        "meta": {
            "ort_version": ort_version(),
            "runs": a.runs,
            "warmup": a.warmup,
            "threads": a.threads,
            "batch": BATCH,
            "cpu_features": cpu_features(),
            "board_start": board_state(),
        },
        "configs": {},
    }

    order = list(MODELS.keys())
    for name in order:
        rel = MODELS[name]
        size_mb = (ROOT / rel).stat().st_size / 1024 / 1024
        for th in a.threads:
            print(f"\n[{name}] threads={th} size={size_mb:.2f}MB")
            st_before = board_state()
            m = run_one(ROOT / rel, th, a.runs, a.warmup, BATCH)
            st_after = board_state()
            key = f"{name}_t{th}"
            results["configs"][key] = {
                "model": name,
                "path": rel,
                "size_mb": round(size_mb, 2),
                "threads": th,
                **m,
                "board_before": st_before,
                "board_after": st_after,
            }
            print(f"  mean={m['mean_ms']}ms p95={m['p95_ms']}ms fps={m['fps']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
