#!/usr/bin/env python3
"""
scripts/benchmark_pi_canonical.py
Single Source of Truth (SSOT) Physical Edge Benchmark on Raspberry Pi 5.
Runs 15 warmups followed by 100 timed iterations per configuration.
Records exact mean, std, p50, p95, p99, fps, and peak_rss_mb for both 1T and 4T.
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

MODELS = {
    # MobileNetV2
    "mobilenetv2_fp32": "models/onnx/mobilenetv2_gate0_fp32.onnx",
    "mobilenetv2_dyn_int8": "models/ptq/mobilenetv2_gate0_int8.onnx",
    "mobilenetv2_static_int8": "models/ptq/mobilenetv2_gate0_static_per_channel_minmax.onnx",
    "mobilenetv2_s30_fp32": "models/pruned/s30/mobilenetv2.onnx",
    "mobilenetv2_s30_dyn_int8": "models/pruned/s30/mobilenetv2_int8.onnx",
    "mobilenetv2_s50_fp32": "models/pruned/s50/mobilenetv2.onnx",
    "mobilenetv2_s50_dyn_int8": "models/pruned/s50/mobilenetv2_int8.onnx",

    # EfficientNet-B0
    "efficientnet_b0_fp32": "models/onnx/efficientnet_b0_gate0_fp32.onnx",
    "efficientnet_b0_dyn_int8": "models/ptq/efficientnet_b0_gate0_int8.onnx",
    "efficientnet_b0_static_int8": "models/ptq/efficientnet_b0_gate0_static_per_channel_minmax.onnx",
    "efficientnet_b0_s30_fp32": "models/pruned/s30/efficientnet_b0.onnx",
    "efficientnet_b0_s30_static_int8": "models/pruned/s30/efficientnet_b0_static_int8.onnx",
    "efficientnet_b0_s50_fp32": "models/pruned/s50/efficientnet_b0.onnx",

    # ShuffleNetV2
    "shufflenetv2_fp32": "models/onnx/shufflenetv2_gate0_fp32.onnx",
    "shufflenetv2_dyn_int8": "models/ptq/shufflenetv2_gate0_int8.onnx",
    "shufflenetv2_static_int8": "models/ptq/shufflenetv2_gate0_static_per_channel_minmax.onnx",
    "shufflenetv2_s30_fp32": "models/pruned/s30/shufflenetv2.onnx",
}

def get_board_state():
    try:
        temp = subprocess.getoutput("vcgencmd measure_temp 2>/dev/null").strip()
        throttled = subprocess.getoutput("vcgencmd get_throttled 2>/dev/null").strip()
        return {"temp": temp, "throttled": throttled}
    except Exception:
        return {}

def benchmark_single(model_path, threads, runs=100, warmup=15):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    
    inp_name = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    for _ in range(warmup):
        sess.run(None, {inp_name: dummy})
        
    latencies = []
    t_start = time.perf_counter()
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {inp_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000.0)
    wall = time.perf_counter() - t_start
    
    arr = np.array(latencies)
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    
    return {
        "runs": runs,
        "mean_ms": round(float(arr.mean()), 2),
        "std_ms": round(float(arr.std()), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "fps": round(1000.0 / float(arr.mean()), 2),
        "peak_rss_mb": round(float(peak_rss_mb), 2),
        "wall_s": round(wall, 2)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", type=str)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=15)
    args = parser.parse_args()

    if args.worker:
        res = benchmark_single(args.model, args.threads, args.runs, args.warmup)
        print(json.dumps(res))
        sys.exit(0)

    root = Path(__file__).resolve().parent
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "rpi5_canonical_benchmark.json"

    print("=== Physical Edge Hardware Benchmarking on Raspberry Pi 5 ===")
    board_init = get_board_state()
    print(f"Board State Start: {board_init}")

    all_results = {
        "device": "Raspberry Pi 5 Model B Rev 1.0 (BCM2712, 4x Cortex-A76 @ 2.4GHz)",
        "board_start": board_init,
        "models": {}
    }

    for key, rel_path in MODELS.items():
        model_path = root / rel_path
        if not model_path.exists():
            print(f"WARNING: {model_path} not found, skipping {key}")
            continue
        file_size_mb = round(os.path.getsize(model_path) / (1024 * 1024), 2)
        all_results["models"][key] = {
            "path": rel_path,
            "file_size_mb": file_size_mb,
            "th1": None,
            "th4": None,
        }
        print(f"\n[{key}] Size: {file_size_mb} MB")
        for th in [1, 4]:
            cmd = [
                sys.executable, str(__file__),
                "--worker",
                "--model", str(model_path),
                "--threads", str(th),
                "--runs", str(args.runs),
                "--warmup", str(args.warmup),
            ]
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if p.returncode == 0:
                data = json.loads(p.stdout.strip().split("\n")[-1])
                all_results["models"][key][f"th{th}"] = data
                print(f"  {th}T: Mean={data['mean_ms']}ms, P50={data['p50_ms']}ms, P95={data['p95_ms']}ms, FPS={data['fps']}, RSS={data['peak_rss_mb']}MB")
            else:
                print(f"  {th}T FAILED: {p.stderr}")

    all_results["board_end"] = get_board_state()
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSUCCESS: Canonical benchmark written to {out_file}")
