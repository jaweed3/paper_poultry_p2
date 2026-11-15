#!/usr/bin/env python3
"""Benchmark baseline and pruned models on Raspberry Pi 5.
Measures latency (mean, std, P50, P95, P99), throughput (FPS), and peak RSS memory.
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

def get_board_state():
    try:
        temp = subprocess.getoutput("vcgencmd measure_temp 2>/dev/null").strip()
        throttled = subprocess.getoutput("vcgencmd get_throttled 2>/dev/null").strip()
        return {"temp": temp, "throttled": throttled}
    except Exception:
        return {}

def benchmark_worker(model_path, threads, runs, warmup, batch=1):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    
    inp_name = sess.get_inputs()[0].name
    inp_shape = [batch, 3, 224, 224]
    dummy = np.random.randn(*inp_shape).astype(np.float32)
    
    # Warmup
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
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 # on Linux, ru_maxrss is in KB
    
    res = {
        "runs": runs,
        "mean_ms": round(float(arr.mean()), 3),
        "std_ms": round(float(arr.std()), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "fps": round(1000.0 / float(arr.mean()), 2),
        "peak_rss_mb": round(float(peak_rss_mb), 2),
        "wall_s": round(wall, 2)
    }
    print(json.dumps(res))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", type=str)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--runs", type=int, default=150)
    parser.add_argument("--warmup", type=int, default=15)
    args = parser.parse_args()
    
    if args.worker:
        benchmark_worker(args.model, args.threads, args.runs, args.warmup)
        sys.exit(0)
        
    # Orchestrator
    root = Path(__file__).resolve().parent
    models_to_test = {
        # Baseline
        "mobilenetv2_fp32": "models/onnx/mobilenetv2_gate0_fp32.onnx",
        "mobilenetv2_int8": "models/ptq/mobilenetv2_gate0_int8.onnx",
        "efficientnet_b0_fp32": "models/onnx/efficientnet_b0_gate0_fp32.onnx",
        "efficientnet_b0_int8": "models/ptq/efficientnet_b0_gate0_int8.onnx",
        "efficientnet_b0_static_int8": "models/ptq/efficientnet_b0_gate0_static_per_channel_minmax.onnx",
        "shufflenetv2_fp32": "models/onnx/shufflenetv2_gate0_fp32.onnx",
        "shufflenetv2_int8": "models/ptq/shufflenetv2_gate0_int8.onnx",
        # Pruned s30
        "mobilenetv2_s30_fp32": "models/pruned/s30/mobilenetv2.onnx",
        "mobilenetv2_s30_int8": "models/pruned/s30/mobilenetv2_int8.onnx",
        "efficientnet_b0_s30_fp32": "models/pruned/s30/efficientnet_b0.onnx",
        "efficientnet_b0_s30_static_int8": "models/pruned/s30/efficientnet_b0_static_int8.onnx",
        "shufflenetv2_s30_fp32": "models/pruned/s30/shufflenetv2.onnx",
        "shufflenetv2_s30_int8": "models/pruned/s30/shufflenetv2_int8.onnx",
        # Pruned s50
        "mobilenetv2_s50_fp32": "models/pruned/s50/mobilenetv2.onnx",
        "mobilenetv2_s50_int8": "models/pruned/s50/mobilenetv2_int8.onnx",
        "efficientnet_b0_s50_fp32": "models/pruned/s50/efficientnet_b0.onnx",
        "efficientnet_b0_s50_int8": "models/pruned/s50/efficientnet_b0_int8.onnx",
    }
    
    results = {
        "device": "Raspberry Pi 5 Model B Rev 1.0 (BCM2712, 4x Cortex-A76 @ 2.4GHz)",
        "os": subprocess.getoutput("uname -srm"),
        "initial_board_state": get_board_state(),
        "benchmarks": {}
    }
    
    thread_settings = [1, 4]
    
    for key, rel_path in models_to_test.items():
        full_path = root / rel_path
        if not full_path.exists():
            print(f"Skipping {key}: file {full_path} not found")
            continue
            
        file_size_mb = round(os.path.getsize(full_path) / (1024 * 1024), 2)
        results["benchmarks"][key] = {
            "path": str(rel_path),
            "file_size_mb": file_size_mb,
            "threads": {}
        }
        
        print(f"=== Benchmarking {key} ({file_size_mb} MB) ===")
        for th in thread_settings:
            cmd = [
                sys.executable,
                str(__file__),
                "--worker",
                "--model", str(full_path),
                "--threads", str(th),
                "--runs", str(args.runs),
                "--warmup", str(args.warmup)
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode == 0:
                try:
                    bench_res = json.loads(proc.stdout.strip().split("\n")[-1])
                    results["benchmarks"][key]["threads"][f"th{th}"] = bench_res
                    print(f"  Threads {th}: mean={bench_res['mean_ms']}ms, P50={bench_res['p50_ms']}ms, P99={bench_res['p99_ms']}ms, FPS={bench_res['fps']}, RSS={bench_res['peak_rss_mb']}MB")
                except Exception as e:
                    print(f"  Threads {th} parse error: {e}, stdout: {proc.stdout}")
            else:
                print(f"  Threads {th} failed: {proc.stderr}")
                
    results["final_board_state"] = get_board_state()
    out_file = root / "results" / "rpi5_pruned_benchmark.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll benchmarks finished! Saved to {out_file}")
