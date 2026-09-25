#!/usr/bin/env python3
"""A4: Pi5 benchmark for sterile gate0 ONNX (sweep dipangkas).

12 configs: 3 models x {FP32, INT8-dynamic} x threads {1,4} x batch {1}.
200 runs + 10 warmup per config (sama kayak paper).
Output: results/rpi5_benchmark_gate0/rpi5_gate0.json

Usage on Pi:
  ~/bench-venv/bin/python benchmark_pi_gate0.py --runs 200 --warmup 10
  ~/bench-venv/bin/python benchmark_pi_gate0.py --runs 20 --warmup 5  # smoke
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import benchmark_rpi5 as B

B.MODEL_CONFIGS = [
    ("mobilenetv2", "FP32", "models/onnx/mobilenetv2_gate0_fp32.onnx"),
    ("mobilenetv2", "INT8", "models/ptq/mobilenetv2_gate0_int8.onnx"),
    ("shufflenetv2", "FP32", "models/onnx/shufflenetv2_gate0_fp32.onnx"),
    ("shufflenetv2", "INT8", "models/ptq/shufflenetv2_gate0_int8.onnx"),
    ("efficientnet_b0", "FP32", "models/onnx/efficientnet_b0_gate0_fp32.onnx"),
    ("efficientnet_b0", "INT8", "models/ptq/efficientnet_b0_gate0_int8.onnx"),
]
B.RESULTS_DIR = Path("results/rpi5_benchmark_gate0")
B.THREAD_COUNTS = [1, 4]
B.BATCH_SIZES = [1]
B.NUM_RUNS = 200
B.NUM_WARMUP = 10

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    a = ap.parse_args()
    B.NUM_RUNS = a.runs
    B.NUM_WARMUP = a.warmup
    B.THREAD_COUNTS = a.threads

    class Args:
        download = False
        force_download = False
    B.run_benchmark(Args())
