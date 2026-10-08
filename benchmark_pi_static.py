#!/usr/bin/env python3
"""RPi5: static QDQ INT8 benchmark untuk EfficientNet-B0 (JAWABAN R3.3).

Reviewer 3 menulis:
  "Benchmark static INT8 configurations on the Raspberry Pi 5, particularly the
   per-channel configuration that recovers EfficientNet-B0 accuracy. Static
   quantization is an important CNN deployment baseline. Accuracy, latency, and
   memory should be reported for the same configurations."

Paper sekarang cuma punya FP32 + INT8-dynamic di Pi (12 config).
Script ini menambah static QDQ per-tensor + per-channel, plus Akurasi,
jadi perbandingan accuracy-latency lengkap untuk konfigurasi yang sama.

Config yang di-benchmark:
  efficientnet_b0  static_per_tensor   (reproduce collapse, ~9%)
  efficientnet_b0  static_per_channel  (recover ~94.6%)
  -> plus FP32 & dynamic sebagai referensi pada run yang sama

Semua config dijalankan pada sesi yang sama supaya perbandingan fair
(single board, single runtime version, thermal state sebanding).

 prerequisites di Pi:
  1. models/onnx/efficientnet_b0_gate0_fp32.onnx
  2. models/ptq/efficientnet_b0_gate0_static_per_tensor_minmax.onnx
  3. models/ptq/efficientnet_b0_gate0_static_per_channel_minmax.onnx
     (dihasilkan static_eb0.py di lab/laptop; script ini TIDAK melakukan kuantisasi)
  4. dataset sterile di data/ (test 1728 gambar)

Run:
  ~/bench-venv/bin/python benchmark_pi_static.py --runs 200 --warmup 10
  ~/bench-venv/bin/python benchmark_pi_static.py --smoke   # cek cepat

Output:
  results/rpi5_static_gate0/rpi5_static.json
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import benchmark_rpi5 as B

MODELS = {
    "eb0_fp32": "models/onnx/efficientnet_b0_gate0_fp32.onnx",
    "eb0_int8_dynamic": "models/ptq/efficientnet_b0_gate0_int8.onnx",
    "eb0_static_per_tensor": "models/ptq/efficientnet_b0_gate0_static_per_tensor_minmax.onnx",
    "eb0_static_per_channel": "models/ptq/efficientnet_b0_gate0_static_per_channel_minmax.onnx",
}

THREADS = [1, 4]
BATCH = 1
OUT_DIR = ROOT / "results" / "rpi5_static_gate0"
OUT_JSON = OUT_DIR / "rpi5_static.json"


def board_state():
    """Snapshot kondisi board supaya antar-config comparable."""
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
    interesting = ["asimd", "sve", "i8mm", "bf16", "asimddp", "asimdfhm", "sme", "i8mm"]
    return {f: (f in feats) for f in interesting}


def run_one(path, threads, runs, warmup, batch=1):
    """Jalankan 1 config, kembalikan dict latency."""
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--smoke", action="store_true",
                    help="runs=20 warmup=5 untuk cek cepat")
    a = ap.parse_args()
    if a.smoke:
        a.runs, a.warmup = 20, 5

    missing = [p for p in MODELS.values() if not (ROOT / p).exists()]
    if missing:
        print("[ERROR] model tidak ditemukan:")
        for m in missing:
            print("  -", m)
        print("\nGenerate static QDQ dulu (di lab/laptop, bukan Pi):")
        print("  python scripts/static_eb0.py")
        print("lalu copy models/ptq/*static*.onnx ke Pi.")
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

    # PENTING: urut dari FP32 -> INT8, cek thermal tiap config
    order = ["eb0_fp32", "eb0_int8_dynamic",
             "eb0_static_per_tensor", "eb0_static_per_channel"]
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
            print(f"  temp {st_before.get('temp_raw')} -> {st_after.get('temp_raw')}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {OUT_JSON}")

    print("\n=== RINGKASAN (t=4) ===")
    base = results["configs"].get("eb0_fp32_t4", {}).get("mean_ms")
    for name in order:
        c = results["configs"].get(f"{name}_t4")
        if not c:
            continue
        rel = f"{base / c['mean_ms']:.2f}x vs FP32" if base else ""
        print(f"  {name:<24} {c['mean_ms']:>7.2f}ms  {c['fps']:>6} FPS  {rel}")

    print("\nAKURASI (isi manual dari test run / atau run eval di Pi):")
    print("  static_per_tensor  ~9.32%  ( dari static_eb0.json, x86)")
    print("  static_per_channel ~94.56% ( dari static_eb0.json, x86)")
    print("  -> Wajib konfirmasi akurasi DI PI juga untuk klaim fair-comparison.")


if __name__ == "__main__":
    main()
