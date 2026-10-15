#!/usr/bin/env python3
"""
Hardware Benchmark Script for Raspberry Pi 5
=============================================
Benchmarks FP32 and INT8 ONNX models across multiple configurations:
- Thread count: 1, 2, 3, 4
- Batch size: 1, 4, 8, 16

Metrics: Latency (mean/P50/P95/P99), FPS, Memory (RSS), CPU%, Power, Temperature
"""

import os
import sys
import json
import time
import subprocess
import argparse
import tarfile
import urllib.request
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[WARN] psutil not installed. Memory/CPU metrics will be limited.")
    print("       Install: pip install psutil")

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] onnxruntime not installed. Install: pip install onnxruntime")
    sys.exit(1)


# ─── Configuration ───────────────────────────────────────────────────────────

RELEASE_URL = "https://anonymous.4open.science/r/paper_poultry_p2-C1F9/releases/models_all.tar.gz"  # anonymized mirror; original release URL redacted for blind review

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results/rpi5_benchmark")

MODEL_CONFIGS = [
    ("mobilenetv2", "FP32",   "models/onnx/mobilenetv2_fp32.onnx"),
    ("mobilenetv2", "INT8",   "models/ptq/mobilenetv2_int8.onnx"),
    ("shufflenetv2", "FP32",  "models/onnx/shufflenetv2_fp32.onnx"),
    ("shufflenetv2", "INT8",  "models/ptq/shufflenetv2_int8.onnx"),
    ("efficientnet_b0", "FP32", "models/onnx/efficientnet_b0_fp32.onnx"),
    ("efficientnet_b0", "INT8", "models/ptq/efficientnet_b0_int8.onnx"),
]

THREAD_COUNTS = [1, 2, 3, 4]
BATCH_SIZES = [1, 4, 8, 16]
NUM_RUNS = 200
NUM_WARMUP = 10


# ─── Hardware Info ───────────────────────────────────────────────────────────

def get_system_info():
    """Collect Raspberry Pi 5 hardware and OS information."""
    info = {
        "hostname": subprocess.getoutput("hostname"),
        "os": subprocess.getoutput("cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'\"' -f2"),
        "kernel": subprocess.getoutput("uname -r"),
        "architecture": subprocess.getoutput("uname -m"),
        "cpu_model": _read_file("/proc/cpuinfo").split("\n")[0] if _read_file("/proc/cpuinfo") else "unknown",
        "cpu_cores": os.cpu_count(),
    }

    # Memory
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        info["ram_total_mb"] = round(mem.total / (1024 * 1024), 1)
        info["ram_available_mb"] = round(mem.available / (1024 * 1024), 1)

    # Thermal zone
    info["thermal_zone"] = _read_file("/sys/class/thermal/thermal_zone0/temp", strip=True)
    info["thermal_type"] = _read_file("/sys/class/thermal/thermal_zone0/type", strip=True)

    # vcgencmd
    for cmd, key in [
        ("vcgencmd measure_temp", "gpu_temp"),
        ("vcgencmd measure_volts core", "voltage_core"),
        ("vcgencmd get_throttled", "throttled"),
    ]:
        try:
            out = subprocess.getoutput(cmd)
            info[key] = out.strip() if out.strip() else "N/A"
        except Exception:
            info[key] = "N/A"

    # ONNX Runtime
    info["onnxruntime_version"] = ort.__version__
    info["providers"] = ort.get_available_providers()

    return info


def _read_file(path, strip=True):
    try:
        with open(path) as f:
            return f.read().strip() if strip else f.read()
    except (FileNotFoundError, PermissionError):
        return None


def _get_temperature():
    """Read current CPU temperature in Celsius."""
    raw = _read_file("/sys/class/thermal/thermal_zone0/temp", strip=True)
    if raw and raw.isdigit():
        return round(int(raw) / 1000.0, 1)
    try:
        out = subprocess.getoutput("vcgencmd measure_temp")
        # "temp=45.0'C" → 45.0
        return float(out.split("=")[1].split("'")[0])
    except Exception:
        return None


def _get_power_estimate():
    """
    Estimate power consumption via vcgencmd (RPi5 official PSU).
    Returns dict with voltage, current, power if available.
    """
    result = {"voltage_v": None, "current_a": None, "power_w": None}

    try:
        # vcgencmd measure_volts core
        voltag_out = subprocess.getoutput("vcgencmd measure_volts core")
        if "V" in voltag_out:
            result["voltage_v"] = float(voltag_out.split("=")[1].split("V")[0])
    except Exception:
        pass

    try:
        # vcgencmd pmic_read_adc 0 (RPi5 with PMIC)
        current_out = subprocess.getoutput("vcgencmd pmic_read_adc 0 2>/dev/null")
        if current_out.strip():
            # Parse current reading (varies by firmware)
            parts = current_out.split()
            for p in parts:
                try:
                    val = float(p.replace("A", "").replace("V", ""))
                    if val > 0.1:
                        result["current_a"] = val
                        break
                except ValueError:
                    continue
    except Exception:
        pass

    # Fallback: read from sysfs if available
    if result["current_a"] is None:
        for zone_path in Path("/sys/class/power_supply/").glob("*/current_now"):
            try:
                raw = _read_file(str(zone_path), strip=True)
                if raw and raw.isdigit():
                    result["current_a"] = round(int(raw) / 1_000_000.0, 3)
                    break
            except Exception:
                pass

    if result["voltage_v"] and result["current_a"]:
        result["power_w"] = round(result["voltage_v"] * result["current_a"], 3)

    return result


# ─── Model Download ──────────────────────────────────────────────────────────

def download_models(force=False):
    """Download and extract models from GitHub release."""
    tar_path = MODELS_DIR / "models_all.tar.gz"

    if tar_path.exists() and not force:
        print(f"[INFO] Models archive already exists: {tar_path}")
        print("       Use --force-download to re-download.")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading models from GitHub release...")
    print(f"       URL: {RELEASE_URL}")

    try:
        urllib.request.urlretrieve(RELEASE_URL, str(tar_path))
        print(f"[OK] Downloaded: {tar_path.stat().st_size / (1024*1024):.1f} MB")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("        Make sure models are in the models/ directory manually.")
        sys.exit(1)

    print(f"[INFO] Extracting to {MODELS_DIR}/...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=str(MODELS_DIR))
        print("[OK] Extraction complete.")
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        sys.exit(1)


# ─── Benchmark Core ──────────────────────────────────────────────────────────

def get_model_input_info(onnx_path):
    """Read input shape from ONNX model metadata."""
    session = ort.InferenceSession(str(onnx_path))
    input_meta = session.get_inputs()[0]
    return input_meta.name, input_meta.shape


def create_session(onnx_path, num_threads):
    """Create ONNX Runtime session with specified thread count."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Disable logging noise
    opts.log_severity_level = 3

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), opts, providers=providers)
    return session


def benchmark_single(session, input_name, dummy_input, num_runs, num_warmup):
    """
    Run benchmark for a single configuration.
    Returns latency array (ms) and per-run memory readings.
    """
    # Warmup
    for _ in range(num_warmup):
        session.run(None, {input_name: dummy_input})

    # Force sync before timing
    if HAS_PSUTIL:
        psutil.cpu_percent(interval=None)

    latencies = []
    mem_readings = []
    cpu_readings = []

    for i in range(num_runs):
        if HAS_PSUTIL:
            mem_before = psutil.Process().memory_info().rss
            cpu_snapshot = psutil.cpu_percent(interval=None)

        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        end = time.perf_counter()

        latencies.append((end - start) * 1000.0)  # ms

        if HAS_PSUTIL:
            mem_after = psutil.Process().memory_info().rss
            mem_readings.append(mem_after / (1024 * 1024))  # MB

    # Overall CPU% for the benchmark window
    if HAS_PSUTIL:
        cpu_overall = psutil.cpu_percent(interval=0.1)
        cpu_readings = [cpu_overall] * num_runs

    return {
        "latencies": np.array(latencies),
        "mem_readings": np.array(mem_readings) if mem_readings else None,
        "cpu_readings": np.array(cpu_readings) if cpu_readings else None,
    }


def benchmark_config(model_name, quant_type, onnx_path, num_threads, batch_size, num_runs, num_warmup):
    """Benchmark a single model/quantization/thread/batch configuration."""
    session = create_session(onnx_path, num_threads)
    input_name, original_shape = get_model_input_info(onnx_path)

    # Build input shape: replace dynamic dims with batch_size
    input_shape = []
    for d in original_shape:
        if isinstance(d, str) or d is None or d <= 0:
            input_shape.append(batch_size)
        else:
            input_shape.append(d)
    input_shape = tuple(input_shape)

    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Collect thermal + power before
    temp_before = _get_temperature()
    power = _get_power_estimate()

    # Run benchmark
    results = benchmark_single(session, input_name, dummy_input, num_runs, num_warmup)

    # Collect thermal after
    temp_after = _get_temperature()

    # Size on disk
    model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

    # Compile metrics
    lats = results["latencies"]
    output = {
        "model": model_name,
        "quantization": quant_type,
        "threads": num_threads,
        "batch_size": batch_size,
        "input_shape": list(input_shape),
        "model_size_mb": round(model_size_mb, 3),
        "latency_mean_ms": round(float(lats.mean()), 2),
        "latency_std_ms": round(float(lats.std()), 2),
        "latency_p50_ms": round(float(np.percentile(lats, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(lats, 95)), 2),
        "latency_p99_ms": round(float(np.percentile(lats, 99)), 2),
        "latency_min_ms": round(float(lats.min()), 2),
        "latency_max_ms": round(float(lats.max()), 2),
        "throughput_fps": round(1000.0 / float(lats.mean()), 2),
        "num_runs": num_runs,
    }

    # Memory
    if results["mem_readings"] is not None:
        mem = results["mem_readings"]
        output["memory_rss_mean_mb"] = round(float(mem.mean()), 2)
        output["memory_rss_peak_mb"] = round(float(mem.max()), 2)

    # CPU
    if results["cpu_readings"] is not None:
        output["cpu_percent"] = round(float(results["cpu_readings"].mean()), 1)

    # Thermal
    if temp_before is not None and temp_after is not None:
        output["temp_before_c"] = temp_before
        output["temp_after_c"] = temp_after

    # Power
    output["power"] = power

    # Cleanup
    del session

    return output


# ─── Main ────────────────────────────────────────────────────────────────────

def run_benchmark(args):
    """Main benchmark loop."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # System info
    print("=" * 70)
    print("  RASPBERRY PI 5 — ONNX Model Hardware Benchmark")
    print("=" * 70)

    sys_info = get_system_info()
    print(f"\n[SYSTEM] Hostname  : {sys_info['hostname']}")
    print(f"[SYSTEM] OS        : {sys_info['os']}")
    print(f"[SYSTEM] Kernel    : {sys_info['kernel']}")
    print(f"[SYSTEM] Arch      : {sys_info['architecture']}")
    print(f"[SYSTEM] CPU Cores : {sys_info['cpu_cores']}")
    if "ram_total_mb" in sys_info:
        print(f"[SYSTEM] RAM       : {sys_info['ram_total_mb']} MB")
    print(f"[SYSTEM] ONNX RT   : {sys_info['onnxruntime_version']}")
    print(f"[SYSTEM] Providers : {sys_info['providers']}")
    print(f"[SYSTEM] Temp      : {_get_temperature()}°C")
    print()

    # Save system info
    sys_info["temp_initial_c"] = _get_temperature()
    sys_info["power_initial"] = _get_power_estimate()
    sys_info["timestamp"] = datetime.now().isoformat()
    with open(RESULTS_DIR / "rpi5_system_info.json", "w") as f:
        json.dump(sys_info, f, indent=2)

    # Download models
    if args.download or args.force_download:
        download_models(force=args.force_download)

    # Filter available models
    available_configs = []
    for model_name, quant_type, path in MODEL_CONFIGS:
        full_path = Path(path)
        if not full_path.exists():
            print(f"[WARN] Model not found, skipping: {full_path}")
            continue
        available_configs.append((model_name, quant_type, full_path))

    if not available_configs:
        print("\n[ERROR] No models found! Run with --download first.")
        sys.exit(1)

    print(f"\n[INFO] Found {len(available_configs)} model(s)")
    print(f"[INFO] Thread counts : {THREAD_COUNTS}")
    print(f"[INFO] Batch sizes   : {BATCH_SIZES}")
    print(f"[INFO] Runs per config: {NUM_RUNS} (+ {NUM_WARMUP} warmup)")

    total_configs = len(available_configs) * len(THREAD_COUNTS) * len(BATCH_SIZES)
    print(f"[INFO] Total configurations: {total_configs}")
    print()

    # Benchmark loop
    all_results = []
    config_idx = 0

    for model_name, quant_type, onnx_path in available_configs:
        for num_threads in THREAD_COUNTS:
            for batch_size in BATCH_SIZES:
                config_idx += 1
                print(
                    f"[{config_idx}/{total_configs}] "
                    f"{model_name} ({quant_type}) | "
                    f"threads={num_threads} | batch={batch_size} ... ",
                    end="", flush=True
                )

                try:
                    result = benchmark_config(
                        model_name=model_name,
                        quant_type=quant_type,
                        onnx_path=onnx_path,
                        num_threads=num_threads,
                        batch_size=batch_size,
                        num_runs=NUM_RUNS,
                        num_warmup=NUM_WARMUP,
                    )
                    all_results.append(result)
                    print(
                        f"{result['latency_mean_ms']:>8.2f} ms | "
                        f"{result['throughput_fps']:>7.2f} FPS | "
                        f"{result.get('temp_after_c', '?')}°C"
                    )
                except Exception as e:
                    print(f"ERROR: {e}")
                    all_results.append({
                        "model": model_name,
                        "quantization": quant_type,
                        "threads": num_threads,
                        "batch_size": batch_size,
                        "error": str(e),
                    })

    # Save results
    output_data = {
        "experiment": "RPi5 Hardware Benchmark",
        "device": "Raspberry Pi 5",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_runs": NUM_RUNS,
            "num_warmup": NUM_WARMUP,
            "thread_counts": THREAD_COUNTS,
            "batch_sizes": BATCH_SIZES,
        },
        "system_info": sys_info,
        "results": all_results,
    }

    results_path = RESULTS_DIR / "rpi5_benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[OK] Results saved: {results_path}")

    # Print summary table
    print_summary_table(all_results)

    # Print speedup analysis (INT8 vs FP32)
    print_speedup_table(all_results)

    print(f"\n[DONE] Benchmark complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def print_summary_table(results):
    """Print a formatted summary table."""
    print("\n" + "=" * 110)
    print("  SUMMARY TABLE")
    print("=" * 110)
    header = f"{'Model':<18} {'Quant':<6} {'Threads':<8} {'Batch':<6} {'Size(MB)':<10} {'Lat(ms)':<10} {'P50(ms)':<10} {'P95(ms)':<10} {'FPS':<10} {'RSS(MB)':<10}"
    print(header)
    print("-" * 110)

    for r in results:
        if "error" in r:
            continue
        print(
            f"{r['model']:<18} "
            f"{r['quantization']:<6} "
            f"{r['threads']:<8} "
            f"{r['batch_size']:<6} "
            f"{r['model_size_mb']:<10.2f} "
            f"{r['latency_mean_ms']:<10.2f} "
            f"{r['latency_p50_ms']:<10.2f} "
            f"{r['latency_p95_ms']:<10.2f} "
            f"{r['throughput_fps']:<10.2f} "
            f"{r.get('memory_rss_mean_mb', 'N/A')}"
        )
    print("=" * 110)


def print_speedup_table(results):
    """Print INT8 vs FP32 speedup comparison."""
    print("\n" + "=" * 80)
    print("  INT8 vs FP32 SPEEDUP ANALYSIS")
    print("=" * 80)
    header = f"{'Model':<18} {'Threads':<8} {'Batch':<6} {'FP32(ms)':<12} {'INT8(ms)':<12} {'Speedup':<10} {'FP32 FPS':<12} {'INT8 FPS':<12}"
    print(header)
    print("-" * 80)

    # Group by model+threads+batch
    fp32_lookup = {}
    for r in results:
        if "error" in r or r["quantization"] != "FP32":
            continue
        key = (r["model"], r["threads"], r["batch_size"])
        fp32_lookup[key] = r

    for r in results:
        if "error" in r or r["quantization"] != "INT8":
            continue
        key = (r["model"], r["threads"], r["batch_size"])
        fp32 = fp32_lookup.get(key)
        if not fp32:
            continue

        speedup = fp32["latency_mean_ms"] / r["latency_mean_ms"] if r["latency_mean_ms"] > 0 else 0
        marker = "✓" if speedup > 1 else "✗ (slower)"

        print(
            f"{r['model']:<18} "
            f"{r['threads']:<8} "
            f"{r['batch_size']:<6} "
            f"{fp32['latency_mean_ms']:<12.2f} "
            f"{r['latency_mean_ms']:<12.2f} "
            f"{speedup:<10.2f}x "
            f"{fp32['throughput_fps']:<12.2f} "
            f"{r['throughput_fps']:<12.2f} "
            f" {marker}"
        )

    print("=" * 80)
    print("  Speedup > 1.0x = INT8 is faster | Speedup < 1.0x = INT8 is slower (dequant overhead)")
    print("=" * 80)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX models on Raspberry Pi 5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_rpi5.py --download              # Download models + benchmark
  python benchmark_rpi5.py                          # Benchmark only (models in models/)
  python benchmark_rpi5.py --download --force       # Force re-download + benchmark
  python benchmark_rpi5.py --runs 100 --warmup 5    # Quick benchmark
  python benchmark_rpi5.py --threads 1 2            # Only test 1 and 2 threads
  python benchmark_rpi5.py --batches 1 4            # Only test batch 1 and 4
        """,
    )
    parser.add_argument("--download", action="store_true", help="Download models before benchmark")
    parser.add_argument("--force-download", action="store_true", help="Force re-download even if models exist")
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help=f"Inference runs per config (default: {NUM_RUNS})")
    parser.add_argument("--warmup", type=int, default=NUM_WARMUP, help=f"Warmup iterations (default: {NUM_WARMUP})")
    parser.add_argument("--threads", type=int, nargs="+", default=THREAD_COUNTS, help=f"Thread counts to test (default: {THREAD_COUNTS})")
    parser.add_argument("--batches", type=int, nargs="+", default=BATCH_SIZES, help=f"Batch sizes to test (default: {BATCH_SIZES})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override globals from CLI args
    NUM_RUNS = args.runs
    NUM_WARMUP = args.warmup
    THREAD_COUNTS = args.threads
    BATCH_SIZES = args.batches

    run_benchmark(args)
