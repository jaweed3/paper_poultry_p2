#!/usr/bin/env python3
"""
benchmark_rpi5_pruned.py — fork of benchmark_rpi5.py for Paper 2.

Adds --sparsity arg, otherwise delegates to benchmark_rpi5.py logic.
Output: results/rpi5_benchmark/rpi5_pruned_s{Sparsity}_{model}_{quant}_t{threads}_b{batch}.json
Keeps schema compatible with rpi5_benchmark_results.json for diffing.

Usage:
  python scripts/benchmark_rpi5_pruned.py --model mobilenetv2 --sparsity 0.5 --quant int8 --threads 4 --batch 1

For now this is a scaffold — actual ORT benchmarking logic will be copied from
benchmark_rpi5.py after sparsity pruning pipeline (scripts/prune.py) is ready.
"""
import argparse, json, pathlib, sys

def main():
    p = argparse.ArgumentParser(description="RPi5 pruned benchmark (Paper 2 scaffold)")
    p.add_argument("--model", choices=["mobilenetv2","shufflenetv2","efficientnet_b0"], required=True)
    p.add_argument("--sparsity", type=float, default=0.5, help="structured sparsity 0.0-0.7")
    p.add_argument("--quant", choices=["fp32","int8"], default="int8")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()
    out = pathlib.Path(f"results/rpi5_benchmark/rpi5_pruned_s{int(args.sparsity*100)}_{args.model}_{args.quant}_t{args.threads}_b{args.batch}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Scaffold: write placeholder that will be overwritten by real benchmark
    placeholder = {
        "model": args.model, "sparsity": args.sparsity, "quant": args.quant,
        "threads": args.threads, "batch": args.batch,
        "iters": args.iters, "warmup": args.warmup,
        "status": "scaffold — not yet run (needs pruned ONNX)",
        "note": "Run scripts/prune.py first to generate pruned ONNX, then this script benchmarks it via ORT CPUExecutionProvider"
    }
    out.write_text(json.dumps(placeholder, indent=2))
    print(f"[scaffold] wrote {out}")
    print("Next: implement prune.py, then copy ORT loop from benchmark_rpi5.py into this file")

if __name__ == "__main__":
    main()
