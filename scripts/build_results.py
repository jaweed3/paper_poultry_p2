#!/usr/bin/env python3
"""Jalur A1: build results/results.json tunggal (SSOT untuk Tabel I + gambar).

Menggabungkan:
- results/eval_gate0.json (test steril FP32+INT8, per-class, CM, latency x86-ref, size)
- results/splits_manifest.json (sterile-8153 counts)
- results/phash_dedup.json -> stats ringkas (tidak bawa 5k grup)
- results/sterile_split_manifest.json -> strategy + counts
- logs/*_gate0_history.json -> best_val_acc, total_time, epoch_time
- runtime provenance: git hash, ORT/onnx/torch versi lab, seed, thread info

Output: results/results.json
Lalu scripts/gen_table1.py membaca results.json -> Tabel I LaTeX + markdown.
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

def load(p):
    with open(RES / p) as f:
        return json.load(f)

def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"

def main():
    eval0 = load("eval_gate0.json")
    splits = load("splits_manifest.json")
    sterile = load("sterile_split_manifest.json")
    phash = load("phash_dedup.json")

    # history ringkas (logs di-ignore git, tapi dibaca kalau ada)
    hist = {}
    for m in ["mobilenetv2", "shufflenetv2", "efficientnet_b0"]:
        hp = ROOT / "logs" / f"{m}_gate0_history.json"
        if hp.exists():
            h = json.loads(hp.read_text())
            hist[m] = {
                "best_val_acc": h["best_val_acc"],
                "total_time_s": h["total_time_s"],
                "finished": h["finished"],
                "seed": h.get("seed", 42),
            }

    out = {
        "meta": {
            "dataset": "sterile-8153 (dedup dari Dianyo/poultry-fecal-fl 8770)",
            "git_hash": git_hash(),
            "seed": 42,
            "protocol_train": "50ep Adam 1e-4 cosine, batch 32, 224px, ImageNet init",
            "protocol_quant": "ORT dynamic PTQ (MatMul-only), CPU",
            "runtime_lab": {"onnxruntime": "1.29.0", "onnx": "1.22.0",
                            "torch": "2.5.1+cu121", "gpu": "RTX 4060",
                            "python": "3.12.14"},
            "note_latency": "latency_ms = referensi x86 WSL lab, BUKAN angka paper. "
                            "Angka paper tetap dari RPi5 + x86 bench khusus.",
            "leakage_disclosure": {
                "publisher_claim": "8770 deduplicated leak-free (FecalFed, Chi 2026)",
                "audit": "pHash Hamming<=5 over 8770 files",
                "near_pairs": 617, "exact_pairs": 393,
                "cross_split_groups": 316,
                "byte_identical_cross": 203,
                "test_identical_to_train": 95,
                "test_identical_to_val": 30,
                "est_unique": 8153,
            },
        },
        "split": {
            "train": splits["actual"]["train"],
            "val": splits["actual"]["val"],
            "test": splits["actual"]["test"],
            "strategy": sterile["strategy"],
        },
        "models": {},
    }
    for m, r in eval0["models"].items():
        out["models"][m] = {
            "params": r["params"],
            "width": {"shufflenetv2": "x1.0"}.get(m, "standard"),
            "fp32_test_acc": r["fp32_test_acc"],
            "fp32_per_class": r["fp32_per_class"],
            "fp32_confusion": r["fp32_confusion"],
            "int8_test_acc": r["int8_test_acc"],
            "int8_per_class": r["int8_per_class"],
            "int8_confusion": r["int8_confusion"],
            "acc_drop_pp": r["acc_drop_pp"],
            "fp32_lat_ms_ref": r["fp32_lat_ms"],
            "int8_lat_ms_ref": r["int8_lat_ms"],
            "fp32_size_mb": r["fp32_size_mb"],
            "int8_size_mb": r["int8_size_mb"],
            "best_val_acc": hist.get(m, {}).get("best_val_acc"),
            "train_time_s": hist.get(m, {}).get("total_time_s"),
        }

    with open(RES / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {RES / 'results.json'}")
    for m, r in out["models"].items():
        print(f"  {m}: FP32 {r['fp32_test_acc']}% -> INT8 {r['int8_test_acc']}% "
              f"(drop {r['acc_drop_pp']}pp)")

if __name__ == "__main__":
    main()
