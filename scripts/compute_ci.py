#!/usr/bin/env python3
"""Jalur A2: bootstrap 95% CI for test accuracy from confusion matrices.

Reconstructs the (true,pred) multiset from each CM in results.json, then
bootstraps accuracy (B=10000, seed 42). Writes results/bootstrap_ci.json
and prints a Tabel-I-ready line per model.

McNemar FP32-vs-INT8 NOT computed here: it needs paired predictions
(agreement/disagreement 2x2), which two separate CMs cannot provide.
That runs in the lab (A3/A4 step) where predictions exist.
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
B = 10000
SEED = 42

def pairs_from_cm(cm):
    pairs = []
    for t, row in enumerate(cm):
        for p, n in enumerate(row):
            pairs.extend([(t, p)] * n)
    return np.array(pairs)

def boot_acc(pairs, rng):
    n = len(pairs)
    idx = rng.integers(0, n, size=(B, n))
    correct = (pairs[idx, 0] == pairs[idx, 1]).sum(axis=1)
    return correct / n * 100.0

def main():
    R = json.load(open(RES / "results.json"))
    rng = np.random.default_rng(SEED)
    out = {"B": B, "seed": SEED, "models": {}}
    print(f"{'model':<16}{'cond':<6}{'acc':>8}{'[95% CI]':>20}")
    for m, r in R["models"].items():
        for cond in ["fp32", "int8"]:
            pairs = pairs_from_cm(r[f"{cond}_confusion"])
            assert len(pairs) == 1728, (m, cond, len(pairs))
            dist = boot_acc(pairs, rng)
            lo, hi = np.percentile(dist, [2.5, 97.5])
            acc = r[f"{cond}_test_acc"]
            out["models"].setdefault(m, {})[cond] = {
                "acc": acc, "ci_lo": round(float(lo), 2),
                "ci_hi": round(float(hi), 2)}
            print(f"{m:<16}{cond:<6}{acc:>8.2f}  [{lo:.2f}, {hi:.2f}]")
    with open(RES / "bootstrap_ci.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {RES / 'bootstrap_ci.json'}")

if __name__ == "__main__":
    main()
