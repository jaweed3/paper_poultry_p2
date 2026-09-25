#!/usr/bin/env python3
"""Jalur A2: plot 2x3 confusion matrices (FP32 vs INT8) from results.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)
CLASSES = ["cocci", "healthy", "ncd", "salmo"]

R = json.load(open(ROOT / "results" / "results.json"))
models = ["mobilenetv2", "shufflenetv2", "efficientnet_b0"]

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
for j, m in enumerate(models):
    r = R["models"][m]
    for i, cond in enumerate(["fp32", "int8"]):
        cm = np.array(r[f"{cond}_confusion"])
        ax = axes[i][j]
        im = ax.imshow(cm, cmap="Blues")
        acc = r[f"{cond}_test_acc"]
        ax.set_title(f"{m} {cond.upper()} ({acc}%)")
        ax.set_xticks(range(4), CLASSES, rotation=20)
        ax.set_yticks(range(4), CLASSES)
        ax.set_xlabel("pred")
        if j == 0:
            ax.set_ylabel("true")
        for a in range(4):
            for b in range(4):
                v = cm[a, b]
                ax.text(b, a, str(v), ha="center", va="center",
                        color="white" if v > cm.max() / 2 else "black",
                        fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle("Confusion matrices — sterile test 1728 (rows=true, cols=pred)", fontsize=14)
plt.tight_layout()
out = FIGDIR / "confusion_gate0.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved -> {out}")
