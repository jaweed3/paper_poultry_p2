#!/usr/bin/env python3
"""Fig 1 (2-panel): EfficientNet-B0 FP32 vs dynamic INT8 confusion matrices.

Replaces the previous 2x3 grid (six panels), which was illegible at column
width. The other four panels remain available in the repository
(figures/confusion_gate0.png and scripts/plot_confusion.py).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "figures"
CLASSES = ["Cocci", "Healthy", "NCD", "Salmo"]

R = json.load(open(ROOT / "results" / "results.json"))
e = R["models"]["efficientnet_b0"]

# Compact canvas: small native size keeps glyphs large once scaled to column width.
fig, axes = plt.subplots(1, 2, figsize=(3.7, 1.95))

for ax, cond, label in zip(axes, ["fp32", "int8"], ["FP32", "INT8 dynamic"]):
    cm = np.array(e[f"{cond}_confusion"])
    acc = e[f"{cond}_test_acc"]
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_title(f"{label} — {acc:.2f}%", fontsize=8.5, pad=3)
    ax.set_xticks(range(4))
    ax.set_xticklabels(CLASSES, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(4))
    ax.set_yticklabels(CLASSES, fontsize=7)
    ax.set_xlabel("predicted", fontsize=7.5)
    ax.set_ylabel("true", fontsize=7.5)
    ax.tick_params(length=2, width=0.6)
    for a in range(4):
        for b in range(4):
            v = cm[a, b]
            ax.text(b, a, str(v), ha="center", va="center", fontsize=7,
                    color="white" if v > cm.max() * 0.55 else "black")

fig.tight_layout(pad=0.4, w_pad=1.2)
out = FIGDIR / "confusion_eb0_2panel.png"
fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.02)
print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")
