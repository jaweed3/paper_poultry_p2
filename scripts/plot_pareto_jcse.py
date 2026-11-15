#!/usr/bin/env python3
"""
scripts/plot_pareto_jcse.py
Generates the comprehensive Accuracy vs. Model Size trade-off frontier for JCSE paper.
Outputs:
  - figures/pareto_frontier_jcse.png
  - figures/pareto_frontier_jcse.eps (vector graphic for LaTeX submission)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Data points: (Model, Variant, Size_MB, Acc_Percent, Color, Marker)
data = [
    # MobileNetV2
    ("MNV2 Baseline FP32", 8.48, 97.92, "#1f77b4", "o", (10, 5)),
    ("MNV2 INT8 Dense", 2.30, 97.28, "#1f77b4", "s", (10, -12)),
    ("MNV2 Pruned s30 INT8", 1.23, 91.09, "#1f77b4", "^", (10, 5)),
    ("MNV2 Pruned s50 INT8 (Cliff)", 0.72, 85.47, "#1f77b4", "v", (10, -15)),
    
    # ShuffleNetV2
    ("SNV2 Baseline FP32", 4.89, 97.69, "#2ca02c", "o", (10, 5)),
    ("SNV2 INT8 Dense", 1.47, 97.63, "#2ca02c", "s", (10, -12)),
    
    # EfficientNet-B0
    ("EB0 Baseline FP32", 15.30, 98.55, "#d62728", "o", (-130, 5)),
    ("EB0 INT8 Static QDQ", 4.67, 94.56, "#d62728", "s", (10, 5)),
    ("EB0 Pruned s30 Static QDQ", 2.62, 63.72, "#d62728", "^", (10, -5)),
    ("EB0 INT8 Dynamic (Collapse)", 4.16, 10.42, "#7f7f7f", "X", (10, -5)),
]

fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

for label, size, acc, color, marker, text_offset in data:
    ax.scatter([size], [acc], color=color, marker=marker, s=140, edgecolors="black", linewidths=1.2, zorder=5)
    ax.annotate(
        label,
        (size, acc),
        textcoords="offset points",
        xytext=text_offset,
        fontsize=9,
        fontweight="semibold" if "Pruned" in label or "Collapse" in label else "normal",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        zorder=6
    )

# Connect compression trajectories
# MobileNetV2 trajectory
ax.plot([8.48, 2.30, 1.23, 0.72], [97.92, 97.28, 91.09, 85.47], linestyle="--", color="#1f77b4", alpha=0.6, label="MobileNetV2 Trajectory")
# EfficientNet-B0 recovery trajectory
ax.plot([15.30, 4.67, 2.62], [98.55, 94.56, 63.72], linestyle="--", color="#d62728", alpha=0.6, label="EfficientNet-B0 (Static QDQ)")
# ShuffleNetV2 trajectory
ax.plot([4.89, 1.47], [97.69, 97.63], linestyle="--", color="#2ca02c", alpha=0.6, label="ShuffleNetV2 Trajectory")

# Annotate Sweet Spot and Collapse
ax.axvspan(0.5, 1.5, color="#2ca02c", alpha=0.08, label="Ultra-Edge Target Zone (< 1.5 MB)")
ax.annotate(
    "Sweet Spot:\n6.89x Compression\n91.09% Acc (1.23 MB)",
    xy=(1.23, 91.09), xytext=(2.5, 82),
    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5),
    fontsize=9.5, fontweight="bold", color="#1f77b4"
)

ax.annotate(
    "Quantization Collapse:\nPer-Tensor Depthwise\nSQNR Drops to 8.4 dB",
    xy=(4.16, 10.42), xytext=(6.0, 18),
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5),
    fontsize=9.5, fontweight="bold", color="#d62728"
)

ax.set_xlabel("Model Storage Footprint on Disk (MB)", fontsize=11, fontweight="bold")
ax.set_ylabel("Sterile Test Top-1 Accuracy (%)", fontsize=11, fontweight="bold")
ax.set_title("Pareto Compression Frontier: Accuracy vs. Storage Footprint Across Architectures", fontsize=12, fontweight="bold", pad=12)
ax.set_xlim(0, 16.5)
ax.set_ylim(0, 102)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)

plt.tight_layout()
png_out = FIG_DIR / "pareto_frontier_jcse.png"
eps_out = FIG_DIR / "pareto_frontier_jcse.eps"
fig.savefig(png_out, dpi=300)
fig.savefig(eps_out, format="eps")
print(f"Saved: {png_out} and {eps_out}")
