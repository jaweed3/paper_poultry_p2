#!/usr/bin/env python3
"""
scripts/plot_pareto_jcse.py
Generates the 2-panel Edge Deployment Pareto Frontier for JCSE paper:
  Panel (a): Test Top-1 Accuracy vs. Hardware Inference Latency (4T on RPi5)
  Panel (b): Newcastle Disease Sensitivity (Minority Lethal Recall) vs. Hardware Inference Latency
Outputs:
  - figures/pareto_frontier_jcse.png
  - figures/pareto_frontier_jcse.eps
  - paper/figures/pareto_frontier_jcse.png
  - paper/figures/pareto_frontier_jcse.eps
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG_DIRS = [ROOT / "figures", ROOT / "paper" / "figures"]
for d in FIG_DIRS:
    d.mkdir(parents=True, exist_ok=True)

# Data: (Name, Latency_4T_ms, Top1_Acc, NCD_Recall, Color, Marker, Offset_a, Offset_b)
points = [
    # Baseline Uncompressed
    ("ShuffleNetV2 FP32", 9.03, 97.69, 95.74, "#2ca02c", "o", (10, 5), (10, -12)),
    ("MobileNetV2 FP32", 19.48, 97.92, 94.33, "#1f77b4", "o", (10, 5), (10, 5)),
    ("EfficientNet-B0 FP32", 39.31, 98.55, 95.74, "#d62728", "o", (10, 5), (10, 5)),
    
    # Quantized
    ("ShuffleNetV2 INT8 Dyn", 11.63, 97.45, 93.62, "#2ca02c", "s", (10, -12), (10, -12)),
    ("MobileNetV2 INT8 Dyn", 42.01, 97.11, 91.49, "#1f77b4", "s", (10, -12), (10, -12)),
    ("EfficientNet-B0 Static QDQ", 19.40, 94.56, 92.91, "#d62728", "D", (10, -12), (10, -12)),
    
    # Pruned Variants
    ("MobileNetV2 s30 FP32", 11.07, 93.81, 75.18, "#1f77b4", "^", (10, 5), (10, 5)),
    ("MobileNetV2 s30 INT8", 25.93, 91.09, 61.70, "#9467bd", "v", (10, -14), (10, -14)),
    ("MobileNetV2 s50 FP32", 6.90, 90.05, 52.48, "#1f77b4", "<", (10, 5), (10, 5)),
    ("MobileNetV2 s50 INT8", 18.88, 85.47, 31.21, "#9467bd", ">", (10, -14), (10, -14)),
    
    ("EfficientNet-B0 s30 FP32", 26.50, 93.69, 82.27, "#d62728", "^", (10, 5), (10, 5)),
    ("EfficientNet-B0 s30 Static", 16.55, 63.72, 37.59, "#e377c2", "D", (10, 5), (10, 5)),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Panel (a): Overall Top-1 Accuracy vs Latency
for name, lat, top1, ncd, color, marker, off_a, _ in points:
    ax1.scatter([lat], [top1], color=color, marker=marker, s=120, edgecolors="black", linewidths=1.0, zorder=5)
    ax1.annotate(
        name,
        (lat, top1),
        textcoords="offset points",
        xytext=off_a,
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"),
        zorder=6
    )

# Connect Pareto Frontier in (a)
pareto_a_x = [6.90, 9.03, 19.48, 39.31]
pareto_a_y = [90.05, 97.69, 97.92, 98.55]
ax1.plot(pareto_a_x, pareto_a_y, color="green", linestyle="--", linewidth=1.8, alpha=0.7, label="Empirical Pareto Frontier")

ax1.axvline(33.33, color="gray", linestyle=":", alpha=0.7, label="30 FPS Real-Time Boundary")
ax1.set_xlabel("Raspberry Pi 5 Inference Latency, 4 Threads (ms)", fontsize=10, fontweight="bold")
ax1.set_ylabel("Sterile Test Top-1 Accuracy (%)", fontsize=10, fontweight="bold")
ax1.set_title("(a) Top-1 Accuracy vs. Hardware Execution Latency", fontsize=11, fontweight="bold")
ax1.set_xlim(0, 50)
ax1.set_ylim(60, 101)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

# Panel (b): Minority Lethal Recall (NCD) vs Latency
for name, lat, top1, ncd, color, marker, _, off_b in points:
    ax2.scatter([lat], [ncd], color=color, marker=marker, s=120, edgecolors="black", linewidths=1.0, zorder=5)
    ax2.annotate(
        name,
        (lat, ncd),
        textcoords="offset points",
        xytext=off_b,
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"),
        zorder=6
    )

# Connect Pareto Frontier in (b)
pareto_b_x = [9.03, 19.48, 39.31]
pareto_b_y = [95.74, 94.33, 95.74]
ax2.plot([9.03, 39.31], [95.74, 95.74], color="green", linestyle="--", linewidth=1.8, alpha=0.7, label="Safety Pareto Frontier (≥90% Recall)")

# Critical Safety Threshold
ax2.axhline(90.0, color="red", linestyle="--", linewidth=1.4, alpha=0.8, label="Veterinary Safety Threshold (90% NCD Recall)")
ax2.axvline(33.33, color="gray", linestyle=":", alpha=0.7, label="30 FPS Boundary")

# Shaded unsafe region
ax2.axhspan(0, 90.0, color="red", alpha=0.06)
ax2.text(3, 40, "CRITICAL BIOSECURITY HAZARD ZONE\n(Minority Pathogen Under-Reported)", color="#b30000", fontsize=9, fontweight="bold", alpha=0.8)

ax2.set_xlabel("Raspberry Pi 5 Inference Latency, 4 Threads (ms)", fontsize=10, fontweight="bold")
ax2.set_ylabel("Newcastle Disease Recall [Sensitivity] (%)", fontsize=10, fontweight="bold")
ax2.set_title("(b) Lethal Minority Sensitivity (NCD) vs. Execution Latency", fontsize=11, fontweight="bold")
ax2.set_xlim(0, 50)
ax2.set_ylim(25, 102)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

plt.tight_layout()

for d in FIG_DIRS:
    fig.savefig(d / "pareto_frontier_jcse.png", dpi=300)
    fig.savefig(d / "pareto_frontier_jcse.eps", format="eps")
    print(f"Saved: {d / 'pareto_frontier_jcse.png'} and {d / 'pareto_frontier_jcse.eps'}")

if __name__ == '__main__':
    pass
