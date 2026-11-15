#!/usr/bin/env python3
"""Plot physical Raspberry Pi 5 pruned latency and throughput comparison."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"

# Comparison groups for MobileNetV2 and EfficientNet-B0
# Latency in 4 threads (mean ms) and FPS
models = [
    "MNV2\nBase FP32", "MNV2\ns30 FP32", "MNV2\ns50 FP32", "MNV2\ns30 INT8", "MNV2\ns50 INT8",
    "EB0\nBase FP32", "EB0\ns30 FP32", "EB0\ns50 FP32", "EB0\nBase Static", "EB0\ns30 Static"
]
latencies_th4 = [19.48, 11.07, 6.90, 25.93, 18.88, 39.31, 26.50, 17.83, 19.40, 16.55]
fps_th4 = [51.3, 90.3, 144.8, 38.6, 53.0, 25.4, 37.7, 56.1, 51.5, 60.4]
sizes_mb = [8.48, 4.20, 2.23, 1.19, 0.72, 15.30, 7.67, 4.04, 4.67, 2.62]

x = np.arange(len(models))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

# Color palettes
colors_mnv2 = ["#1f77b4", "#4a90e2", "#70b0ff", "#2ca02c", "#5cd65c"]
colors_eb0 = ["#d62728", "#e55353", "#ff7f7f", "#9467bd", "#b388ff"]
bar_colors = colors_mnv2 + colors_eb0

# Plot 1: Inference Latency (ms)
bars1 = ax1.bar(x, latencies_th4, color=bar_colors, edgecolor="black", linewidth=0.8, width=0.65)
ax1.set_ylabel("Inference Latency (ms, 4 Threads)", fontsize=11, fontweight="bold")
ax1.set_title("Physical Inference Latency on Raspberry Pi 5 (Cortex-A76)", fontsize=12, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=8.5)
ax1.grid(axis="y", linestyle=":", alpha=0.6)
ax1.axvline(4.5, color="gray", linestyle="--", linewidth=1.2)
ax1.text(2.0, max(latencies_th4) * 0.93, "MobileNetV2 Family", ha="center", fontsize=10, fontweight="bold", color="#1f77b4")
ax1.text(7.2, max(latencies_th4) * 0.93, "EfficientNet-B0 Family", ha="center", fontsize=10, fontweight="bold", color="#d62728")

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f"{yval:.1f}ms", ha="center", va="bottom", fontsize=8, fontweight="semibold")

# Plot 2: Throughput (FPS)
bars2 = ax2.bar(x, fps_th4, color=bar_colors, edgecolor="black", linewidth=0.8, width=0.65)
ax2.set_ylabel("Throughput (Frames Per Second)", fontsize=11, fontweight="bold")
ax2.set_title("Physical Inference Throughput on Raspberry Pi 5", fontsize=12, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=8.5)
ax2.grid(axis="y", linestyle=":", alpha=0.6)
ax2.axvline(4.5, color="gray", linestyle="--", linewidth=1.2)
ax2.text(2.0, max(fps_th4) * 0.93, "MobileNetV2 Family", ha="center", fontsize=10, fontweight="bold", color="#1f77b4")
ax2.text(7.2, max(fps_th4) * 0.93, "EfficientNet-B0 Family", ha="center", fontsize=10, fontweight="bold", color="#d62728")

# Highlight real-time line (30 FPS)
ax2.axhline(30.0, color="green", linestyle="--", linewidth=1.2, label="Real-Time Threshold (30 FPS)")
ax2.legend(loc="upper right", fontsize=9)

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 2.0, f"{yval:.1f}", ha="center", va="bottom", fontsize=8, fontweight="semibold")

plt.tight_layout()
fig.savefig(FIG_DIR / "rpi5_pruned_latency_fps.png", dpi=300)
fig.savefig(FIG_DIR / "rpi5_pruned_latency_fps.eps", format="eps")
fig.savefig(ROOT / "paper" / "figures" / "rpi5_pruned_latency_fps.png", dpi=300)
print("Saved rpi5_pruned_latency_fps.png and .eps")
