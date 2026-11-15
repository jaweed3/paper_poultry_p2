#!/usr/bin/env python3
"""
scripts/plot_sparsity_sweep.py
Plots continuous degradation curves across structured channel sparsity s in {0.0, 0.20, 0.30, 0.40, 0.50, 0.60}:
  - Panel (a): Overall Top-1 Accuracy (%) vs. Sparsity Ratio s
  - Panel (b): Newcastle Disease Sensitivity (%) vs. Sparsity Ratio s
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG_DIRS = [ROOT / "figures", ROOT / "paper" / "figures"]
for d in FIG_DIRS:
    d.mkdir(parents=True, exist_ok=True)

# Data from empirical sweep logs
sparsities = [0.0, 0.20, 0.30, 0.40, 0.50, 0.60]

# MobileNetV2 FP32
mnv2_fp32_acc = [97.92, 95.78, 93.81, 90.68, 90.05, 86.57]
mnv2_fp32_ncd = [94.33, 85.82, 75.18, 55.32, 52.48, 32.62]

# MobileNetV2 Dynamic INT8
mnv2_int8_acc = [97.11, 92.48, 91.09, 88.19, 85.47, 81.13]
mnv2_int8_ncd = [91.49, 71.63, 61.70, 51.77, 31.21, 29.08]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

# Panel (a): Top-1 Accuracy
ax1.plot(sparsities, mnv2_fp32_acc, marker="o", color="#1f77b4", linewidth=2.2, label="MobileNetV2 FP32")
ax1.plot(sparsities, mnv2_int8_acc, marker="s", color="#ff7f0e", linewidth=2.2, linestyle="--", label="MobileNetV2 Dynamic INT8")

for s, a in zip(sparsities, mnv2_fp32_acc):
    ax1.annotate(f"{a:.1f}%", (s, a), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5)
for s, a in zip(sparsities, mnv2_int8_acc):
    ax1.annotate(f"{a:.1f}%", (s, a), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5, color="#d95f02")

ax1.set_xlabel("Structured Channel Sparsity Ratio ($s$)", fontsize=10.5, fontweight="bold")
ax1.set_ylabel("Sterile Test Top-1 Accuracy (%)", fontsize=10.5, fontweight="bold")
ax1.set_title("(a) Overall Top-1 Diagnostic Accuracy", fontsize=11, fontweight="bold")
ax1.set_xlim(-0.05, 0.65)
ax1.set_ylim(75, 102)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="lower left", fontsize=9.5)

# Panel (b): Newcastle Disease Recall
ax2.plot(sparsities, mnv2_fp32_ncd, marker="o", color="#1f77b4", linewidth=2.2, label="MobileNetV2 FP32")
ax2.plot(sparsities, mnv2_int8_ncd, marker="s", color="#ff7f0e", linewidth=2.2, linestyle="--", label="MobileNetV2 Dynamic INT8")

for s, n in zip(sparsities, mnv2_fp32_ncd):
    ax2.annotate(f"{n:.1f}%", (s, n), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5)
for s, n in zip(sparsities, mnv2_int8_ncd):
    ax2.annotate(f"{n:.1f}%", (s, n), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5, color="#d95f02")

ax2.axhline(90.0, color="red", linestyle=":", linewidth=1.5, label="Clinical Safety Threshold (90%)")
ax2.axhspan(0, 90.0, color="red", alpha=0.06)
ax2.text(0.02, 45, "CRITICAL SENSITIVITY DEGRADATION\n(Lethal Pathogen Under-Reported)", color="#b30000", fontsize=8.5, fontweight="bold", alpha=0.85)

ax2.set_xlabel("Structured Channel Sparsity Ratio ($s$)", fontsize=10.5, fontweight="bold")
ax2.set_ylabel("Newcastle Disease Recall [Sensitivity] (%)", fontsize=10.5, fontweight="bold")
ax2.set_title("(b) Lethal Minority Class Sensitivity ($n=141$)", fontsize=11, fontweight="bold")
ax2.set_xlim(-0.05, 0.65)
ax2.set_ylim(20, 102)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper right", fontsize=9.5)

plt.tight_layout()

for d in FIG_DIRS:
    fig.savefig(d / "sparsity_sweep_degradation.png", dpi=300)
    fig.savefig(d / "sparsity_sweep_degradation.eps", format="eps")
    print(f"Saved: {d / 'sparsity_sweep_degradation.png'} and {d / 'sparsity_sweep_degradation.eps'}")

if __name__ == '__main__':
    pass
