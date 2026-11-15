#!/usr/bin/env python3
"""Generate Fig. 1 System Pipeline Architecture for JCSE paper."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"

fig, ax = plt.subplots(figsize=(12, 3.8), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 32)
ax.axis("off")

# Stage boxes
stages = [
    {
        "title": "Stage 1: Sterile Data Hygiene",
        "subtitle": "Subject-Separated Split\nNo Cross-Split Leakage\n1,728 Sterile Test Images\nClass-Balanced Sampling",
        "x": 2, "w": 17, "color": "#e8f4f8", "border": "#1f77b4"
    },
    {
        "title": "Stage 2: Baseline & Failure Analysis",
        "subtitle": "Dense CNNs (MNV2, EB0, SNV2)\nDynamic INT8 Evaluation\nSQNR Layer-wise Audit\nDepthwise Collapse Localized",
        "x": 21.5, "w": 18, "color": "#fdeded", "border": "#d62728"
    },
    {
        "title": "Stage 3: Structured Pruning",
        "subtitle": "Torch-Pruning DepGraph\nL1-Norm Channel Saliency\nIterative S30 / S50 Targets\nTopology-Aware Constraints",
        "x": 42, "w": 18, "color": "#eef8ee", "border": "#2ca02c"
    },
    {
        "title": "Stage 4: Post-Training Calibration",
        "subtitle": "Static MinMax Calibration\nPer-Channel QDQ Scheme\nActivation Range Profiling\nQuantization Recovery (+30%)",
        "x": 62.5, "w": 18, "color": "#fcf3e6", "border": "#ff7f0e"
    },
    {
        "title": "Stage 5: Physical Edge Deployment",
        "subtitle": "Raspberry Pi 5 (Cortex-A76)\nONNX Runtime 1.29 Engine\nMulti-Thread Scaling (1-4T)\n144.8 FPS / 0.72 MB Footprint",
        "x": 83, "w": 15, "color": "#f3eafa", "border": "#9467bd"
    }
]

for s in stages:
    # Outer box
    box = patches.FancyBboxPatch(
        (s["x"], 3), s["w"], 26,
        boxstyle="round,pad=0.5,rounding_size=1.2",
        linewidth=1.8, edgecolor=s["border"], facecolor=s["color"]
    )
    ax.add_patch(box)
    
    # Title banner
    title_box = patches.FancyBboxPatch(
        (s["x"] + 0.3, 23.5), s["w"] - 0.6, 5.0,
        boxstyle="round,pad=0.2,rounding_size=0.8",
        linewidth=0, facecolor=s["border"]
    )
    ax.add_patch(title_box)
    ax.text(s["x"] + s["w"]/2.0, 26.0, s["title"], ha="center", va="center", color="white", fontsize=8.2, fontweight="bold")
    
    # Body text
    ax.text(s["x"] + s["w"]/2.0, 13.0, s["subtitle"], ha="center", va="center", color="#222222", fontsize=7.8, linespacing=1.6)

# Arrows between stages
for i in range(len(stages) - 1):
    x_start = stages[i]["x"] + stages[i]["w"] + 0.3
    x_end = stages[i+1]["x"] - 0.3
    y_mid = 16.0
    ax.annotate(
        "", xy=(x_end, y_mid), xytext=(x_start, y_mid),
        arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2.0, color="#444444")
    )

plt.tight_layout()
fig.savefig(FIG_DIR / "system_pipeline_jcse.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "system_pipeline_jcse.eps", format="eps", bbox_inches="tight")
fig.savefig(ROOT / "paper" / "figures" / "system_pipeline_jcse.png", dpi=300, bbox_inches="tight")
print("Saved system_pipeline_jcse.png and .eps")
