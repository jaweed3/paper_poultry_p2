#!/usr/bin/env python3
"""A5: regenerate steril figures for main.tex from results/results.json.

Outputs (figures/):
- rpi5_thread_scaling_gate0.png (FP32 solid vs INT8 dashed, t=1/t=4)
- pareto_gate0.png (FP32 acc vs RPi5 t=4 latency, sterile)
- quant_acc_size_gate0.png (FP32 vs INT8 acc + size, zoomed y-axis)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
R = json.load(open(ROOT / "results" / "results.json"))

ORDER = ["mobilenetv2", "shufflenetv2", "efficientnet_b0"]
PRETTY = {"mobilenetv2": "MobileNetV2", "shufflenetv2": "ShuffleNetV2",
          "efficientnet_b0": "EfficientNet-B0"}
COLORS = {"mobilenetv2": "tab:blue", "shufflenetv2": "tab:green",
          "efficientnet_b0": "tab:red"}

# ---- 1. thread scaling (sterile, t=1/t=4) ----
pi = {(c["model"], c["quant"], c["threads"]): c
      for c in R["rpi5_gate0"]["configs"]}
fig, ax = plt.subplots(figsize=(9, 5.5))
for m in ORDER:
    f = [pi[(m, "FP32", 1)]["mean_ms"], pi[(m, "FP32", 4)]["mean_ms"]]
    i = [pi[(m, "INT8", 1)]["mean_ms"], pi[(m, "INT8", 4)]["mean_ms"]]
    ax.plot([1, 4], f, "o-", color=COLORS[m], label=f"{PRETTY[m]} FP32")
    ax.plot([1, 4], i, "s--", color=COLORS[m], label=f"{PRETTY[m]} INT8")
ax.set_xticks([1, 4])
ax.set_xlabel("Threads")
ax.set_ylabel("Latency (ms)")
ax.set_title("RPi5 thread scaling, batch=1 (sterile models, ORT 1.29.0)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "rpi5_thread_scaling_gate0.png", dpi=120, bbox_inches="tight")
print("saved rpi5_thread_scaling_gate0.png")

# ---- 2. Pareto (sterile FP32 acc vs RPi5 t=4 latency) ----
fig, ax = plt.subplots(figsize=(8, 5.5))
for m in ORDER:
    acc = R["models"][m]["fp32_test_acc"]
    lat = pi[(m, "FP32", 4)]["mean_ms"]
    ax.scatter([lat], [acc], s=120, color=COLORS[m], label=PRETTY[m])
    ax.annotate(PRETTY[m], (lat, acc), textcoords="offset points",
                xytext=(8, 6), fontsize=10)
ax.set_xlabel("RPi5 latency @4 threads, batch=1 (ms)")
ax.set_ylabel("Sterile test accuracy (%)")
ax.set_title("Accuracy-latency Pareto (sterile, FP32)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "pareto_gate0.png", dpi=120, bbox_inches="tight")
print("saved pareto_gate0.png")

# ---- 3. quant acc + size (zoomed) ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
x = np.arange(3)
w = 0.35
facc = [R["models"][m]["fp32_test_acc"] for m in ORDER]
iacc = [R["models"][m]["int8_test_acc"] for m in ORDER]
axes[0].bar(x - w / 2, facc, w, label="FP32", color="tab:blue")
axes[0].bar(x + w / 2, iacc, w, label="INT8 dynamic", color="tab:orange")
axes[0].set_xticks(x, [PRETTY[m] for m in ORDER], rotation=10)
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_ylim(0, 100)
axes[0].set_title("FP32 vs INT8 accuracy (sterile test 1728)")
axes[0].legend()
for xi, fa, ia in zip(x, facc, iacc):
    axes[0].text(xi - w / 2, fa + 0.8, f"{fa:.1f}", ha="center", fontsize=9)
    axes[0].text(xi + w / 2, ia + 0.8, f"{ia:.1f}", ha="center", fontsize=9)
fsz = [R["models"][m]["fp32_size_mb"] for m in ORDER]
isz = [R["models"][m]["int8_size_mb"] for m in ORDER]
axes[1].bar(x - w / 2, fsz, w, label="FP32", color="tab:blue")
axes[1].bar(x + w / 2, isz, w, label="INT8 dynamic", color="tab:orange")
axes[1].set_xticks(x, [PRETTY[m] for m in ORDER], rotation=10)
axes[1].set_ylabel("Model size (MB)")
axes[1].set_title("FP32 vs INT8 model size")
axes[1].legend()
fig.suptitle("Quantization impact (sterile split)")
fig.tight_layout()
fig.savefig(FIG / "quant_acc_size_gate0.png", dpi=120, bbox_inches="tight")
print("saved quant_acc_size_gate0.png")
