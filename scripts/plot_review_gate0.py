#!/usr/bin/env python3
"""Review plot: 50ep sterile training curves for 3 models."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LOGDIR = ROOT / "logs"
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)

models = ["mobilenetv2", "shufflenetv2", "efficientnet_b0"]
H = {}
for m in models:
    with open(LOGDIR / f"{m}_gate0_history.json") as f:
        H[m] = json.load(f)

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
for j, m in enumerate(models):
    h = H[m]
    ep = range(1, len(h["train_loss"]) + 1)
    axes[0][j].plot(ep, h["train_loss"], label="train loss")
    axes[0][j].plot(ep, h["val_loss"], label="val loss")
    axes[0][j].set_title(f"{m} — loss (best val {h['best_val_acc']:.2f}%)")
    axes[0][j].set_xlabel("epoch")
    axes[0][j].legend()
    axes[0][j].grid(True, alpha=0.3)
    axes[1][j].plot(ep, h["train_acc"], label="train acc")
    axes[1][j].plot(ep, h["val_acc"], label="val acc")
    axes[1][j].set_title(f"{m} — accuracy")
    axes[1][j].set_xlabel("epoch")
    axes[1][j].set_ylabel("acc (%)")
    axes[1][j].legend()
    axes[1][j].grid(True, alpha=0.3)

plt.suptitle("Gate0 sterile split (8153) — 50ep training curves, seed 42", fontsize=14)
plt.tight_layout()
out = FIGDIR / "training_review_gate0.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved -> {out}")

# combined val acc
fig2, ax = plt.subplots(figsize=(10, 5))
for m in models:
    h = H[m]
    ax.plot(range(1, len(h["val_acc"]) + 1), h["val_acc"], label=f"{m} (best {h['best_val_acc']:.2f}%)")
ax.set_xlabel("epoch")
ax.set_ylabel("val acc (%)")
ax.set_title("Val accuracy comparison — sterile split")
ax.legend()
ax.grid(True, alpha=0.3)
out2 = FIGDIR / "val_acc_compare_gate0.png"
plt.tight_layout()
plt.savefig(out2, dpi=120, bbox_inches="tight")
print(f"saved -> {out2}")

# numeric summary
for m in models:
    h = H[m]
    va = h["val_acc"]
    print(f"{m}: best={h['best_val_acc']:.2f}% @ep{va.index(max(va))+1} "
          f"| final train_acc={h['train_acc'][-1]:.2f}% final_val={va[-1]:.2f}% "
          f"| min_val_loss={min(h['val_loss']):.4f} final_val_loss={h['val_loss'][-1]:.4f} "
          f"| time={h['total_time_s']/60:.0f}min")
