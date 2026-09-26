#!/usr/bin/env python3
"""Gen Tabel I LaTeX + markdown dari results/results.json (SSOT).

Output: results/table1.tex + results/table1.md
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = json.load(open(ROOT / "results" / "results.json"))

ORDER = ["mobilenetv2", "shufflenetv2", "efficientnet_b0"]
PRETTY = {"mobilenetv2": "MobileNetV2", "shufflenetv2": "ShuffleNetV2",
          "efficientnet_b0": "EfficientNet-B0"}

rows = []
for m in ORDER:
    r = R["models"][m]
    fci = r["fp32_ci"]
    ici = r["int8_ci"]
    rows.append({
        "model": PRETTY[m],
        "fp32": f"{r['fp32_test_acc']:.2f} [{fci[0]:.2f}, {fci[1]:.2f}]",
        "int8d": f"{r['int8_test_acc']:.2f} [{ici[0]:.2f}, {ici[1]:.2f}]",
        "drop": f"{r['acc_drop_pp']:.2f}",
        "fp32md": f"{r['fp32_test_acc']:.2f} ({fci[0]:.2f}--{fci[1]:.2f})",
        "int8md": f"{r['int8_test_acc']:.2f} ({ici[0]:.2f}--{ici[1]:.2f})",
    })

eb0s = R["eb0_static"]["configs"]
srow = {k: v["test_acc"] for k, v in eb0s.items()}
srow["dynamic"] = R["eb0_static"]["dynamic_int8_test_acc"]

pi = {(c["model"], c["quant"], c["threads"]): c for c in R["rpi5_gate0"]["configs"]}

tex = r"""\begin{table}[htbp]
\caption{Test accuracy on sterile split (1,728 images) with 95\% bootstrap CI (B=10,000). Dynamic INT8 PTQ (ORT).}
\label{tab:results}
\centering
\begin{tabular}{lccc}
\toprule
Model & FP32 (\%) & INT8 dynamic (\%) & Drop (pp) \\
\midrule
"""
for r in rows:
    tex += f"{r['model']} & {r['fp32']} & {r['int8d']} & {r['drop']} \\\\\n"
tex += r"""\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\caption{EfficientNet-B0: static QDQ INT8 diagnosis (MinMax, calib 100). Per-tensor collapses; per-channel recovers.}
\label{tab:eb0static}
\centering
\begin{tabular}{lcc}
\toprule
Config & Test acc. (\%) & McNemar vs FP32 $p$ \\
\midrule
"""
tex += f"Dynamic INT8 & {srow['dynamic']:.2f} & --- \\\\\n"
tex += (f"Static QDQ per-tensor & "
        f"{srow['static_per_tensor_minmax']:.2f} & 0.0 \\\\\n")
tex += (f"Static QDQ per-channel & "
        f"{srow['static_per_channel_minmax']:.2f} & 1.37e-13 \\\\\n")
tex += r"""\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\caption{Raspberry Pi~5 latency, batch=1 (mean of 200 runs, P50 in parentheses). ORT 1.29.0, throttled=0x0.}
\label{tab:rpi5}
\centering
\footnotesize
\setlength{\tabcolsep}{3.2pt}
\resizebox{\linewidth}{!}{\begin{tabular}{lcccccc}
\toprule
Model & Quant & t=1 & t=2 & t=3 & t=4 & FP32$\to$INT8 \\
 & & (ms) & (ms) & (ms) & (ms) & t=4 \\
\midrule
"""
# t=2/t=3 from legacy rpi5 file for context (marked) — here gate0 only has t=1/t=4
for m in ORDER:
    pm = PRETTY[m]
    f1, f4 = pi[(m, "FP32", 1)], pi[(m, "FP32", 4)]
    i1, i4 = pi[(m, "INT8", 1)], pi[(m, "INT8", 4)]
    slow = i4["mean_ms"] / f4["mean_ms"]
    tex += (f"{pm} & FP32 & {f1['mean_ms']:.2f} ({f1['p50_ms']:.2f}) & --- & --- & "
            f"\\textbf{{{f4['mean_ms']:.2f}}} ({f4['p50_ms']:.2f}) & --- \\\\\n")
    tex += (f" & INT8 & {i1['mean_ms']:.2f} ({i1['p50_ms']:.2f}) & --- & --- & "
            f"{i4['mean_ms']:.2f} ({i4['p50_ms']:.2f}) & {slow:.2f}$\\times$ slower \\\\\n")
tex += r"""\bottomrule
\end{tabular}}
\end{table}
"""
with open(ROOT / "results" / "table1.tex", "w") as f:
    f.write(tex)

md = "| Model | FP32 % [95% CI] | INT8 dyn % [95% CI] | Drop pp |\n|---|---|---|---|\n"
for r in rows:
    md += f"| {r['model']} | {r['fp32md']} | {r['int8md']} | {r['drop']} |\n"
md += (f"\nEB0 static: per-tensor {srow['static_per_tensor_minmax']:.2f}% vs "
       f"per-channel {srow['static_per_channel_minmax']:.2f}% "
       f"(dynamic {srow['dynamic']:.2f}%).\n")
with open(ROOT / "results" / "table1.md", "w") as f:
    f.write(md)
print(tex[:1500])
print("...")
print(f"saved -> results/table1.tex + table1.md")
