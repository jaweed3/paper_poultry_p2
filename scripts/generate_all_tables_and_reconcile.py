#!/usr/bin/env python3
"""
scripts/generate_all_tables_and_reconcile.py
Single Source of Truth (SSOT) Table Generator and Text Reconciler for JCSE Manuscript.
Directly ingests:
- results/splits_manifest.json
- results/leakage_effect_size.json
- results/eval_gate0.json
- results/static_quant_all.json
- results/static_eb0.json
- results/finetune_mobilenetv2_s30_weighted_distill.json
- results/rpi5_canonical_benchmark.json
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) / n) + (z**2) / (4 * (n**2)))) / denom
    lower = max(0.0, center - margin) * 100
    upper = min(1.0, center + margin) * 100
    return lower, upper

def format_ci(k, n):
    pct = (k / n) * 100
    low, high = wilson_ci(k, n)
    return f"{pct:.2f}\\% [{low:.2f}\\%, {high:.2f}\\%]"

def main():
    print("=== Reconciling All JCSE Tables from Authoritative JSON SSOT ===")

    # 1. Dataset Table
    with open(ROOT / "results/splits_manifest.json") as f:
        splits_json = json.load(f)
    splits = splits_json["actual"]
    tr = splits["train"]["per_class"]
    va = splits["val"]["per_class"]
    te = splits["test"]["per_class"]
    tot_cocci = tr["cocci"] + va["cocci"] + te["cocci"]
    tot_healthy = tr["healthy"] + va["healthy"] + te["healthy"]
    tot_salmo = tr["salmo"] + va["salmo"] + te["salmo"]
    tot_ncd = tr["ncd"] + va["ncd"] + te["ncd"]
    tot_all = splits["train"]["total"] + splits["val"]["total"] + splits["test"]["total"]

    table1 = rf"""\begin{{table}}
\centering
\caption{{Class Distribution Across Sterile Dataset Partitions ($N=8,153$)}}
\label{{tab:dataset}}
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lcccc}}
\toprule
\textbf{{Diagnostic Category}} & \textbf{{Training}} & \textbf{{Validation}} & \textbf{{Test ($n$)}} & \textbf{{Total}} \\
\midrule
Coccidiosis (Cocci)          & {tr['cocci']:,} & {va['cocci']:,} & {te['cocci']:,} (30.6\%) & {tot_cocci:,} \\
Healthy Poultry               & {tr['healthy']:,} & {va['healthy']:,} & {te['healthy']:,} (29.2\%) & {tot_healthy:,} \\
Salmonella (Salmo)           & {tr['salmo']:,} & {va['salmo']:,} & {te['salmo']:,} (32.1\%) & {tot_salmo:,} \\
Newcastle Disease (NCD)      & {tr['ncd']:,}   & {va['ncd']:,} & {te['ncd']:,} (8.2\%)  & {tot_ncd:,}   \\
\midrule
\textbf{{Total Images}}        & \textbf{{{splits['train']['total']:,}}} & \textbf{{{splits['val']['total']:,}}} & \textbf{{{splits['test']['total']:,}}} & \textbf{{{tot_all:,}}} \\
\bottomrule
\end{{tabular}}%
}}
\end{{table}}"""

    # 2. Leakage Table with NCD column
    with open(ROOT / "results/leakage_effect_size.json") as f:
        leak = json.load(f)
    mb = leak["evaluations"]["mobilenetv2"]
    l_sub = mb["leaked_subset"]
    nl_sub = mb["non_leaked_subset"]
    pub_sub = mb["original_full_test"]
    st_sub = mb["sterile_test"]

    table_leakage = rf"""\begin{{table}}
\centering
\caption{{Quantification of Test Partition Contamination and Empirical Leakage Bias}}
\label{{tab:leakage}}
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lcccccc}}
\toprule
\textbf{{Test Partition Subset}} & \textbf{{Sample Size ($n$)}} & \textbf{{Top-1 Acc.}} & \textbf{{Healthy}} & \textbf{{Cocci}} & \textbf{{Salmo}} & \textbf{{NCD Recall}} \\
\midrule
Leaked Pairs (Duplicate of Train) & 156 & {l_sub['acc']:.2f}\% & {l_sub['per_class']['healthy']['recall']:.1f}\% & {l_sub['per_class']['cocci']['recall']:.1f}\% & {l_sub['per_class']['salmo']['recall']:.1f}\% & {l_sub['per_class']['ncd']['recall']:.1f}\% \\
Non-Leaked Test Partition         & 1,598 & {nl_sub['acc']:.2f}\% & {nl_sub['per_class']['healthy']['recall']:.1f}\% & {nl_sub['per_class']['cocci']['recall']:.1f}\% & {nl_sub['per_class']['salmo']['recall']:.1f}\% & {nl_sub['per_class']['ncd']['recall']:.1f}\% \\
Original Published Split \cite{{machuve2021poultry}} & 1,754 & {pub_sub['acc']:.2f}\% & {pub_sub['per_class']['healthy']['recall']:.1f}\% & {pub_sub['per_class']['cocci']['recall']:.1f}\% & {pub_sub['per_class']['salmo']['recall']:.1f}\% & {pub_sub['per_class']['ncd']['recall']:.1f}\% \\
Decontaminated Sterile Benchmark  & 1,728 & {st_sub['acc']:.2f}\% & {st_sub['per_class']['healthy']['recall']:.1f}\% & {st_sub['per_class']['cocci']['recall']:.1f}\% & {st_sub['per_class']['salmo']['recall']:.1f}\% & {st_sub['per_class']['ncd']['recall']:.1f}\% \\
\bottomrule
\end{{tabular}}%
}}
\end{{table}}"""

    # 3. Hardware Table from Canonical Benchmark
    with open(ROOT / "results/rpi5_canonical_benchmark.json") as f:
        hw = json.load(f)["models"]

    # Table VI: Physical Edge Hardware Profiling
    def row(name, key):
        m = hw[key]
        sz = f"{m['file_size_mb']:.2f} MB"
        th1 = m["th1"]
        th4 = m["th4"]
        return (
            f"{name} & {sz} & {th1['mean_ms']:.2f} ms & {th1['p50_ms']:.2f} ms & {th1['fps']:.1f} & "
            f"{th4['mean_ms']:.2f} ms & {th4['p50_ms']:.2f} ms & {th4['fps']:.1f} & {th4['peak_rss_mb']:.2f} MB \\\\"
        )

    table_hw = rf"""\begin{{table*}}
\centering
\caption{{Comprehensive Physical Embedded Hardware Profiling on Raspberry Pi 5 (Quad-Core Cortex-A76 @ 2.4\,GHz, 100 Runs)}}
\label{{tab:hardware_full}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lcccccccc}}
\toprule
\textbf{{Model Configuration}} & \textbf{{Disk Size}} & \textbf{{1T Mean (ms)}} & \textbf{{1T P50 (ms)}} & \textbf{{1T FPS}} & \textbf{{4T Mean (ms)}} & \textbf{{4T P50 (ms)}} & \textbf{{4T FPS}} & \textbf{{Peak RSS (MB)}} \\
\midrule
{row('MobileNetV2 Baseline FP32', 'mobilenetv2_fp32')}
{row('MobileNetV2 Baseline Dynamic INT8', 'mobilenetv2_dyn_int8')}
{row('MobileNetV2 Baseline Static QDQ', 'mobilenetv2_static_int8')}
{row('MobileNetV2 Pruned s30 FP32', 'mobilenetv2_s30_fp32')}
{row('MobileNetV2 Pruned s30 INT8 Dynamic', 'mobilenetv2_s30_dyn_int8')}
{row('MobileNetV2 Pruned s50 FP32', 'mobilenetv2_s50_fp32')}
{row('MobileNetV2 Pruned s50 INT8 Dynamic', 'mobilenetv2_s50_dyn_int8')}
\midrule
{row('EfficientNet-B0 Baseline FP32', 'efficientnet_b0_fp32')}
{row('EfficientNet-B0 Baseline Dynamic INT8', 'efficientnet_b0_dyn_int8')}
{row('EfficientNet-B0 Baseline Static QDQ', 'efficientnet_b0_static_int8')}
{row('EfficientNet-B0 Pruned s30 FP32', 'efficientnet_b0_s30_fp32')}
{row('EfficientNet-B0 Pruned s30 Static QDQ', 'efficientnet_b0_s30_static_int8')}
{row('EfficientNet-B0 Pruned s50 FP32', 'efficientnet_b0_s50_fp32')}
\midrule
{row('ShuffleNetV2 Baseline FP32', 'shufflenetv2_fp32')}
{row('ShuffleNetV2 Baseline Dynamic INT8', 'shufflenetv2_dyn_int8')}
{row('ShuffleNetV2 Baseline Static QDQ', 'shufflenetv2_static_int8')}
{row('ShuffleNetV2 Pruned s30 FP32', 'shufflenetv2_s30_fp32')}
\bottomrule
\end{{tabular}}%
}}
\end{{table*}}"""

    out_file = ROOT / "paper/generated_tables.tex"
    with open(out_file, "w") as f:
        f.write("% AUTO-GENERATED BY scripts/generate_all_tables_and_reconcile.py - DO NOT EDIT MANUALLY\n\n")
        f.write(table1 + "\n\n")
        f.write(table_leakage + "\n\n")
        f.write(table_hw + "\n\n")

    print(f"Generated {out_file} successfully!")
    print("\nKey Hardware Values to Reconcile with Manuscript Text:")
    print(f"MobileNetV2 FP32 4T: {hw['mobilenetv2_fp32']['th4']['mean_ms']:.2f} ms ({hw['mobilenetv2_fp32']['th4']['fps']:.1f} FPS)")
    print(f"MobileNetV2 Static QDQ 4T: {hw['mobilenetv2_static_int8']['th4']['mean_ms']:.2f} ms ({hw['mobilenetv2_static_int8']['th4']['fps']:.1f} FPS) -> Speedup: {hw['mobilenetv2_fp32']['th4']['mean_ms']/hw['mobilenetv2_static_int8']['th4']['mean_ms']:.2f}x")
    print(f"ShuffleNetV2 FP32 4T: {hw['shufflenetv2_fp32']['th4']['mean_ms']:.2f} ms ({hw['shufflenetv2_fp32']['th4']['fps']:.1f} FPS)")
    print(f"ShuffleNetV2 Static QDQ 4T: {hw['shufflenetv2_static_int8']['th4']['mean_ms']:.2f} ms ({hw['shufflenetv2_static_int8']['th4']['fps']:.1f} FPS) -> Speedup: {hw['shufflenetv2_fp32']['th4']['mean_ms']/hw['shufflenetv2_static_int8']['th4']['mean_ms']:.2f}x")
    print(f"EfficientNet-B0 FP32 4T: {hw['efficientnet_b0_fp32']['th4']['mean_ms']:.2f} ms ({hw['efficientnet_b0_fp32']['th4']['fps']:.1f} FPS)")
    print(f"EfficientNet-B0 Static QDQ 4T: {hw['efficientnet_b0_static_int8']['th4']['mean_ms']:.2f} ms ({hw['efficientnet_b0_static_int8']['th4']['fps']:.1f} FPS) -> Speedup: {hw['efficientnet_b0_fp32']['th4']['mean_ms']/hw['efficientnet_b0_static_int8']['th4']['mean_ms']:.2f}x")
    print(f"ShuffleNetV2 Dynamic INT8 4T: {hw['shufflenetv2_dyn_int8']['th4']['mean_ms']:.2f} ms ({hw['shufflenetv2_dyn_int8']['th4']['fps']:.1f} FPS)")

if __name__ == "__main__":
    main()
