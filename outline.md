# Outline — Paper 2: Pruning Mitigates INT8 Overhead (MLCIPR)

## Target
MLCIPR 2026-10-05, IEEE 4-6pp A4, 15+ refs (5yr), Scopus.

## Title (working)
Pruning Mitigates INT8 Quantization Overhead on ARM CPUs for Poultry Disease Detection

## Story
Paper 1 found the disease (INT8 slowdown via expansion+overhead). Paper 2 proves the cure (structured pruning before PTQ).

## Sections (skeleton — fill after numbers exist)
1. Intro: poultry loss, edge gap, RQ prescriptive
2. Related: poultry DL, lightweight CNN, quant PTQ vs QAT, structured pruning (Liang, LightPrune, DepGraph, He survey)
3. Method: dataset Machuve 8770, 3 arch, pipeline prune->finetune->export->PTQ, sparsity grid 0/30/50/70
4. Exp: 136 config matrix, x86 24 + RPi5 96 + spot 16
5. Results: (a) sparsity vs acc, (b) sparsity vs expansion/overhead, (c) thread scaling hero, (d) per-class NCD
6. Discussion: why pruning fixes what PTQ breaks, deployment guideline
7. Conclusion: pruning-then-PTQ makes INT8 viable on CPU w/o accelerator
