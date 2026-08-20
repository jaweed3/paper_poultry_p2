# Experiment Plan — Paper 2: Structured Pruning before INT8 PTQ

> Kontrak eksperimen. Semua run harus ikut plan ini supaya apple-to-apple dengan Paper 1 dan bisa diklaim mitigation.

## RQ

Apakah structured channel pruning sebelum INT8 dynamic PTQ bisa menurunkan node expansion & overhead kernel quant (Paper 1: 1.37-3.05x, 20-29%) sehingga thread scaling INT8 di Cortex-A76 kembali normal dan latency INT8 viable di CPU tanpa VNNI/DOTPROD. Turunan: (a) sparsity berapa trade-off terbaik, (b) apakah penurunan expansion linear dengan overhead & scaling, (c) apakah SNV2 tetap paling tahan setelah di-prune.

## Hipotesis

H1: Tiap channel yang di-prune hilangkan rantai DQLinear+ConvInteger+Mul+Cast -> expansion turun proporsional. H2: Overhead MN2 29.2% -> <24% di 50%. H3: Scaling INT8 1.11x (1->4) -> >1.7x di 50%. H4: Acc global drop <1% sampai 50%, recall NCD drop duluan.

## Variabel

Independen: arch {MN2 2.2M, SNV2 1.3M, EB0 4.0M} × sparsity {0,30,50,70}% × quant {FP32, INT8 PTQ} × thread {1,2,3,4} × batch {1 (+ spot 4,8,16 hanya MN2)}. Dependen: acc top-1 + per-class P/R/F1, latency mean/P50/P95/P99 + FPS, size MB, RSS mean/peak, node FP32/INT8 + expansion, overhead %, scaling efficiency. Kontrol: dataset Machuve HF 8770 (5614/1402/1754), 224×224 ImageNet norm, hardware x86 i5-12400F + RPi5 BCM2712 4×A76 2.4GHz 8GB Debian13 trixie kernel 6.18 perf governor throttled 0x0, ORT 1.29.0 CPUExecutionProvider.

## Pipeline

Checkpoint FP32 50ep Adam 1e-4 wd 1e-4 cosine bs32 (Paper 1) -> **structured channel L1-norm** via Torch-Pruning/DepGraph (Conv+BN, skip head, rule SE untuk EB0) -> finetune 10-20ep lr 5e-5 -> export ONNX opset17 -> dynamic PTQ ORT per-tensor INT8 (tanpa calibration, sama persis Paper 1) -> benchmark+profiling.

## Matriks feasible

Full factorial 384 config tidak feasible. **Full sweep hanya batch=1**: 3×4×2×4×1=96 RPi5 +24 x86 =120. Spot batch scaling: MN2 sparsity 0 & 50% ×2 quant ×4 batch ×1 thread =16. Total **136 config**, 200 iter +10 warmup per config, estimasi 6-8 jam RPi5 berurutan, split per model.

## Protokol

- x86: CPUExecutionProvider, 3×224×224, report mean/std/P50/P95/P99/FPS/size.
- RPi5: fork `benchmark_rpi5.py` -> `scripts/benchmark_rpi5_pruned.py` (+ arg sparsity, json kompatibel `rpi5_benchmark_results.json`), 200+10, report latency/FPS/RSS/CPU%/suhu, 96/96 tanpa throttling.
- Level-2: fork `profiling_nodecount.py` -> `scripts/profiling_pruned.py`, 10 run batch1 1thread, hitung FP32/INT8 nodes, expansion, DynQuant/CvInt, overhead, simpan `results/profiling/level2_pruned_artifact.json` format sama dengan Paper 1.

## Kriteria sukses

Primer: sparsity 50% expansion turun ≥30% vs baseline, overhead turun ≥5pp, scaling INT8 1->4 >1.7x, acc drop <1%. Sekunder: recall NCD drop <3%, RSS tidak naik. Gagal tetap publishable sebagai temuan negatif + per-operator analysis.

## Baseline & ablation

Baseline: Paper 1 FP32/INT8 (MN2 1.85/55.75ms, SNV2 3.17/20.34ms, EB0 3.50/80.66ms x86). Ablation: (1) sparsity sweep 0/30/50/70 cari elbow, (2) Pruned-FP32 vs Pruned-INT8 pisahkan efek, (3) arch comparison.

## Repro

Semua json schema sama dengan Paper 1, seed fix, ORT 1.29.0 + commit hash di json, figure generate dari json, data Paper 1 via symlink.

## Figure & tabel

Tabel: 3×4 sparsity — acc FP32/Pruned-FP32/Pruned-INT8 | size | latency FP32/INT8 | slowdown | expansion | overhead | scaling eff. Fig1 sparsity vs acc, Fig2 sparsity vs expansion+overhead (dual axis), Fig3 hero thread scaling RPi5 batch1 FP32 vs INT8 vs Pruned-INT8 50% (MN2), Fig4 per-class recall vs sparsity + CM 50%.

## Urutan eksekusi

MN2 30% ->50%->70% (23ep tercepat) -> SNV2 -> EB0 terakhir (SE tricky).

## Risiko

Unstructured tidak speedup (mitigasi: hanya structured), NCD anjlok (monitor per-class, siap weighted loss), waktu RPi5 habis (batasi batch1), EB0 collapse lagi 98->35% (jangan prune SE agresif di 70% atau fallback 50%).

## Checklist siap jalan

Positioning+planning terkunci, Torch-Pruning/DepGraph ter-install, fork script jalan 1 config RPi5 sampai json, Zotero 5+ refs.
