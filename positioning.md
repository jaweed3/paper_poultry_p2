# Positioning — Paper 2 vs Paper 1 (MLCIPR 2026-10-05)

Paper 1 = characterization. Paper 2 = diagnosis + sterile rebuild. File ini mengunci delta supaya tidak dianggap incremental / self-plagiarism.

## Paper 1 dalam 1 paragraf

Paper 1 (ICITISEE/AMICT) menjawab 3 RQ deskriptif: (1) lightweight mana paling akurat di fecal poultry, (2) seberapa sensitif tiap arsitektur ke INT8 dynamic PTQ di CPU, (3) dimana Pareto accuracy-latency di x86 + RPi5. Dataset Machuve 8770 image (5614/1402/1754), 3 arsitektur MN2/SNV2/EB0, 96 config RPi5 + level-2 profiling node & kernel. Temuan: INT8 malah lambat 6.4-30.1x x86, 1.9-2.8x RPi5; thread scaling INT8 flat 1.1-1.2x (1->4) vs 2.3x FP32; mekanisme terbukti node expansion 1.37-3.05x + overhead quant 20-29%; SNV2 paling tahan (homogen). Klaim: hardware-aware selection.

## Delta Paper 2 yang wajib dikunci

| Dimensi | Paper 1 | Paper 2 (harus beda) |
|---------|---------|----------------------|
| RQ | Deskriptif: seberapa parah bottleneck | Preskriptif: apakah **structured pruning sebelum PTQ** bisa memotong bottleneck sehingga INT8 viable di CPU tanpa VNNI/DOTPROD |
| Metode | Export ONNX -> dynamic PTQ | **Prune (structured channel) -> finetune 10-20ep -> export -> PTQ** |
| Pruning | — | **Structured channel L1-norm** via Torch-Pruning/DepGraph; sparsity 0/30/50/70%; skip head; rule khusus SE untuk EB0 |
| Metrik baru | acc, size, latency, RSS, node, overhead | **Korelasi sparsity vs expansion vs overhead vs thread scaling** |
| Hardware | x86 i5-12400F + RPi5 96 config | **Sama persis** (apple-to-apple), tambah dimensi sparsity |
| Matriks | 3×2×4×4=96 | **3×4×2×4×1=96 RPi5 +24 x86 =120 +16 spot batch** |
| Klaim | Pemilihan model | **Mitigasi overhead** (pruning mitigates quant overhead) |
| Figure hero | Thread scaling FP32 vs INT8, batch scaling | **Overlay FP32 vs INT8 vs Pruned-INT8 + sparsity vs expansion/overhead** |

Jika salah satu baris masih sama dengan Paper 1, Paper 2 akan ditolak sebagai extended version.

## Hipotesis & trade-off

H1: Tiap Conv yang di-prune menghilangkan rantai DynamicQuantizeLinear+ConvInteger+Mul+Cast setelah PTQ -> expansion turun (MN2 2.86x, EB0 3.05x harus turun proporsional). H2: Overhead 29.2% (MN2) turun <24% di 50%. H3: Scaling INT8 1.11x (1->4) kembali >1.7x di 50%. H4: Acc global drop <1% sampai 50%, tapi recall NCD (kelas 8.3% Zenodo) akan drop duluan — monitor per-class.

## Novelty check (Aug 2026)

Mango leaf Dec 2025: two-stage pruning+PTQ hybrid CNN mobile — domain mango, eval general mobile, tanpa ORT level-2 + 96 RPi5 config. LightPrune ICCVW 2025: latency-aware structured pruning embedded — tidak coupling dengan PTQ overhead. Survey Liang 2021 & He 2023: bahas kombinasi teknik tanpa konteks poultry fecal. Dhungana/Tasdelen/Degu: lomba akurasi tanpa kompresi sistematis. Gap kosong: **bukti mekanistik pruning-then-PTQ memulihkan latency INT8 di Cortex-A76 dengan profiling node expansion di dataset Machuve** — itu yang kita isi.

## Story yang dipilih

**Story A — Mitigation** (dipilih): "Pruning Mitigates INT8 Quantization Overhead on ARM CPUs for Poultry Disease Detection" — Paper 1 nemu penyakit, Paper 2 kasih obat. Paling kuat karena pakai aset unik Paper 1 (level-2 artifact + scaling flat). Story B Pareto co-design dan Story C SE-robustness jadi sekunder.
