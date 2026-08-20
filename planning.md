# Planning — Paper 2: Pruning-then-PTQ Mitigation (MLCIPR 2026-10-05)

Target: buktikan hipotesis Paper 2 lewat eksperimen yang bisa direproduksi. Semua checklist di bawah harus dicentang 1 per 1 sebelum lanjut. File ini hidup — centang dengan `[x]` saat selesai.

## 0. Prinsip

- Paper 1 = characterization (penyakit). Paper 2 = mitigation (obat). Jangan copy kalimat Paper 1 verbatim.
- Semua hipotesis harus diuji via eksperimen, bukan klaim. Angka yang tidak ada di `results/` tidak boleh masuk ke `paper/main.tex`.
- Baseline Paper 1 ada di `results/baseline_p1/` (symlink ke ~/paper_poultry_edge/results). Jangan copy file 30MB.

---

## 1. Fork & baseline — DONE hari ini

- [x] Fork ~/paper_poultry_edge -> ~/paper_poultry_p2 (history 601dbb2 kejaga)
- [x] `git remote remove origin` (remote baru jaweed3/paper_poultry_p2 nanti saat repo dibuat)
- [x] `results/baseline_p1 -> ~/paper_poultry_edge/results` (symlink, bukti APPLE-TO-APPLE)
- [x] `literature/` subdir + `FORK.md` lineage
- [ ] `git init` sudah ada; commit awal `chore: fork p2 from p1 601dbb2 + scaffolding` (tunggu planning+experiment_plan final)

## 2. Literature pack — 7 refs (kumpulin di literature/, kalau paywall simpan link di literature/README.md)

Kriteria: 5 tahun terakhir, prioritas pruning terstruktur + quant + edge CPU.

- [ ] [1] Liang et al. 2021 — Pruning and Quantization survey (arxiv:2101.09671) — PDF open
- [ ] [2] LightPrune — Latency-Aware Structured Pruning ICCVW 2025 — PDF open
- [ ] [3] Channel pruning + group vector quantization — Neural Computing & Applications (Springer) — cek paywall, kumpulkan link+DOI
- [ ] [4] Pruning+Quantization Hybrid CNN Mango Leaf — ResearchGate 2025-12-16 — link, minta PDF manual
- [ ] [5] Optimized CNN at IoT edge via pruning+quantization — Multimedia Tools Appl. Springer 2024 — link/DOI
- [ ] [6] Structured pruning for edge CPU 2024-2025 (cari 1 paper tambahan, mis. DepGraph / Torch-Pruning original)
- [ ] [7] QAT vs PTQ trade-off (Maulana & Ramasamy 2026 sudah ada di Paper 1, reuse + tambah 1 QAT paper 2024-2025 untuk contrast)

Action per ref: `curl -L -o literature/<slug>.pdf <url>` kalau open, kalau 403/timeout tulis di README.md dengan format `- [title](url) — paywall, download manual`. Nanti user kumpulin manual ke literature/.

- [ ] literature/README.md terisi (tabel: # | title | DOI/url | status pdf/link | relevansi 1 kalimat)

## 3. Pruning method lock — sebelum sweep

- [ ] Pilih library: Torch-Pruning vs DepGraph — putuskan 1 (rekomendasi: Torch-Pruning L1-norm, paling reproducible untuk deadline)
- [ ] Lock sparsity grid: 0% (baseline Paper 1), 30%, 50%, 70%
- [ ] Lock finetune: 10-20 epoch, lr 5e-5, cosine atau constant, batch 32 (sama kayak Paper 1)
- [ ] Lock scope: prune Conv+BN channel, skip classifier head + SE block handling (catat rule untuk EfficientNet-B0)
- [ ] Tulis `scripts/prune.py` skeleton (arg: --model {mobilenetv2,shufflenetv2,efficientnet_b0} --sparsity 0.3 --finetune 15)

## 4. Infra fork — benchmark & profiling (APPLE-TO-APPLE dengan Paper 1)

- [ ] Fork `benchmark_rpi5.py` -> `scripts/benchmark_rpi5_pruned.py` (tambah arg --sparsity, output json kompatibel dengan rpi5_benchmark_results.json)
- [ ] Fork `profiling_nodecount.py` -> `scripts/profiling_pruned.py` (hitung FP32 nodes, INT8 nodes, expansion, DynQuant/CvInt, overhead, simpan ke results/profiling/level2_pruned_artifact.json)
- [ ] Test 1 config end-to-end di RPi5: MobileNetV2 sparsity 30% FP32 1 thread batch 1 -> json terisi, suhu & RSS ke-log
- [ ] Verify artifact level-2: 10 run batch1 1thread, Conv_quant_kernel % + overhead % ke-log

## 5. Sweep matrix — 136 config feasible (batch 1 full, batch scaling spot check)

- [ ] Full sweep batch=1: 3 model × 4 sparsity × 2 quant × 4 thread = 96 config RPi5 + 24 x86 = 120
- [ ] Spot check batch scaling: MobileNetV2 sparsity 0% & 50% × 2 quant × 4 batch (1,4,8,16) × 1 thread = 16
- [ ] Tiap config 200 iter + 10 warmup (sama Paper 1); estimasi 6-8 jam RPi5 berurutan, split per model

Urutan eksekusi (hemat waktu):

- [ ] MobileNetV2 30% -> 50% -> 70% (convergence tercepat 23 epoch)
- [ ] ShuffleNetV2 30/50/70
- [ ] EfficientNet-B0 30/50/70 (taruh terakhir, SE tricky)

## 6. Metrik & kriteria sukses (yang diuji hipotesisnya)

Primer (harus ada angka di results/):
- [ ] Sparsity 50%: expansion turun ≥30% vs baseline (mis. MN2 2.86x -> <2.0x)
- [ ] Overhead turun ≥5pp (mis. 29.2% -> <24%)
- [ ] Thread scaling INT8 1->4 thread: dari ~1.11x -> >1.7x
- [ ] Akurasi global drop <1% sampai 50%

Sekunder:
- [ ] Recall NCD drop <3% (monitor per-class)
- [ ] RSS tidak naik signifikan

Kalau gagal tetap publishable: laporkan temuan negatif + analisis per-operator.

## 7. Figure & tabel (generate dari json, jangan manual)

- [ ] Tabel utama: 3×4 sparsity — acc FP32 / Pruned-FP32 / Pruned-INT8 | size | latency FP32/INT8 | slowdown | expansion | overhead | scaling eff
- [ ] Fig 1: sparsity vs accuracy (3 model)
- [ ] Fig 2: sparsity vs expansion + overhead (dual axis)
- [ ] Fig 3 hero: thread scaling RPi5 batch1 — FP32 vs INT8 vs Pruned-INT8 50% (MobileNetV2)
- [ ] Fig 4: per-class recall vs sparsity (fokus NCD) + CM sparsity 50%

## 8. Writing — baru setelah angka ada

- [ ] Copy `paper/main.tex` IEEE A4 dari Paper 1 -> Paper 2, ganti title/abstract ke mitigation story
- [ ] Isi 5-5.5 halaman (4-6 limit MLCIPR), 15+ refs 5 tahun terakhir
- [ ] `tectonic paper/main.tex` + `pdfinfo` A4 + `pdffonts` 15 embedded
- [ ] positioning.md & experiment_plan.md final

## 9. Submit MLCIPR 2026-10-05

- [ ] EDAS submit + supplementary level2_pruned_artifact.json kalau diminta
- [ ] Checklist: no verbatim Paper 1, semua angka ada di results/, figure punya std dari 200 run

---

## Cara pakai checklist ini

Jalanin dari atas ke bawah. Tiap selesai satu `[ ]`, ganti jadi `[x]` dan commit. Kalau stuck >1 hari, catat di `results/README.md` sebagai blocking issue, jangan lompat ke Burn.
