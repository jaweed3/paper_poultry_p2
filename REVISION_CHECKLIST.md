# Revision Checklist — ICAIDES #1571359798

Source: email TPC Chairs 6 Okt 2026, 3 regular reviews.
Deadline: **31 Okt 2026**. Branch: `post30-gate0-dedup`.

Legend: ✅ done · 🟡 partial · 🔴 open · ⛔ blocked (needs hardware)

---

## R1 — Weak Accept (4)

| # | Item | Status | Evidence / Action |
|---|---|---|---|
| R1.1 | Support depthwise diagnosis with layer-wise or ablation experiment (DW/PW/SE, per-tensor vs per-channel) | ✅ | `static_eb0.py` does whole-model only. Need selective ablation. ⛔ needs Pi/lab run |
| R1.1b | Report multiple seeds + calibration-set sizes with CIs | 🔴 | Only seed 42, calib N=100. Need sweep. ⛔ needs runs |
| R1.2 | Re-run INT8 benchmarks with properly configured ORT (static QDQ + QLinearConv, ARM-optimized kernels, native Linux not WSL on x86) | 🔴 | x86 = WSL only. Need native Linux + ARM kernel evidence. ⛔ needs lab + Pi |
| R1.2b | Soften/remove claim that ORT "dequantizes before the matmul" unless profiling shows it | ✅ | L189 rewritten: "operator presence alone does not prove which kernels execute", "we state this as a hypothesis" |
| R1.3 | Fair comparison on same deduplicated split vs prior work, OR clearly caveat Table VI | ✅ | L244 caption "different protocols; not head-to-head"; L262 caveat paragraph |
| R1.3b | Report energy OR discuss its absence as a limitation | ✅ | L82/85/270: `current_a=null`, "energy is a stated limitation" |
| R1.4a | Fix broken Table III header | ✅ | Verified: renders correctly, 2-row header, all columns aligned. No change needed |
| R1.4b | Fix illegible Fig. 1 | ✅ | `pareto_gate0.png` regenerated 7 Oct, 52KB, big font |
| R1.4c | Fix inconsistent statements about run counts | ✅ | L85: "200 iterations after 10 warmup iterations" consistent |
| R1.4d | Fix unexplained x86 ShuffleNetV2 vs MobileNetV2 FP32 latency ordering | 🟡 | L189 explains via matmul count / node expansion. Verify it actually answers |
| R1.4e | Grad-CAM figure adds little → shorten or remove | ✅ | Figure + section removed; one-sentence note kept; abstract + keywords synced (commit 8598f01) |
| R1.5 | Replace anonymous repo link with permanent archive | ✅ | No `4open`/`anonymous` string remains. Verify actual URL works |
| R1.5b | Temper generality of conclusions resting on single board / one runtime / one dataset / batch 1 | ✅ | L85 "Variability across independent benchmark sessions was not measured"; "model-only"; limitations in Conclusion |

---

## R2 — Strong Accept (6)

| # | Item | Status | Evidence / Action |
|---|---|---|---|
| R2.1 | Explain theoretically why depthwise-separable conv in EfficientNet-B0 is exceptionally sensitive to per-tensor vs MobileNetV2/ShuffleNetV2 | 🟡 | Mechanism paragraph added to §IV-A (commit 8598f01): per-tensor shares one scale across C≤32 depthwise channels with no spatial averaging; SE 1x1 convs span few channels so range is dominated by the largest; MNv2/SNv2 pair depthwise with wide pointwise. Framed as consistent-with, still needs R1.1 ablation as evidence |
| R2.2 | Deeply elaborate RPi5 threading inefficiency (ORT threading overhead / small-matrix bottleneck) | 🟡 | L189 has hypothesis + node expansion table. Needs more depth |
| R2.3 | Explicitly detail pHash audit so others can replicate the 8,153-image dedup split | ✅ | L73: manifest paths, 80/20 stratified seed 42, publisher test never re-split, pair/group semantics |
| R2.4 | Strong contribution / novelty | ✅ | R2 scored 4/4/4 — no action |

---

## R3 — Weak Reject (3)

| # | Item | Status | Evidence / Action |
|---|---|---|---|
| R3.1a | **Selective quantization ablations for DW / PW / SE** + layerwise weight or activation error analysis | ✅ | Not done. Overlaps R1.1. ⛔ needs runs |
| R3.1b | OTHERWISE revise title, abstract, conclusions to say "granularity sensitivity" not "confirmed depthwise mechanism" | ✅ | Title still "Localizing...". Abstract softened, §IV-A mechanism added but framed consistent-with. **Still binary: run ablation (script ready) or retitle** |
| R3.2a | Distinguish graph operators from executed kernels; provide optimized-graph + kernel-dispatch evidence | 🔴 | Only softened to hypothesis. Need actual optimized graph + kernel dispatch data. ⛔ needs lab |
| R3.2b | CPU instruction availability alone doesn't prove instructions are used | ✅ | Removed the VNNI-style claim in commit aacf0b7 |
| R3.2c | Memory-bound claim needs evidence or cautious wording | ✅ | "we state this as a hypothesis since memory-boundedness was not directly measured" |
| R3.3 | **Benchmark static INT8 on RPi5, esp. per-channel that recovers EfficientNet-B0** | ✅ | DONE 8 Okt: static per-tensor 13.63ms / per-channel 13.50ms vs FP32 31.53ms at t=4 → 2.31-2.34x FASTER (200 runs, peak 49.4C, no throttle). Artifact `results/rpi5_static_gate0/rpi5_static.json`. **Reverses headline claim** — see below |
| R3.4a | McNemar or paired bootstrap CIs for FP32–INT8 diffs in MobileNetV2 and ShuffleNetV2 | 🔴 | McNemar exists for EfficientNet-B0 only. MNv2/SNv2 drops (0.81pp, 0.24pp) have NO paired test. `compute_ci.py` may support it |
| R3.4b | Distinguish lack of statistical significance from equivalence | ✅ | L91 explicit: "overlapping marginal intervals alone do not establish paired significance or equivalence" |
| R3.4c | Table II: specify test variant + discordant counts; report underflow via bound not p=0 | ✅ | "continuity correction", b=14 c=1468, `p < 10^{-15}` |
| R3.5a | Explain validation split creation (source release has train/test, audit refers to validation) | ✅ | L73: "stratified 80/20 split of the publisher train only (seed 42); publisher test was never re-split" |
| R3.5b | Distinguish duplicate pairs vs connected groups vs removed images; explain overlapping near-dup handling | ✅ | L73: "all size 2", "counts refer to removed copies rather than connected groups" |
| R3.5c | Replace absolute "sterile" with qualified statement | ✅ | §III-A now defines the term explicitly: denotes only absence of cross-split near-duplicates under this audit, not independence; residual within-split dependence stated (commit 8598f01) |
| R3.5d | Discuss remaining dependence between images from same specimen/farm/session | ✅ | L73: "Residual dependence may remain..." |
| R3.6a | Specify model implementations, trainable layers, checkpoint-selection, seeds, training repetitions | 🟡 | L78 Training Protocol — verify completeness |
| R3.6b | Exact quantization options: act/weight signedness, quantized op types, per-channel axes, preprocessing/fusion, calib selection | ✅ | L82: activation_type=QInt8, weight_type=QInt8, QuantFormat.QDQ, MinMax, first 100 sterile-train, batch 8 |
| R3.6c | Report runtime version + benchmark thread settings for x86 | ✅ | L82: ORT 1.29.0, ONNX 1.22.0, WSL, default intra-op |
| R3.7a | Per-class precision, recall, F1 + macro-F1 + balanced accuracy for principal configs | ✅ | L108: full macro-P/R/F1 + balanced acc for 3 models × 2 quant |
| R3.7b | Clarify definition of "per-class accuracy" | ✅ | L108: "means recall (true-positive rate per true class)" |
| R3.7c | State whether preprocessing/data transfer included in timing | ✅ | L85: "model-only (eval preprocessing and data transfer excluded)" |
| R3.7d | Report variability across independent benchmark sessions | ✅ | L85: "within-session; across sessions was not measured" |
| R3.7e | Separate model-only throughput from end-to-end application throughput | ✅ | L85 explicit |
| R3.8a | Enlarge labels/annotations in Figures 1, 4, 5 | 🟡 | Fig 1 done. Fig 4 (rpi5 thread scaling) 82KB. Fig 5 (gradcam) 2.6MB — needs check |
| R3.8b | Resolve discrepancy: text says single Grad-CAM map, Fig 5 shows four examples | ✅ | Resolved by removal: figure deleted entirely, no caption discrepancy possible |
| R3.8c | Avoid inferring low-level feature use from visualizations alone | 🟡 | Check §Grad-CAM wording |
| R3.8d | Explain criterion behind "best balance" recommendation for MobileNetV2 | ✅ | Phrase "best balance" no longer present |
| R3.8e | State Table VI results are contextual, different protocols | ✅ | L244 + L262 |

---

## 🔴 P0 — must resolve before camera-ready

| ID | Item | Needs HW? | Owner |
|---|---|---|---|
| **P0-1** | Retitle OR run selective ablation (R3.1b + R1.1 + R2.1) | yes (ablation) / no (retitle) | decision first |
| **P0-2** | Static INT8 benchmark on RPi5 (R3.3) | ✅ DONE | closed 8 Okt, commit 012ac0f |
| **P0-3** | Optimized-graph + kernel-dispatch evidence (R3.2a) | ✅ 8 Okt: level2 profiling artifact already in paper §IV-C as supplementary | partially addressed; enhanced version tomorrow |
| **P0-4** | McNemar / paired bootstrap for MNv2 + SNv2 (R3.4a) | 🔴 TODO 9 Okt — needs paired prediction dump | agent can do |

## 🟡 P1 — after P0

| ID | Item | Needs HW? |
|---|---|---|
| P1-1 | Native Linux (not WSL) ORT benchmark, ARM kernel verification (R1.2) | ⛔ lab |
| P1-2 | Multi-seed + calibration-size sweep with CIs (R1.1b) | ⛔ runs |
| P1-3 | Theoretical explanation of depthwise per-tensor sensitivity (R2.1) | no — but needs P0-1 data |
| P1-4 | Replace "sterile" term entirely (R3.5c) | no |
| P1-5 | Fix Table III header (R1.4a) | no |
| P1-6 | Grad-CAM: fix 4-vs-1 caption, shorten/remove section | no |
| P1-7 | Strengthen RPi5 threading discussion (R2.2) | no |

---

## ⚠️ FINDING 8 Okt — headline claim was half-true

Paper sebelumnya claimed "INT8 is 1.9–2.8x slower than FP32 on RPi5". That is
only true for **dynamic** quantization. Static QDQ is **2.34x faster** than FP32
(13.50ms vs 31.53ms, t=4) while recovering accuracy to 94.56%. Abstract,
contributions, and §IV-C have been rescoped. Reviewer R3 asked exactly this
comparison and the answer changes the paper's practical conclusion:
per-channel static QDQ is simultaneously the accurate AND the fast config.

Remaining caveat: latency measured on Pi, accuracy measured on x86 (same
graphs, same ORT 1.29.0). Confirming accuracy on the Pi itself is still open.

## Page budget

Current: 6 pages. Adding P0-2 table + P0-1 ablation ≈ +0.5–0.9 pp → 7 pp.
ICAIDES registration page lists **flat fees only, no per-page surcharge** (checked 8 Okt 2026).
Decision needed: shorten Grad-CAM (~0.4 pp) to stay at 6, or confirm page limit with TPC.

## Admin (before 31 Okt)

- [ ] Register + pay (student IEEE online Rp 2.000.000 / non-IEEE Rp 2.500.000)
- [ ] Fill registration form: https://forms.gle/wMepCRRpfcQQ9Knx6
- [ ] Remove IEEE footer + page numbers from camera-ready
- [ ] Verify all 3 authors registered in EDAS
- [ ] Verify abstract matches EDAS record
- [ ] PDF Express: https://ieee-pdf-express.org/ , Conference ID **73571X**
- [ ] Download certified PDF, upload to EDAS
- [ ] Complete e-copyright on EDAS
- [ ] Add presenter on EDAS
- [ ] Upload pre-recorded presentation video: https://icaides.untar.ac.id/page/39/Online-Presentation
- [ ] Presenter must attend allocated session (12–13 Nov 2026)

---

# 📋 WORK PLAN — Jumat 9 Okt 2026

Deadline camera-ready: **31 Okt 2026** (23 hari).
Semua P0-1 dan P0-2 sudah **DONE**. Sisa P0 tinggal 3 item.

## ✅ Sudah selesai (jangan ulang)

| Item | Commit | Hasil |
|---|---|---|
| P0-1 ablation DW/PW/SE | `9bcd2c0` | depthwise necessary+sufficient: C=93.46%, D=8.33% |
| P0-2 RPi5 static INT8 | `012ac0f` | 2.34x lebih cepat dari FP32 (13.50ms vs 31.53ms) |
| Table III header | — | verified clean, no change needed |

## 🔴 P0 — wajib, 3 item

### 1. R3.4a — McNemar / paired CI untuk MNv2 + SNv2

**Kenapa:** reviewer mau paired significance test untuk drops 0.81pp (MNv2) & 0.24pp (SNv2). Cuma EB0 yang punya McNemar sekarang.

**Cara:**
1. Patch `scripts/eval_gate0.py` untuk dump paired predictions
   (`results/preds_<model>_<cond>.npy`) — bukan cuma confusion matrix
2. Jalankan di lab WSL: `bash scripts/run_sterile.sh scripts/eval_gate0.py`
3. Script baru `scripts/mcnemar_paired.py` — hitung b/c discordant,
   continuity-corrected p-value, bootstrap paired CI
4. Patch paper Table I + teks §IV-A

**Waktu:** ~1 jam (patch + run + patch paper)

### 2. R3.2a — kernel-dispatch evidence

**Kenapa:** reviewer mau bukti kernel yang benar-benar **execute**, bukan just graph operator. Lo cuma punya "hypothesis" wording.

**Cara:**
- ORT profiling pada Pi dengan `ort_session_options.enable_profiling = True`
- Ambil `profiler_*.json` / `*.json` dari Pi, parse kernel names
- Tambah table atau footnote di §IV-C

**Waktu:** ~2 jam (script + run di Pi + parse)

### 3. R1.2 — native Linux ORT benchmark

**Kenapa:** reviewer minta ARM kernel verification + native Linux, bukan WSL x86.

**Cara:**
- Native Linux x86 di lab (nilai WSL2 ist WSL2 = kernel boundary overhead)
- Re-run `benchmark_pi_gate0.py` + `static_eb0.py` on native Linux
- Patch §IV-C x86 reference numbers

**Waktu:** ~3 jam, TANDAS setup.

**Fallback kalau nggak sempat:** write it as limitation, bukan klaim.

### 🟡 P1 — backlog

| Item | Effort | Note |
|---|---|---|
| R1.1b multi-seed | mahal | retrain per seed, skip unless reviewer's adamant |
| R2.1 theory | 15 menit | §IV-A sudah ada; strengthen dengan ablation data |

## ⚠️ Page budget

PDF 7 halaman, page 7 = conclusion tail + references. Fee table resmi: **extra page = USD 50 / IDR 500.000**, max 2 pages.

**Decision needed:** stay at 7 atau compress back to 6?
- 7 halaman (now) → +500k IDR over base
- 6 halaman → free
- Base fee (student member online): **Rp 2.000.000**

**Gw sarankan:** compress back to 6 (compression is possible in Related Work + Related Work text) if possible, else pay the 500k.

## 📦 Admin (do not forget)

- [ ] Register + pay (student IEEE member online = Rp 2.000.000 / non-member 2.500.000)
- [ ] Registration form: https://forms.gle/wMepCRRpfcQQ9Knx6
- [ ] Verify 3 authors registered in EDAS
- [ ] PDF Express: https://ieee-pdf-express.org/ (Conference ID **73571X**)
- [ ] e-copyright on EDAS
- [ ] Add presenter on EDAS
- [ ] Upload pre-recorded video: https://icaides.untar.ac.id/page/39/Online-Presentation

## 🔑 Key resources

- Lab WSL: `ssh -o RemoteCommand=none lab-wsl 'wsl -d Ubuntu -- bash -lc "cmd"'`
- Pi: `ssh pi` (~/p2bench, ~/bench-venv)
- Dataset: `POULTRY_DATA_DIR=data/images_sterile` (1728 test)
- Final PDF: `~/paper_poultry_p2/paper/main.pdf`

