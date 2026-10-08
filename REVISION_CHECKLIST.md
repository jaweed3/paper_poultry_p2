# Revision Checklist — ICAIDES #1571359798

Source: email TPC Chairs 6 Okt 2026, 3 regular reviews.
Deadline: **31 Okt 2026**. Branch: `post30-gate0-dedup`.

Legend: ✅ done · 🟡 partial · 🔴 open · ⛔ blocked (needs hardware)

---

## R1 — Weak Accept (4)

| # | Item | Status | Evidence / Action |
|---|---|---|---|
| R1.1 | Support depthwise diagnosis with layer-wise or ablation experiment (DW/PW/SE, per-tensor vs per-channel) | 🔴 | `static_eb0.py` does whole-model only. Need selective ablation. ⛔ needs Pi/lab run |
| R1.1b | Report multiple seeds + calibration-set sizes with CIs | 🔴 | Only seed 42, calib N=100. Need sweep. ⛔ needs runs |
| R1.2 | Re-run INT8 benchmarks with properly configured ORT (static QDQ + QLinearConv, ARM-optimized kernels, native Linux not WSL on x86) | 🔴 | x86 = WSL only. Need native Linux + ARM kernel evidence. ⛔ needs lab + Pi |
| R1.2b | Soften/remove claim that ORT "dequantizes before the matmul" unless profiling shows it | ✅ | L189 rewritten: "operator presence alone does not prove which kernels execute", "we state this as a hypothesis" |
| R1.3 | Fair comparison on same deduplicated split vs prior work, OR clearly caveat Table VI | ✅ | L244 caption "different protocols; not head-to-head"; L262 caveat paragraph |
| R1.3b | Report energy OR discuss its absence as a limitation | ✅ | L82/85/270: `current_a=null`, "energy is a stated limitation" |
| R1.4a | Fix broken Table III header | 🔴 | Needs visual inspection of rendered PDF |
| R1.4b | Fix illegible Fig. 1 | ✅ | `pareto_gate0.png` regenerated 7 Oct, 52KB, big font |
| R1.4c | Fix inconsistent statements about run counts | ✅ | L85: "200 iterations after 10 warmup iterations" consistent |
| R1.4d | Fix unexplained x86 ShuffleNetV2 vs MobileNetV2 FP32 latency ordering | 🟡 | L189 explains via matmul count / node expansion. Verify it actually answers |
| R1.4e | Grad-CAM figure adds little → shorten or remove | 🟡 | Softened to "descriptive attention only" but full section + 2.6MB figure still present |
| R1.5 | Replace anonymous repo link with permanent archive | ✅ | No `4open`/`anonymous` string remains. Verify actual URL works |
| R1.5b | Temper generality of conclusions resting on single board / one runtime / one dataset / batch 1 | ✅ | L85 "Variability across independent benchmark sessions was not measured"; "model-only"; limitations in Conclusion |

---

## R2 — Strong Accept (6)

| # | Item | Status | Evidence / Action |
|---|---|---|---|
| R2.1 | Explain theoretically why depthwise-separable conv in EfficientNet-B0 is exceptionally sensitive to per-tensor vs MobileNetV2/ShuffleNetV2 | 🔴 | NOT answered. §IV-A says "consistent with" but no mechanism explanation. Needs a paragraph + R1.1 ablation as evidence |
| R2.2 | Deeply elaborate RPi5 threading inefficiency (ORT threading overhead / small-matrix bottleneck) | 🟡 | L189 has hypothesis + node expansion table. Needs more depth |
| R2.3 | Explicitly detail pHash audit so others can replicate the 8,153-image dedup split | ✅ | L73: manifest paths, 80/20 stratified seed 42, publisher test never re-split, pair/group semantics |
| R2.4 | Strong contribution / novelty | ✅ | R2 scored 4/4/4 — no action |

---

## R3 — Weak Reject (3)

| # | Item | Status | Evidence / Action |
|---|---|---|---|
| R3.1a | **Selective quantization ablations for DW / PW / SE** + layerwise weight or activation error analysis | 🔴 | Not done. Overlaps R1.1. ⛔ needs runs |
| R3.1b | OTHERWISE revise title, abstract, conclusions to say "granularity sensitivity" not "confirmed depthwise mechanism" | 🔴 | Title still "Localizing...". Abstract says "consistent with". **Binary choice: run ablation or retitle** |
| R3.2a | Distinguish graph operators from executed kernels; provide optimized-graph + kernel-dispatch evidence | 🔴 | Only softened to hypothesis. Need actual optimized graph + kernel dispatch data. ⛔ needs lab |
| R3.2b | CPU instruction availability alone doesn't prove instructions are used | ✅ | Removed the VNNI-style claim in commit aacf0b7 |
| R3.2c | Memory-bound claim needs evidence or cautious wording | ✅ | "we state this as a hypothesis since memory-boundedness was not directly measured" |
| R3.3 | **Benchmark static INT8 on RPi5, esp. per-channel that recovers EfficientNet-B0** | 🔴 | Zero data. Biggest gap. ⛔ needs Pi + static ONNX |
| R3.4a | McNemar or paired bootstrap CIs for FP32–INT8 diffs in MobileNetV2 and ShuffleNetV2 | 🔴 | McNemar exists for EfficientNet-B0 only. MNv2/SNv2 drops (0.81pp, 0.24pp) have NO paired test. `compute_ci.py` may support it |
| R3.4b | Distinguish lack of statistical significance from equivalence | ✅ | L91 explicit: "overlapping marginal intervals alone do not establish paired significance or equivalence" |
| R3.4c | Table II: specify test variant + discordant counts; report underflow via bound not p=0 | ✅ | "continuity correction", b=14 c=1468, `p < 10^{-15}` |
| R3.5a | Explain validation split creation (source release has train/test, audit refers to validation) | ✅ | L73: "stratified 80/20 split of the publisher train only (seed 42); publisher test was never re-split" |
| R3.5b | Distinguish duplicate pairs vs connected groups vs removed images; explain overlapping near-dup handling | ✅ | L73: "all size 2", "counts refer to removed copies rather than connected groups" |
| R3.5c | Replace absolute "sterile" with qualified statement | 🟡 | Says "under this audit" but the word "sterile" itself still used 23×. Consider replacing term entirely |
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
| R3.8b | Resolve discrepancy: text says single Grad-CAM map, Fig 5 shows four examples | 🔴 | Must check caption vs figure |
| R3.8c | Avoid inferring low-level feature use from visualizations alone | 🟡 | Check §Grad-CAM wording |
| R3.8d | Explain criterion behind "best balance" recommendation for MobileNetV2 | ✅ | Phrase "best balance" no longer present |
| R3.8e | State Table VI results are contextual, different protocols | ✅ | L244 + L262 |

---

## 🔴 P0 — must resolve before camera-ready

| ID | Item | Needs HW? | Owner |
|---|---|---|---|
| **P0-1** | Retitle OR run selective ablation (R3.1b + R1.1 + R2.1) | yes (ablation) / no (retitle) | decision first |
| **P0-2** | Static INT8 benchmark on RPi5 (R3.3) | ⛔ Pi + static ONNX | Fatih |
| **P0-3** | Optimized-graph + kernel-dispatch evidence (R3.2a) | ⛔ lab ORT profiling | Fatih |
| **P0-4** | McNemar / paired bootstrap for MNv2 + SNv2 (R3.4a) | no — data exists | agent can do |

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
