# Camera-Ready Status — ICAIDES #1571359798

**Deadline: 31 Okt 2026** (~23 hari)
**Branch:** `post30-gate0-dedup` (clean; verify with `git status` before release)
**Title:** "Localizing INT8 Quantization Collapse in EfficientNet-B0 for Edge Poultry Disease Detection"

## Review scores

| Reviewer | Novelty | Tech | Presentation | Rekomendasi |
|---|---|---|---|---|
| R1 | 3 Moderate | 3 Good | 3 Above avg | **Weak Accept** |
| R2 | 4 High | 4 Exceptional | 4 Excellent | **Strong Accept** |
| R3 | 2 Limited | 2 Sufficient | 3 Above avg | **Weak Reject** |

R3 yang bikin revisi pedas. R2 bilang "merit serious consideration for outstanding paper award".

---

## SUDAH SELESAI (terbukti di main.tex)

| Item review | Bukti di main.tex |
|---|---|
| R3: paired significance test | `McNemar` 5x + `bootstrap` 4x; b=14, c=1468; p underflow diberi bound `p < 10^{-15}` bukan `p=0` |
| R3: "overlapping CI ≠ insignificant" | L91: eksplisit bilang overlapping marginal intervals tidak cukup, no significance claim tanpa paired test |
| R3: "sterile" → qualified | L38/55/73 pakai "sterile split" + disclaimer "under this audit" |
| R3: grafik label kecil | `pareto_gate0.png` 52KB, `rpi5_thread_scaling_gate0.png` 82KB — already regenerated |
| R1: Grad-CAMfigura | L108 Grad-CAM = descriptive attention only; 2.6MB PNG masih ada tapi caption sudah di-soften |
| R3: Table VI caveat | L244 caption: "different protocols; not head-to-head" |
| R1: energy limitation | L82/85/270: current telemetry `current_a=null`, energy = stated limitation |
| R2: pHash dedup reproducibility | L73: split manifest + audit artifact paths, 80/20 stratified seed 42, publisher test never re-split |
| R2: threading inefficiency hypothesis | L189: "we state this as a hypothesis since memory-boundedness was not directly measured" |
| R1: anonymous repo → permanent | Tidak ada lagi string `4open`/`anonymous` — sudah diganti |
| R3: per-class P/R/F1 + macro + balanced acc | L108 lengkap untuk 3 model × 2 kuantisasi |
| R3: reproducibility details | L82: opset 13, activation/weight type, QuantFormat, 100 img calib, batch 8, ORT 1.29.0, ONNX 1.22.0, 200 runs + 10 warmup |
| R3: split construction | L73: validasi = stratified 80/20 dari publisher train, bukan re-split test |

---

## BELUM SELESAI (sorted by severity)

### 🔴 P0 — WAJIB, kalau tidak kemungkinan besar di-reject camera-ready

**1. Judul dan selective ablation** — DONE. Selective per-family ablation is now in `paper/main.tex` and supports the mechanism claim.

The old "selective ablations remain future work" wording was removed from the current paper; the current title is supported by the selective per-family ablation reported in Table IV.

No retitling is required for this evidence-backed version.

**2. Static INT8 di RPi5** — DONE. Per-channel static QDQ: 13.50 ms versus FP32 31.53 ms at four threads (2.34× faster), with 94.56% ONNX accuracy.

Artifact: `results/rpi5_static_gate0/rpi5_static.json`.

**3. Runtime explanation** — DONE for the submitted scope. Executed-kernel profiling is in `results/profiling/kernel_dispatch_artifact.json` and described in §IV-C.

Remaining claims about memory-boundedness are explicitly labelled as hypotheses because power/energy was not measured.

**4. Energy belum diukur** — R1 item 3: *"report energy or at least discuss its absence as a limitation."*

L82/85 sudah bilang limitation (`current_a=null`). Kalau RPi5 lo masih punya akses → ukur p Crack. Kalau tidak, limitation statement sudah cukup secara formal.

---

### 🟡 P1 — setelah P0 selesai

**5. Multi-seed + calibration-size sweep** — R1 item 1: *"report multiple seeds and calibration-set sizes with confidence intervals."*

Lo cuma punya 1 seed (42) dan 1 calib size (100). Reviewer mau sweep.

**6. Fair comparison prior work** — R1 item 3: *"Add a fair comparison on the same deduplicated split against prior work."*

L244 sekarang cuma caveat. Kalau mau jawab: re-run 1 model dari paper prior (misal Tasdelen MobileNetV2) di split lo.

---

### 🟢 P2 — polish

| Item | Status |
|---|---|
| R3: Fig 5 Grad-CAM "4 contoh vs deskripsi 1 map" | perlu dicek caption Fig 5 |
| R3: "per-class accuracy" definisi | ✅ L108 sudah dijelasin = recall |
| R3: preprocessing masuk timing? | ✅ L85 "model-only, preprocessing excluded" |
| R3: "best balance" MobileNetV2 | ✅ kata "best balance" tidak ada, sudah diganti |
| R1: run counts inconsistent | ✅ L85 "200 iterations after 10 warmup" konsisten |

---

## XPLOR (jawaban per reviewer)

**Soal selective ablation (P0-1)** — ini kunci. Kalau lo punya waktu Pi/board, ablation depthwise/pointwise/SE + static INT8 di RPi5 bisa jadi **satu paper IEEE kedua** dari data yang sama. Kalau tidak, rename judul dan survival.

**Deadline keras: 31 Okt.** Registrasi + video pre-recorded + PDF Express + e-copyright + presenter di EDAS. Semua itu butuh waktu, jangan ditunda ke minggu terakhir.

---

## Struktur files

```
paper_poultry_p2/
  paper/main.tex        (329 lines, 6pp A4)
  paper/*.png           (5 figures)
  scripts/              (benchmark_pi_gate0.py, dll)
  results/              (phash_dedup.json, sterile_split_manifest.json, profiling/)
```
