# Localizing INT8 Quantization Collapse & Pruning for Edge Poultry Disease Detection

[![Target Journal](https://img.shields.io/badge/Target_Journal-IJEEI_(Scopus_Q3)-blue.svg)](http://www.ijeei.org/)
[![Conference Baseline](https://img.shields.io/badge/Conference-ICAIDES_2026-green.svg)](https://icaides.org/)
[![Hardware](https://img.shields.io/badge/Edge_Device-Raspberry_Pi_5-red.svg)](https://www.raspberrypi.com/products/raspberry-pi-5/)
[![Dataset Status](https://img.shields.io/badge/Dataset-Sanitized_Sterile_(8,153_images)-orange.svg)](#dataset-integrity-audit)

Research codebase and experimental artifact for poultry-fecal disease classification under extreme edge resource constraints. This repository contains the complete experimental pipeline:
1. **Conference Baseline (ICAIDES 2026):** Discovery and diagnosis of INT8 dynamic quantization collapse in EfficientNet-B0 down to 10.42%, mitigated by static per-channel QDQ calibration (94.56%).
2. **Journal Extension (Target: IJEEI, Scopus Q3):** Systematic SQNR signal-to-quantization-noise ratio layer ablation, L1 structured channel pruning (Torch-Pruning) fine-tuned on an NVIDIA RTX 4060, reducing model footprint to **1.23 MB** at **91.09% test accuracy**, and end-to-end Raspberry Pi 5 ARM CPU benchmarking.

---

## 1. Key Empirical Findings

### A. Model Accuracy & Quantization Robustness (Sterile Split, $n = 1,728$)

| Architecture | FP32 Acc (%) | Model Size (MB) | INT8 Dynamic Acc (%) | INT8 Dynamic Size (MB) | INT8 Static Per-Channel (%) | INT8 Static Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ShuffleNetV2 0.5x** | 97.69% | 4.89 | 97.63% (-0.06%) | 1.47 | 96.88% (-0.81%) | 1.52 |
| **MobileNetV2 1.0x** | 97.92% | 8.48 | 97.28% (-0.64%) | 2.30 | 97.11% (-0.81%) | 2.38 |
| **EfficientNet-B0** | 98.55% | 15.30 | **10.42% (COLLAPSE)** | 4.16 | **94.56% (RECOVERED)** | 4.67 |

> **Diagnostic Root Cause:** Depthwise separable convolutions in EfficientNet-B0 exhibit extreme cross-channel dynamic range (weight dynamic range $\Delta_w > 14.8$). Per-tensor quantization clamps delicate activation channels, collapsing Signal-to-Quantization-Noise Ratio (SQNR) to **8.4 dB**. Static per-channel QDQ calibration allocates dedicated scale factors per depthwise slice, restoring SQNR to **21.8 dB** and recovering accuracy to **94.56%** ($p < 10^{-15}$, paired McNemar test).

---

### B. Structured Pruning + INT8 Quantization (Journal Extension)

Using dependency-graph-aware L1 structured pruning (`torch_pruning`), channels were pruned and fine-tuned for 12 epochs on the sterile split:

| Pipeline Variant | Params (M) | Sparsity | FP32 Acc (%) | INT8 Acc (%) | Model Size (INT8) | Total Compression |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense MobileNetV2 (Baseline)** | 2.23M | 0.0% | 97.92% | 97.11% | 2.38 MB | 3.56x |
| **Pruned MobileNetV2 (30% Sparsity)** | **1.11M** | **49.8%** | **93.81%** | **91.09%** | **1.23 MB** | **6.89x** |

- Artifact: `results/eval_pruned_s30.json`
- Script: `scripts/train_pruned.py` & `scripts/eval_pruned.py`

---

### C. Edge Latency on Raspberry Pi 5 (ONNX Runtime 1.29.0, 4 Threads)

| Architecture | FP32 Latency | INT8 Dynamic Latency | INT8 Static Per-Channel | Static Speedup vs FP32 |
| :--- | :---: | :---: | :---: | :---: |
| **ShuffleNetV2** | 6.19 ms | 11.95 ms (1.93x slower) | **4.98 ms** | **1.24x** |
| **MobileNetV2** | 14.23 ms | 39.83 ms (2.80x slower) | **8.12 ms** | **1.75x** |
| **EfficientNet-B0** | 31.53 ms | 63.99 ms (2.03x slower) | **13.50 ms** | **2.34x** |

> **Warning on Dynamic Quantization on ARM:** Dynamic INT8 causes substantial latency regressions (up to 2.80x slower) due to runtime quantize/dequantize memory roundtrips. Static QDQ precomputes scales and enables optimized NEON/GEMM integer kernel execution.

---

## 2. Dataset Integrity Audit

The published 8,770-image poultry benchmark (`Dianyo/poultry-fecal-fl`) claimed zero data leakage. Our perceptual hash audit (pHash, Hamming distance $\le 5$) revealed:
- **617 near-duplicate image pairs**.
- **94 test set images byte-identical to training set images**.

To establish scientific validity, all experiments use the sanitized **8,153-image sterile split** (1,728 isolated test images, stratified 4-class distribution: Coccidiosis, Healthy, New Castle Disease, Salmonella).
- Manifest: `results/sterile_split_manifest.json`
- Dedup Log: `results/phash_dedup.json`

---

## 3. Repository Structure

```text
.
├── paper/                         # Journal manuscripts & camera-ready source
│   ├── MANUSCRIPT_IJEEI.docx      # Formatted submission manuscript for IJEEI (single-column, Scopus Q3)
│   ├── MANUSCRIPT_IJEEI.md        # Complete journal text draft with comprehensive analysis
│   ├── manuscript_ijeei.tex       # LaTeX journal version
│   ├── main.tex & main.pdf        # Original 5-page ICAIDES conference paper
│   └── figures/                   # Confusion matrices, Pareto frontier, thread scaling, Grad-CAM
├── scripts/                       # Reproducible experiment workflows
│   ├── train_pruned.py            # L1 structured pruning + fine-tuning pipeline
│   ├── eval_pruned.py             # Validation & ONNX INT8 export for pruned models
│   ├── build_ijeei_docx.py        # Automated docx builder matching IJEEI journal template
│   ├── ablate_layers.py           # Layer-by-layer SQNR & error propagation ablation
│   ├── static_eb0.py              # Per-tensor vs per-channel calibration & McNemar test
│   └── benchmark_rpi5.py          # Multithreaded Raspberry Pi 5 benchmark harness
├── results/                       # Empirical logs, evaluation JSONs, and test manifests
│   ├── eval_pruned_s30.json       # Pruned model validation & quantization accuracy
│   ├── sterile_split_manifest.json# Sanitized deduplicated dataset split
│   └── rpi5_benchmark/            # Latency, throughput, and power logs on RPi5
├── archive/                       # Archived conference files (camera ready, QA prep, legacy zips)
│   ├── icaides_camera_ready/
│   └── icaides_qa_prep/
├── sample_data/                   # Attributed sample images for quick local inference testing
├── config.yaml                    # Hyperparameter & quantization configuration
└── pyproject.toml                 # Dependency specifications (uv / pip)
```

---

## 4. Quickstart & Reproduction

### Prerequisites
- Python 3.10+
- PyTorch 2.5+, ONNX, ONNX Runtime, Torch-Pruning, torchvision

```bash
# Clone repository
git clone https://gitlab.com/wedjaw22/paper-poultry-journal.git
cd paper-poultry-journal

# Install dependencies using uv or pip
uv pip install -e .
# or
pip install -r requirements.txt
```

### Reproduce Experiments

1. **Verify Dataset Integrity:**
   ```bash
   python scripts/phash_dedup.py
   python verify_dataset.py
   ```

2. **Run Static Per-Channel Quantization Recovery:**
   ```bash
   python scripts/static_eb0.py
   ```

3. **Train & Fine-Tune Structured Pruning (RTX 4060 / CUDA):**
   ```bash
   python scripts/train_pruned.py --model mobilenetv2 --sparsity 0.3 --epochs 12
   python scripts/eval_pruned.py
   ```

4. **Compile IJEEI Submission Manuscript:**
   ```bash
   python scripts/build_ijeei_docx.py
   # Generates paper/MANUSCRIPT_IJEEI.docx matching IJEEI single-column format
   ```

---

## 5. Target Publication

- **Journal:** *International Journal on Electrical Engineering and Informatics (IJEEI)*
- **Publisher:** School of Electrical Engineering and Informatics, Institut Teknologi Bandung (ITB)
- **Indexing:** Scopus Q3, CiteScore 1.7, SJR 0.22
- **Manuscript File:** [`paper/MANUSCRIPT_IJEEI.docx`](paper/MANUSCRIPT_IJEEI.docx)

---

## 6. Citation

```bibtex
@article{mumtaz2026poultry,
  title={Structured Channel Pruning and Static INT8 Calibration for Resilient Edge Poultry Disease Diagnosis},
  author={Mumtaz, Fatih Jawwad Al and Aziz, Amirul Al and Hidayatullah, Muhammad Iqbal},
  journal={International Journal on Electrical Engineering and Informatics (IJEEI)},
  year={2026},
  note={Under Review}
}
```
