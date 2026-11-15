# Localizing INT8 Quantization Collapse & Channel Pruning for Resilient Edge Poultry Disease Diagnosis

[![Target Journal](https://img.shields.io/badge/Target_Journal-JCSE_(Scopus_Q4/ESCI)-blue.svg)](https://jcse.kiise.org/)
[![Publisher](https://img.shields.io/badge/Publisher-KIISE-navy.svg)](https://www.kiise.or.kr/)
[![Conference Baseline](https://img.shields.io/badge/Conference-ICAIDES_2026-green.svg)](https://icaides.org/)
[![Hardware](https://img.shields.io/badge/Edge_Device-Raspberry_Pi_5-red.svg)](https://www.raspberrypi.com/products/raspberry-pi-5/)
[![Dataset Status](https://img.shields.io/badge/Dataset-Sanitized_Sterile_(8,153_images)-orange.svg)](#dataset-integrity-audit)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Official research repository and reproducible artifact for the journal paper:
> **"Localizing INT8 Quantization Collapse and Channel Pruning for Resilient Edge Poultry Disease Diagnosis"**  
> *Fatih Jawwad Al Mumtaz and Farrel Ghozy Afifuddin*  
> Submitted to **Journal of Computing Science and Engineering (JCSE)**, Korean Institute of Information Scientists and Engineers (KIISE).

---

## 1. Key Empirical Findings

### A. Model Accuracy & Quantization Robustness (Sterile Split, $n = 1,728$)

| Architecture | FP32 Acc (%) | Model Size (MB) | INT8 Dynamic Acc (%) | INT8 Dynamic Size (MB) | INT8 Static Per-Channel (%) | INT8 Static Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ShuffleNetV2 0.5x** | 97.69% | 4.89 | 97.63% (-0.06%) | 1.47 | 96.88% (-0.81%) | 1.52 |
| **MobileNetV2 1.0x** | 97.92% | 8.48 | 97.28% (-0.64%) | 2.30 | 97.11% (-0.81%) | 2.38 |
| **EfficientNet-B0** | 98.55% | 15.30 | **10.42% (COLLAPSE)** | 4.16 | **94.56% (RECOVERED)** | 4.67 |

> **Diagnostic Root Cause:** Depthwise separable convolutions in EfficientNet-B0 exhibit extreme cross-channel dynamic range (weight dynamic range $\Delta_w > 14.8$). Per-tensor quantization clamps delicate activation channels, collapsing Signal-to-Quantization-Noise Ratio (SQNR) to **8.4 dB**. Static per-channel QDQ calibration allocates dedicated scale factors per depthwise slice, restoring SQNR to **21.8 dB** and recovering accuracy to **94.56%** ($p < 10^{-15}$, paired McNemar test).

---

### B. Structured Pruning + INT8 Quantization Matrix

Using dependency-graph-aware L1 structured pruning (`torch_pruning`), channels were pruned and fine-tuned for 12 epochs on the sterile split:

| Pipeline Variant | Params | Active (%) | FP32 Acc (%) | INT8 Acc (%) | INT8 Size (MB) | Total Compression |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV2 Baseline** | 2,228,996 | 100.0% | 97.92% | 97.11% | 2.38 MB | 3.56x |
| **MobileNetV2 Pruned s30 (Sweet Spot)** | **1,110,790** | **49.8%** | **93.81%** | **91.09%** | **1.23 MB** | **6.89x** |
| **MobileNetV2 Pruned s50 (Accuracy Cliff)** | **583,332** | **26.2%** | **90.05%** | **85.47%** | **0.72 MB** (733 KB) | **11.78x** |
| **EfficientNet-B0 Baseline** | 4,012,672 | 100.0% | 98.55% | 94.56% | 4.67 MB | 3.28x |
| **EfficientNet-B0 Pruned s30 (Static QDQ)**| **2,005,801** | **50.0%** | **93.69%** | **63.72%** | **2.62 MB** | **5.84x** |

---

### C. Physical Edge Profiling on Raspberry Pi 5 (Quad-Core Cortex-A76 @ 2.4 GHz)

Evaluated on physical hardware running ONNX Runtime 1.29.0 (`CPUExecutionProvider`, 100 timed runs + 15 warmup):

| Architecture Variant | Model Size (MB) | 1-Thread Latency | 4-Thread Latency | 4-Thread FPS | Peak RSS (MB) | Energy / Inf. (mJ) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV2 Baseline FP32** | 8.48 | 32.85 ms | 19.48 ms | 51.33 | 86.05 | 113.0 mJ |
| **MobileNetV2 Pruned s30 FP32** | 4.20 | 19.52 ms | **11.07 ms** | **90.33** | 79.97 | 64.2 mJ |
| **MobileNetV2 Pruned s30 INT8** | **1.19** | 28.41 ms | 25.93 ms | 38.56 | 80.23 | 150.4 mJ |
| **MobileNetV2 Pruned s50 FP32** | 2.23 | 11.80 ms | **6.90 ms** | **144.84** | 72.97 | **40.0 mJ** |
| **MobileNetV2 Pruned s50 INT8** | **0.72** | 19.61 ms | 18.88 ms | 52.97 | 77.52 | 109.5 mJ |
| **EfficientNet-B0 Baseline FP32** | 15.30 | 68.99 ms | 39.31 ms | 25.44 | 103.34 | 228.0 mJ |
| **EfficientNet-B0 Static QDQ INT8** | 4.67 | 24.64 ms | **19.40 ms** | **51.54** | 89.05 | 112.5 mJ |
| **EfficientNet-B0 Pruned s30 Static**| **2.62** | 22.59 ms | **16.55 ms** | **60.44** | 82.84 | **96.0 mJ** |
| **ShuffleNetV2 Baseline FP32** | 4.89 | 14.13 ms | **9.03 ms** | **110.69** | 75.27 | 52.4 mJ |

---

## 2. Dataset Integrity Audit & Sterile Partitioning

The published 8,770-image poultry benchmark (`Dianyo/poultry-fecal-fl`) claimed zero data leakage. Our perceptual hash audit (pHash, Hamming distance $\le 5$) revealed:
- **617 near-duplicate image pairs** caused by burst photography.
- **94 test set images byte-identical to training set images**.

To establish scientific validity, all experiments use the sanitized **8,153-image sterile split** (1,728 isolated test images, stratified 4-class distribution: Coccidiosis, Healthy, New Castle Disease, Salmonella).
- Manifest: `results/sterile_split_manifest.json`
- Dedup Log: `results/phash_dedup.json`

---

## 3. Repository Structure

```text
.
├── paper/                         # Complete JCSE LaTeX submission package
│   ├── manuscript_jcse.tex        # 12-page pure academic narrative manuscript
│   ├── manuscript_jcse.pdf        # Compiled camera-ready PDF with embedded figures & author bios
│   ├── jcse.cls & jcse.bst        # Official KIISE JCSE LaTeX template files
│   ├── setup.tex & geometry.sty   # Macro definitions and geometry configurations
│   ├── logo.pdf & logo.eps        # Official JCSE publication masthead
│   └── figures/                   # All 8 high-resolution publication figures & author photos
├── scripts/                       # Reproducible experiment workflows
│   ├── benchmark_pi_pruned.py     # Multi-thread Raspberry Pi 5 benchmark harness
│   ├── plot_system_pipeline.py    # Generates Fig. 1 system architecture diagram
│   ├── plot_pareto_jcse.py        # Generates Fig. 6 Pareto compression frontier
│   ├── plot_rpi5_pruned.py        # Generates Fig. 8 physical hardware latency & FPS plot
│   ├── train_pruned.py            # L1 structured pruning + fine-tuning pipeline
│   ├── eval_pruned.py             # Validation & ONNX INT8 export for pruned models
│   ├── ablate_layers.py           # Layer-by-layer SQNR & error propagation ablation
│   └── static_eb0.py              # Per-tensor vs per-channel calibration & McNemar test
├── results/                       # Empirical logs, evaluation JSONs, and test manifests
│   ├── rpi5_pruned_benchmark.json # Physical Raspberry Pi 5 latency, FPS, & RSS measurements
│   ├── sterile_split_manifest.json# Sanitized deduplicated dataset split (n=1,728)
│   └── phash_dedup.json           # Perceptual hash duplicate audit logs
├── figures/                       # Vector EPS & PNG publication figures
├── onnx_pruned/                   # Exported pruned & quantized ONNX models (s30, s50)
├── config.yaml                    # Hyperparameter & quantization configuration
└── pyproject.toml                 # Dependency specifications
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

# Install dependencies
pip install -e .
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

3. **Train & Fine-Tune Structured Pruning (CUDA):**
   ```bash
   python scripts/train_pruned.py --model mobilenetv2 --sparsity 0.3 --epochs 12
   python scripts/eval_pruned.py
   ```

4. **Benchmark on Raspberry Pi 5 (via SSH):**
   ```bash
   ssh pi "python3 ~/p2bench/benchmark_pi_pruned.py --runs 100 --warmup 15"
   ```

5. **Compile JCSE Manuscript:**
   ```bash
   cd paper
   pdflatex manuscript_jcse.tex
   pdflatex manuscript_jcse.tex
   # Produces 12-page manuscript_jcse.pdf
   ```

---

## 5. Zenodo Artifact Submission

This repository contains the complete experimental artifact, sterile split manifests, trained model checkpoints, and calibration code for archival on Zenodo.
- Recommended Title: *Replication Package for: Localizing INT8 Quantization Collapse and Channel Pruning for Resilient Edge Poultry Disease Diagnosis*
- Keywords: Edge AI, Neural Network Compression, Post-Training Quantization, Structured Pruning, Precision Agriculture, Raspberry Pi 5.

---

## 6. Citation

```bibtex
@article{almumtaz2026localizing,
  title={Localizing INT8 Quantization Collapse and Channel Pruning for Resilient Edge Poultry Disease Diagnosis},
  author={Al Mumtaz, Fatih Jawwad and Afifuddin, Farrel Ghozy},
  journal={Journal of Computing Science and Engineering (JCSE)},
  volume={20},
  number={4},
  pages={1--12},
  year={2026},
  publisher={Korean Institute of Information Scientists and Engineers (KIISE)},
  note={Submitted for Publication}
}
```
