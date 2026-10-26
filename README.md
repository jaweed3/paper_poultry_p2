# Localizing INT8 Quantization Collapse in EfficientNet-B0 for Edge Poultry Disease Detection

Camera-ready research artifact for **ICAIDES 2026**. Lightweight CNNs (MobileNetV2, ShuffleNetV2, EfficientNet-B0) for poultry-fecal disease classification, quantized to INT8 and benchmarked on a Raspberry Pi 5.

**Headline result:** dynamic INT8 barely touches MobileNetV2/ShuffleNetV2 but collapses EfficientNet-B0 to 10.42%. Static per-channel quantization recovers it to 94.56% — and runs 2.34x faster than FP32 on the Pi.

## Accuracy (sterile test split, n = 1728)

| Model | FP32 | INT8 dynamic | FP32 size | INT8 size |
|---|---|---|---|---|
| MobileNetV2 | 97.92% | 97.28% | 8.48 MB | 2.30 MB |
| ShuffleNetV2 | 97.69% | 97.63% | 4.89 MB | 1.47 MB |
| EfficientNet-B0 | 98.55% | 10.42% | 15.30 MB | 4.16 MB |

Static quantization on EfficientNet-B0 (100-image calibration, ONNX Runtime 1.29.0):

| Scheme | Accuracy | Size | RPi5 latency (4 threads) |
|---|---|---|---|
| Per-tensor | 9.32% | 4.29 MB | 13.63 ms |
| Per-channel | **94.56%** | 4.67 MB | **13.50 ms** |

Per-channel vs. dynamic, paired McNemar on the same 1728 test images: b = 14, c = 1468, p < 1e-15. Against FP32 the gap is still significant (b = 75, c = 7, p ~ 1e-13) — recovery, not parity.

A retrained EfficientNet-B0 (FP32 98.15%) with a stratified 25-images/class calibration slice confirms the pattern: per-tensor 27.14% vs. per-channel 93.40% (`results/balanced_replication/`).

## Latency on Raspberry Pi 5 (ONNX Runtime 1.29.0, 200 runs, batch 1, 4 threads)

| Model | FP32 | INT8 dynamic | Slowdown |
|---|---|---|---|
| MobileNetV2 | 14.23 ms | 39.83 ms | 2.80x |
| ShuffleNetV2 | 6.19 ms | 11.95 ms | 1.93x |
| EfficientNet-B0 | 31.53 ms | 63.99 ms | 2.03x |

Dynamic INT8 is 1.93–2.80x *slower* than FP32 here — the runtime's quantize/dequantize path around every convolution costs more than it saves (kernel-level evidence in `results/kernel_dispatch_artifact.json`, Sec. IV-C of the paper). Static per-channel QDQ flips that: 13.50 ms vs. 31.53 ms FP32, 2.34x faster.

## Dataset integrity audit

The published 8,770-image benchmark (Dianyo/poultry-fecal-fl, derived from Machuve et al. Zenodo 4628934) was claimed leak-free. A pHash audit (Hamming <= 5) found **617 near-duplicate pairs**, including **94 test images byte-identical to training images**. All reported numbers use the deduplicated **8,153-image split** (test n = 1728, seed 42). Artifacts: `results/phash_dedup.json`, `results/sterile_split_manifest.json`.

Two attributed sample images per class (demo only, not the benchmark) live in [`sample_data/`](sample_data/).

## Reproduce

```bash
pip install torch torchvision onnx onnxruntime numpy matplotlib seaborn tqdm Pillow
python train_pipeline.py          # train + ONNX export + INT8 PTQ + Grad-CAM
```

Key scripts in [`scripts/`](scripts/): `rebuild_sterile.py` (dedup split), `train_gate0.py`, `eval_gate0.py`, `static_eb0.py` (per-tensor vs. per-channel + McNemar), `benchmark_pi_gate0.py` (RPi5 sweep), `retrain_eb0_balanced.py` (stratified-calibration replication).

Manuscript source: [`paper/main.tex`](paper/main.tex) (5 pages, A4) — compiled PDF at [`paper/main.pdf`](paper/main.pdf).

## Layout

```
train_pipeline.py          training + export + PTQ + Grad-CAM
eval_int8.py               FP32 vs INT8 accuracy
models/                    fp32 (.pth), onnx (FP32), ptq (INT8)
results/                   metrics, manifests, RPi5 + static + McNemar artifacts
figures/                   training curves, confusion matrices, Grad-CAM, Pareto
paper/                     LaTeX source + compiled PDF
scripts/                   rebuild / train / eval / benchmark / plot / retrain
sample_data/               8 attributed demo images (2 per class) + manifest
```

## Citation

If you use this artifact, cite the ICAIDES 2026 paper and the Zenodo record (DOI: `10.5281/zenodo.23374216`). Machine-readable metadata: [`CITATION.cff`](CITATION.cff).

## License

Research use only.
