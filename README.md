# Poultry Fecal Disease Detection — Edge ML

Diagnosing INT8 Quantization Collapse in EfficientNet-B0: Per-Channel Recovery and Scheme x Runtime Characterization for Poultry Fecal Disease Detection on Raspberry Pi 5.

## Models (sterile split, test 1728)

| Model | FP32 Acc | INT8 Acc | FP32 Size | INT8 Size | RPi5 FP32 t=4 |
|-------|----------|----------|-----------|-----------|---------------|
| MobileNetV2 | 97.92% | 97.11% | 8.48 MB | 2.30 MB | 14.23 ms |
| ShuffleNetV2 | 97.69% | 97.45% | 4.89 MB | 1.47 MB | 6.19 ms |
| EfficientNet-B0 | 98.55% | 10.42% | 15.30 MB | 4.16 MB | 31.56 ms |

Key diagnosis: EB0 static QDQ per-tensor 9.32% vs per-channel 94.56% (McNemar vs dynamic p~1e-312).

Benchmark: Raspberry Pi 5, ONNX Runtime 1.29.0, 200 runs, sterile-split models.

## Dataset

Machuve et al. (2022) — 8,770 images, 4 classes (Coccidiosis, Healthy, NCD, Salmonellosis).

## Project Structure

```
├── train_pipeline.py          # Training + ONNX export + INT8 PTQ + Grad-CAM
├── eval_int8.py               # FP32 vs INT8 accuracy evaluation
├── eval_int8_batched.py       # Batched version (faster)
├── models/
│   ├── fp32/                  # PyTorch weights (.pth)
│   ├── onnx/                  # ONNX FP32 models
│   └── ptq/                   # ONNX INT8 quantized models
├── results/
│   ├── experiment_results.json
│   ├── int8_real_accuracy.json
│   └── int8_real_accuracy_lab.json
├── figures/                   # Training curves, confusion matrices, Grad-CAM, Pareto
├── paper/                     # LaTeX source + figures
└── data/images/               # Dataset (train/val/test split)
```

## Reproduce

```bash
pip install torch torchvision onnx onnxruntime numpy matplotlib seaborn tqdm Pillow pytorch_grad_cam
python train_pipeline.py
```

## Key Finding

INT8 PTQ on CPU introduces 6.4–30.1× latency overhead due to dequantization costs. Quantization does not always improve inference speed on CPU without hardware acceleration (VNNI/Tensor Cores).

## License

Research use only.
