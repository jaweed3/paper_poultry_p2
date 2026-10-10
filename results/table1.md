| Model | FP32 % [95% CI] | INT8 dyn % [95% CI] | Drop pp |
|---|---|---|---|
| MobileNetV2 | 97.92 (97.22--98.55) | 97.28 | 0.64 |
| ShuffleNetV2 | 97.69 (96.93--98.38) | 97.63 | 0.06 |
| EfficientNet-B0 | 98.55 (97.97--99.07) | 10.42 (8.97--11.92) | 88.13 |

EB0 static: per-tensor 9.32% vs per-channel 94.56% (dynamic 10.42%).
