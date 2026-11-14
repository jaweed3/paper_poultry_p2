# ICAIDES 2026 — Video Script (12 slides, ~7:15 @ 130 wpm, max 10:00)

# Paper ID 1571359798. Bahasa: Inggris. 1 slide = 1 take. Divider (4 slide) dipotong saat edit.

# Sumber angka: paper/main.tex + results/*.json. Jangan tambah isi — sisa waktu buat jeda/napas.

## 1. Title (0:20)

Hello, everyone. I'm Fatih Jawwad Al Mumtaz from Universitas Darussalam Gontor,
presenting joint work with Amirul Al Aziz and Muhammad Iqbal Hidayatullah.
Today I'll show where INT8 quantization breaks EfficientNet-B0 on a Raspberry Pi 5, and what actually fixes it.

## 2. The gap this paper closes (0:40)

Poultry diseases like Coccidiosis, Salmonellosis, and Newcastle disease cause major losses,
and small farms often cannot afford laboratory diagnosis.
Fecal image classifiers already reach about 99 percent accuracy, but they target the cloud.
A farm needs a model that runs on a cheap board.
So we ask two questions. Which lightweight architectures survive INT8 on a CPU?
And what is the accuracy-latency tradeoff on a Raspberry Pi 5?
The scope is deliberately narrow: three models, one board, one runtime.

## 3. The benchmark (0:35)

Before any quantization, we audited the dataset.
It is published as 8,770 leak-free images. Using perceptual hashing, we found 617 near-duplicate pairs,
and 94 test images are byte-identical to training images.
We removed them, leaving 8,153 images, with 1,728 in the test set.
Every number in this talk uses this deduplicated split.

## 4. Dynamic INT8 table (0:35)

First result. With dynamic INT8 in ONNX Runtime,
MobileNetV2 and ShuffleNetV2 lose less than one percentage point.
EfficientNet-B0 falls from 98.55 percent to 10.42 percent, a drop of 88 points.
MobileNetV2's small drop is detectable with a paired McNemar test,
p equals 0.0455. ShuffleNetV2's is not. Same pipeline, same data, and one model is destroyed.

## 5. Confusion matrices (0:25)

The confusion matrix shows how it fails.
On the right, 1,676 of 1,728 test images are predicted as Newcastle.
Recall is 100 percent for Newcastle and below 5 percent for every other class.
The model has become a near single-class predictor.

## 6. Why: hypothesis (0:55)

Why does this happen? This is a hypothesis, and I want to be clear about that.
Per-tensor quantization uses one scale for the whole tensor,
with 127 levels on each side. The widest channel sets that scale.
A channel with a much smaller range gets squeezed into about one level,
so its information is effectively erased.
EfficientNet-B0's depthwise layers have a median channel-range ratio of about 40,
and the worst layer reaches 112.
For MobileNetV2 the median is 9, for ShuffleNetV2 it is 5, and both survive.
Our SQNR analysis and ablation support this, but a single-layer ablation would test it directly.
[SLIDE SOURCE: worst channel at 0 dB = depthwise_range_audit.json worst_sqnr_db, juga disebut di paper Sec IV-A. Kalau ditanya, tunjuk JSON + paper.]

## 7. Per-channel recovers it (0:45)

Now the fix. Per-channel static quantization gives each channel its own scale.
Accuracy goes from 9.32 percent with per-tensor to 94.56 percent with per-channel,
for only 0.4 megabytes more. This is recovery, not parity: it is still 3.9 points below FP32. One caveat up front.
Our calibration set was 100 images, all Coccidiosis, because the data loader was class-ordered.
So treat this as diagnostic evidence, not a robustness claim.
A balanced calibration replication shows the same pattern: 27 percent per-tensor, 93 percent per-channel.

## 8. Ablation (0:50)

Which layers matter? We quantized selectively.
Depthwise layers per-channel and everything else per-tensor gives 93.46 percent, no collapse.
Everything per-channel except depthwise collapses to 8.33 percent.
Leaving depthwise in FP32 also works, while leaving squeeze-and-excitation in FP32 does not help.
So in this setup, depthwise granularity decides the collapse.
Two limits: this ablation is at family level, on EfficientNet-B0 only,
and depthwise sensitivity is already known from Sheng and colleagues.
Our contribution is the case study and the runtime behavior.

## 9. On the Pi: dynamic INT8 is slower (0:40)

Now the Raspberry Pi 5, CPU only, batch size one.
Dynamic INT8 is slower than FP32 for all three models, by 1.9 to 2.8 times.
It also scales badly with threads: from one to four threads, FP32 speeds up 2.2 to 2.3 times, but INT8 only 1.1 to 1.2.
The reason is that each convolution expands into DynamicQuantizeLinear, ConvInteger,
a multiply, and a cast, and these quantization kernels add 21 to 29 percent overhead.

## 10. Static flips it (0:35)

Static quantization flips this. On EfficientNet-B0 with four threads,
FP32 takes about 31.5 milliseconds.
Static per-channel takes 13.5 milliseconds, 2.34 times faster, at 94.56 percent accuracy.
Per-tensor is just as fast, but useless at 9 percent accuracy.
So the slowdown belongs to dynamic quantization on this runtime, not to INT8 in general.
Again: one board, one runtime, one seed.

## 11. Which model do you ship? (0:40)

So which model do you ship? Not the INT8 EfficientNet.
ShuffleNetV2 in FP32 runs in 6.19 milliseconds at 97.69 percent accuracy.
That beats EfficientNet static per-channel on both speed and accuracy.
FP32 accuracy differs by less than one point across all three models,
so hardware and quantization scheme should drive the decision.
INT8 shrinks files by 3.3 to 3.7 times, but memory at inference barely changes.
Limitations: single-class calibration, one seed, and energy was not measured.

## 12. Takeaway (0:35)

To conclude: INT8 is not one thing. Scheme, granularity, and runtime decide together.
Dynamic INT8 preserves two models and collapses EfficientNet-B0.
Per-channel static recovers it and is faster than FP32, but still loses to ShuffleNetV2 FP32.
In this setup, depthwise granularity is what prevents the collapse.
Code, split manifests, and logs are on GitHub and Zenodo.
Thank you, and I welcome your questions.
[SLIDE TEXT: "decides collapse in this setup" — selaras dengan paper "strong sensitivity", bukan "necessary and sufficient".]

---

# Checklist rekam (deadline upload EDAS 6 Nov)

- [ ] Hari ini: baca skrip keras-keras + timer 3x (target 7:00-8:30)
- [ ] Besok: tes mic 30 dtk, rekam per slide (1 slide = 1 take, divider dipotong)
- [ ] Lusa: edit, HandBrake, cek < 150 MB, 720p MP4 16:9
- [ ] Cadangan 1 hari penuh buat masalah teknis — jangan upload hari terakhir
- Jangan hafal kata per kata. Paham alur, baca angka dari teks.
