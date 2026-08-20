# Paper Outline: Edge-Optimized Lightweight CNNs for Poultry Fecal Disease Detection
# UPDATED — Based on Literature Review (August 16, 2026)

## Target Venue
**AMICT 2026** — 3rd International Conference on Advances in Machine Intelligence and Cybersecurity Technologies
- Deadline: Sep 1, 2026
- Format: IEEE, 4-6 pages, English
- Publication: IEEE Xplore + Scopus
- Location: Labuan, Malaysia (hybrid)
- Fee: ~$150 student

## Working Title
**"Characterization of Lightweight CNN Architectures with INT8 Quantization for Poultry Fecal Disease Detection on Edge Devices"**

## Positioning Against Literature

| Paper | Their Approach | Our Difference |
|---|---|---|
| Dhungana (AgriEngineering 2025) | EB0=99.12%, YOLO11+EB0 pipeline, web-based | We do 3-arch comparison + INT8 + Pareto + Grad-CAM |
| Tasdelen (Elsevier 2025) | 6 archs, MobileNetV2=97.1%, no quantization | We include ShuffleNetV2, INT8 analysis, CPU latency |
| Degu (Smart Agri Tech 2023) | YOLO-V3+ResNet50=98.7%, heavy models | We use lightweight models, edge-focused |
| Hoang (Nature Sci Reports 2026) | 7 archs, tomato disease, XAI+PSS metric | We apply to poultry, add INT8 Pareto |
| Maulana (MDPI Computers 2026) | Review: QAT > PTQ | We demonstrate PTQ limitations empirically |

**Our unique angle:** First systematic comparison of lightweight CNNs + INT8 PTQ characterization + Pareto frontier + Grad-CAM interpretability specifically for poultry fecal disease detection.

---

## Abstract (Draft)

Poultry fecal disease detection using deep learning has shown promising accuracy, yet existing approaches rely on cloud-based inference with heavy architectures impractical for resource-constrained farm environments. This paper presents a systematic characterization of three lightweight convolutional neural network architectures—MobileNetV2, ShuffleNetV2, and EfficientNet-B0—for classifying poultry fecal images into four disease categories: Coccidiosis, Salmonellosis, Newcastle Disease, and Healthy. We evaluate each architecture under post-training INT8 quantization and analyze the accuracy-latency Pareto frontier on consumer CPU hardware. Our results demonstrate that all three architectures achieve over 97% test accuracy through ImageNet transfer learning, with EfficientNet-B0 achieving the highest accuracy (98.3%) and MobileNetV2 offering the best latency-accuracy trade-off (1.85 ms FP32, 541 FPS). Critically, we find that INT8 quantization on CPU introduces 6-30× latency overhead due to dequantization costs without hardware acceleration, contradicting the common assumption that quantization always improves inference speed. Grad-CAM visualizations reveal that each architecture learns distinct feature representations from fecal images, with EfficientNet-B0 attending to multi-scale diagnostic features. These findings provide practical deployment guidelines for edge-based poultry disease detection in resource-constrained farming environments.

---

## 1. Introduction (0.5-0.75 pages)

### 1.1 Background
- Poultry industry: USD 70.2B in US alone (Dhungana 2025), critical for food security
- Disease outbreaks cause billions in losses — early detection is key
- Manual fecal inspection: labor-intensive, requires expertise, not scalable
- Deep learning enables automated detection but most solutions require cloud/GPU

### 1.2 Problem Statement
- Existing poultry disease detection uses heavy models (ResNet50, Xception, ViT): 23M+ params
- Deployment on farm-level edge devices (RPi, smartphones, MCUs) is unexplored
- No systematic comparison of lightweight architectures for this specific domain
- INT8 quantization effectiveness on CPU is hardware-dependent and poorly characterized

### 1.3 Research Questions
1. How do lightweight CNN architectures (MobileNetV2, ShuffleNetV2, EfficientNet-B0) compare for poultry fecal disease classification?
2. What is the quantization sensitivity of each architecture under INT8 PTQ on CPU?
3. Where does each architecture sit on the accuracy-latency Pareto frontier?
4. How do different architectures learn domain-specific features from fecal images?

### 1.4 Contributions
1. First systematic comparison of three lightweight CNNs for poultry fecal disease with INT8 characterization
2. Empirical demonstration that INT8 PTQ on CPU is counterproductive (6-30× slower)
3. Accuracy-latency Pareto frontier enabling hardware-aware model selection
4. Grad-CAM interpretability analysis per architecture for domain-specific feature understanding

---

## 2. Related Work (0.5-0.75 pages)

### 2.1 Poultry Disease Detection
- Machuve et al. (2022): MobileNet TFLite, 94.12%, dataset source [OUR DATASET]
- Dhungana et al. (2025): EB0=99.12%, YOLO11+EB0 pipeline, augmentation for NCD
- Tasdelen & Arslan (2025): 6 archs comparison, MobileNetV2=97.1% best
- Degu & Simegn (2023): YOLO-V3+ResNet50=98.7%, smartphone approach
- **Gap:** No lightweight architecture comparison + quantization + Pareto analysis

### 2.2 Lightweight CNN Architectures
- MobileNetV2 (Sandler 2018): Inverted residuals, depthwise separable conv
- ShuffleNetV2 (Ma 2018): Channel shuffle, four design principles
- EfficientNet (Tan & Le 2019): Compound scaling, balanced depth/width/resolution
- Hoang et al. (2026): 7 archs for tomato disease, MobilePlantViT best for edge
- **Gap:** No comparison specifically for fecal image domain

### 2.3 Model Quantization for Edge Deployment
- Jacob et al. (2018): INT8 quantization fundamentals
- ONNX Runtime docs (2024): INT8 slower on CPU without VNNI/TensorCore
- GitHub Issues #6732, #12854, #26143: Community confirmation of CPU INT8 overhead
- Maulana & Ramasamy (2026): Systematic review — QAT > PTQ
- **Gap:** No quantization analysis in poultry domain

### 2.4 Explainable AI in Agriculture
- Selvaraju et al. (2017): Grad-CAM fundamentals
- Hoang et al. (2026): PSS metric for XAI comparison, Grad-CAM/SHAP/LIME
- **Gap:** No per-architecture Grad-CAM comparison for fecal images

---

## 3. Methodology (1-1.25 pages)

### 3.1 Dataset
- **Source:** Machuve et al. (2022), Zenodo record 4628934
- **Size:** 8,770 images (7,016 train → 5,614 train + 1,402 val, 1,754 test)
- **Classes:** Coccidiosis (30.7%), Healthy (29.2%), Salmonella (32.5%), Newcastle Disease (8.3%)
- **Note:** NCD class significantly underrepresented — analysis of per-class performance included
- **Preprocessing:** Resize 224×224, ImageNet normalization
- **Augmentation:** Random horizontal flip, rotation ±15°, color jitter

### 3.2 Model Architectures

| Model | Params | FLOPs | Key Mechanism |
|---|---|---|---|
| MobileNetV2 | 2.2M | 300M | Inverted residuals, depthwise separable conv |
| ShuffleNetV2 | 1.3M | 150M | Channel shuffle, split operation |
| EfficientNet-B0 | 4.0M | 400M | Compound scaling + squeeze-excitation |

- All pre-trained on ImageNet, classification head replaced for 4-class output
- Frozen backbone + fine-tuning approach

### 3.3 Training Protocol
- Optimizer: Adam (lr=1e-4, weight_decay=1e-4)
- Scheduler: Cosine annealing, 50 epochs
- Batch size: 32, Image size: 224×224
- Hardware: NVIDIA RTX 4060 (8GB VRAM)

### 3.4 Quantization
- ONNX export (FP32) → INT8 dynamic quantization (ONNX Runtime)
- Benchmark: 200 runs, CPU Execution Provider
- Metrics: mean latency, P50 latency, FPS

### 3.5 Evaluation Metrics
- **Accuracy:** Top-1 test accuracy, per-class precision/recall/F1
- **Efficiency:** Model size (MB), inference latency (ms), throughput (FPS)
- **Interpretability:** Grad-CAM visualization per architecture
- **Trade-off:** Pareto frontier (accuracy vs latency vs size)

---

## 4. Experiments (0.5 pages)

- **E1:** Baseline FP32 training (all 3 architectures, 50 epochs)
- **E2:** INT8 PTQ comparison (latency + size reduction)
- **E3:** Pareto frontier construction
- **E4:** Grad-CAM interpretability analysis

---

## 5. Results and Discussion (1.5-2 pages)

### 5.1 Baseline Performance

| Model | Test Acc | Train-Val Gap | Best Epoch | Convergence |
|---|---|---|---|---|
| EfficientNet-B0 | **98.3%** | 1.6% | 43 | Slow but stable |
| MobileNetV2 | 97.8% | 2.1% | 23 | Fast convergence |
| ShuffleNetV2 | 97.4% | 2.9% | 23 | Cold start problem |

- All models >97% — transfer learning from ImageNet effective for fecal domain
- EB0 best generalization (smallest train-val gap) despite most params
- MN2 fastest convergence — inverted residuals adapt quickly

### 5.2 Quantization Impact

| Model | FP32 Size | INT8 Size | Reduction | FP32 Latency | INT8 Latency | Slowdown |
|---|---|---|---|---|---|---|
| MobileNetV2 | 8.48 MB | 2.30 MB | 3.7× | 1.85 ms | 55.8 ms | 30.2× |
| ShuffleNetV2 | 4.89 MB | 1.47 MB | 3.3× | 3.17 ms | 20.3 ms | 6.4× |
| EfficientNet-B0 | 15.3 MB | 4.16 MB | 3.7× | 3.50 ms | 80.7 ms | 23.1× |

**Key finding:** INT8 PTQ on CPU without hardware INT8 acceleration is counterproductive. ShuffleNetV2 most resilient (6.4× slowdown vs 30.2× for MN2) due to homogeneous operations.

### 5.3 Pareto Frontier
[Figure: accuracy vs latency, accuracy vs size, size vs latency]
- Sweet spot: MobileNetV2 — fastest FP32, competitive accuracy
- Memory-constrained: ShuffleNetV2 INT8 (1.47 MB)
- Accuracy-critical: EfficientNet-B0 (98.3%)

### 5.4 Grad-CAM Interpretability
[Figure: Grad-CAM per architecture per class]
- MN2: focuses on color gradients (low-level features)
- SNV2: cross-channel patterns (texture mixing)
- EB0: multi-scale features (SE attention to diagnostic regions)

### 5.5 Comparison with Related Work

| Paper | Dataset | Best Model | Accuracy | Edge Analysis |
|---|---|---|---|---|
| Dhungana 2025 | Machuve (same) | EB0 | 99.12% | Web-based only |
| Tasdelen 2025 | Machuve (subset) | MN2 | 97.1% | No quantization |
| Degu 2023 | Machuve (augmented) | ResNet50 | 98.7% | Heavy model |
| **Ours** | **Machuve** | **EB0** | **98.3%** | **+INT8+Pareto+Grad-CAM** |

- Our accuracy competitive with SOTA while providing additional edge deployment analysis
- First to characterize INT8 PTQ behavior for this domain

### 5.6 Practical Implications
- Deployment recommendations by hardware constraint
- Why INT8 on CPU is not recommended without GPU/VNNI
- When to choose which architecture

---

## 6. Conclusion (0.25-0.5 pages)

### Summary
1. Three lightweight CNNs characterized on poultry fecal domain — all >97% accuracy
2. INT8 PTQ on CPU introduces 6-30× latency overhead — hardware-dependent finding
3. Pareto frontier identified: MobileNetV2 for latency, ShuffleNetV2 for memory, EB0 for accuracy
4. Grad-CAM reveals architecture-specific feature learning patterns

### Limitations
- CPU-only benchmark (RPi5/ESP32 deployment planned for future work)
- Class imbalance not addressed (NCD 8.3%) — augmentation study as future work
- PTQ only — QAT comparison as future work

### Future Work
- Deploy on Raspberry Pi 5 and ESP32-S3 (Paper 2)
- QAT comparison to recover INT8 accuracy
- Dataset augmentation for NCD class imbalance
- Real-world field testing

---

## References (15+ refs, 5 years recent)

### Core Domain (Poultry)
1. Machuve et al., "Poultry diseases diagnostics models using deep learning," Frontiers in AI, 2022. DOI: 10.3389/frai.2022.733345
2. Dhungana et al., "An Integrated DL Approach for Poultry Disease Detection," AgriEngineering, 2025. DOI: 10.3390/agriengineering7090278
3. Tasdelen & Arslan, "Detection of high-risk diseases in poultry feces through transfer learning," Eng Sci Tech Int J, 2025. DOI: 10.1016/j.jestch.2025.102002
4. Degu & Simegn, "Smartphone based detection of poultry diseases from chicken fecal images," Smart Agri Tech, 2023. DOI: 10.1016/j.atech.2023.100221

### Architecture
5. Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks," CVPR, 2018.
6. Ma et al., "ShuffleNetV2: Practical Guidelines for Efficient CNN Architecture Design," ECCV, 2018.
7. Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs," ICML, 2019.

### Quantization
8. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," CVPR, 2018.
9. ONNX Runtime, "Quantize ONNX Models," Microsoft Docs, 2024.
10. Maulana & Ramasamy, "Systematic Review of Quantization-Optimized Lightweight Architectures," Computers, 2026. DOI: 10.3390/computers15010069

### XAI
11. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks," ICCV, 2017.
12. Hoang et al., "A comprehensive evaluation of lightweight DL for tomato disease on edge," Scientific Reports, 2026. DOI: 10.1038/s41598-026-42439-6

### Edge AI
13. Partovi Nia et al., "Rethinking Pareto Frontier for Performance Evaluation of DNNs," ICML Workshop, 2022.
14. "Edge AI and IoT for Real-Time Crop Disease Detection: Survey," IJRIAS, 2025.
15. "Optimizing lightweight neural networks for mobile edge computing," PMC, 2025.

---

## Experimental Figures Checklist

- [ ] Figure 1: Training curves (3 models, loss + accuracy)
- [ ] Figure 2: Confusion matrices normalized (3 models)
- [ ] Figure 3: Grad-CAM visualizations (3 models × 4 classes)
- [ ] Figure 4: Pareto frontier (accuracy vs latency, accuracy vs size)
- [ ] Figure 5: Quantization comparison (FP32 vs INT8 size + latency)

---

## Writing Timeline

| Day | Task | Status |
|---|---|---|
| Aug 16 | Literature review + outline | ✅ Done |
| Aug 17 | Write Methodology + Experiments | |
| Aug 18 | Write Results + figures generation | |
| Aug 19 | Write Intro + Related Work | |
| Aug 20 | Write Conclusion + Abstract | |
| Aug 21 | Review + formatting | |
| Aug 22 | Final review + submit | |
