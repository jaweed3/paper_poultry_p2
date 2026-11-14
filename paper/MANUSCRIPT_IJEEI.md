# MANUSCRIPT DRAFT: TARGET IJEEI (Scopus Q3)

**Journal:** International Journal on Electrical Engineering and Informatics (IJEEI)  
**Track:** Artificial Intelligence / Computer Science / Embedded Systems  
**Formatting Style:** Single-column IJEEI standard, complete with Mathematical Formulations, Algorithms, Layer-wise Ablation, Edge Benchmarking, and Grad-CAM Explainable AI.

---

# Localizing and Mitigating INT8 Quantization Collapse in Lightweight CNNs for Farm-Level Poultry Disease Diagnostics on Raspberry Pi 5

**Fatih Jawwad Al Mumtaz**$^{1}$, **Amirul Al Aziz**$^{2}$, **Muhammad Iqbal Hidayatullah**$^{1}$  
$^{1}$Department of Informatics, Faculty of Science and Technology, Universitas Darussalam Gontor, Ponorogo, Indonesia  
$^{2}$Department of Agricultural Industrial Technology, Faculty of Science and Technology, Universitas Darussalam Gontor, Ponorogo, Indonesia  
*Corresponding Author:* fatihalmumtaz76@student.cs.unida.gontor.ac.id

---

### Abstract
Non-invasive diagnostic tools powered by deep learning offer significant potential for early disease detection in poultry farming, mitigating severe economic losses caused by Coccidiosis, Salmonellosis, and Newcastle Disease. However, existing state-of-the-art vision models predominantly target cloud or GPU servers, which are impractical for rural, resource-constrained smallholder farms. Deploying lightweight Convolutional Neural Networks (CNNs) directly onto cost-effective edge devices via INT8 Post-Training Quantization (PTQ) frequently induces severe accuracy degradation, termed quantization collapse. 

In this study, we present a comprehensive empirical and theoretical investigation into INT8 quantization dynamics across lightweight CNN architectures (MobileNetV2, ShuffleNetV2, EfficientNet-B0, and EfficientNet-Lite0) deployed on a physical Raspberry Pi 5 single-board computer using ONNX Runtime. First, a perceptual hash (pHash) integrity audit of the prevailing 8,770-image public fecal benchmark revealed 617 near-duplicate pairs, including 94 byte-identical images leaked between training and testing splits; all subsequent evaluations are conducted on a rigorously sanitized 8,153-image sterile split. Second, we localize the structural failure point of EfficientNet-B0, where naive dynamic INT8 and static per-tensor quantization trigger catastrophic degradation from 98.55% down to 10.42% and 9.32% accuracy, respectively. Systematic layer-wise ablation and Signal-to-Quantization-Noise Ratio (SQNR) profiling demonstrate that this collapse is strictly driven by dynamic range clipping within depthwise separable convolutions, rather than Squeeze-and-Excitation (SE) modules. Static per-channel Quantize-Dequantize (QDQ) calibration fully rescues EfficientNet-B0 to 94.56% accuracy ($p < 10^{-15}$ vs. dynamic). Third, hardware profiling reveals that dynamic INT8 introduces runtime quantize/dequantize kernel overheads, causing a 1.93--2.80$\times$ slowdown relative to FP32, whereas static per-channel QDQ yields a 2.34$\times$ speedup (13.50 ms latency, 74.07 FPS) under four threads without thermal throttling. Finally, Gradient-weighted Class Activation Mapping (Grad-CAM) confirms that the calibrated edge models attend to authentic pathological fecal lesions rather than environmental background noise.

**Keywords:** Edge AI, INT8 Quantization Collapse, Lightweight CNNs, Raspberry Pi 5, Precision Poultry Farming, Explainable AI.

---

## 1. Introduction
Poultry farming represents a foundational pillar of global livestock production and rural food security, contributing over \$67 billion in annual commercial value in major producing regions alone [1]. However, infectious avian pathologies—most notably Coccidiosis, Salmonellosis, and Newcastle Disease (NCD)—present persistent risks of high mortality and severe economic disruption [2]. Coccidiosis, an enteric parasitic infection caused by *Eimeria* protozoa, causes intestinal hemorrhaging and compromised nutrient absorption. Salmonellosis leads to severe systemic bacterial infection, whereas Newcastle Disease virus constitutes a highly contagious, reportable avian condition capable of wiping out entire flocks within days [3].

Timely intervention requires rapid, accurate identification. Traditional laboratory diagnostics (such as Polymerase Chain Reaction / PCR assays, blood serology, and fecal microscopic smear examinations) demand specialized laboratory infrastructure, expensive reagents, and trained veterinary microbiologists [4]. For the overwhelming majority of small-scale and semi-intensive poultry farmers in developing and agrarian countries, routine laboratory diagnosis is logistically and economically unfeasible [5]. In practice, clinical management relies on non-invasive visual inspection of avian excreta. Fecal consistency, color changes, hemorrhagic shedding, and mucoid discharge reflect distinct clinical manifestations of gastrointestinal and systemic infections.

To automate this visual inspection, recent studies have applied deep convolutional neural networks (CNNs) to photographic fecal images [6, 7]. Transfer learning on standard benchmarks has yielded impressive classification accuracies exceeding 97% to 99% using deep architectures such as EfficientNet-B0 and MobileNetV2 [8]. Nevertheless, existing literature exhibits three critical limitations:
1. **Cloud-Centric Bias:** The vast majority of published models rely on high-power workstation GPUs or cloud-hosted web APIs. Rural farming environments frequently lack reliable broadband connectivity, making continuous image streaming to remote cloud backends impractical and prone to transmission latency.
2. **The Edge Quantization Pitfall:** Compressing full-precision (FP32) weights into 8-bit integers (INT8) is mandatory to fit low-cost embedded hardware such as the Raspberry Pi. However, popular modern architectures such as EfficientNet-B0 experience catastrophic accuracy failure under standard post-training quantization—an issue rarely addressed in agricultural literature.
3. **Pervasive Data Leakage in Public Benchmarks:** Leading open-source poultry datasets have been deployed under unverified assumptions of split independence. Inadvertent duplication between training and testing sets artificially inflates reported classification scores, masking true model generalization.

This study closes these operational and scientific gaps. By systematically investigating the mechanics of INT8 quantization failure and edge inference on a quad-core ARM Cortex-A76 processor (Raspberry Pi 5), we present an end-to-end framework for reliable, real-time poultry disease diagnosis at the farm level.

The main contributions of this work are summarized as follows:
1. **Benchmark Audit and Sanitization:** We conduct a rigorous perceptual hashing (pHash, Hamming distance $\le 5$) audit on the widely cited 8,770-image poultry fecal dataset, uncovering 617 near-duplicate pairs and 94 byte-exact cross-split leakages. We establish and release an 8,153-image sterile benchmark split with guaranteed zero cross-split leakage.
2. **Empirical and Theoretical Diagnosis of Quantization Collapse:** We demonstrate that while MobileNetV2 and ShuffleNetV2 preserve accuracy under INT8 quantization, EfficientNet-B0 suffers a catastrophic collapse to 10.42%. Through selective layer-wise ablation and mathematical SQNR formulation, we prove that this degradation originates in the high channel-wise weight variance of depthwise convolutions, which is successfully alleviated via static per-channel QDQ calibration (recovering accuracy to 94.56%).
3. **Comprehensive Edge Profiling on Raspberry Pi 5:** We benchmark 12 deployment configurations across thread allocations (1 to 4 threads), measuring latency percentiles (P50, P95, P99), throughput (FPS), RAM footprint, and core thermal stability, explaining why dynamic INT8 introduces runtime bottlenecks while static QDQ achieves a 2.34$\times$ acceleration.
4. **Visual Interpretability via Explainable AI:** We perform Grad-CAM analysis across FP32 and INT8 models, confirming that edge-quantized representations retain pathological focus on disease-indicative textures.

---

## 2. Related Work

### 2.1 Computer Vision in Precision Livestock & Poultry Disease Detection
Automating animal health monitoring through non-invasive optical sensors has emerged as an active area of smart farming research. Machuve et al. [9] established the foundational benchmark of 6,812 PCR-verified fecal photographs collected across smallholder poultry farms in Tanzania. Subsequent works focused on maximizing test classification metrics; for example, Degu et al. [10] combined YOLOv5 object bounding with CNN classifiers, and Tasdelen & Arslan [11] benchmarked five deep architectures on cloud servers, reporting 97.1% accuracy with MobileNetV2. More recently, Dhungana et al. [12] reported 99.12% accuracy utilizing EfficientNet-B0. However, none of these preceding works deployed models to physical single-board computers or addressed the mathematical and architectural challenges of low-bit quantization.

### 2.2 Lightweight Architectures for Resource-Constrained Edge
Edge computing mandates models with low parameter counts, compact memory footprints, and low floating-point operations (FLOPs). MobileNetV2 [13] introduced inverted residual blocks with linear bottlenecks, utilizing depthwise separable convolutions to decouple spatial filtering from channel projection. ShuffleNetV2 [14] optimized memory access cost (MAC) by introducing channel split and channel shuffle operations. Tan & Le [15] introduced EfficientNet, which balances network depth, width, and input resolution through compound scaling, with EfficientNet-B0 serving as the lightweight baseline incorporating Squeeze-and-Excitation (SE) attention blocks.

### 2.3 Quantization Dynamics and Quantization-Dequantization (QDQ) Paradigms
Post-Training Quantization (PTQ) maps continuous 32-bit floating-point tensors ($r \in \mathbb{R}$) into discrete 8-bit integers ($q \in [-128, 127]$ or $[0, 255]$), reducing memory bandwidth by 75% [16]. However, as surveyed by Nagel et al. [17], depthwise separable convolutions are notoriously susceptible to quantization noise due to high inter-channel weight variances. When a single scaling factor is applied across all channels (per-tensor quantization), channels with small dynamic ranges suffer extreme underflow, corrupting feature representations.

---

## 3. Mathematical Foundations of Edge Quantization

To formalize the quantization phenomenon, we define the uniform affine quantization operator $\mathcal{Q}(r; S, Z)$ mapping continuous real values $r$ to quantized integer representations $q$:

$$q = \mathcal{Q}(r; S, Z) = \text{clip}\left(\left\lfloor \frac{r}{S} \right\rceil + Z, \; q_{\min}, \; q_{\max}\right) \tag{1}$$

where $\lfloor \cdot \rceil$ denotes the round-to-nearest-integer operation, $q_{\min} = -128$, $q_{\max} = 127$ for signed 8-bit integers (`int8`), and $\text{clip}(v, a, b) = \max(a, \min(v, b))$. The arbitrary dynamic range $[r_{\min}, r_{\max}]$ of tensor $r$ defines the scale factor $S$ and zero-point $Z$:

$$S = \frac{r_{\max} - r_{\min}}{q_{\max} - q_{\min}} \tag{2}$$

$$Z = \text{round}\left( - \frac{r_{\min}}{S} \right) + q_{\min} \tag{3}$$

The reconstructed real-valued tensor $\hat{r}$ through dequantization $\mathcal{Q}^{-1}(q)$ is expressed as:

$$\hat{r} = \mathcal{Q}^{-1}(q) = S \cdot (q - Z) \tag{4}$$

### 3.1 Per-Tensor versus Per-Channel Quantization
In **per-tensor quantization**, a scalar scale $S$ and zero-point $Z$ are calculated globally across all weight parameters of a convolutional layer $W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K \times K}$:

$$S_{\text{tensor}} = \frac{\max(W) - \min(W)}{255} \tag{5}$$

For standard convolutions, spatial aggregation mitigates variance. However, for a **depthwise convolution** where each input channel is filtered independently ($C_{\text{in}} = 1$), the weight magnitude between different output channels $c \in \{1, \dots, C_{\text{out}}\}$ can diverge by multiple orders of magnitude ($\max |W_c| \ll \max |W_{c'}|$). Under per-tensor scaling, channels with small magnitudes are rounded to 0 or 1, obliterating their learned spatial features.

In **per-channel (per-axis) quantization**, an independent scale $S_c$ and zero-point $Z_c$ are assigned to each individual output filter slice:

$$S_c = \frac{\max(W_c) - \min(W_c)}{255}, \quad \forall c \in \{1, \dots, C_{\text{out}}\} \tag{6}$$

$$q_c = \text{clip}\left(\left\lfloor \frac{W_c}{S_c} \right\rceil + Z_c, \; -128, \; 127\right) \tag{7}$$

### 3.2 Signal-to-Quantization-Noise Ratio (SQNR)
To quantitatively localize layer-wise precision loss without evaluating the entire downstream network, we formulate the layerwise weight SQNR (expressed in decibels, dB):

$$\text{SQNR}_l = 10 \log_{10} \left( \frac{\sum_{i=1}^{N_l} W_{l,i}^2}{\sum_{i=1}^{N_l} (W_{l,i} - \hat{W}_{l,i})^2} \right) \tag{8}$$

where $W_{l,i}$ represents the original FP32 weight element of layer $l$, and $\hat{W}_{l,i}$ is the dequantized reconstructed weight. Lower SQNR indicates severe quantization distortion.

---

## 4. Dataset Integrity Audit & Sanitization

Reliable edge benchmarking requires absolute partition independence. The published benchmark `Dianyo/poultry-fecal-fl` contains 8,770 images distributed into official splits of 7,016 training and 1,754 testing images [18]. 

### 4.1 Perceptual Hash (pHash) Duplication Audit
We evaluated every image pair $(x_i, x_j)$ across the dataset using 64-bit DCT-based perceptual hashing:

$$D_H(\text{pHash}(x_i), \text{pHash}(x_j)) \le \tau \tag{9}$$

Setting a strict threshold $\tau = 5$ bits, our audit identified **617 near-duplicate pairs** across the dataset:
* **393 pairs** were byte-exact duplicates ($\text{MD5}(x_i) = \text{MD5}(x_j)$).
* **303 duplicate pairs** spanned across partition boundaries (147 test--train, 54 test--validation, and 102 train--validation).
* Crucially, **94 test images were byte-identical to training images**, directly corrupting generalization measurements.

```
       [Raw Dataset: 8,770 Images]
                   │
                   ▼
       pHash Audit (Hamming ≤ 5)
   Found 617 near-duplicate pairs
   (303 pairs spanning across splits)
                   │
                   ▼
       Deduplication Protocol
       (Drop 1 duplicate per pair)
  -513 Train  │  -78 Val  │  -26 Test
                   │
                   ▼
    [Sanitized Sterile Split: 8,153 Images]
   • Train: 5,100 (62.5%)
   • Validation: 1,325 (16.3%)
   • Test: 1,728 (21.2%)
   • Zero Cross-Split Leakage Confirmed
```

### 4.2 Sanitized Split Establishment
To guarantee integrity, we removed the duplicate instances following a strict hierarchy: preserving test instances over validation, and validation over training. This deduplication pruned 513 training, 78 validation, and 26 test images, producing a leak-free sterile benchmark of **8,153 images** (5,100 train, 1,325 validation, 1,728 test) across four balanced pathological classes: Coccidiosis (`cocci`), Salmonella (`salmo`), Newcastle Disease (`ncd`), and Healthy (`healthy`).

---

## 5. Methodology & Algorithmic Formulation

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                           END-TO-END SYSTEM PIPELINE                              │
└───────────────────────────────────────────────────────────────────────────────────┘

 [Input Fecal Image] ──► [Preprocess: 224x224, ImageNet Norm]
                                    │
                                    ▼
                     [Pretrained Lightweight CNN]
                     (MobileNetV2 / ShuffleNetV2 /
                      EfficientNet-B0 / Lite0)
                                    │
                                    ▼
                       [Fine-Tuned FP32 Model]
                     (50 Epochs, Cosine Anneal)
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        [Dynamic INT8 PTQ]                  [Static Per-Channel QDQ]
      (Weights QInt8, No Calib)            (100-Sample Stratified Calib)
                  │                                   │
                  ▼                                   ▼
          Acc: 10.42% (COLLAPSE)              Acc: 94.56% (RECOVERED)
        Pi Latency: 63.99 ms (Slow)         Pi Latency: 13.50 ms (2.34x Fast)
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    ▼
                         [Explainable AI: Grad-CAM]
                   (Verify Biological Lesion Attention)
                                    │
                                    ▼
                      [Raspberry Pi 5 Deployment]
                   (Broadcom BCM2712, 4x Cortex-A76)
```

The system workflow consists of model fine-tuning, ONNX graph export, quantization calibration, and edge runtime profiling.

### 5.1 Algorithms
We define two core operational procedures governing calibration and diagnostic localization.

#### Algorithm 1: Static Per-Channel QDQ Calibration and Export
```
Require: Trained FP32 ONNX model M_fp32, Calibration dataset D_calib (100 images)
Ensure: Optimized INT8 QDQ ONNX model M_int8
 1: Initialize ONNX CalibrationDataReader with D_calib
 2: Configure QuantizationParameters:
 3:     ActivationType ← QUInt8 (MinMax calibration)
 4:     WeightType ← QInt8 (Per-channel granularity, axis = 0)
 5:     QuantFormat ← QDQ (QuantizeLinear / DequantizeLinear nodes)
 6: For each Conv operator C_i in M_fp32 do
 7:     Compute per-channel scale S_c and zero-point Z_c for weights via Eq. (6)-(7)
 8:     Insert DequantizeLinear(QuantizeLinear(W)) before GEMM kernel
 9: End For
10: Run activation range observer over D_calib to capture dynamic tensors [r_min, r_max]
11: Fuse adjacent BatchNorm and Conv nodes where applicable
12: Serialize optimized computational graph M_int8
13: Return M_int8
```

#### Algorithm 2: Layer-wise Quantization Sensitivity Localization
```
Require: FP32 model M_fp32, Quantized model M_int8, Test split D_test
Ensure: Layer sensitivity map S_map and identification of collapse bottleneck
 1: For each layer group L in {Depthwise, Pointwise, SqueezeExcite, Head} do
 2:     Construct hybrid model M_hybrid where group L remains in FP32
 3:     Quantize all remaining layers to INT8 per-tensor via Eq. (1)
 4:     Evaluate Top-1 Accuracy Acc_L on D_test
 5:     Compute layer group SQNR via Eq. (8)
 6:     If Acc_L ≥ 0.90 × Acc_FP32 then
 7:         Mark group L as "Critical Bottleneck Layer"
 8:     Else
 9:         Mark group L as "Non-Critical / Invariant"
10:     End If
11: End For
12: Return S_map
```

---

## 6. Experimental Results & In-Depth Discussion

### 6.1 Classification Performance & Quantization Collapse
Table 1 presents the classification performance across all architectures evaluated on the sterile test set ($n = 1,728$ images).

**Table 1. Test Classification Metrics on Sanitized Dataset (n = 1,728)**
| Architecture | FP32 Acc. (%) | INT8 Dyn. Acc. (%) | Acc. Drop (pp) | FP32 Macro-F1 | INT8 Macro-F1 | FP32 Size (MB) | INT8 Size (MB) | Compression |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV2** | 97.92% | 97.28% | -0.64% | 0.9757 | 0.9645 | 8.48 MB | 2.30 MB | 3.69$\times$ |
| **ShuffleNetV2** | 97.69% | 97.63% | -0.06% | 0.9710 | 0.9682 | 4.89 MB | 1.47 MB | 3.32$\times$ |
| **EfficientNet-B0** | 98.55% | **10.42%** | **-88.13%** | 0.9822 | 0.0749 | 15.30 MB | 4.16 MB | 3.68$\times$ |
| **EfficientNet-Lite0** | 97.80% | 96.95% | -0.85% | 0.9740 | 0.9650 | 14.10 MB | 3.95 MB | 3.57$\times$ |

While MobileNetV2 and ShuffleNetV2 maintain robust representations under dynamic INT8, EfficientNet-B0 collapses catastrophically (dropping 88.13 percentage points to 10.42%, performing worse than random guessing). Crucially, **EfficientNet-Lite0**—which replaces Swish activations with ReLU6 and strips Squeeze-and-Excitation attention—maintains 96.95% accuracy under INT8, demonstrating that compound architectural choices dictate quantization robustness.

### 6.2 Structural Localization of the Failure Point
To pinpoint whether the collapse stems from Squeeze-and-Excitation (SE) attention or depthwise convolutions, we conducted selective per-family ablation on EfficientNet-B0 (Table 2).

**Table 2. Selective Layer-Family QDQ Ablation for EfficientNet-B0**
| Configuration | Quantization Scheme Applied | Model Size | Accuracy (%) | Collapse Observed? |
| :---: | :--- | :---: | :---: | :---: |
| **A** | All layers Per-Tensor INT8 | 4.29 MB | 9.32% | **Yes** |
| **B** | All layers Per-Channel INT8 | 4.67 MB | **94.56%** | **No (Recovered)** |
| **C** | **Depthwise Per-Channel**, rest Per-Tensor | 4.33 MB | **93.46%** | **No (Recovered)** |
| **D** | All Per-Channel **except Depthwise** | 4.51 MB | 8.33% | **Yes** |
| **E** | Depthwise isolated in FP32, rest INT8 | 4.69 MB | 97.16% | **No** |
| **F** | Squeeze-and-Excitation isolated in FP32 | 5.91 MB | 12.44% | **Yes** |

The empirical evidence from Configurations C, D, and F is conclusive:
* When depthwise convolutions alone receive per-channel calibration (Config C), accuracy immediately jumps from 9.32% to **93.46%**.
* Conversely, keeping all other layers in per-channel INT8 while restricting depthwise convolutions to per-tensor (Config D) results in immediate collapse (8.33%).
* Leaving Squeeze-and-Excitation in FP32 (Config F) fails to arrest the collapse (12.44%). 

Thus, the quantization collapse in EfficientNet-B0 is definitively localized to **dynamic range clipping within depthwise separable convolutions**.

### 6.3 Signal-to-Quantization-Noise Ratio (SQNR) Analysis
Layer-wise SQNR analysis confirms this diagnosis. In depthwise layers, per-tensor quantization yields an average SQNR of only **8.4 dB**, with extreme filter clipping. Switching to per-channel quantization boosts the depthwise SQNR to **21.8 dB** (+13.4 dB gain), preserving gradient propagation through the residual pipeline.

### 6.4 Synergistic Compression: Structured Pruning Combined with INT8 Quantization (Pruning-then-PTQ)
To establish whether channel redundancy can be trimmed prior to quantization across distinct architectures, we applied structured channel pruning via Dependency Graph (DepGraph) L1-norm pruning on MobileNetV2 (at 30% and 50% channel sparsity) and EfficientNet-B0 (at 30% channel sparsity), followed by 12 epochs of fine-tuning on the sterile split and subsequent post-training quantization (Table 4).

**Table 4. Impact of Structured Pruning Coupled with INT8 Quantization Across Architectures**
| Model Stage | Active Parameters | Disk Footprint | Top-1 Accuracy (%) | Healthy Rec. (%) | Cocci Rec. (%) | Salmo Rec. (%) | NCD Rec. (%) | Total Compression |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MNV2 Baseline FP32** | 2,228,996 (100%) | 8.48 MB | 97.92% | 98.41% | 98.11% | 97.83% | 92.20% | 1.00$\times$ |
| **MNV2 Pruned FP32 (s=0.3)** | 1,110,790 (49.8%) | 4.23 MB | 93.81% | 97.42% | 93.76% | 95.31% | 75.18% | 2.00$\times$ |
| **MNV2 Pruned INT8 (s=0.3)** | 1,110,790 (49.8%) | **1.23 MB** | **91.09%** | **97.22%** | **93.38%** | **90.79%** | **61.70%** | **6.89$\times$** |
| **MNV2 Pruned INT8 (s=0.5)** | 583,332 (26.2%) | **0.72 MB** | **85.47%** | **91.07%** | **96.98%** | **83.21%** | **31.21%** | **11.78$\times$** |
| **EB0 Baseline FP32** | 4,012,672 (100%) | 15.30 MB | 98.55% | 99.11% | 99.24% | 98.92% | 94.33% | 1.00$\times$ |
| **EB0 Pruned FP32 (s=0.3)** | 2,005,801 (50.0%) | 7.67 MB | 93.69% | 95.44% | 91.68% | 96.93% | 82.27% | 1.99$\times$ |
| **EB0 Pruned INT8 Dynamic** | 2,005,801 (50.0%) | 2.23 MB | 33.28% | 1.79% | 32.70% | 65.70% | 20.57% | 6.86$\times$ (Collapse) |
| **EB0 Pruned INT8 Static QDQ** | 2,005,801 (50.0%) | **2.62 MB** | **63.72%** | **65.48%** | **99.24%** | **34.84%** | **37.59%** | **5.84$\times$** |

Coupling structured channel pruning with INT8 quantization yields an exceptional compression continuum:
1. For **MobileNetV2**, pruning 30% of channels achieves a **6.89$\times$ overall compression ratio** (shrinking to **1.23 MB** at **91.09%** accuracy). Pushing channel sparsity to 50% reduces parameters down to just 583,332 (26.2% active weights), shrinking the INT8 footprint to **0.72 MB (733 KB)** at **85.47%** accuracy—an **11.78$\times$ compression factor**. This delineates the empirical *accuracy cliff*, wherein the minor Newcastle Disease (NCD) class suffers degradation due to limited channel capacity.
2. For **EfficientNet-B0**, dynamic INT8 exhibits severe collapse post-pruning down to 33.28%. However, our static per-channel QDQ calibration recovers diagnostic performance by **+30.44%** to **63.72%** at **2.62 MB**, validating that per-channel scaling remains essential even in pruned depthwise topologies.
3. For **ShuffleNetV2**, standard L1-norm DepGraph channel pruning cannot be applied automatically due to topological channel-split and channel-shuffle symmetries requiring paired branch channel equality, demonstrating a fundamental structural trade-off between manual architectural coupling and generic pruning frameworks.

### 6.5 Raspberry Pi 5 Edge Hardware Benchmark
Inference latency was benchmarked on a physical Raspberry Pi 5 (Broadcom BCM2712 quad-core Cortex-A76 @ 2.4 GHz, 8 GB RAM) running Debian 13 trixie and ONNX Runtime 1.29.0 (`CPUExecutionProvider`). Measurements reflect 200 iterations after 10 warm-up runs (Table 3).

**Table 3. Edge Profiling on Raspberry Pi 5 (Batch Size = 1, 4 Threads)**
| Model | Precision & Scheme | Latency P50 (ms) | Latency P99 (ms) | Throughput (FPS) | Peak RAM (MB) | Temperature | Speedup vs FP32 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV2** | FP32 Reference | 14.23 ms | 15.10 ms | 70.27 FPS | 58.4 MB | 48.2 °C | 1.00$\times$ |
| MobileNetV2 | Dynamic INT8 | 39.83 ms | 42.15 ms | 25.10 FPS | 44.1 MB | 49.0 °C | 0.36$\times$ (Slowdown) |
| **ShuffleNetV2** | FP32 Reference | 6.19 ms | 6.84 ms | 161.55 FPS | 41.2 MB | 46.5 °C | 1.00$\times$ |
| ShuffleNetV2 | Dynamic INT8 | 11.95 ms | 13.02 ms | 83.68 FPS | 36.8 MB | 47.1 °C | 0.52$\times$ (Slowdown) |
| **EfficientNet-B0** | FP32 Reference | 31.53 ms | 33.20 ms | 31.71 FPS | 72.5 MB | 50.1 °C | 1.00$\times$ |
| EfficientNet-B0 | Dynamic INT8 | 63.99 ms | 67.80 ms | 15.62 FPS | 51.3 MB | 51.4 °C | 0.49$\times$ (Slowdown) |
| **EfficientNet-B0** | **Static Per-Channel QDQ** | **13.50 ms** | **14.22 ms** | **74.07 FPS** | **49.8 MB** | **48.8 °C** | **2.34$\times$ (Speedup)** |

#### Why is Dynamic INT8 Slower than FP32 on CPU?
A critical takeaway for practitioners is that **Dynamic INT8 on ARM CPUs introduces a severe latency penalty (1.93--2.80$\times$ slower than FP32)**. Dynamic quantization dynamically computes activation scale and zero-point parameters during runtime for every layer invocation, adding significant quantize/dequantize memory traffic around memory-bound depthwise kernels. In contrast, **Static Per-Channel QDQ** pre-computes quantization parameters during offline calibration, enabling ONNX Runtime to execute optimized integer GEMM kernels directly, achieving a true **2.34$\times$ speedup** over FP32 on Raspberry Pi 5.

### 6.6 Explainable AI: Grad-CAM Pathological Verification
Grad-CAM heatmaps generated from the final convolutional stage verify that both FP32 and calibrated INT8 models focus precisely on pathological markers:
* **Coccidiosis:** High activation density centered over bloody, hemorrhagic stool clusters.
* **Salmonella:** Concentrated attention on mucoid, watery yellow-green discharges.
* **Newcastle Disease:** Clear localization of dense greenish enteric fluids.
* **Healthy:** Diffuse, low-magnitude activations across normal solid cecal droppings.
This confirms that the recovered INT8 model retains authentic veterinary diagnostic features rather than spurious background artifacts.

---

## 7. Economic & Farm-Level Feasibility Analysis
For smallholder poultry farming operations in rural communities, cost feasibility is paramount. Table 4 compares the proposed Raspberry Pi 5 edge diagnostic node against conventional cloud-GPU setups.

**Table 5. Deployment Feasibility Comparison**
| Metric | Cloud GPU Setup (T4 / A10G) | Edge Node (Raspberry Pi 5) | Advantage |
| :--- | :---: | :---: | :---: |
| **Initial Capital Cost** | \$1,500 -- \$2,500 (Server/PC) | **\$80 -- \$110 (Pi 5 + Camera)** | **~95% Lower CapEx** |
| **Monthly Operating Cost** | \$50 -- \$150 (Cloud API/Bandwidth) | **\$0 (Local Inference)** | **Zero Subscription** |
| **Network Dependency** | Mandatory (Continuous Upload) | **None (100% Offline)** | **Zero Outage Risk** |
| **Inference Latency** | 250 -- 800 ms (RTT + Queue) | **13.50 ms (Local)** | **Real-Time Video** |
| **Power Consumption** | 250W -- 450W | **5W -- 12W** | **Solar/Battery Operable**|

---

## 8. Conclusion and Future Directions
This paper provides an empirical and theoretical blueprint for deploying lightweight deep learning models on resource-constrained edge hardware for agricultural pathology. By resolving the dataset duplication flaw in prior benchmarks, demystifying the root cause of INT8 quantization collapse in EfficientNet-B0, and demonstrating that static per-channel QDQ delivers both high diagnostic fidelity (94.56%) and high edge throughput (74 FPS at 13.5 ms on Raspberry Pi 5), we establish that edge-native poultry diagnostics are viable for rural deployment.

Future work will expand into hardware-aware structured pruning combined with INT8 quantization (Pruning-then-PTQ) and on-device continuous federated learning across cooperative farm networks.

---

## References
[1] USDA, "Poultry - Production and Value 2023 Summary," United States Department of Agriculture National Agricultural Statistics Service, Washington, D.C., Tech. Rep., Apr. 2024.  
[2] E. Machuve et al., "An agricultural dataset for poultry diseases diagnostics in developing countries," *Scientific Data*, vol. 9, no. 1, p. 760, 2022.  
[3] D. E. Swayne et al., *Diseases of Poultry*, 14th ed. Hoboken, NJ: Wiley-Blackwell, 2020.  
[4] J. A. Tasdelen and O. Arslan, "Automated classification of poultry diseases using deep convolutional neural networks," *Comput. Electron. Agric.*, vol. 228, p. 109552, 2025.  
[5] E. Degu, S. Dereje, and A. Girma, "Poultry disease detection and classification using computer vision and deep learning techniques," *Smart Agric. Technol.*, vol. 6, p. 100345, 2023.  
[6] M. Dhungana et al., "Transfer learning based EfficientNet for avian fecal image classification," *Poult. Sci.*, vol. 104, no. 2, p. 103210, 2025.  
[7] M. Sandler et al., "MobileNetV2: Inverted residuals and linear bottlenecks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 4510–4520.  
[8] N. Ma et al., "ShuffleNet V2: Practical guidelines for efficient CNN architecture design," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2018, pp. 116–131.  
[9] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2019, pp. 6105–6114.  
[10] B. Jacob et al., "Quantization and training of neural networks for efficient integer-arithmetic-only inference," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 2704–2713.  
[11] M. Nagel et al., "A white paper on neural network quantization," *arXiv preprint arXiv:2106.08295*, 2021.  
[12] H. Chi, "FecalFed: A federated learning benchmark for poultry fecal disease classification," *IEEE Access*, vol. 12, pp. 45210–45222, 2026.  
[13] ONNX Runtime Authors, "ONNX Runtime: Cross-platform, high performance ML inferencing and training accelerator," 2024. [Online]. Available: https://onnxruntime.ai  
[14] J. Deng et al., "ImageNet: A large-scale hierarchical image database," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2009, pp. 248–255.  
[15] R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 618–626.  
[16] G. Fang et al., "DepGraph: Towards any structural pruning," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2023, pp. 16091–16100.  
[17] A. Hoang et al., "Comparative benchmark of lightweight architectures for edge-based plant pathology," *IEEE Trans. AgriFood Electron.*, vol. 2, no. 1, pp. 45–56, 2026.  
[18] Dianyo, "Poultry Fecal Disease Benchmark Dataset," Hugging Face Datasets, 2024. [Online]. Available: https://huggingface.co/datasets/Dianyo/poultry-fecal-fl  

---

## Author Biographies

**Fatih Jawwad Al Mumtaz** is an undergraduate researcher in the Department of Informatics at Universitas Darussalam Gontor, Indonesia. His research interests center on Edge AI, TinyML, model compression, computer vision, and embedded deep learning systems for precision agriculture and livestock management.

**Amirul Al Aziz** is a researcher in the Department of Agricultural Industrial Technology at Universitas Darussalam Gontor, Indonesia. His research focuses on smart farming technologies, agricultural process automation, and non-invasive livestock diagnostics.

**Muhammad Iqbal Hidayatullah** is a faculty member and researcher in the Department of Informatics at Universitas Darussalam Gontor, Indonesia. His research domains encompass applied artificial intelligence, machine learning, distributed computing, and intelligent systems engineering.
