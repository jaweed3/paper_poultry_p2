# Literature — Paper 2 (pruning-then-PTQ mitigation)

> Cara pakai: PDF yang open sudah ter-download di folder ini. Yang paywall hanya link — download manual lalu taruh di `literature/` dengan nama yang sama.

## PDF open (sudah ter-download)

| # | File | Title | DOI / URL | Relevansi 1 kalimat |
|---|------|-------|-----------|---------------------|
| 1 | `01_liang_pruning_quant_survey_2021.pdf` | Pruning and Quantization for Deep Neural Network Acceleration: A Survey (Liang et al., 2021) | https://arxiv.org/abs/2101.09671 | Survey gabungan pruning+quant — jadi backbone related work, bedakan structured vs unstructured dan PTQ vs QAT |
| 2 | `02_lightprune_iccvw2025.pdf` | LightPrune: Latency-Aware Structured Pruning for Efficient Deep Inference on Embedded (Belhadi et al., ICCVW 2025) | https://openaccess.thecvf.com/.../LightPrune...pdf | Latency-aware structured pruning di embedded — pembanding terdekat untuk story mitigation, tapi tidak coupling dengan PTQ overhead |
| 3 | `03_nagel_whitepaper_quant_2021.pdf` | A White Paper on Neural Network Quantization (Nagel et al., 2021) | https://arxiv.org/abs/2106.08295 | White paper quant — jelaskan kenapa PTQ hancur di depthwise/SE (relevan untuk EfficientNet-B0 collapse 62.7%) |
| 6 | `06_depgraph_structured_pruning_2023.pdf` | DepGraph: Towards Any Structural Pruning (Fang et al., CVPR 2023) | https://arxiv.org/abs/2301.10700 | Dependency graph untuk structured pruning — dasar implementasi Torch-Pruning/DepGraph di Paper 2 |
| 7 | `07_he_structured_pruning_survey_2023.pdf` | Structured Pruning for Deep CNNs — Survey (He et al., 2023) | https://arxiv.org/abs/2303.00566 | Survey structured pruning khusus CNN — untuk positioning pruning method (L1-norm, filter vs channel) |

## Link paywall / manual download (taruh PDF dengan nama yang disarankan)

| # | Nama file yang disarankan | Title | DOI / URL | Status | Relevansi |
|---|---------------------------|-------|-----------|--------|-----------|
| 4 | `04_mango_pruning_quant_2025.pdf` | Pruning and Quantization of Lightweight Hybrid CNN Models for Real-Time Inference on Low-Power Mobile Devices for Farmer-Centric Mango Leaf Disease Diagnostics | https://www.researchgate.net/publication/398757560_Pruning_and_Quantization_of_Lightweight_Hybrid_CNN_Models_for_Real-Time_Inference_on_Low-Power_Mobile_Devices_for_Farmer-Centric_Mango_Leaf_Disease_Diagnostics (2025-12-16) | **Paywall RG — download manual** | Two-stage pruning+PTQ di domain pertanian (mango) — pembanding domain terdekat, tapi bukan poultry dan tanpa ORT level-2 96-config |
| 5a | `05a_channel_pruning_groupvq_2023.pdf` | A lightweight deep neural network model and its applications based on channel pruning and group vector quantization | https://doi.org/10.1007/s00521-023-09332-z (Neural Computing & Applications, Springer) | **Paywall Springer — download via campus/VPN** | Channel pruning + group vector quantization — contoh kompresi hybrid yang rapi untuk related work |
| 5b | `05b_pruning_quant_iot_edge_2024.pdf` | Optimized CNN at the IoT edge for image detection using pruning and quantization | https://doi.org/10.1007/s11042-024-20523-1 (Multimedia Tools Appl., Springer 2024) | **Paywall Springer — download via campus/VPN** | Pruning+quant di IoT edge — untuk klaim edge deployment |

## Referensi yang sudah ada di Paper 1 (reuse, tidak perlu download lagi)

- Maulana & Ramasamy 2026 — Systematic Review Quantization-Optimized Lightweight Transformer (QAT > PTQ) — sudah di Paper 1, reuse untuk contrast PTQ vs QAT
- Hoang et al. 2026, Dhungana 2025, Tasdelen 2025, Degu 2023 — untuk gap poultry domain

## Checklist

- [x] 5 PDF open ter-download (01, 02, 03, 06, 07)
- [ ] 3 PDF paywall manual (04, 05a, 05b) — download dan taruh di literature/ dengan nama di atas
- [ ] Import ke Zotero + verify bib entries (cross-check DOI)
