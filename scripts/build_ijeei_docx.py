#!/usr/bin/env python3
"""Build polished IJEEI formatted Word Document (.docx) from manuscript draft."""

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "paper" / "MANUSCRIPT_IJEEI.docx"

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_document():
    doc = docx.Document()
    
    # Page setup: A4, 2.5cm margins
    sections = doc.sections
    for section in sections:
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.top_margin = Inches(0.98)
        section.bottom_margin = Inches(0.98)
        section.left_margin = Inches(0.98)
        section.right_margin = Inches(0.98)
        
    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(10)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(4)
    normal_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(12)
    run_title = p_title.add_run("Localizing and Mitigating INT8 Quantization Collapse in Lightweight CNNs for Farm-Level Poultry Disease Diagnostics on Raspberry Pi 5")
    run_title.font.name = 'Times New Roman'
    run_title.font.size = Pt(14)
    run_title.font.bold = True

    # Authors
    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.space_after = Pt(4)
    r1 = p_auth.add_run("Fatih Jawwad Al Mumtaz")
    r1.bold = True
    p_auth.add_run("1,*, ")
    r2 = p_auth.add_run("Amirul Al Aziz")
    r2.bold = True
    p_auth.add_run("2, and ")
    r3 = p_auth.add_run("Muhammad Iqbal Hidayatullah")
    r3.bold = True
    p_auth.add_run("1\n")
    
    # Affiliations
    r_aff = p_auth.add_run("1 Department of Informatics, Faculty of Science and Technology, Universitas Darussalam Gontor, Ponorogo, Indonesia\n2 Department of Agricultural Industrial Technology, Faculty of Science and Technology, Universitas Darussalam Gontor, Ponorogo, Indonesia\n*Corresponding Author: fatihalmumtaz76@student.cs.unida.gontor.ac.id")
    r_aff.font.size = Pt(9)
    r_aff.font.italic = True

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # Abstract Box
    p_abs = doc.add_paragraph()
    p_abs.paragraph_format.space_after = Pt(6)
    r_abs_label = p_abs.add_run("Abstract: ")
    r_abs_label.bold = True
    r_abs_label.italic = True
    r_abs_text = p_abs.add_run(
        "Non-invasive diagnostic tools powered by deep learning offer significant potential for early disease detection in poultry farming, mitigating severe economic losses caused by Coccidiosis, Salmonellosis, and Newcastle Disease. However, existing state-of-the-art vision models predominantly target cloud or GPU servers, which are impractical for rural, resource-constrained smallholder farms. Deploying lightweight Convolutional Neural Networks (CNNs) directly onto cost-effective edge devices via INT8 Post-Training Quantization (PTQ) frequently induces severe accuracy degradation, termed quantization collapse. In this study, we present a comprehensive empirical and theoretical investigation into INT8 quantization dynamics across lightweight CNN architectures (MobileNetV2, ShuffleNetV2, EfficientNet-B0, and EfficientNet-Lite0) deployed on a physical Raspberry Pi 5 single-board computer using ONNX Runtime. First, a perceptual hash (pHash) integrity audit of the prevailing 8,770-image public fecal benchmark revealed 617 near-duplicate pairs, including 94 byte-identical images leaked between training and testing splits; all subsequent evaluations are conducted on a rigorously sanitized 8,153-image sterile split. Second, we localize the structural failure point of EfficientNet-B0, where naive dynamic INT8 and static per-tensor quantization trigger catastrophic degradation from 98.55% down to 10.42% and 9.32% accuracy, respectively. Systematic layer-wise ablation and Signal-to-Quantization-Noise Ratio (SQNR) profiling demonstrate that this collapse is strictly driven by dynamic range clipping within depthwise separable convolutions, rather than Squeeze-and-Excitation (SE) modules. Static per-channel Quantize-Dequantize (QDQ) calibration fully rescues EfficientNet-B0 to 94.56% accuracy (p < 1e-15 vs. dynamic). Third, hardware profiling reveals that dynamic INT8 introduces runtime quantize/dequantize kernel overheads, causing a 1.93-2.80x slowdown relative to FP32, whereas static per-channel QDQ yields a 2.34x speedup (13.50 ms latency, 74.07 FPS) under four threads without thermal throttling. Furthermore, coupling structured channel pruning (s=0.3) with INT8 yields a 6.89x total compression down to 1.23 MB with 91.09% accuracy. Finally, Gradient-weighted Class Activation Mapping (Grad-CAM) confirms that the calibrated edge models attend to authentic pathological fecal lesions rather than environmental background noise."
    )
    r_abs_text.font.size = Pt(9.5)

    # Keywords
    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.space_after = Pt(12)
    r_kw_lbl = p_kw.add_run("Keywords: ")
    r_kw_lbl.bold = True
    r_kw_lbl.italic = True
    r_kw_val = p_kw.add_run("Edge AI, INT8 Quantization Collapse, Lightweight CNNs, Raspberry Pi 5, Precision Poultry Farming, Explainable AI, Structured Pruning.")
    r_kw_val.font.size = Pt(9.5)

    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(11)
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(text)
        r.bold = True
        r.italic = True
        r.font.size = Pt(10.5)
        return p

    def add_eq(formula, eq_num):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(f"    {formula}    ({eq_num})")
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10)
        r.italic = True
        return p

    # 1. Introduction
    add_h1("1. Introduction")
    doc.add_paragraph(
        "Poultry production is a vital sector of global livestock agriculture, supplying accessible animal protein and sustaining rural livelihoods across both developed and developing economies [1]. However, commercial broiler and layer operations remain highly vulnerable to widespread outbreaks of infectious diseases, notably Coccidiosis, Salmonellosis, and Newcastle Disease (NCD) [2]. Coccidiosis, caused by apicomplexan parasites of the genus Eimeria, inflicts profound intestinal mucosa lesions, precipitating severe hemorrhagic diarrhea, malabsorption, and acute flock mortality. Salmonellosis causes septicemic and enteric deterioration, posing concurrent zoonotic hazards to human food supply chains. Newcastle Disease, an avian paramyxovirus infection, is characterized by extreme contagiousness and flock fatality rates often approaching 100% in unvaccinated flocks [3]."
    )
    doc.add_paragraph(
        "Early and accurate clinical identification is imperative to initiate therapeutic intervention and biosecurity containment before pathogens disseminate throughout an entire poultry housing facility. Conventional diagnostic protocols rely on laboratory assays, including Polymerase Chain Reaction (PCR), enzyme-linked immunosorbent assay (ELISA), and microbiological culturing [4]. While definitive, these centralized laboratory methodologies impose substantial financial costs, require specialized diagnostic personnel, and incur turnaround times of 24 to 72 hours. For smallholder and semi-intensive farming operations in rural communities, regular laboratory testing is financially and logistically prohibitive [5]."
    )
    doc.add_paragraph(
        "In practical poultry management, visual examination of avian excreta provides the primary non-invasive proxy for gastrointestinal and systemic pathology. Pathological conditions exhibit distinct fecal manifestations: Coccidiosis typically causes bloody, hemorrhagic droppings; Salmonellosis produces fluid, mucoid, yellowish-green droppings; while virulent Newcastle Disease induces distinctive greenish watery discharge with high urate concentration. Automating this visual inspection via deep convolutional neural networks (CNNs) has yielded impressive accuracies in laboratory benchmarks [6, 7]. Nevertheless, translating these deep learning pipelines to resource-constrained smallholder farms faces three critical bottlenecks: (1) cloud-dependent server architectures incompatible with rural connectivity deficits, (2) severe quantization collapse when converting models to 8-bit integers for edge microprocessors, and (3) pervasive cross-split data leakage in public vision datasets that artificially distorts reported benchmark scores."
    )
    doc.add_paragraph(
        "This paper addresses these operational and scientific bottlenecks through an end-to-end empirical and theoretical investigation on physical Raspberry Pi 5 edge hardware. We audit public benchmarks, localize the exact mathematical cause of INT8 quantization collapse in compound-scaled CNNs, demonstrate complete recovery via static per-channel QDQ calibration, establish a synergistic structured-pruning pipeline, and validate pathological feature grounding through Explainable AI."
    )

    # 2. Related Work
    add_h1("2. Related Work")
    add_h2("A. Precision Poultry Pathology via Deep Learning")
    doc.add_paragraph(
        "Automating poultry health diagnostics through optical computer vision has developed rapidly following open-access data collection initiatives. Machuve et al. [2] introduced the foundational benchmark of 6,812 PCR-verified fecal photographs from smallholder farms in Tanzania. Subsequent works benchmarked classification heads; Tasdelen and Arslan [4] evaluated multiple deep CNNs, achieving 97.1% accuracy with MobileNetV2, while Dhungana et al. [6] reported 99.12% with EfficientNet-B0. However, these benchmarks were conducted exclusively on high-performance workstation GPUs without addressing physical edge latency or integer quantization constraints. Chi [8] introduced FecalFed, a federated-learning benchmark claiming 8,770 rigorously deduplicated images, which serves as the target of our dataset integrity audit in Section 4."
    )
    add_h2("B. Lightweight CNN Topologies for Embedded Systems")
    doc.add_paragraph(
        "Resource-constrained edge computing demands compact neural network topologies that minimize parameters, memory access cost (MAC), and floating-point operations (FLOPs). MobileNetV2 [9] introduced inverted residual blocks with linear bottlenecks, utilizing depthwise separable convolutions to decouple spatial filtering from channel projection. ShuffleNetV2 [10] optimized memory access cost by introducing channel split and channel shuffle operations. Tan and Le [11] established EfficientNet, which balances network depth, width, and resolution using automated compound scaling, incorporating Squeeze-and-Excitation (SE) attention mechanisms to optimize representational power."
    )
    add_h2("C. Quantization Mechanics and Edge Latency Overheads")
    doc.add_paragraph(
        "Post-Training Quantization (PTQ) converts 32-bit floating-point (FP32) weights into 8-bit integer equivalents, theoretically compressing memory footprint by 4x and accelerating integer SIMD execution on edge CPUs [12]. However, as surveyed by Nagel et al. [13], depthwise separable convolutions exhibit acute sensitivity to quantization noise when weights across individual filters exhibit high variance. Furthermore, runtime overheads from dynamic quantize/dequantize transformations frequently degrade execution speed on ARM architectures unless static calibration is utilized [14]."
    )

    # 3. Mathematical Foundations
    add_h1("3. Theoretical Formulation of Edge Quantization")
    doc.add_paragraph(
        "Uniform affine quantization maps real-valued continuous tensors r onto discrete integer representations q within bounded limits [q_min, q_max]:"
    )
    add_eq("q = clip( round(r / S) + Z,  q_min,  q_max )", "1")
    doc.add_paragraph(
        "where S represents the real-valued scale factor and Z denotes the integer zero-point parameter, formulated from the dynamic dynamic bounds [r_min, r_max]:"
    )
    add_eq("S = (r_max - r_min) / (q_max - q_min)", "2")
    add_eq("Z = round( - r_min / S ) + q_min", "3")
    doc.add_paragraph(
        "The corresponding dequantization operation reconstructs the approximate real-valued tensor:"
    )
    add_eq("r_hat = S * (q - Z)", "4")

    add_h2("A. Per-Tensor versus Per-Channel Granularity in Depthwise Layers")
    doc.add_paragraph(
        "In per-tensor quantization, a single global scale factor is shared across all output filters of a convolutional tensor. While adequate for standard convolutions where inter-channel spatial pooling mitigates variance, this approach proves fatal for depthwise convolutions where each filter operates independently. When weight ranges across channels differ substantially, a shared scale causes severe underflow in low-magnitude channels. Under per-channel quantization, an independent scale vector is computed along the output channel axis:"
    )
    add_eq("S_c = (max(W_c) - min(W_c)) / 255,  for all c in {1, ..., C_out}", "5")
    add_eq("q_c = clip( round(W_c / S_c) + Z_c,  -128,  127 )", "6")

    add_h2("B. Signal-to-Quantization-Noise Ratio (SQNR)")
    doc.add_paragraph(
        "To quantitatively assess precision degradation across individual network stages, we formulate the layer-wise weight SQNR (expressed in decibels, dB):"
    )
    add_eq("SQNR_l = 10 * log10( sum(W_l^2) / sum((W_l - W_hat_l)^2) )", "7")
    doc.add_paragraph(
        "where W_l represents original FP32 weights and W_hat_l denotes dequantized integer reconstructions. Lower SQNR directly correlates with severe feature distortion."
    )

    # 4. Dataset Integrity Audit
    add_h1("4. Dataset Integrity Audit and Sanitization")
    doc.add_paragraph(
        "To ensure uncompromising evaluation integrity, we audited the public 8,770-image benchmark distributed under Dianyo/poultry-fecal-fl [8]. We computed 64-bit DCT perceptual hashes across all images and evaluated pairwise Hamming distances:"
    )
    add_eq("D_H( pHash(x_i),  pHash(x_j) ) <= tau", "8")
    doc.add_paragraph(
        "With a strict threshold tau = 5 bits, our audit uncovered 617 near-duplicate image pairs across the benchmark. Among these, 393 pairs were byte-exact clones (MD5 identity), and 303 duplicate pairs spanned across official partition boundaries (147 test-train, 54 test-validation, and 102 train-validation pairs). Most crucially, 94 test images were byte-identical to training images, creating artificial accuracy inflation."
    )
    doc.add_paragraph(
        "To construct a leak-free sterile benchmark, duplicates were pruned following a strict preservation priority: test images were prioritized over validation, and validation over training. This removed 513 training, 78 validation, and 26 test duplicates, establishing a leak-free sterile benchmark of 8,153 images (5,100 training, 1,325 validation, 1,728 testing) across four classes: Coccidiosis (cocci), Salmonella (salmo), Newcastle Disease (ncd), and Healthy (healthy)."
    )

    # 5. Proposed Edge AI Architecture & Optimization Framework
    add_h1("5. Proposed Edge AI Diagnostic Framework")
    doc.add_paragraph(
        "The end-to-end framework encompasses model transfer learning, ONNX graph export, quantization calibration, and edge runtime deployment on Raspberry Pi 5. Two formal algorithmic pipelines govern calibration and diagnostic sensitivity mapping."
    )
    doc.add_paragraph(
        "Algorithm 1 (Static Per-Channel QDQ Calibration and Export Pipeline):\n"
        "1. Initialize ONNX CalibrationDataReader with 100 stratified training samples.\n"
        "2. Configure Quantization: ActivationType = QUInt8 (MinMax), WeightType = QInt8 (Per-channel, axis=0), Format = QDQ.\n"
        "3. For each Conv layer, compute per-channel scale S_c and zero-point Z_c via Eq. (5)-(6).\n"
        "4. Insert QuantizeLinear and DequantizeLinear nodes into the computational graph.\n"
        "5. Record activation clipping ranges [r_min, r_max] via offline observer over calibration set.\n"
        "6. Fuse adjacent BatchNorm and Conv nodes, and serialize optimized ONNX model.\n\n"
        "Algorithm 2 (Layer-Wise Quantization Sensitivity Localization):\n"
        "1. For each target layer group G in {Depthwise, Pointwise, SqueezeExcite, Head}:\n"
        "2. Construct hybrid model M_G where group G is isolated in FP32 while remaining layers are quantized to INT8 per-tensor.\n"
        "3. Evaluate Top-1 Accuracy on sterile test set (n=1,728) and compute weight SQNR via Eq. (7).\n"
        "4. If Acc_G >= 0.90 * Acc_FP32, flag group G as Primary Collapse Driver; else flag as Robust."
    )

    # 6. Experimental Results
    add_h1("6. Experimental Results and Discussion")
    add_h2("A. Test Classification Performance across Architectures")
    doc.add_paragraph(
        "All models were trained from ImageNet-pretrained weights on the sanitized dataset using Adam (lr=1e-4, cosine annealing, 50 epochs, batch size 32). Table 1 presents classification metrics on the 1,728 sterile test images."
    )

    # Table 1
    t1 = doc.add_table(rows=5, cols=7)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Architecture", "FP32 Acc.", "INT8 Dyn. Acc.", "Acc. Drop", "FP32 F1", "INT8 F1", "Model Size"]
    for i, h in enumerate(headers):
        cell = t1.cell(0, i)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        set_cell_background(cell, "EAEAEA")
        set_cell_margins(cell)
    
    rows_data = [
        ["MobileNetV2", "97.92%", "97.28%", "-0.64%", "0.9757", "0.9645", "8.48 -> 2.30 MB"],
        ["ShuffleNetV2", "97.69%", "97.63%", "-0.06%", "0.9710", "0.9682", "4.89 -> 1.47 MB"],
        ["EfficientNet-B0", "98.55%", "10.42%", "-88.13%", "0.9822", "0.0749", "15.30 -> 4.16 MB"],
        ["EfficientNet-Lite0", "97.80%", "96.95%", "-0.85%", "0.9740", "0.9650", "14.10 -> 3.95 MB"],
    ]
    for r_idx, r_data in enumerate(rows_data, start=1):
        for c_idx, val in enumerate(r_data):
            cell = t1.cell(r_idx, c_idx)
            cell.text = val
            set_cell_margins(cell)
            if c_idx in [2, 3] and "10.42%" in val:
                cell.paragraphs[0].runs[0].bold = True

    p_t1 = doc.add_paragraph("Table 1. Test Classification Performance on Sterile Benchmark (n=1,728)")
    p_t1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t1.paragraph_format.space_after = Pt(8)

    doc.add_paragraph(
        "As reported in Table 1, MobileNetV2 and ShuffleNetV2 preserve high diagnostic fidelity under dynamic INT8, whereas EfficientNet-B0 collapses catastrophically to 10.42% accuracy. In contrast, EfficientNet-Lite0—which replaces Swish activations with ReLU6 and omits Squeeze-and-Excitation blocks—maintains 96.95% accuracy under INT8, demonstrating that compound architectural design dictates quantization robustness."
    )

    add_h2("B. Dissecting the EfficientNet-B0 Quantization Collapse")
    doc.add_paragraph(
        "To isolate whether the collapse stems from Squeeze-and-Excitation (SE) attention or depthwise convolutions, we executed selective layer-wise QDQ ablations on EfficientNet-B0 (Table 2)."
    )

    # Table 2
    t2 = doc.add_table(rows=7, cols=5)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    h2 = ["Config", "Ablation Description", "Model Size", "Accuracy", "Collapse Observed?"]
    for i, h in enumerate(h2):
        cell = t2.cell(0, i)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        set_cell_background(cell, "EAEAEA")
        set_cell_margins(cell)
    
    rows_t2 = [
        ["A", "All layers Per-Tensor INT8", "4.29 MB", "9.32%", "Yes (Collapse)"],
        ["B", "All layers Per-Channel INT8", "4.67 MB", "94.56%", "No (Recovered)"],
        ["C", "Depthwise Per-Channel, rest Per-Tensor", "4.33 MB", "93.46%", "No (Recovered)"],
        ["D", "All Per-Channel except Depthwise", "4.51 MB", "8.33%", "Yes (Collapse)"],
        ["E", "Depthwise preserved in FP32, rest INT8", "4.69 MB", "97.16%", "No"],
        ["F", "Squeeze-and-Excitation preserved in FP32", "5.91 MB", "12.44%", "Yes (Collapse)"],
    ]
    for r_idx, r_data in enumerate(rows_t2, start=1):
        for c_idx, val in enumerate(r_data):
            cell = t2.cell(r_idx, c_idx)
            cell.text = val
            set_cell_margins(cell)
            if c_idx == 3 and val in ["94.56%", "93.46%"]:
                cell.paragraphs[0].runs[0].bold = True

    p_t2 = doc.add_paragraph("Table 2. Selective Layer-Group QDQ Ablation for EfficientNet-B0")
    p_t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t2.paragraph_format.space_after = Pt(8)

    doc.add_paragraph(
        "The empirical findings from Table 2 confirm: (1) Per-channel calibration applied solely to depthwise convolutions (Config C) restores accuracy to 93.46%, (2) Excluding depthwise layers from per-channel scaling (Config D) induces immediate collapse to 8.33%, and (3) Preserving SE blocks in FP32 (Config F) does not prevent collapse (12.44%). Layer-wise SQNR profiling reveals that depthwise layers in EfficientNet-B0 suffer an average SQNR drop to 8.4 dB under per-tensor scaling, whereas per-channel quantization restores SQNR to 21.8 dB (+13.4 dB gain)."
    )

    add_h2("C. Synergistic Compression: Structured Pruning Coupled with INT8 Quantization")
    doc.add_paragraph(
        "To explore whether parameter redundancy can be trimmed prior to quantization across distinct architectures, we applied structured channel pruning via Dependency Graph (DepGraph) L1-norm pruning on MobileNetV2 (at 30% and 50% channel sparsity) and EfficientNet-B0 (at 30% channel sparsity), followed by a 12-epoch fine-tuning on the sterile split and subsequent post-training quantization (Table 3)."
    )

    # Table 3
    t3 = doc.add_table(rows=8, cols=8)
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    h3 = ["Model & Compression Stage", "Parameters", "Disk Size", "Top-1 Acc.", "Healthy", "Cocci", "Salmo", "NCD"]
    for i, h in enumerate(h3):
        cell = t3.cell(0, i)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        set_cell_background(cell, "EAEAEA")
        set_cell_margins(cell)

    rows_t3 = [
        ["MNV2 Baseline FP32", "2,228,996 (100%)", "8.48 MB", "97.92%", "98.41%", "98.11%", "97.83%", "92.20%"],
        ["MNV2 Pruned FP32 (s=0.3)", "1,110,790 (49.8%)", "4.23 MB", "93.81%", "97.42%", "93.76%", "95.31%", "75.18%"],
        ["MNV2 Pruned INT8 (s=0.3)", "1,110,790 (49.8%)", "1.23 MB", "91.09%", "97.22%", "93.38%", "90.79%", "61.70%"],
        ["MNV2 Pruned INT8 (s=0.5)", "583,332 (26.2%)", "0.72 MB", "85.47%", "91.07%", "96.98%", "83.21%", "31.21%"],
        ["EB0 Baseline FP32", "4,012,672 (100%)", "15.30 MB", "98.55%", "99.11%", "99.24%", "98.92%", "94.33%"],
        ["EB0 Pruned FP32 (s=0.3)", "2,005,801 (50.0%)", "7.67 MB", "93.69%", "95.44%", "91.68%", "96.93%", "82.27%"],
        ["EB0 Pruned INT8 Static QDQ", "2,005,801 (50.0%)", "2.62 MB", "63.72%", "65.48%", "99.24%", "34.84%", "37.59%"],
    ]
    for r_idx, r_data in enumerate(rows_t3, start=1):
        for c_idx, val in enumerate(r_data):
            cell = t3.cell(r_idx, c_idx)
            cell.text = val
            set_cell_margins(cell)
            if c_idx in [2, 3] and ("1.23 MB" in val or "0.72 MB" in val or "2.62 MB" in val):
                cell.paragraphs[0].runs[0].bold = True

    p_t3 = doc.add_paragraph("Table 3. Structured Channel Pruning Coupled with INT8 Quantization Across MobileNetV2 and EfficientNet-B0")
    p_t3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t3.paragraph_format.space_after = Pt(8)

    doc.add_paragraph(
        "Coupling structured channel pruning with INT8 quantization yields an exceptional compression continuum: for MobileNetV2, 30% sparsity achieves a 6.89x total compression ratio (1.23 MB at 91.09% accuracy), while aggressive 50% sparsity reaches an 11.78x reduction (0.72 MB / 733 KB at 85.47% accuracy), revealing the critical accuracy cliff primarily driven by the low-prevalence Newcastle disease class. For EfficientNet-B0, dynamic INT8 collapses severely to 33.28% post-pruning, whereas static per-channel QDQ calibration recovers diagnostic performance to 63.72% (+30.44% accuracy recovery) at 2.62 MB. For ShuffleNetV2, automatic DepGraph channel pruning is topologically constrained due to paired branch dimension requirements in channel-split operations, highlighting a crucial architectural trade-off between manual channel dimension matching and automated pruning pipelines."
    )

    add_h2("D. Physical Hardware Profiling on Raspberry Pi 5")
    doc.add_paragraph(
        "Inference latency was benchmarked on a physical Raspberry Pi 5 (Broadcom BCM2712 quad-core Cortex-A76 @ 2.4 GHz, 8 GB RAM) running Debian 13 trixie and ONNX Runtime 1.29.0 across 200 iterations (Table 4)."
    )

    # Table 4
    t4 = doc.add_table(rows=8, cols=7)
    t4.alignment = WD_TABLE_ALIGNMENT.CENTER
    h4 = ["Model & Precision", "Latency P50", "Latency P99", "Throughput", "RAM Peak", "Temp.", "Speedup"]
    for i, h in enumerate(h4):
        cell = t4.cell(0, i)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        set_cell_background(cell, "EAEAEA")
        set_cell_margins(cell)

    rows_t4 = [
        ["MobileNetV2 FP32", "14.23 ms", "15.10 ms", "70.27 FPS", "58.4 MB", "48.2 °C", "1.00x"],
        ["MobileNetV2 INT8 Dyn.", "39.83 ms", "42.15 ms", "25.10 FPS", "44.1 MB", "49.0 °C", "0.36x"],
        ["ShuffleNetV2 FP32", "6.19 ms", "6.84 ms", "161.55 FPS", "41.2 MB", "46.5 °C", "1.00x"],
        ["ShuffleNetV2 INT8 Dyn.", "11.95 ms", "13.02 ms", "83.68 FPS", "36.8 MB", "47.1 °C", "0.52x"],
        ["EfficientNet-B0 FP32", "31.53 ms", "33.20 ms", "31.71 FPS", "72.5 MB", "50.1 °C", "1.00x"],
        ["EfficientNet-B0 INT8 Dyn.", "63.99 ms", "67.80 ms", "15.62 FPS", "51.3 MB", "51.4 °C", "0.49x"],
        ["EfficientNet-B0 Static QDQ", "13.50 ms", "14.22 ms", "74.07 FPS", "49.8 MB", "48.8 °C", "2.34x"],
    ]
    for r_idx, r_data in enumerate(rows_t4, start=1):
        for c_idx, val in enumerate(r_data):
            cell = t4.cell(r_idx, c_idx)
            cell.text = val
            set_cell_margins(cell)
            if r_idx == 7:
                cell.paragraphs[0].runs[0].bold = True

    p_t4 = doc.add_paragraph("Table 4. Physical Hardware Profiling on Raspberry Pi 5 (4 Threads, Batch Size 1)")
    p_t4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t4.paragraph_format.space_after = Pt(8)

    doc.add_paragraph(
        "Dynamic INT8 exhibits a 1.93-2.80x slowdown on ARM CPUs due to runtime quantize/dequantize memory traffic around convolution kernels. Static per-channel QDQ eliminates runtime parameter computation, unlocking a 2.34x acceleration (13.50 ms, 74.07 FPS), which easily exceeds real-time camera streaming requirements (30 FPS) with low thermal overhead (< 49 °C)."
    )

    add_h2("E. Visual Explainability via Grad-CAM")
    doc.add_paragraph(
        "Grad-CAM heatmaps generated from the final convolutional stage verify that both FP32 and calibrated INT8 models focus precisely on pathological markers: high activation density over bloody hemorrhagic stool clusters (Coccidiosis), concentrated attention on watery mucosal discharge (Salmonellosis), and clear alignment with dark greenish fecal pools (Newcastle Disease). Diffuse, low-magnitude heatmaps across healthy samples confirm the absence of spurious environmental background triggers."
    )

    # 7. Economic Feasibility
    add_h1("7. Economic Feasibility and Farm-Level Deployment")
    doc.add_paragraph(
        "Table 5 contrasts the proposed Raspberry Pi 5 edge diagnostic node against conventional cloud-GPU server configurations."
    )

    # Table 5
    t5 = doc.add_table(rows=6, cols=3)
    t5.alignment = WD_TABLE_ALIGNMENT.CENTER
    h5 = ["Evaluation Dimension", "Cloud GPU Infrastructure", "Raspberry Pi 5 Edge Node"]
    for i, h in enumerate(h5):
        cell = t5.cell(0, i)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        set_cell_background(cell, "EAEAEA")
        set_cell_margins(cell)

    rows_t5 = [
        ["Initial Capital Cost", "$1,500 - $2,500 (Server/PC)", "$80 - $110 (Pi 5 + Camera)"],
        ["Monthly Operating Cost", "$50 - $150 (Cloud API/Bandwidth)", "$0 (Local Inference)"],
        ["Network Dependency", "Mandatory (Continuous Upload)", "None (100% Offline)"],
        ["Inference Latency", "250 - 800 ms (RTT + Queue)", "13.50 ms (Local Real-Time)"],
        ["Power Consumption", "250W - 450W", "5W - 12W (Solar/Battery Operable)"],
    ]
    for r_idx, r_data in enumerate(rows_t5, start=1):
        for c_idx, val in enumerate(r_data):
            cell = t5.cell(r_idx, c_idx)
            cell.text = val
            set_cell_margins(cell)

    p_t5 = doc.add_paragraph("Table 5. Practical Deployment and Economic Feasibility Comparison")
    p_t5.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t5.paragraph_format.space_after = Pt(8)

    # 8. Conclusion
    add_h1("8. Conclusion")
    doc.add_paragraph(
        "This study presented a rigorous empirical and architectural framework for deploying lightweight deep learning models on resource-constrained edge hardware for precision poultry disease diagnostics. By auditing and sanitizing public benchmarks, localizing the root cause of INT8 quantization collapse to depthwise convolution dynamic ranges, and demonstrating that static per-channel QDQ calibration achieves 94.56% accuracy with 13.50 ms latency (74 FPS) on a physical Raspberry Pi 5, we show that edge-native poultry diagnostics are viable for rural deployment. Furthermore, structured pruning coupled with INT8 achieves a 6.89x compression down to 1.23 MB while preserving over 91% accuracy."
    )

    # References
    add_h1("References")
    refs = [
        "[1] USDA, 'Poultry - Production and Value 2023 Summary,' United States Department of Agriculture National Agricultural Statistics Service, Washington, D.C., Tech. Rep., Apr. 2024.",
        "[2] E. Machuve et al., 'An agricultural dataset for poultry diseases diagnostics in developing countries,' Scientific Data, vol. 9, no. 1, p. 760, 2022.",
        "[3] D. E. Swayne et al., Diseases of Poultry, 14th ed. Hoboken, NJ: Wiley-Blackwell, 2020.",
        "[4] J. A. Tasdelen and O. Arslan, 'Automated classification of poultry diseases using deep convolutional neural networks,' Comput. Electron. Agric., vol. 228, p. 109552, 2025.",
        "[5] E. Degu, S. Dereje, and A. Girma, 'Poultry disease detection and classification using computer vision and deep learning techniques,' Smart Agric. Technol., vol. 6, p. 100345, 2023.",
        "[6] M. Dhungana et al., 'Transfer learning based EfficientNet for avian fecal image classification,' Poult. Sci., vol. 104, no. 2, p. 103210, 2025.",
        "[7] J. Deng et al., 'ImageNet: A large-scale hierarchical image database,' in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2009, pp. 248-255.",
        "[8] H. Chi, 'FecalFed: A federated learning benchmark for poultry fecal disease classification,' IEEE Access, vol. 12, pp. 45210-45222, 2026.",
        "[9] M. Sandler et al., 'MobileNetV2: Inverted residuals and linear bottlenecks,' in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 4510-4520.",
        "[10] N. Ma et al., 'ShuffleNet V2: Practical guidelines for efficient CNN architecture design,' in Proc. Eur. Conf. Comput. Vis. (ECCV), 2018, pp. 116-131.",
        "[11] M. Tan and Q. V. Le, 'EfficientNet: Rethinking model scaling for convolutional neural networks,' in Proc. Int. Conf. Mach. Learn. (ICML), 2019, pp. 6105-6114.",
        "[12] B. Jacob et al., 'Quantization and training of neural networks for efficient integer-arithmetic-only inference,' in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 2704-2713.",
        "[13] M. Nagel et al., 'A white paper on neural network quantization,' arXiv preprint arXiv:2106.08295, 2021.",
        "[14] ONNX Runtime Authors, 'ONNX Runtime: Cross-platform, high performance ML inferencing accelerator,' 2024. [Online]. Available: https://onnxruntime.ai",
        "[15] R. R. Selvaraju et al., 'Grad-CAM: Visual explanations from deep networks via gradient-based localization,' in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 618-626.",
        "[16] G. Fang et al., 'DepGraph: Towards any structural pruning,' in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 16091-16100.",
    ]
    for r in refs:
        p_ref = doc.add_paragraph(r)
        p_ref.paragraph_format.space_after = Pt(2)

    # Author Biographies
    add_h1("Author Biographies")
    bios = [
        ("Fatih Jawwad Al Mumtaz", "is an undergraduate researcher in the Department of Informatics at Universitas Darussalam Gontor, Indonesia. His research interests center on Edge AI, TinyML, model compression, computer vision, and embedded deep learning systems for precision agriculture and livestock management."),
        ("Amirul Al Aziz", "is a researcher in the Department of Agricultural Industrial Technology at Universitas Darussalam Gontor, Indonesia. His research focuses on smart farming technologies, agricultural process automation, and non-invasive livestock diagnostics."),
        ("Muhammad Iqbal Hidayatullah", "is a faculty member and researcher in the Department of Informatics at Universitas Darussalam Gontor, Indonesia. His research domains encompass applied artificial intelligence, machine learning, distributed computing, and intelligent systems engineering.")
    ]
    for name, bio in bios:
        p_b = doc.add_paragraph()
        r_n = p_b.add_run(f"{name} ")
        r_n.bold = True
        p_b.add_run(bio)
        p_b.paragraph_format.space_after = Pt(6)

    doc.save(str(OUT_PATH))
    print(f"Successfully generated IJEEI formatted Word document: {OUT_PATH}")

if __name__ == "__main__":
    create_document()
