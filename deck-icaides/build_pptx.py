#!/usr/bin/env python3
"""
Build deck ICAIDES 2026 sebagai PPTX native (teks bisa diedit).
Font: Geist Mono (Regular + Bold).
Semua angka diambil dari paper/main.tex & deck-icaides/slides.tex.
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import ImageFont

# ---------------- palet ----------------
OLIVE = RGBColor(0x5A, 0x74, 0x11)
DARK  = RGBColor(0x1C, 0x1C, 0x1A)
GREY  = RGBColor(0x6B, 0x6B, 0x63)
FAINT = RGBColor(0x9A, 0x9A, 0x91)
LINE  = RGBColor(0xD8, 0xD8, 0xCE)
CREAM = RGBColor(0xF0, 0xEE, 0xDC)
PALE  = RGBColor(0xDD, 0xE3, 0xC2)
RUST  = RGBColor(0xC1, 0x50, 0x2E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Geist Mono"
REG = "/Users/wedjaw/Library/Fonts/GeistMono-400.ttf"
BLD = "/Users/wedjaw/Library/Fonts/GeistMono-700.ttf"

W, H = 13.333, 7.5
ML, MR = 0.72, 0.72
CW = W - ML - MR

prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)
BLANK = prs.slide_layouts[6]

PROBLEMS = []


# ---------------- pengukur teks ----------------
LINE_FACTOR = 1.30   # Geist Mono: (ascent - descent) / upm = 1.3


def measure(text, pt, width_in, bold=False, line_mult=1.10):
    """Tinggi teks (inch). line_mult adalah kelipatan single line spacing,
    dan single line spacing font ini = 1.3 x ukuran font."""
    f = ImageFont.truetype(BLD if bold else REG, int(pt))
    max_w = width_in * 72.0
    lines, cur = 0, ""
    for para in text.split("\n"):
        if not para.strip():
            lines += 1
            continue
        cur = ""
        for word in para.split(" "):
            trial = (cur + " " + word).strip()
            if f.getlength(trial) <= max_w or not cur:
                cur = trial
            else:
                lines += 1
                cur = word
        lines += 1
    return lines * (pt * LINE_FACTOR * line_mult) / 72.0


def tx(slide, x, y, w, h, text, size=12, color=DARK, bold=False,
       align=PP_ALIGN.LEFT, line_mult=1.10, space_after=0, anchor=MSO_ANCHOR.TOP,
       check=True, name=""):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    tf.auto_size = None

    paras = text.split("\n")
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.line_spacing = line_mult
        para.space_after = Pt(space_after)
        run = para.add_run()
        run.text = p
        run.font.name = FONT
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color

    if check:
        need = measure(text, size, w, bold, line_mult)
        if need > h + 0.012:
            PROBLEMS.append(
                f"OVERFLOW {name or text[:34]!r}: butuh {need:.2f}in, kotak {h:.2f}in")
    return box


def rule(slide, x, y, w, color=LINE, pt=0.75):
    ln = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Pt(pt))
    ln.fill.solid()
    ln.fill.fore_color.rgb = color
    ln.line.fill.background()
    ln.shadow.inherit = False
    return ln


def block(slide, x, y, w, h, fill, line_color=None, radius=False):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line_color is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line_color
        shp.line.width = Pt(0.75)
    shp.shadow.inherit = False
    if radius:
        try:
            shp.adjustments[0] = 0.06
        except Exception:
            pass
    return shp


def head(slide, kicker, page, dark=False):
    """Kicker kiri atas + nomor halaman kanan atas."""
    c = PALE if dark else OLIVE
    n = PALE if dark else FAINT
    tx(slide, ML, 0.34, CW * 0.7, 0.22, kicker.upper(), 8.5, c, bold=True,
       check=False)
    tx(slide, W - MR - 1.2, 0.34, 1.2, 0.22, f"{page:02d} / 12", 8.5, n,
       align=PP_ALIGN.RIGHT, check=False)


def title(slide, text, y=0.66, size=23, color=DARK, w=None):
    h = measure(text, size, w or CW, True) + 0.06
    tx(slide, ML, y, w or CW, h, text, size, color, bold=True, name="title")
    return y + h


def lead(slide, text, y, size=13.5, color=OLIVE, w=None):
    h = measure(text, size, w or CW, True) + 0.05
    tx(slide, ML, y, w or CW, h, text, size, color, bold=True, name="lead")
    return y + h


def body(slide, text, y, size=12, color=DARK, w=None, x=None, name=""):
    h = measure(text, size, w or CW) + 0.06
    tx(slide, x if x is not None else ML, y, w or CW, h, text, size, color,
       name=name)
    return y + h


def aside(slide, text, y, size=9, color=GREY, w=None, x=None):
    ww = w or CW
    h = measure(text, size, ww) + 0.05
    tx(slide, x if x is not None else ML, y, ww, h, text, size, color,
       line_mult=1.18, name="aside")
    return y + h


def table(slide, x, y, cols, rows, size=11, row_h=0.30, head_color=OLIVE,
          accent_rows=(), accent_color=OLIVE, bold_rows=(), col_align=None):
    """Tabel hairline: header + baris, garis atas/bawah tebal tipis."""
    widths = [c[1] for c in cols]
    total = sum(widths)
    if total > CW + 0.001:
        PROBLEMS.append(f"TABLE TERLALU LEBAR: total {total:.2f}in > {CW:.2f}in")
    if x + total > W - MR + 0.001:
        PROBLEMS.append(f"TABLE KELUAR KANAN: x={x:.2f} + {total:.2f} = {x+total:.2f}in")
    xs, acc = [], x
    for wd in widths:
        xs.append(acc)
        acc += wd

    # garis atas
    rule(slide, x, y, total, DARK, 1.0)

    # header
    hy = y + 0.07
    for i, (label, _) in enumerate(cols):
        al = col_align[i] if col_align else (PP_ALIGN.LEFT if i == 0 else PP_ALIGN.RIGHT)
        tx(slide, xs[i], hy, widths[i], 0.22, label, size - 0.5, head_color,
           bold=True, align=al, check=False)
    y2 = hy + 0.24
    rule(slide, x, y2, total, DARK, 0.9)
    y2 += 0.08

    for ri, row in enumerate(rows):
        is_acc = ri in accent_rows
        is_bold = ri in bold_rows
        col = accent_color if is_acc else DARK
        for ci, cell in enumerate(row):
            al = col_align[ci] if col_align else (PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.RIGHT)
            tx(slide, xs[ci], y2, widths[ci], row_h - 0.04, cell, size, col,
               bold=is_bold or is_acc, align=al, check=False)
        y2 += row_h
        if ri < len(rows) - 1:
            rule(slide, x, y2 - 0.05, total, LINE, 0.5)

    rule(slide, x, y2 - 0.04, total, DARK, 1.0)
    return y2 + 0.12


def card(slide, x, y, w, h, kicker, value, note="", accent=False):
    block(slide, x, y, w, h, CREAM if not accent else PALE)
    tx(slide, x + 0.20, y + 0.17, w - 0.40, 0.20, kicker.upper(), 8, OLIVE,
       bold=True, check=False)
    tx(slide, x + 0.20, y + 0.42, w - 0.40, 0.34, value, 15, DARK, bold=True,
       check=False)
    if note:
        nh = measure(note, 8.5, w - 0.40, line_mult=1.18) + 0.06
        tx(slide, x + 0.20, y + 0.80, w - 0.40, nh, note, 8.5, GREY,
           line_mult=1.35, name="cardnote")


# =====================================================================
# 01 — JUDUL
# =====================================================================
s = prs.slides.add_slide(BLANK)
block(s, 0, 0, W, H, OLIVE)
head(s, "ICAIDES 2026  ·  Paper ID 1571359798", 1, dark=True)

tx(s, ML, 1.52, CW * 0.74, 0.22,
   "EDGE POULTRY DISEASE DETECTION ON RASPBERRY PI 5", 9.5, PALE, bold=True,
   check=False)

ty = 1.92
th = measure("Localizing INT8\nQuantization Collapse\nin EfficientNet-B0",
             34, 7.4, True, 1.02) + 0.1
tx(s, ML, ty, 7.6, th,
   "Localizing INT8\nQuantization Collapse\nin EfficientNet-B0",
   34, CREAM, bold=True, line_mult=1.02, name="cover-title")

rule(s, ML, 5.28, 2.4, PALE, 1.5)

tx(s, ML, 5.56, 6.4, 0.22, "Fatih Jawwad Al Mumtaz  ·  Amirul Al Aziz  ·  Muhammad Iqbal Hidayatullah",
   10, CREAM, check=False)
tx(s, ML, 5.86, 6.4, 0.22, "Universitas Darussalam Gontor, Indonesia", 9, PALE,
   check=False)

tx(s, W - MR - 1.9, 6.86, 1.9, 0.20, "November 2026", 9, PALE,
   align=PP_ALIGN.RIGHT, check=False)

# =====================================================================
# 02 — GAP
# =====================================================================
s = prs.slides.add_slide(BLANK)
head(s, "Why edge, why this benchmark", 2)
y = title(s, "The gap this paper closes")
y = lead(s, "Farm-level detection needs models that run on a Pi, not a GPU.", y + 0.10)

y = body(s, "Poultry disease — Coccidiosis, Salmonellosis, Newcastle — causes major "
            "losses, yet laboratory diagnosis is out of reach for small farms. "
            "Prior work reaches 99.12% with EfficientNet-B0 and 97.1% with "
            "MobileNetV2, but targets the cloud rather than the edge.",
         y + 0.24, 12.5)

y += 0.34
rule(s, ML, y, CW, LINE, 0.5)
y += 0.30

tx(s, ML, y, 3.0, 0.20, "TWO QUESTIONS", 8.5, OLIVE, bold=True, check=False)
tx(s, ML + 3.35, y, 3.0, 0.20, "SCOPE", 8.5, OLIVE, bold=True, check=False)

y += 0.28
body(s, "Which lightweight architectures survive INT8 on CPU — and what does "
        "the accuracy–latency tradeoff look like on a Raspberry Pi 5?",
     y, 12, w=3.1)
body(s, "Three models, one board, one runtime (ONNX Runtime 1.29.0). "
        "No claimed generality beyond the measured stack.",
     y, 11, GREY, w=5.2, x=ML + 3.35)

aside(s, "All accuracy figures in this talk come from a deduplicated split, "
         "described on the next slide.", 6.72, 9)

# =====================================================================
# 03 — BENCHMARK AUDIT
# =====================================================================
s = prs.slides.add_slide(BLANK)
head(s, "Why edge, why this benchmark", 3)
y = title(s, "The benchmark, and what was wrong with it")
y = lead(s, "8,770 published images — audited, then deduplicated to 8,153.", y + 0.10)

y += 0.42
cw3 = (CW - 0.70) / 3
card(s, ML, y, cw3, 1.34, "claimed", "8,770 images",
     "FecalFed benchmark, published as leak-free")
card(s, ML + cw3 + 0.35, y, cw3, 1.34, "audit", "617 pairs",
     "pHash near-duplicates; 94 test images byte-identical to train")
card(s, ML + 2 * (cw3 + 0.35), y, cw3, 1.34, "used", "8,153 images",
     "test n = 1,728, four classes, seed 42", accent=True)

y += 1.62
y = body(s, "Every number reported in this talk uses the deduplicated split. "
            "Four classes: Coccidiosis, Healthy, Salmonella, Newcastle — "
            "with Newcastle underrepresented at 141 test images.",
         y, 12.5)

aside(s, "Removing the duplicates matters: without it, 94 test images would have "
         "been scored against images the model had already seen.", 6.80, 9)

# =====================================================================
# 04 — DYNAMIC INT8
# =====================================================================
s = prs.slides.add_slide(BLANK)
head(s, "What survives INT8, and what collapses", 4)
y = title(s, "Dynamic INT8: two survive, one collapses")
y = lead(s, "Same pipeline, same data, same protocol — one model is destroyed.", y + 0.10)

y += 0.58
y = table(s, ML, y,
          [("Model", 4.6), ("FP32", 2.1), ("INT8 dynamic", 2.4), ("Drop", 2.1)],
          [["MobileNetV2", "97.92%", "97.28%", "0.64 pp"],
           ["ShuffleNetV2", "97.69%", "97.63%", "0.06 pp"],
           ["EfficientNet-B0", "98.55%", "10.42%", "88.13 pp"]],
          size=13, row_h=0.52, accent_rows=(2,), accent_color=RUST)

y += 0.36
y = body(s, "EfficientNet-B0 degenerates into a near-single-class predictor: "
            "1,676 of 1,728 test images land in the Newcastle column.",
         y, 13.5)

y += 0.30
aside(s, "Paired McNemar on the same 1,728 images — MobileNetV2: b = 18, c = 7, "
         "p = 0.0455. ShuffleNetV2: b = 6, c = 5, p = 1.0. "
         "MobileNetV2's drop is detectable; ShuffleNetV2's is not.", y, 9.5)

# =====================================================================
# 05 — CONFUSION
# =====================================================================
s = prs.slides.add_slide(BLANK)
head(s, "What survives INT8, and what collapses", 5)
y = title(s, "Where exactly EfficientNet-B0 fails")

fig_w = 9.9
fig_h = fig_w / 2.12
s.shapes.add_picture("figures/confusion_eb0_2panel.png",
                     Inches((W - fig_w) / 2), Inches(y + 0.22),
                     width=Inches(fig_w), height=Inches(fig_h))

cy = y + 0.22 + fig_h + 0.20
tx(s, ML, cy, CW, 0.20,
   "FP32 (left) versus dynamic INT8 (right). Per-class INT8 recall: "
   "Coccidiosis 2.27%, Healthy 4.76%, Salmonella 0.54%, Newcastle 100%.",
   10.5, DARK, check=True, name="conf-cap")

# =====================================================================
# 06 — WHY
# =====================================================================
s = prs.slides.add_slide(BLANK)
head(s, "What survives INT8, and what collapses", 6)
y = title(s, "Why: one shared scale, 127 levels")
y = lead(s, "Per-tensor, the widest channel sets the scale for every channel.", y + 0.10)

y += 0.34
# bar skala
bx, bw = ML + 0.05, 6.2
tx(s, ML, y, 7.2, 0.20, "PER-TENSOR — widest channel spans all 127 levels each side",
   8.5, GREY, bold=True, check=False)
block(s, bx, y + 0.30, bw, 0.13, OLIVE)

ty2 = y + 0.72
tx(s, ML, ty2, 7.2, 0.20,
   "PER-TENSOR — a channel with a 112× smaller range gets 127 / 112 ≈ 1 level",
   8.5, GREY, bold=True, check=False)
block(s, bx + bw / 2 - 0.035, ty2 + 0.30, 0.07, 0.13, RUST)

ty3 = ty2 + 0.72
tx(s, ML, ty3, 7.2, 0.20,
   "PER-CHANNEL — that same channel gets its own scale, full range back",
   8.5, GREY, bold=True, check=False)
block(s, bx, ty3 + 0.30, bw, 0.13, OLIVE)

ay = ty3 + 0.66
rule(s, bx, ay, bw, FAINT, 0.75)
for frac, lab in ((0.0, "−127"), (0.5, "0"), (1.0, "+127")):
    xx = bx + bw * frac
    rule(s, xx - 0.004, ay - 0.05, 0.008, FAINT, 3.0)
    lx = min(max(xx - 0.42, ML), W - MR - 0.84)
    tx(s, lx, ay + 0.09, 0.84, 0.18, lab, 8.5, FAINT,
       align=PP_ALIGN.CENTER, check=False)

# panel kanan
px = ML + 7.55
pw = W - MR - px
tx(s, px, y + 0.06, pw, 0.20, "DEPTHWISE CHANNEL-RANGE RATIO", 8.5, OLIVE,
   bold=True, check=False)
ry = y + 0.38
for name, val, col, verdict in (
        ("EfficientNet-B0", "39.94×", RUST, "collapses"),
        ("MobileNetV2", "9.20×", OLIVE, "survives"),
        ("ShuffleNetV2", "5.32×", OLIVE, "survives")):
    tx(s, px, ry, pw * 0.62, 0.20, name, 9.5, DARK, check=False)
    tx(s, px + pw * 0.62, ry, pw * 0.38, 0.20, val, 9.5, col, bold=True,
       align=PP_ALIGN.RIGHT, check=False)
    tx(s, px, ry + 0.19, pw, 0.18, verdict, 8, GREY, check=False)
    ry += 0.60

tx(s, px, ry + 0.02, pw, 0.20, "worst layer 112×", 9, RUST, bold=True,
   check=False)

aside(s, "Median of per-layer P95/P5 channel-range ratios. EfficientNet-B0's worst "
         "depthwise layer reaches 112×, enough to quantize a channel to a constant "
         "(0 dB SQNR in the audit JSON). Supported by SQNR analysis and the family "
         "ablation below; a single-layer ablation would test it directly.",
      6.66, 9)

# =====================================================================
# 07 — PER-CHANNEL
# =====================================================================
s = prs.slides.add_slide(BLANK)
head(s, "What survives INT8, and what collapses", 7)
y = title(s, "Per-channel recovers it: 9.32% versus 94.56%")
y = lead(s, "Granularity, not quantization itself, decides the outcome.", y + 0.10)

y += 0.52
y = table(s, ML, y,
          [("EfficientNet-B0", 4.5), ("Accuracy", 2.2), ("Size", 2.0),
           ("Pi, t = 4", 2.4)],
          [["dynamic INT8", "10.42%", "4.16 MB", "63.99 ms"],
           ["static per-tensor", "9.32%", "4.29 MB", "13.63 ms"],
           ["static per-channel", "94.56%", "4.67 MB", "13.50 ms"]],
          size=13, row_h=0.50, accent_rows=(2,))

y += 0.34
y = body(s, "Recovery, not parity: still 3.9 pp below FP32 (98.50% on the exported "
            "ONNX graph). A balanced 25-per-class replication on a retrained model "
            "shows the same pattern — per-tensor 27.14%, per-channel 93.40%.",
         y, 13.5)

y += 0.34
block(s, ML, y, CW, 0.94, CREAM)
tx(s, ML + 0.22, y + 0.15, CW - 0.44, 0.20, "CAVEAT, UP FRONT", 8, RUST,
   bold=True, check=False)
tx(s, ML + 0.22, y + 0.38, CW - 0.44, 0.44,
   "Calibration used 100 images, all Coccidiosis, because the loader was "
   "class-ordered. Treat this as diagnostic evidence, not a robustness claim.",
   10, DARK, name="caveat")

# =====================================================================
# 08 — ABLATION
# =====================================================================
s = prs.slides.add_slide(BLANK)
head(s, "What survives INT8, and what collapses", 8)
y = title(s, "Which layers matter: the selective ablation")
y = lead(s, "Depthwise granularity decides collapse in this setup.", y + 0.10)

y += 0.52
y = table(s, ML, y,
          [("Configuration", 6.0), ("Size", 1.9), ("Accuracy", 2.0),
           ("Collapse", 1.99)],
          [["C  depthwise per-channel, rest per-tensor", "4.33 MB", "93.46%", "no"],
           ["D  all per-channel except depthwise", "4.51 MB", "8.33%", "yes"],
           ["E  depthwise left in FP32", "4.69 MB", "97.16%", "no"],
           ["F  squeeze-and-excitation left in FP32", "5.91 MB", "12.44%", "yes"]],
          size=11.5, row_h=0.50,
          accent_rows=(0,), accent_color=OLIVE)

y += 0.10
tx(s, ML, y, CW, 0.18,
   "red = collapsed model", 8.5, FAINT, check=False)

y += 0.34
y = body(s, "Leaving depthwise in FP32 (E) works; leaving squeeze-and-excitation in "
            "FP32 (F) does not. The ablation is at family level, on EfficientNet-B0 "
            "only — depthwise sensitivity is already known from Sheng et al.; our "
            "contribution is the case study and the runtime behaviour.",
         y, 13)

# =====================================================================
# 09 — PI DYNAMIC
# =====================================================================
s = prs.slides.add_slide(BLANK)
head(s, "What the Pi actually does", 9)
y = title(s, "On the Pi: dynamic INT8 is slower than FP32")
y = lead(s, "The slowdown belongs to dynamic quantization on this runtime.", y + 0.10)

y += 0.38
y = table(s, ML, y,
          [("Model", 3.0), ("FP32, t = 4", 2.3), ("INT8, t = 4", 2.3),
           ("Scaling 1→4", 2.2), ("Slowdown", 2.09)],
          [["MobileNetV2", "14.23 ms", "39.83 ms", "1.10×", "2.80×"],
           ["ShuffleNetV2", "6.19 ms", "11.95 ms", "1.14×", "1.93×"],
           ["EfficientNet-B0", "31.56 ms", "64.39 ms", "1.19×", "2.04×"]],
          size=12, row_h=0.36)

y += 0.18
y = body(s, "FP32 scales 2.20–2.33× going from 1 to 4 threads; INT8 manages only "
            "1.10–1.19×. Each convolution expands into DynamicQuantizeLinear, "
            "ConvInteger, a multiply and a cast — the quantization kernels add "
            "20.7–29.2% overhead.",
         y, 12.5)

aside(s, "Raspberry Pi 5, ONNX Runtime 1.29.0, CPU only, batch size 1, model-only "
         "timing, 200 runs after 10 warm-up iterations, governor pinned to performance.",
      6.72, 9)

# =====================================================================
# 10 — STATIC FLIPS IT
# =====================================================================
s = prs.slides.add_slide(BLANK)
head(s, "What the Pi actually does", 10)
y = title(s, "Static flips it: 2.34× faster than FP32")
y = lead(s, "EfficientNet-B0 on the Raspberry Pi 5, four threads, batch size 1.",
         y + 0.10)

# bar chart
bx0 = ML + 0.55
bw_max = 6.1
base = 5.42
scale = 3.28 / 63.99  # ms -> inch

bars = [("FP32", 31.53, "98.50%", GREY),
        ("dynamic INT8", 63.99, "10.42%", RUST),
        ("static per-tensor", 13.63, "9.32%", RUST),
        ("static per-channel", 13.50, "94.56%", OLIVE)]

rule(s, bx0 - 0.12, base, bw_max + 0.30, DARK, 1.0)

slot = bw_max / 4
for i, (label, ms, acc, col) in enumerate(bars):
    bh = ms * scale
    bx = bx0 + slot * i + slot * 0.16
    block(s, bx, base - bh, slot * 0.62, bh, col)
    tx(s, bx - 0.22, base - bh - 0.26, slot * 0.62 + 0.44, 0.20, f"{ms:.2f} ms",
       9.5, DARK, bold=True, align=PP_ALIGN.CENTER, check=False)
    tx(s, bx - 0.30, base + 0.12, slot * 0.62 + 0.60, 0.34, label, 8.5, GREY,
       align=PP_ALIGN.CENTER, line_mult=1.05, name="barlabel")
    tx(s, bx - 0.30, base + 0.54, slot * 0.62 + 0.60, 0.20, acc, 9, col,
       bold=True, align=PP_ALIGN.CENTER, check=False)

# panel kanan
px = ML + 7.35
pw = W - MR - px
tx(s, px, 2.30, pw, 0.20, "WHAT THIS MEANS", 8.5, OLIVE, bold=True, check=False)
tx(s, px, 2.60, pw, 1.10,
   "Static per-channel is 4.7× faster than dynamic INT8 and 2.34× faster than "
   "FP32 — at 94.56% instead of 10.42%.", 11, DARK, line_mult=1.12,
   name="staticside")

rule(s, px, 3.92, pw, LINE, 0.5)
tx(s, px, 4.10, pw, 1.40,
   "One board, one runtime, one seed. FP32 here is a separate session "
   "(31.53 ms) and its accuracy is measured on the exported ONNX graph "
   "(98.50%); the earlier sweep and the PyTorch evaluation give 31.56 ms "
   "and 98.55%.", 9, GREY, line_mult=1.16, name="staticcaveat")

# =====================================================================
# 11 — WHICH TO SHIP
# =====================================================================
s = prs.slides.add_slide(BLANK)
head(s, "What the Pi actually does", 11)
y = title(s, "So which model do you ship?")
y = lead(s, "INT8 here is a diagnostic result, not a shipping recommendation.",
         y + 0.10)

y += 0.52
cw3 = (CW - 0.70) / 3
card(s, ML, y, cw3, 1.86, "ship this", "ShuffleNetV2 FP32",
     "6.19 ms · 97.69% — fastest and accurate enough", accent=True)
card(s, ML + cw3 + 0.35, y, cw3, 1.86, "most accurate", "EfficientNet-B0 FP32",
     "31.56 ms · 98.55% — five times slower")
card(s, ML + 2 * (cw3 + 0.35), y, cw3, 1.86, "dominated", "EB0 static per-channel",
     "13.50 ms · 94.56% — loses on both axes")

y += 2.26
y = body(s, "EfficientNet-B0 static per-channel loses to ShuffleNetV2 FP32 on both "
            "speed and accuracy, and to MobileNetV2 FP32 (97.92%) on accuracy at "
            "similar latency. FP32 accuracy spans only 0.86 pp across the three "
            "models, so hardware and quantization scheme should drive the decision, "
            "not accuracy alone.", y, 13.5)

y += 0.22
aside(s, "INT8 shrinks the model 3.3–3.7×, but RAM at inference barely moves. "
         "Limitations: single Coccidiosis-only calibration slice, one seed, "
         "energy not measured.", y, 9.5)

# =====================================================================
# 12 — TAKEAWAY
# =====================================================================
s = prs.slides.add_slide(BLANK)
head(s, "Close", 12)
y = title(s, "Takeaway and artifact")
y = lead(s, "INT8 is not one thing: scheme × granularity × runtime decides.",
         y + 0.10)

y += 0.46
items = [
    ("Dynamic INT8", "preserves MobileNetV2 and ShuffleNetV2, collapses "
                     "EfficientNet-B0 to 10.42%"),
    ("Per-channel static", "recovers EfficientNet-B0 to 94.56% and runs 2.34× "
                           "faster than FP32 — but still loses to ShuffleNetV2 FP32"),
    ("Depthwise granularity", "decides collapse in this setup: 93.46% versus 8.33%"),
]
for kicker, text in items:
    tx(s, ML, y, 3.05, 0.20, kicker.upper(), 8.5, OLIVE, bold=True, check=False)
    hh = measure(text, 11.5, CW - 3.35, line_mult=1.12) + 0.04
    tx(s, ML + 3.25, y - 0.02, CW - 3.35, hh, text, 11.5, DARK,
       line_mult=1.12, name="takeaway")
    y += max(hh, 0.30) + 0.24
    rule(s, ML, y - 0.12, CW, LINE, 0.5)

y += 0.18
tx(s, ML, y, CW, 0.20, "ARTIFACT", 8.5, OLIVE, bold=True, check=False)
tx(s, ML, y + 0.24, CW, 0.22,
   "github.com/jaweed3/paper_poultry_p2   ·   Zenodo 10.5281/zenodo.23374216",
   11, DARK, check=False)

block(s, 0, 6.66, W, 0.84, OLIVE)
tx(s, ML, 6.90, CW, 0.24, "Thank you — questions welcome. The failure modes are in the paper.",
   11, CREAM, bold=True, check=False)

# =====================================================================
out = "/Users/wedjaw/paper_poultry_p2/deck-icaides/slides-icaides.pptx"
prs.save(out)

print(f"tersimpan: {out}")
print(f"slide: {len(prs.slides)}")
if PROBLEMS:
    print("\nMASALAH LAYOUT:")
    for p in PROBLEMS:
        print("  " + p)
else:
    print("\nsemua teks muat di kotaknya")
