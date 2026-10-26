# ICAIDES 2026 Deck — INT8 Collapse in EfficientNet-B0

Paper ID 1571359798. Presenter: Fatih. Video: max 10 min, MP4 720p 16:9.

## Files

- `slides.tex` — deck source (11 content frames + dividers, midcentury-olive theme)
- `slides-icaides-2026.pdf` — compiled deck, ready to record against
- `VIDEO_SCRIPT_QA.md` — ~8-min narration script + predicted Q&A + recording checklist
- `qa/contact-sheet.jpg` — 15-slide thumbnail overview

## Build

Needs the bundled fonts (NOT in git — 18 MB):

```bash
# 1. Drop fonts/ here: fonts/montserrat/*.otf + fonts/EBGaramond-0.016/otf/*.otf
#    (ask Fatih, or re-extract midcentury_olive__1_.zip)
# 2. Build twice (XeLaTeX overlay positioning needs two passes):
tectonic -X compile slides.tex --keep-intermediates --outdir _build
```

The committed PDF was built with all fonts present. Rebuilding without
`fonts/` fails loudly by design (template refuses silent font substitution).

## Before recording

1. Replace `figures/logo-trim.png` (ugly "YOUR LOGO HERE" placeholder) with the
   UNIDA/Gontor logo, then re-measure `titlegraphic height` + `date makebox`
   in `slides.tex` (see template README logic).
2. Numbers in deck are synced to `../paper/main.tex` + `../results/*.json`
   as of 26 Oct 2026. If the paper changes, re-sync.

## Theme attribution (required, CC BY 4.0)

Theme "midcentury modern" by Jules Leguy
(https://github.com/jules-leguy/midcenturymodern), used under CC BY 4.0.
Changes vs upstream are listed in `tdstyle.tex`. Keep that header if you copy
the `.sty` anywhere.
