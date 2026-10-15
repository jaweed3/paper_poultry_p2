#!/usr/bin/env bash
# fetch_p2_assets.sh — bertahap, tmux/mosh-friendly, fail-loud.
#
# Usage:
#   ./scripts/fetch_p2_assets.sh deps     # Phase 1: pip light deps (onnxruntime==1.29.0, numpy, pillow, tqdm) — 2 sec, no torch
#   ./scripts/fetch_p2_assets.sh models   # Phase 2: ONNX models (30 MB total, no torch)
#   ./scripts/fetch_p2_assets.sh data     # Phase 3: sterile data 8153 jpgs (scp from Pi if reachable, else HF)
#   ./scripts/fetch_p2_assets.sh all      # all 3 in order
#   ./scripts/fetch_p2_assets.sh check    # just verify via check_p2_prereqs.py
#
# Di tmux: tmux new -s p2-fetch './scripts/fetch_p2_assets.sh all'
# Cek:     tmux capture-pane -t p2-fetch -p | tail -n 50
# Mosh:    mosh ghozy@100.78.190.70 -- tmux attach -t p2-fetch

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

log()  { echo -e "\033[36m[fetch]\033[0m $*"; }
ok()   { echo -e "\033[32m[OK]\033[0m $*"; }
warn() { echo -e "\033[33m[WARN]\033[0m $*"; }
die()  { echo -e "\033[31m[FAIL]\033[0m $*"; exit 1; }

PHASE="${1:-all}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

# ---- Phase 1: light deps (no torch) ----
phase_deps() {
  log "Phase 1: light deps (onnxruntime==1.29.0, numpy, pillow, tqdm) — no torch"
  # prefer project venv via uv, fallback to system pip
  if [ -f ".venv/bin/python" ]; then
    PY=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
  else
    die "no python found"
  fi

  # Install pinned light deps only — 2 sec, not 10 min (no torch/cuda)
  # Use --no-deps trick? No, just direct pip. Pin ORT exactly.
  log "pip install onnxruntime==1.29.0 numpy pillow tqdm"
  if command -v uv >/dev/null 2>&1 && [ -f ".venv/bin/python" ]; then
    uv pip install --python "$PY" "onnxruntime==1.29.0" "numpy" "pillow" "tqdm" 2>&1 | tail -n 20
  else
    # fallback: pip (may need --break-system-packages on Ubuntu)
    if "$PY" -m pip install "onnxruntime==1.29.0" "numpy" "pillow" "tqdm" 2>&1 | tail -n 30; then
      :
    elif "$PY" -m pip install --break-system-packages "onnxruntime==1.29.0" "numpy" "pillow" "tqdm" 2>&1 | tail -n 30; then
      :
    else
      die "pip install failed — try: pip install --break-system-packages onnxruntime==1.29.0"
    fi
  fi

  # verify pinned version
  "$PY" -c "import onnxruntime; assert onnxruntime.__version__=='1.29.0', onnxruntime.__version__; print(f'onnxruntime {onnxruntime.__version__} OK')" \
    || die "onnxruntime version mismatch — need 1.29.0"
  "$PY" -c "import numpy, PIL; print(f'numpy {numpy.__version__} pillow {PIL.__version__} OK')"
  ok "Phase 1 done"
}

# ---- Phase 2: models (ONNX) ----
phase_models() {
  log "Phase 2: ONNX models (FP32 + INT8 dynamic, ~30 MB)"
  need_cmd scp
  # try download_models.sh if exists, else scp from Pi
  if [ -x "./download_models.sh" ]; then
    log "Running ./download_models.sh --archive onnx (if supported)"
    ./download_models.sh --archive onnx 2>&1 | tail -n 50 || warn "download_models.sh failed, trying scp fallback"
  fi
  # Verify, fallback to scp from Pi if missing
  MISSING=0
  for f in models/onnx/mobilenetv2_gate0_fp32.onnx models/onnx/shufflenetv2_gate0_fp32.onnx models/onnx/efficientnet_b0_gate0_fp32.onnx; do
    if [ ! -f "$f" ]; then
      log "Missing $f — trying scp from Pi"
      # try tailscale first, then local IP
      if scp -o ConnectTimeout=6 techronity@100.74.103.104:~/p2bench/models/onnx/$(basename "$f") "$f" 2>&1 | tail -n 5; then
        ok "fetched $f via tailscale"
      elif scp -o ConnectTimeout=6 techronity@192.168.119.43:~/p2bench/models/onnx/$(basename "$f") "$f" 2>&1 | tail -n 5; then
        ok "fetched $f via 192.168.119.43"
      else
        warn "cannot fetch $f from Pi — Pi offline? will retry later"
        MISSING=1
      fi
    else
      ok "$f exists ($(du -h "$f" | cut -f1))"
    fi
  done
  for f in models/ptq/mobilenetv2_gate0_int8.onnx models/ptq/shufflenetv2_gate0_int8.onnx models/ptq/efficientnet_b0_gate0_int8.onnx; do
    if [ ! -f "$f" ]; then
      log "Missing $f — trying scp from Pi"
      if scp -o ConnectTimeout=6 techronity@100.74.103.104:~/p2bench/models/ptq/$(basename "$f") "$f" 2>&1 | tail -n 5; then
        ok "fetched $f via tailscale"
      elif scp -o ConnectTimeout=6 techronity@192.168.119.43:~/p2bench/models/ptq/$(basename "$f") "$f" 2>&1 | tail -n 5; then
        ok "fetched $f via 192.168.119.43"
      else
        warn "cannot fetch $f from Pi"
        MISSING=1
      fi
    else
      ok "$f exists ($(du -h "$f" | cut -f1))"
    fi
  done
  if [ "$MISSING" -eq 1 ]; then
    warn "Some models still missing — re-run after Pi is reachable, or: ./download_models.sh"
  else
    ok "Phase 2 done"
  fi
}

# ---- Phase 3: sterile data 8153 ----
phase_data() {
  log "Phase 3: sterile data 8153 (5088/1337/1728)"
  if [ -d "data/images_sterile/test/cocci" ] && [ "$(find data/images_sterile/test -name '*.jpg' 2>/dev/null | wc -l)" -eq 1728 ]; then
    ok "data/images_sterile already complete (1728 test)"
    return 0
  fi
  # Try scp from Pi data if Pi has it, else HF materialize (heavy)
  log "Checking Pi for data/images_sterile via scp"
  # Pi data location is not in p2bench — check common spots
  for host in "techronity@100.74.103.104" "techronity@192.168.119.43"; do
    if ssh -o ConnectTimeout=4 "$host" "ls ~/p2bench/data/images_sterile/test/cocci 2>&1 | head -n 2" 2>&1 | grep -q "jpg\|cocci"; then
      log "Found Pi data, rsync sterile split (this copies 8153 jpgs, ~1-2 GB, use tmux)"
      mkdir -p data
      # use rsync if available, else scp recursive
      if command -v rsync >/dev/null 2>&1; then
        rsync -avz --progress "$host:~/p2bench/data/images_sterile/" data/images_sterile/ 2>&1 | tail -n 30 && ok "rsync done" && return 0
      else
        scp -r -o ConnectTimeout=10 "$host:~/p2bench/data/images_sterile" data/ 2>&1 | tail -n 20 && ok "scp done" && return 0
      fi
    fi
  done
  warn "Pi data not reachable or missing — fallback: materialize from HF (downloads 8770 jpgs, ~1-2 GB)"
  log "Run: uv run --with datasets --with pyarrow --with pillow scripts/materialize_hf.py"
  log "Then: POULTRY_DATA_DIR=data/images uv run scripts/rebuild_sterile.py"
  if command -v uv >/dev/null 2>&1; then
    uv run --with datasets --with pyarrow --with pillow scripts/materialize_hf.py 2>&1 | tail -n 30
    POULTRY_DATA_DIR=data/images uv run scripts/rebuild_sterile.py 2>&1 | tail -n 30
  else
    die "uv not found — install uv, or scp sterile split from Pi/Mac"
  fi
  if [ "$(find data/images_sterile/test -name '*.jpg' 2>/dev/null | wc -l)" -eq 1728 ]; then
    ok "Phase 3 done"
  else
    die "data still incomplete after fetch — check logs"
  fi
}

case "$PHASE" in
  deps)   phase_deps ;;
  models) phase_models ;;
  data)   phase_data ;;
  check)  python scripts/check_p2_prereqs.py ;;
  all)
    phase_deps
    echo
    phase_models
    echo
    phase_data
    echo
    python scripts/check_p2_prereqs.py
    ;;
  *) die "unknown phase: $PHASE (use: deps|models|data|all|check)" ;;
esac
