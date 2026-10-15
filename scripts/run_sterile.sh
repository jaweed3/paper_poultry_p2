#!/usr/bin/env bash
# Runner untuk script yang butuh dataset STERILE (1728 test) di WSL lab.
#
# Kenapa wrapper ini perlu:
#   train_pipeline.py baca POULTRY_DATA_DIR dari env, defaultnya "data/images"
#   (= publisher split, 1754 test). Semua ANGKA DI PAPER kita pakai
#   "data/images_sterile" (1728 test: 529/504/141/554 per kelas). Env HARUS
#   di-set eksplisit, kalau tidak script assert len(test)==1728 gagal.
#
#   run: bash scripts/run_sterile.sh scripts/static_eb0.py
#        bash scripts/run_sterile.sh scripts/eval_gate0.py
set -euo pipefail

cd "$(dirname "$0")/.."

export POULTRY_DATA_DIR="data/images_sterile"

# LD_PRELOAD untuk nvJitLink (torch 2.5.1 + CUDA 12.1 di WSL2).
# Tanpa ini: "libnvJitLink.so.12: cannot open shared object file".
# Path-nya di dalam venv (site-packages/nvidia/nvjitlink/lib), BUKAN
# /usr/lib/wsl/lib — sudah diverifikasi di lab 8 Okt 2026.
_NVJIT="$(ls .venv/lib/python3*/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12 2>/dev/null | head -1 || true)"
if [ -n "${_NVJIT}" ]; then
  export LD_PRELOAD="${_NVJIT}"
fi

if [ ! -d "${POULTRY_DATA_DIR}" ]; then
  echo "[ERROR] ${POULTRY_DATA_DIR} tidak ada. Jalankan scripts/phash_dedup.py dulu." >&2
  exit 1
fi

echo "POULTRY_DATA_DIR=${POULTRY_DATA_DIR}"
echo "LD_PRELOAD=${LD_PRELOAD:-<unset>}"
echo "cwd=$(pwd)"
echo "---"

if [ "$#" -lt 1 ]; then
  echo "usage: bash scripts/run_sterile.sh <script.py> [args...]" >&2
  exit 1
fi

exec .venv/bin/python "$@"