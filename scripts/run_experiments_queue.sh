#!/usr/bin/env bash
# scripts/run_experiments_queue.sh
# Robust training queue runner for PC Putih (RTX 4060)
# Runs inside tmux to ensure continuous execution without SSH disconnection risks.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Virtual environment python
if [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
    PY="${ROOT_DIR}/.venv/bin/python"
else
    PY="python3"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p logs results models/pruned onnx_pruned

echo "=========================================================="
echo " Starting Robust Experiment Queue on $(hostname)"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"
echo " Time: $(date)"
echo "=========================================================="

# -------------------------------------------------------------
# EXPERIMENT 1: MobileNetV2 (50% Sparsity / s50)
# -------------------------------------------------------------
echo ""
echo ">>> [1/2] Starting MobileNetV2 (Sparsity: 0.5, Epochs: 12)..."
echo "Log: logs/train_mobilenetv2_s50.log"
${PY} scripts/train_pruned.py --model mobilenetv2 --sparsity 0.5 --epochs 12 --no-wandb 2>&1 | tee logs/train_mobilenetv2_s50.log

echo ""
echo ">>> [1/2] Evaluating MobileNetV2 s50 on 1728 sterile test images..."
${PY} scripts/eval_pruned.py --model mobilenetv2 --sparsity 0.5 2>&1 | tee -a logs/train_mobilenetv2_s50.log

echo ""
echo ">>> [1/2] Completed MobileNetV2 s50! Flushing GPU cache..."
${PY} -c "import torch, gc; gc.collect(); torch.cuda.empty_cache(); print('VRAM clean')"
sleep 5

# -------------------------------------------------------------
# EXPERIMENT 2: EfficientNet-B0 (30% Sparsity / s30)
# -------------------------------------------------------------
echo ""
echo ">>> [2/2] Starting EfficientNet-B0 (Sparsity: 0.3, Epochs: 12)..."
echo "Log: logs/train_efficientnet_b0_s30.log"
${PY} scripts/train_pruned.py --model efficientnet_b0 --sparsity 0.3 --epochs 12 --no-wandb 2>&1 | tee logs/train_efficientnet_b0_s30.log

echo ""
echo ">>> [2/2] Evaluating EfficientNet-B0 s30 (FP32, Dynamic INT8, & Static Per-Channel QDQ)..."
${PY} scripts/eval_pruned.py --model efficientnet_b0 --sparsity 0.3 --static 2>&1 | tee -a logs/train_efficientnet_b0_s30.log

echo ""
echo ">>> [2/2] Completed EfficientNet-B0 s30!"
${PY} -c "import torch, gc; gc.collect(); torch.cuda.empty_cache(); print('VRAM clean')"

echo "=========================================================="
echo " ALL EXPERIMENTS IN QUEUE COMPLETED SUCCESSFULLY!"
echo " Finished at: $(date)"
echo "=========================================================="
touch results/QUEUE_COMPLETED.flag
