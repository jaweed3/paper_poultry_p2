#!/bin/bash
set -e
cd /home/master_core_ti/paper_poultry_p2
mkdir -p logs results onnx_pruned
for s in 0.2 0.4 0.6; do
    echo "=== [$(date)] Starting MobileNetV2 sparsity $s ===" >> logs/sweep_progress.log
    .venv/bin/python3 scripts/train_pruned.py --model mobilenetv2 --sparsity $s --epochs 12 > logs/train_mobilenetv2_s${s}.log 2>&1
    echo "=== [$(date)] Evaluating MobileNetV2 sparsity $s ===" >> logs/sweep_progress.log
    .venv/bin/python3 scripts/eval_pruned.py --model mobilenetv2 --sparsity $s > logs/eval_mobilenetv2_s${s}.log 2>&1
    echo "=== [$(date)] Finished MobileNetV2 sparsity $s ===" >> logs/sweep_progress.log
done
echo "ALL_SWEEP_DONE" > logs/sweep_done.txt
