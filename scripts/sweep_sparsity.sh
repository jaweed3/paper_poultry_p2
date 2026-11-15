#!/bin/bash
set -e
cd /home/master_core_ti/paper_poultry_p2
mkdir -p logs
for s in 0.2 0.4 0.6; do
    echo "=== Running MobileNetV2 sparsity $s ==="
    .venv/bin/python3 scripts/train_pruned.py --model mobilenetv2 --sparsity $s --epochs 12 > logs/train_mobilenetv2_s${s}.log 2>&1
    echo "=== Finished MobileNetV2 sparsity $s ==="
done
echo "SWEEP_COMPLETE" > logs/sweep_done.txt
