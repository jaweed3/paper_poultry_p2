# Results — Paper 2

## Blocking issues

- ShuffleNetV2 DepGraph 1.6.1 bug: ZeroDivisionError in update_reshape_index_mapping for any pruning_ratio >0.
  Current workaround: fallback no-op (SNV2 runs at 0% sparsity until fix). Track: need torch-pruning upgrade or per-layer ignore for ChannelShuffle/Reshape.
  Impact: hypothesis still testable via MN2 + EB0 (2/3 models). SNV2 baseline 1.37x is already lowest expansion.

## Dry-run artifacts (2026-08-21 morning, synthetic, reproducible via config.yaml seed 42)

- dryrun_mobilenetv2_s30.json: 209->526 = 2.52x (Paper1 2.86x) — DOWN
- dryrun_mobilenetv2_s50.json: 2.52x, 3.87ms, 0.69MB INT8
- dryrun_efficientnet_b0_s30.json: 298->789 = 2.65x (Paper1 3.05x) — DOWN
- dryrun_efficientnet_b0_s50.json: 2.65x, 7.72ms
- dryrun_shufflenetv2_s30.json: 1.35x (fallback no-op, baseline 1.37x) — neutral

All INT8 bench skipped on macOS (ConvInteger NOT_IMPLEMENTED) — expected, will run on RPi5.

## Next

- Tonight: full dataset + finetune 15ep lr 5e-5 for MN2/EB0 s30/s50 (real accuracy).
- SNV2: fix or keep as baseline-only for Paper 2 (still contributes to Pareto and per-class NCD analysis).
