# future-work/ — NOT part of the submitted paper

Early-stage artifacts kept for provenance, excluded from all paper numbers:

- `experiment_results.json`, `int8_real_accuracy*.json` — earlier pipeline runs on the
  non-sterile split (int8_acc == fp32_acc placeholder bug in experiment_results).
- `dryrun_*`, `prune_*`, `finetune_mobilenetv2_s30.json` — structured-pruning
  exploration (broken finetune: 24% val after 2-epoch run). Future work only.

The submitted paper uses only `results/` sterile-split artifacts:
`eval_gate0.json`, `bootstrap_ci.json`, `static_eb0.json`,
`rpi5_benchmark_gate0.json`, `phash_dedup.json`,
`sterile_split_manifest.json`, `splits_manifest.json`, `results.json`,
`profiling/level2_profiling_artifact.json`.
