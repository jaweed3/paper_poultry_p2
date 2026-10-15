# Sample dataset images

Eight small sample images are included for teaching, demonstration, and qualitative inspection only: two images per class from the `test` split of the deduplicated benchmark source.

- `images/cocci_*.jpg` — Coccidiosis
- `images/healthy_*.jpg` — Healthy
- `images/ncd_*.jpg` — Newcastle Disease (NCD)
- `images/salmo_*.jpg` — Salmonella

## Provenance

Source dataset: [Dianyo/poultry-fecal-fl](https://huggingface.co/datasets/Dianyo/poultry-fecal-fl), which documents the leak-free benchmark derived from the original [Machuve et al. dataset on Zenodo](https://doi.org/10.5281/zenodo.4628934). The source dataset is distributed under CC BY 4.0; attribution is retained here for demonstration copies.

The exact source asset URLs and labels are recorded in [`manifest.csv`](manifest.csv). These images are **not** the complete dataset and must not be used to reproduce the reported metrics. Use the Zenodo/Hugging Face source and the repository's split artifacts for experiments.
