#!/usr/bin/env python3
"""Materialize HF Dianyo/poultry-fecal-fl parquet -> data/images/{train,test}/<cls>.

Preserves publisher splits. Label int -> class name resolved by matching
value-counts against splits_manifest.json expected totals (cocci 2676,
healthy 2546, ncd 720, salmo 2828). Fails loudly if counts don't match.

Run on LAB (not Mac): uv run --with datasets --with pyarrow --with pillow scripts/materialize_hf.py
"""
import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "images"
MANIFEST = ROOT / "results" / "splits_manifest.json"

HF_ID = "Dianyo/poultry-fecal-fl"
# Expected TOTAL per class (train+val+test) from splits_manifest (Farrel sync)
EXPECTED_TOTALS = {"cocci": 2676, "healthy": 2546, "ncd": 720, "salmo": 2828}
CLASS_NAMES = ["cocci", "healthy", "ncd", "salmo"]

IMG_EXT = ".jpg"  # HF images are JPEG; keep original bytes, just rename


def resolve_label_map(ds_train, ds_test):
    from collections import Counter
    counts = Counter(ds_train["label"]) + Counter(ds_test["label"])
    print("raw label value-counts:", dict(counts))
    # try ClassLabel names first
    try:
        names = ds_train.features["label"].names
        print("ClassLabel names:", names)
        if names and set(n.lower() for n in names) >= {"cocci", "healthy"}:
            norm = {"coccidiosis": "cocci", "salmonella": "salmo",
                    "newcastle": "ncd", "cocci": "cocci", "healthy": "healthy",
                    "salmo": "salmo", "ncd": "ncd"}
            return {i: norm[n.lower()] for i, n in enumerate(names)}
    except Exception as e:
        print("no ClassLabel names:", e)
    # fallback: match counts to expected totals
    inv = {v: k for k, v in EXPECTED_TOTALS.items()}
    mapping = {}
    for label_val, cnt in counts.items():
        # nearest expected total within 5%
        best = min(EXPECTED_TOTALS.items(), key=lambda kv: abs(kv[1] - cnt))
        if abs(best[1] - cnt) / best[1] > 0.05:
            print(f"FATAL: label {label_val} count {cnt} matches nothing "
                  f"(nearest {best}). Abort.")
            sys.exit(1)
        mapping[label_val] = best[0]
    print("resolved by count-matching:", mapping)
    return mapping


def dump_split(ds, split, label_map):
    out_counts = Counter()
    for i, row in enumerate(ds):
        cls = label_map[int(row["label"])]
        d = DATA_DIR / split / cls
        d.mkdir(parents=True, exist_ok=True)
        img = row["image"]
        # row["image"] is PIL image when datasets decodes; save as JPEG
        p = d / f"{split}_{i:05d}.jpg"
        if not p.exists():
            if hasattr(img, "save"):
                img.save(p, "JPEG", quality=95)
            else:
                with open(p, "wb") as f:
                    f.write(img["bytes"])
        out_counts[cls] += 1
    return out_counts


def main():
    from datasets import load_dataset
    print("loading", HF_ID)
    ds = load_dataset(HF_ID)
    print("splits:", {k: len(v) for k, v in ds.items()})
    assert len(ds["train"]) == 7016, ds["train"]
    assert len(ds["test"]) == 1754, ds["test"]

    if (DATA_DIR / "train" / "cocci").exists() and (DATA_DIR / "test" / "cocci").exists():
        n_train = sum(1 for _ in (DATA_DIR / "train").rglob("*.jpg"))
        n_test = sum(1 for _ in (DATA_DIR / "test").rglob("*.jpg"))
        print(f"already materialized: train={n_train} test={n_test}")
        if n_train == 7016 and n_test == 1754:
            print("counts match publisher. DONE, no re-download.")
            return
        print("counts MISMATCH, re-materializing missing files only (no overwrite).")

    label_map = resolve_label_map(ds["train"], ds["test"])
    c_train = dump_split(ds["train"], "train", label_map)
    c_test = dump_split(ds["test"], "test", label_map)
    print("train:", dict(c_train), "total:", sum(c_train.values()))
    print("test:", dict(c_test), "total:", sum(c_test.values()))

    # verify against manifest
    exp_train = {"cocci": 2141, "healthy": 2037, "ncd": 576, "salmo": 2262}
    exp_test = {"cocci": 535, "healthy": 509, "ncd": 144, "salmo": 566}
    ok = dict(c_train) == exp_train and dict(c_test) == exp_test
    print("match splits_manifest:", ok)
    if not ok:
        print("EXPECTED train:", exp_train)
        print("EXPECTED test:", exp_test)
        sys.exit(2)
    print("MATERIALIZE OK")


if __name__ == "__main__":
    main()
