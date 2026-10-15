#!/usr/bin/env python3
"""Gate 0: pHash dedup audit over materialized data/images/{train,test}/<cls>.

- Computes pHash (64-bit) for every image.
- Exact-duplicate groups (hamming == 0) and near-duplicate groups (hamming <= 5).
- Cross-split leakage check: does any test image duplicate a train image?
- Output: results/phash_dedup.json {stats, exact_groups, near_groups, cross_split_hits}

Run on LAB after materialize: uv run --with ImageHash --with pillow scripts/phash_dedup.py
Read-only on data/. Fast (~8770 images, seconds-minutes).
"""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "images"
OUT = ROOT / "results" / "phash_dedup.json"

NEAR_THRESHOLD = 5  # hamming distance: catches resize/recompress copies


def main():
    from PIL import Image
    import imagehash

    files = sorted(DATA_DIR.rglob("*.jpg"))
    print(f"scanning {len(files)} images in {DATA_DIR}")
    assert len(files) == 8770, f"expected 8770, got {len(files)} — run materialize first"

    hashes = {}  # path -> int phash
    for p in files:
        try:
            with Image.open(p) as im:
                h = imagehash.phash(im)
            hashes[str(p.relative_to(ROOT))] = int(str(h), 16)
        except Exception as e:
            print(f"FAIL {p}: {e}")
    print(f"hashed {len(hashes)} images")

    # exact groups
    by_hash = defaultdict(list)
    for path, h in hashes.items():
        by_hash[h].append(path)
    exact_groups = [sorted(v) for v in by_hash.values() if len(v) > 1]

    # near groups (hamming <= NEAR_THRESHOLD), union-find
    paths = list(hashes)
    parent = {p: p for p in paths}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    n = len(paths)
    hs = [hashes[p] for p in paths]
    for i in range(n):
        hi = hs[i]
        for j in range(i + 1, n):
            if bin(hi ^ hs[j]).count("1") <= NEAR_THRESHOLD:
                union(paths[i], paths[j])
        if i % 2000 == 0:
            print(f"  compared {i}/{n}")
    clusters = defaultdict(list)
    for p in paths:
        clusters[find(p)].append(p)
    near_groups = [sorted(v) for v in clusters.values() if len(v) > 1]

    def split_of(p):
        return p.split("/")[2]  # data/images/<split>/<cls>/file

    cross_split = [g for g in near_groups
                   if len({split_of(p) for p in g}) > 1]

    n_dup_extra = sum(len(g) - 1 for g in near_groups)
    stats = {
        "total": len(hashes),
        "unique_exact": len(by_hash),
        "exact_dup_groups": len(exact_groups),
        "exact_dup_extra_copies": sum(len(g) - 1 for g in exact_groups),
        "near_dup_groups": len(near_groups),
        "near_dup_extra_copies": n_dup_extra,
        "near_threshold": NEAR_THRESHOLD,
        "cross_split_groups": len(cross_split),
        "est_unique_images": len(hashes) - n_dup_extra,
    }
    print(json.dumps(stats, indent=2))

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"stats": stats,
                   "exact_groups": exact_groups,
                   "near_groups": near_groups,
                   "cross_split_groups": cross_split}, f, indent=2)
    print(f"saved -> {OUT}")

    # Gate verdict
    if stats["cross_split_groups"] > 0:
        print("GATE-0 VERDICT: LEAKAGE CONFIRMED — test contains near-copies of train.")
        print("All accuracy narratives must be rewritten; rebuild split from originals.")
    elif stats["near_dup_extra_copies"] > 0:
        print("GATE-0 VERDICT: duplicates exist but contained within splits.")
        print("Rebuild split grouped by origin image anyway.")
    else:
        print("GATE-0 VERDICT: CLEAN — 8770 unique, publisher claim holds.")


if __name__ == "__main__":
    main()
