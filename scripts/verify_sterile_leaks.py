#!/usr/bin/env python3
"""
Verify whether data/images_sterile has any cross-split duplicate pairs (D_H <= 5).
"""
import imagehash
from PIL import Image
from pathlib import Path
from collections import defaultdict
import time

def main():
    root = Path("/home/master_core_ti/paper_poultry_p2/data/images_sterile")
    splits = ["train", "val", "test"]

    paths_by_split = defaultdict(list)
    for s in splits:
        for p in (root / s).rglob("*.jpg"):
            paths_by_split[s].append(p)

    counts = {s: len(v) for s, v in paths_by_split.items()}
    print(f"Splits count: {counts}, Total: {sum(counts.values())}")

    print("Computing pHash for all images in images_sterile...")
    t0 = time.time()
    hashes = {}
    for s in splits:
        for p in paths_by_split[s]:
            with Image.open(p) as im:
                h = imagehash.phash(im)
            hashes[p] = (s, p.parent.name, int(str(h), 16))
    print(f"Computed {len(hashes)} hashes in {time.time()-t0:.2f}s")

    # Check all pairs across splits
    all_paths = list(hashes.keys())
    n = len(all_paths)
    cross_leaks = 0
    cross_details = []

    print("Checking cross-split Hamming distances (<= 5)...")
    for i in range(n):
        p1 = all_paths[i]
        s1, c1, h1 = hashes[p1]
        for j in range(i + 1, n):
            p2 = all_paths[j]
            s2, c2, h2 = hashes[p2]
            if s1 != s2:
                dist = bin(h1 ^ h2).count("1")
                if dist <= 5:
                    cross_leaks += 1
                    cross_details.append((s1, s2, c1, c2, dist, str(p1), str(p2)))

    print(f"Total cross-split pairs with Hamming dist <= 5 in images_sterile: {cross_leaks}")
    if cross_leaks > 0:
        for leak in cross_details[:10]:
            print("  Leak:", leak)
    else:
        print("PERFECT: images_sterile is 100% free of cross-split duplicate leakage!")

if __name__ == "__main__":
    main()
