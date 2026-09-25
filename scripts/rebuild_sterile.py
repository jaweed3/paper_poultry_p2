#!/usr/bin/env python3
"""Gate 0b: rebuild sterile split from pHash dedup verdict.

Strategy (locked):
- Drop 1 per near-duplicate pair (617 pairs, all size 2, class-consistent).
- Priority: keep test > val > train. Within-split: keep lexicographically-first.
- Source: data/images/ (8770, untouched). Output: data/images_sterile/ (8153).
- Manifest: results/sterile_split_manifest.json (every dropped file + reason).
- Verifies: no cross-split pair survives; counts 5088/1337/1728.

Run on LAB: uv run scripts/rebuild_sterile.py
"""
import json
import shutil
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "images"
DST = ROOT / "data" / "images_sterile"
DEDUP = ROOT / "results" / "phash_dedup.json"
OUT_MANIFEST = ROOT / "results" / "sterile_split_manifest.json"

PRIORITY = {"test": 0, "val": 1, "train": 2}


def split_of(p):
    return p.split("/")[2]


def main():
    d = json.load(open(DEDUP))
    pairs = d["near_groups"]
    assert all(len(g) == 2 for g in pairs), "expected all pairs"
    print(f"pairs: {len(pairs)}")

    drop = {}  # dropped_path -> reason
    for a, b in pairs:
        sa, sb = split_of(a), split_of(b)
        if sa != sb:
            # keep higher priority (lower number); drop the other
            loser = a if PRIORITY[sa] > PRIORITY[sb] else b
            winner = b if loser == a else a
            drop[loser] = f"cross-split dup of {winner}"
        else:
            loser = max(a, b)  # lexicographically-last dropped (deterministic)
            drop[loser] = f"within-{sa} dup of {min(a, b)}"

    print(f"dropping {len(drop)} files")
    assert len(drop) == len(pairs), "one drop per pair"

    if DST.exists():
        print(f"removing old {DST}")
        shutil.rmtree(DST)
    copied, dropped = 0, 0
    for p in sorted(SRC.rglob("*.jpg")):
        rel = str(p.relative_to(SRC))
        rel_root = str(Path(rel).as_posix())
        if rel_root in drop:
            dropped += 1
            continue
        dest = DST / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        copied += 1
    print(f"copied={copied} dropped={dropped}")
    assert copied + dropped == 8770, (copied, dropped)

    # verify counts
    c = Counter()
    for p in DST.rglob("*.jpg"):
        c[p.parent.parent.name] += 1
    print(dict(c))
    assert c["train"] == 5088, c
    assert c["val"] == 1337, c
    assert c["test"] == 1728, c

    # verify zero cross-split survivors: re-run exact-hash check on DST
    import hashlib
    by_md5 = {}
    cross = 0
    for p in DST.rglob("*.jpg"):
        h = hashlib.md5(p.read_bytes()).hexdigest()
        s = p.parent.parent.name
        if h in by_md5 and by_md5[h] != s:
            cross += 1
            print("SURVIVOR:", p, by_md5[h])
        by_md5.setdefault(h, s)
    print(f"exact cross-split survivors in sterile: {cross}")
    assert cross == 0, "sterile split still leaks!"

    manifest = {
        "strategy": "drop-1-per-pair, keep test>val>train, within keep lexicographically-first",
        "source_total": 8770,
        "sterile_total": copied,
        "counts": dict(c),
        "dropped": drop,
        "exact_cross_survivors": cross,
    }
    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"saved -> {OUT_MANIFEST}")
    print("STERILE OK — 8153, test steril dari train/val.")


if __name__ == "__main__":
    main()
