import random

def video_disjoint_split(records, seed=0, fractions=(0.8, 0.1, 0.1)):
    groups = {}
    for r in records: groups.setdefault(r["video_path"], []).append(r)
    vids = list(groups); random.Random(seed).shuffle(vids)
    n = len(vids); a, b = int(n*fractions[0]), int(n*(fractions[0]+fractions[1]))
    return {"train": [r for v in vids[:a] for r in groups[v]], "val": [r for v in vids[a:b] for r in groups[v]], "test": [r for v in vids[b:] for r in groups[v]]}

def assert_disjoint(splits):
    sets = {k: {r["video_path"] for r in v} for k,v in splits.items()}
    for a, x in sets.items():
        for b, y in sets.items():
            if a < b and x & y: raise AssertionError(f"video overlap {a}/{b}: {x&y}")
