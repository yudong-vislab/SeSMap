#!/usr/bin/env python3
"""Build MSU-disjoint train/test splits of the merged13 positive pairs.

Reviewer concern: the released split shares MSUs between train_pos and test_pos
(74.8% of test MSUs appear in train), so Co-loc@12 on test_pos is partly a
memorisation check.  Here we partition the *MSU nodes* (stratified by paper),
then keep a pair only if BOTH endpoints fall on the same side.  Pairs crossing
the partition are dropped, so no MSU is ever seen on both sides.
"""
import json, collections, argparse
from pathlib import Path
import numpy as np

BK = Path('/Users/yudong/Desktop/SeSMap/SeSMap-backend')
PAIR = BK / 'data/merged13/stages/02_msu/llm_pairs.json'
FDB = BK / 'data/merged13/stages/02_msu/formdatabase_v2.0.json'


def build(seed, test_frac=0.30):
    p = json.load(open(PAIR))
    pairs = [(int(a), int(b)) for a, b, *_ in p['train_pos'] + p['test_pos']]
    pid = [r['paper_id'] for r in json.load(open(FDB))]

    nodes = sorted({m for ab in pairs for m in ab})
    by_paper = collections.defaultdict(list)
    for m in nodes:
        by_paper[pid[m]].append(m)

    rng = np.random.default_rng(seed)
    test_nodes = set()
    for pp, ms in sorted(by_paper.items()):          # stratify by paper
        ms = np.array(sorted(ms))
        k = int(round(len(ms) * test_frac))
        test_nodes.update(rng.choice(ms, size=k, replace=False).tolist())

    tr = [(a, b) for a, b in pairs if a not in test_nodes and b not in test_nodes]
    te = [(a, b) for a, b in pairs if a in test_nodes and b in test_nodes]
    dropped = len(pairs) - len(tr) - len(te)

    Mtr = {m for ab in tr for m in ab}
    Mte = {m for ab in te for m in ab}
    assert not (Mtr & Mte), 'split is not MSU-disjoint'
    return dict(seed=seed, test_frac=test_frac, train=tr, test=te,
                n_train=len(tr), n_test=len(te), n_dropped=dropped,
                msu_train=len(Mtr), msu_test=len(Mte))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--test-frac', type=float, default=0.30)
    ap.add_argument('--out', type=Path, default=Path(__file__).parent / 'splits_disjoint.json')
    a = ap.parse_args()

    out = {}
    for s in a.seeds:
        d = build(s, a.test_frac)
        out[str(s)] = d
        print(f"seed {s}: train {d['n_train']:5d} pairs / {d['msu_train']:5d} MSUs | "
              f"test {d['n_test']:4d} pairs / {d['msu_test']:4d} MSUs | dropped {d['n_dropped']:5d}")
    a.out.write_text(json.dumps(out))
    print(f'\nsaved -> {a.out}')
