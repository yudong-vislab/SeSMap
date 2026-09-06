#!/usr/bin/env python3
"""Evaluate Co-loc@12 on the audit-verified subset of held-out pairs.

Two independent worries about the headline Co-loc number:
  1. leakage      - test pairs share MSUs with training pairs (see run_disjoint.py)
  2. circularity  - both train and test positives come from the same LLM
                    similarity heuristic, so the metric may only confirm that the
                    model learned that heuristic.
This script attacks (2): it restricts evaluation to pairs the correspondence
audit judged 'accepted' in BOTH rounds.  Scored on the published layouts.
"""
import json, collections
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors

BK = Path('/Users/yudong/Desktop/SeSMap/SeSMap-backend')
HERE = Path(__file__).parent
K = 12
MDL = BK / 'data/merged13/stages/05_model'


def hits(Z, pairs, k=K):
    ng = [set(r) for r in NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)]
    return np.array([1.0 if (b in ng[a] or a in ng[b]) else 0.0 for a, b in pairs])


def wilson(p, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def audit_labels():
    """merged13 pair -> consensus audit label (corpora A+B only)."""
    rep = json.load(open(HERE / 'audit_replay.json'))
    assert rep['_verified'], 'audit replay did not verify'
    raw = json.load(open(BK / 'results/audit_multi_raw.json'))
    fwd = {(k.split(':')[0], int(k.split(':')[1])): v for k, v in rep['_map'].items()}
    lab = {}
    for t in ['A', 'B']:
        r1, r2 = raw[f'{t}_pair_r1'], raw[f'{t}_pair_r2']
        for i, (a, b, c) in enumerate(rep[t]['pairs']):
            k = str(i)
            if k not in r1 or k not in r2:
                continue
            if (t, a) not in fwd or (t, b) not in fwd:
                continue
            l1, l2 = str(r1[k].get('label', 'uncertain')), str(r2[k].get('label', 'uncertain'))
            lab[tuple(sorted((fwd[(t, a)], fwd[(t, b)])))] = l1 if l1 == l2 else 'uncertain'
    return lab


def main():
    lab = audit_labels()
    print('audit labels resolved onto merged13 pairs (A+B):',
          dict(collections.Counter(lab.values())))

    m13 = json.load(open(BK / 'data/merged13/stages/02_msu/llm_pairs.json'))
    held13 = {tuple(sorted((int(a), int(b)))) for a, b, *_ in m13['test_pos']}
    train13 = {tuple(sorted((int(a), int(b)))) for a, b, *_ in m13['train_pos']}
    Mtr = {m for ab in train13 for m in ab}

    acc = {p for p, l in lab.items() if l == 'accepted'}
    (HERE / 'audit_accepted_pairs.json').write_text(json.dumps(sorted(map(list, acc))))

    subsets = {
        'held_all': sorted(held13),
        'held_msu_unseen': [p for p in sorted(held13) if p[0] not in Mtr and p[1] not in Mtr],
        'held_audit_all': sorted(held13 & set(lab)),
        'held_audit_accepted': sorted(held13 & acc),
        'audit_all': sorted(lab),
        'audit_accepted': sorted(acc),
    }

    Zs = {f.stem[2:]: np.load(f) for f in sorted(MDL.glob('Z_*.npy'))}
    out = {}
    for sub, pairs in subsets.items():
        if not pairs:
            continue
        out[sub] = {'n': len(pairs)}
        print(f'\n--- {sub}  (n={len(pairs)}) ---')
        for name in ['Ours_lam0.3', 'Ours_lam0.0', 't-SNE', 'UMAP', 'PCA']:
            if name not in Zs:
                continue
            h = hits(Zs[name], pairs)
            lo, hi = wilson(float(h.mean()), len(pairs))
            out[sub][name] = {'coloc': float(h.mean()), 'ci': [lo, hi], 'n': len(pairs)}
            print(f'  {name:14} Co-loc@12 = {h.mean():.3f}  95%CI [{lo:.3f}, {hi:.3f}]')
        np.save(HERE / f'subset_{sub}.npy', np.array(pairs))

    (HERE / 'audit_subset_results.json').write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {HERE/'audit_subset_results.json'}")


if __name__ == '__main__':
    raise SystemExit(main())
