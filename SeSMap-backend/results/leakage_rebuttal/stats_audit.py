#!/usr/bin/env python3
"""Paired bootstrap on the published layouts: does the cross-paper supervision
gain (lam_con 0.3 vs 0.0) survive on subsets that are audit-verified and/or free
of MSU leakage?  Same pairs under both layouts, so we resample pairs and take
the paired difference."""
import json
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors

BK = Path('/Users/yudong/Desktop/SeSMap/SeSMap-backend')
HERE = Path(__file__).parent
MDL = BK / 'data/merged13/stages/05_model'
K, R = 12, 10000


def hits(Z, pairs, k=K):
    ng = [set(r) for r in NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)]
    return np.array([1.0 if (b in ng[a] or a in ng[b]) else 0.0 for a, b in pairs])


def main():
    Z3, Z0 = np.load(MDL / 'Z_Ours_lam0.3.npy'), np.load(MDL / 'Z_Ours_lam0.0.npy')
    Zt = np.load(MDL / 'Z_t-SNE.npy')
    rng = np.random.default_rng(0)
    out = {}
    for sub in ['held_all', 'held_audit_accepted', 'held_msu_unseen']:
        f = HERE / f'subset_{sub}.npy'
        if not f.exists():
            continue
        pairs = [tuple(x) for x in np.load(f)]
        h3, h0, ht = hits(Z3, pairs), hits(Z0, pairs), hits(Zt, pairs)
        n = len(pairs)
        idx = rng.integers(0, n, size=(R, n))
        res = {'n': n, 'Ours_lam0.3': float(h3.mean()),
               'Ours_lam0.0': float(h0.mean()), 't-SNE': float(ht.mean())}
        for label, hb in [('vs_lam0.0', h0), ('vs_tSNE', ht)]:
            d = h3[idx].mean(1) - hb[idx].mean(1)
            lo, hi = np.percentile(d, [2.5, 97.5])
            res[label] = {'delta': float(h3.mean() - hb.mean()),
                          'ci': [float(lo), float(hi)], 'p_le0': float((d <= 0).mean())}
        out[sub] = res
        print(f'\n{sub}  (n={n})   Ours0.3={h3.mean():.3f}  Ours0.0={h0.mean():.3f}  t-SNE={ht.mean():.3f}')
        for label in ['vs_lam0.0', 'vs_tSNE']:
            r = res[label]
            print(f"   delta {label:10} = {r['delta']:+.3f}  95%CI [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  P(delta<=0)={r['p_le0']:.4f}")
    (HERE / 'audit_stats.json').write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {HERE/'audit_stats.json'}")


if __name__ == '__main__':
    raise SystemExit(main())
