#!/usr/bin/env python3
"""Aggregate the MSU-disjoint retraining runs into reviewer-facing numbers.

Per seed we report Co-loc@12 on
  (i)  the full MSU-disjoint held-out set, and
  (ii) its intersection with the audit-accepted pairs (strictest subset: no MSU
       shared with training AND correspondence confirmed by the audit).
Baselines are unsupervised, so the published PCA/t-SNE/UMAP layouts are scored
on the *same* disjoint test pairs rather than retrained.
"""
import json, argparse
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors

BK = Path('/Users/yudong/Desktop/SeSMap/SeSMap-backend')
H = Path(__file__).parent
MDL = BK / 'data/merged13/stages/05_model'
K, R = 12, 10000
METRIC = {'coloc': None, 'hexadj': None}   # filled in main()


def hits(Z, pairs, k=K):
    ng = [set(r) for r in NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)]
    return np.array([1.0 if (b in ng[a] or a in ng[b]) else 0.0 for a, b in pairs])


def cube_round(x, y, z):
    """Verbatim from code_for_model/eval_merged13.py."""
    rx, ry, rz = np.round(x), np.round(y), np.round(z)
    dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)
    m = (dx > dy) & (dx > dz); rx = np.where(m, -ry - rz, rx)
    m2 = (~m) & (dy > dz); ry = np.where(m2, -rx - rz, ry)
    return rx.astype(int), rz.astype(int)


NB = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]


def hexadj_hits(Z, pairs, scale=32):
    """Per-pair hexAdj@32 indicator; mean matches eval_merged13.hexadj."""
    mu, sd = Z.mean(0), Z.std(0); sd[sd < 1e-9] = 1
    S = (Z - mu) / sd * 1.8; size = 0.15 * scale / 32
    q = (np.sqrt(3) / 3 * S[:, 0] - 1 / 3 * S[:, 1]) / size
    r = (2 / 3 * S[:, 1]) / size
    qq, rr = cube_round(q, -q - r, r); c = list(zip(qq, rr))
    return np.array([1.0 if (c[a] == c[b] or (c[a][0] - c[b][0], c[a][1] - c[b][1]) in NB)
                     else 0.0 for a, b in pairs])


def ms(v):
    v = np.asarray(v, float)
    return f'{v.mean():.3f} ± {v.std(ddof=1):.3f}' if len(v) > 1 else f'{v.mean():.3f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', default=['0', '1', '2', '3', '4'])
    ap.add_argument('--lams', nargs='+', default=['0.0', '0.3'])
    ap.add_argument('--tag', default='')
    ap.add_argument('--metric', default='coloc', choices=['coloc', 'hexadj'])
    a = ap.parse_args()

    score = hits if a.metric == 'coloc' else hexadj_hits
    S = json.load(open(H / 'splits_disjoint.json'))
    acc = {tuple(p) for p in json.load(open(H / 'audit_accepted_pairs.json'))}
    base = {n: np.load(MDL / f'Z_{n}.npy') for n in ['PCA', 't-SNE', 'UMAP']}

    rows, pooled = {}, {}
    for s in a.seeds:
        te = [tuple(sorted(map(int, p))) for p in S[s]['test']]
        te_acc = [p for p in te if p in acc]
        for sub, pairs in [('disjoint', te), ('disjoint_audit', te_acc)]:
            if not pairs:
                continue
            for lam in a.lams:
                f = H / f'Zdis{a.tag}_s{s}_lam{lam}.npy'
                if not f.exists():
                    continue
                h = score(np.load(f), pairs)
                rows.setdefault(f'Ours_lam{lam}', {}).setdefault(sub, []).append(h.mean())
                pooled.setdefault((f'Ours_lam{lam}', sub), []).append(h)
            for n, Z in base.items():
                h = score(Z, pairs)
                rows.setdefault(n, {}).setdefault(sub, []).append(h.mean())
                pooled.setdefault((n, sub), []).append(h)

    print(f'metric = {a.metric}@{K if a.metric=="coloc" else 32}\n')
    print(f'{"method":14} | {"MSU-disjoint held-out":>24} | {"+ audit-accepted":>22}')
    print('-' * 66)
    for m in ['Ours_lam0.3', 'Ours_lam0.0', 't-SNE', 'UMAP', 'PCA']:
        if m not in rows:
            continue
        print(f'{m:14} | {ms(rows[m].get("disjoint", [np.nan])):>24} | '
              f'{ms(rows[m].get("disjoint_audit", [np.nan])):>22}')

    ns = {sub: [len(x) for x in pooled.get(('Ours_lam0.3', sub), [])]
          for sub in ['disjoint', 'disjoint_audit']}
    print(f"\nper-seed n: disjoint={ns['disjoint']}  disjoint_audit={ns['disjoint_audit']}")

    print('\npaired bootstrap, pooled over seeds (lam0.3 - comparator):')
    rng = np.random.default_rng(0)
    stats = {}
    for sub in ['disjoint', 'disjoint_audit']:
        if ('Ours_lam0.3', sub) not in pooled:
            continue
        h3 = np.concatenate(pooled[('Ours_lam0.3', sub)])
        stats[sub] = {'n': int(len(h3)), 'Ours_lam0.3': float(h3.mean())}
        idx = rng.integers(0, len(h3), size=(R, len(h3)))
        for comp in ['Ours_lam0.0', 't-SNE', 'UMAP', 'PCA']:
            if (comp, sub) not in pooled:
                continue
            hb = np.concatenate(pooled[(comp, sub)])
            d = h3[idx].mean(1) - hb[idx].mean(1)
            lo, hi = np.percentile(d, [2.5, 97.5])
            stats[sub][comp] = {'coloc': float(hb.mean()),
                                'delta': float(h3.mean() - hb.mean()),
                                'ci': [float(lo), float(hi)], 'p_le0': float((d <= 0).mean())}
            print(f'  [{sub:15}] vs {comp:12} delta={h3.mean()-hb.mean():+.3f} '
                  f'95%CI [{lo:+.3f}, {hi:+.3f}]  P(delta<=0)={(d<=0).mean():.4f}')

    (H / f'disjoint_summary_{a.metric}.json').write_text(
        json.dumps({'per_seed': rows, 'pooled_bootstrap': stats, 'n_per_seed': ns},
                   indent=2, default=float))
    print(f"\nsaved -> {H}/disjoint_summary_{a.metric}.json")


if __name__ == '__main__':
    raise SystemExit(main())
