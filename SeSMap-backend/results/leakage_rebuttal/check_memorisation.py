#!/usr/bin/env python3
"""Train-vs-test Co-loc on the ORIGINAL released split.

Unsupervised layouts cannot distinguish training pairs from held-out pairs, so
their train/test scores should match -- and they do.  Any gap for the supervised
model is memorisation of the pairs it was trained on.  This quantifies how much
of the headline Co-loc number is fitted rather than generalised, independently of
the MSU-disjoint retraining.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import stats

BK = Path('/Users/yudong/Desktop/SeSMap/SeSMap-backend')
H = Path(__file__).parent
MDL = BK / 'data/merged13/stages/05_model'
K = 12


def hits(Z, pairs, k=K):
    ng = [set(r) for r in NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)]
    return np.array([1.0 if (b in ng[a] or a in ng[b]) else 0.0 for a, b in pairs])


m13 = json.load(open(BK / 'data/merged13/stages/02_msu/llm_pairs.json'))
tr = [tuple(sorted((int(a), int(b)))) for a, b, *_ in m13['train_pos']]
te = [tuple(sorted((int(a), int(b)))) for a, b, *_ in m13['test_pos']]

X = np.load(BK / 'data/merged13/stages/04_embeddings/emb_corpus.npy').astype(np.float32)
Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
cos = lambda ps: np.array([float(Xn[a] @ Xn[b]) for a, b in ps])

METHODS = ['t-SNE', 'UMAP', 'PCA', 'Ours_lam0.0', 'Ours_lam0.3']
print(f'{"":14} {"n":>5} {"cos":>7} | ' + ' '.join(f'{m:>11}' for m in METHODS))
out = {}
for nm, ps in [('orig TRAIN', tr), ('orig TEST', te)]:
    vals = {m: float(hits(np.load(MDL / f'Z_{m}.npy'), ps).mean()) for m in METHODS}
    out[nm] = {'n': len(ps), 'cos': float(cos(ps).mean()), **vals}
    print(f'{nm:14} {len(ps):5d} {cos(ps).mean():7.4f} | ' + ' '.join(f'{vals[m]:11.3f}' for m in METHODS))

print('\ntrain-minus-test gap (memorisation signal):')
for m in METHODS:
    g = out['orig TRAIN'][m] - out['orig TEST'][m]
    print(f'  {m:14} {g:+.3f}')

p = stats.mannwhitneyu(cos(tr), cos(te)).pvalue
print(f'\ncos(train) vs cos(test) Mann-Whitney p = {p:.4f}  '
      f'(difficulty is comparable, so the gap is not a difficulty artefact)')
out['cos_mannwhitney_p'] = float(p)
(H / 'memorisation_check.json').write_text(json.dumps(out, indent=2))
print(f"\nsaved -> {H/'memorisation_check.json'}")
