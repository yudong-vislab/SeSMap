#!/usr/bin/env python3
"""Sanity check for the rebuttal: are the MSU-disjoint held-out pairs harder than
the original held-out pairs?  If they were, the shrinking supervision gain could
be blamed on test difficulty rather than on leakage.  We compare the encoder-space
cosine similarity of the positive pairs in each set."""
import json
from pathlib import Path
import numpy as np
from scipy import stats

BK = Path('/Users/yudong/Desktop/SeSMap/SeSMap-backend')
H = Path(__file__).parent
X = np.load(BK / 'data/merged13/stages/04_embeddings/emb_corpus.npy').astype(np.float32)
Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def cos(ps):
    return np.array([float(Xn[a] @ Xn[b]) for a, b in ps])


m13 = json.load(open(BK / 'data/merged13/stages/02_msu/llm_pairs.json'))
orig = [tuple(sorted((int(a), int(b)))) for a, b, *_ in m13['test_pos']]
S = json.load(open(H / 'splits_disjoint.json'))

o = cos(orig)
hdr = f'{"set":22} {"n":>5} {"cos mean":>9} {"sd":>7} {"p25":>6} {"p50":>6} {"p75":>6}'
print(hdr); print('-' * len(hdr))
def line(name, v):
    print(f'{name:22} {len(v):5d} {v.mean():9.4f} {v.std():7.4f} '
          f'{np.percentile(v,25):6.3f} {np.percentile(v,50):6.3f} {np.percentile(v,75):6.3f}')
line('original test', o)
allv = []
for s in sorted(S):
    v = cos([tuple(sorted(map(int, p))) for p in S[s]['test']])
    allv.append(v); line(f'disjoint seed {s}', v)
a = np.concatenate(allv)
line('disjoint pooled', a)
p = stats.mannwhitneyu(o, a).pvalue
print(f'\nMann-Whitney original vs disjoint-pooled: p = {p:.4f}')
(H / 'difficulty_check.json').write_text(json.dumps(
    {'orig_mean': float(o.mean()), 'orig_n': len(o),
     'disjoint_mean': float(a.mean()), 'disjoint_n': len(a),
     'mannwhitney_p': float(p)}, indent=2))
