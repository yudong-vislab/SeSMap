#!/usr/bin/env python3
"""从 Table 1 语料分层抽样：MSU 与 correspondence pair，各约 20%。"""
import json, random, collections, numpy as np

random.seed(20260827)
DB   = json.load(open('data/stages/02_msu/formdatabase.json'))
X    = np.load('data/stages/04_embeddings/emb_corpus.npy').astype(np.float32)
PJ   = json.load(open('data/stages/02_msu/llm_pairs_10k.json'))
N    = X.shape[0]
FRAC = 0.20

# ---------- MSU：论文 × role 分层 ----------
strata = collections.defaultdict(list)
for i, r in enumerate(DB[:N]):
    s = (r.get('sentence') or '').strip()
    if len(s.split()) < 3:          # 跳过空句/残句，审计无意义
        continue
    strata[(r.get('paper_id'), r.get('category') or 'None')].append(i)

msu_idx = []
for key, ids in strata.items():
    k = max(1, round(len(ids) * FRAC))
    msu_idx += random.sample(ids, min(k, len(ids)))
msu_idx = sorted(set(msu_idx))

# ---------- pair：相似度区间 × 角色组合 分层 ----------
Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
role = [ (r.get('category') or 'None') for r in DB[:N] ]
pos  = [(int(a), int(b)) for a, b, *_ in PJ['train_pos'] + PJ['test_pos']]
held = {(int(a), int(b)) for a, b, *_ in PJ['test_pos']}

def band(c):
    return '0.55-0.65' if c < .65 else '0.65-0.75' if c < .75 else '0.75-0.85' if c < .85 else '0.85-0.95'

pstrata = collections.defaultdict(list)
for a, b in pos:
    c = float(Xn[a] @ Xn[b])
    rc = tuple(sorted((role[a], role[b])))
    pstrata[(band(c), rc)].append((a, b, c))

pair_sample = []
for key, lst in pstrata.items():
    k = max(1, round(len(lst) * FRAC))
    pair_sample += random.sample(lst, min(k, len(lst)))

json.dump({
    'msu_idx': msu_idx,
    'pairs': [[a, b, round(c, 4), (a, b) in held] for a, b, c in pair_sample],
}, open('data/stages/02_msu/audit_sample.json', 'w'))

print(f"语料           : {len(DB)} MSU / {len(set(r.get('paper_id') for r in DB))} 篇")
print(f"MSU 抽样       : {len(msu_idx)}  ({len(msu_idx)/N*100:.1f}%)   分层 {len(strata)} 格")
print(f"  按 role      : {dict(collections.Counter(role[i] for i in msu_idx).most_common())}")
print(f"Pair 总体      : {len(pos)}  正例")
print(f"Pair 抽样      : {len(pair_sample)}  ({len(pair_sample)/len(pos)*100:.1f}%)   分层 {len(pstrata)} 格")
print(f"  其中留出集   : {sum(1 for a,b,c in pair_sample if (a,b) in held)}")
print(f"  相似度分布   : {dict(collections.Counter(band(c) for _,_,c in pair_sample).most_common())}")
