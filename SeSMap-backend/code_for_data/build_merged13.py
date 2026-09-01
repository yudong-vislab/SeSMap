#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并语料 A(data/stages, 6篇) + B(data/general_v11, 7篇) -> merged13 (13篇)。
B 的行索引整体偏移 +len(A)；paper_id 偏移 +6。保留各自原有的 train/test 划分。"""
import json, collections
from pathlib import Path
import numpy as np

BK = Path(__file__).resolve().parents[1]
A_EMB = BK/'data/stages/04_embeddings/emb_corpus.npy'
B_EMB = BK/'data/general_v11/stages/04_embeddings/emb_corpus.npy'
A_PAIR= BK/'data/stages/02_msu/llm_pairs_10k.json'
B_PAIR= BK/'data/general_v11/stages/02_msu/llm_pairs.json'
A_FDB = BK/'data/stages/02_msu/formdatabase_v2.0.json'
B_FDB = BK/'data/general_v11/stages/02_msu/formdatabase_v2.0.json'
OUT   = BK/'data/merged13/stages'

# 排除语料 B 中唯一一句引用生信论文的 MSU（"Zhang et al. introduce an LLM-driven framework
# for circular genome visualization generation."），避免与生信评测语料产生任何接触。该 MSU 不参与任何配对。
B_EXCLUDE = {506}

XA, XB = np.load(A_EMB), np.load(B_EMB); nA, nB = len(XA), len(XB)
fa, fb = json.load(open(A_FDB)), json.load(open(B_FDB))
pa, pb = json.load(open(A_PAIR)), json.load(open(B_PAIR))
print(f"A: emb {XA.shape} fdb {len(fa)} pairs {len(pa['train_pos'])}+{len(pa['test_pos'])}")
print(f"B: emb {XB.shape} fdb {len(fb)} pairs {len(pb['train_pos'])}+{len(pb['test_pos'])}")
assert len(fa)==nA and len(fb)==nB, "formdb 与 embedding 行数不一致，索引不可直接对齐"


keepB = [i for i in range(nB) if i not in B_EXCLUDE]
bmap  = {o:i for i,o in enumerate(keepB)}          # B 旧行号 -> B 新行号
for i in sorted(B_EXCLUDE):
    print(f"  排除 B 行 {i} (p{fb[i].get('paper_id')}): {fb[i].get('sentence','')[:90]}")

def shift(lst, off, n, remap=None):
    out=[]
    for u,v,*r in lst:
        u,v=int(u),int(v)
        assert 0<=u<n and 0<=v<n, f"索引越界 {u},{v} (n={n})"
        if remap is not None:
            if u not in remap or v not in remap: continue   # 端点被排除则丢弃该对
            u,v = remap[u], remap[v]
        out.append([u+off, v+off]+list(r))
    return out

train = shift(pa['train_pos'],0,nA) + shift(pb['train_pos'],nA,nB,bmap)
test  = shift(pa['test_pos'], 0,nA) + shift(pb['test_pos'], nA,nB,bmap)
X = np.concatenate([XA, XB[keepB]],0).astype(np.float32); N=len(X)
fdb=[]
for r in fa: r=dict(r); r['paper_id']=int(r.get('paper_id',0));            fdb.append(r)
for i in keepB:
    r=dict(fb[i]); r['paper_id']=int(r.get('paper_id',0))+6;               fdb.append(r)
for i,r in enumerate(fdb): r['row']=i

# --- 校验 ---
ep=set(u for u,v,*_ in train+test)|set(v for u,v,*_ in train+test)
pid=np.array([r['paper_id'] for r in fdb])
cross=sum(1 for u,v,*_ in train+test if pid[u]!=pid[v])
ov=set((min(u,v),max(u,v)) for u,v,*_ in train) & set((min(u,v),max(u,v)) for u,v,*_ in test)
print(f"\nmerged: N={N}  papers={len(set(pid.tolist()))}  train={len(train)} test={len(test)} total={len(train)+len(test)}")
print(f"  端点覆盖 {len(ep)}/{N} ({len(ep)/N*100:.1f}%)   跨论文对 {cross}/{len(train)+len(test)} ({cross/(len(train)+len(test))*100:.0f}%)")
print(f"  train/test 完全重复对 = {len(ov)}   (必须为 0)"); assert len(ov)==0
epT=set(u for u,v,*_ in test)|set(v for u,v,*_ in test); epR=set(u for u,v,*_ in train)|set(v for u,v,*_ in train)
print(f"  端点泄漏(test 端点也出现在 train) = {len(epT&epR)}/{len(epT)} ({len(epT&epR)/len(epT)*100:.0f}%)  <- 随机按对划分的固有性质")
print(f"  每篇 MSU: {dict(sorted(collections.Counter(pid.tolist()).items()))}")

OUT.joinpath('04_embeddings').mkdir(parents=True,exist_ok=True); OUT.joinpath('02_msu').mkdir(parents=True,exist_ok=True)
np.save(OUT/'04_embeddings/emb_corpus.npy', X)
json.dump(list(range(N)), open(OUT/'04_embeddings/emb_corpus.npy.ids.json','w'))
json.dump({'meta':{'note':'pairs are ROW indices into merged EMB_CACHE',
                   'source_A':str(A_PAIR.relative_to(BK)),'source_B':str(B_PAIR.relative_to(BK)),
                   'n_A':nA,'n_B':nB-len(B_EXCLUDE),'papers':13,'excluded_B_rows':sorted(B_EXCLUDE),'n_pos':len(train)+len(test)},
           'train_pos':train,'test_pos':test}, open(OUT/'02_msu/llm_pairs.json','w'))
json.dump(fdb, open(OUT/'02_msu/formdatabase_v2.0.json','w'), ensure_ascii=False)
print(f"\nsaved -> {OUT}")
