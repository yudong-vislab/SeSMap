#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""留一篇法 + 双领域同协议并排。"""
import json,sys,collections
from pathlib import Path
import numpy as np, torch
BACKEND=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(BACKEND))
from code_for_model.models import ResidualProjectionMapper
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
K=12
def fwd(c,X):
    ck=torch.load(c,map_location='cpu',weights_only=False)
    m=ResidualProjectionMapper(embed_dim=ck['embed_dim'],width=ck['width'],num_blocks=ck['num_blocks'],dropout=ck['dropout'])
    m.load_state_dict(ck['mapper_state']); m.eval()
    with torch.no_grad(): return m(torch.tensor(X,dtype=torch.float32)).numpy().astype(np.float64)
def hits(Z,pairs,k=K):
    ng=[set(r) for r in NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)]
    return np.array([1.0 if (b in ng[a] or a in ng[b]) else 0.0 for a,b in pairs])
def knn_ov(X,Z,k=K):
    a=NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False)
    b=NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)
    return float(np.mean([len(set(u)&set(v))/k for u,v in zip(a,b)]))

Xb=np.load(BACKEND/'data/bio_eval/stages/04_embeddings/emb_corpus.npy').astype(np.float32); N=len(Xb)
pj=json.load(open(BACKEND/'data/bio_eval/stages/02_msu/llm_pairs.json'))
TE=[(int(u),int(v)) for u,v,*_ in pj['test_pos'] if u<N and v<N and u!=v]
fdb=json.load(open(BACKEND/'data/bio_eval/stages/02_msu/formdatabase_v2.0.json'))
pid=np.array([r.get('paper_id',-1) for r in fdb][:N])
titles=sorted(p.name for p in (BACKEND/'data/bio_eval/stages/01_corpus').iterdir() if p.is_dir())
Z0=fwd(BACKEND/'data/bio_eval/stages/05_model/bert2d_mapper_all_v10.pt',Xb)
Z1=fwd(BACKEND/'data/bio_eval/stages/05_model/v11_genomics.pt',Xb)

print("="*84); print("A. 留一篇法：剔除某篇后，监督增益是否还在（检验是否靠单篇撑起）"); print("="*84)
print(f"{'剔除的论文':46}{'MSU':>6}{'对':>5}{'无监督':>8}{'+监督':>8}{'增益':>8}")
print('-'*84)
rowsL=[]
for p in sorted(set(pid.tolist())):
    keep=np.flatnonzero(pid!=p); rm={int(o):i for i,o in enumerate(keep)}
    sp=[(rm[u],rm[v]) for u,v in TE if u in rm and v in rm]
    c0=hits(Z0[keep],sp).mean(); c1=hits(Z1[keep],sp).mean()
    t=titles[p][:44] if p<len(titles) else f"paper {p}"
    print(f"{t:46}{(pid==p).sum():>6}{len(sp):>5}{c0:>8.3f}{c1:>8.3f}{(c1/c0-1)*100:>7.0f}%")
    rowsL.append(dict(paper=t,n_msu=int((pid==p).sum()),n_pairs=len(sp),unsup=float(c0),sup=float(c1),gain=float((c1/c0-1)*100)))
g=[r['gain'] for r in rowsL]
print(f"\n留一增益范围 [{min(g):.0f}%, {max(g):.0f}%]  中位数 {np.median(g):.0f}%  全部为正: {all(x>0 for x in g)}")

print("\n"+"="*84); print("B. 双领域同协议并排（各自语料重拟合，各自留出对）"); print("="*84)
Xm=np.load(BACKEND/'data/stages/04_embeddings/emb_corpus.npy').astype(np.float32)
pm=json.load(open(BACKEND/'data/stages/02_msu/llm_pairs_big.json'))
TM=[(int(u),int(v)) for u,v,*_ in pm['test_pos'] if u<len(Xm) and v<len(Xm) and u!=v]
M0=fwd(BACKEND/'data/stages/05_model/bert2d_mapper_all_v10.pt',Xm)
M1=fwd(BACKEND/'data/stages/05_model/v11_hybrid_0.1_big.pt',Xm)
print(f"{'语料':22}{'篇':>4}{'MSU':>7}{'留出对':>7}{'无监督':>9}{'+监督':>8}{'增益':>8}{'trust↑':>16}{'knnOv↑':>14}")
print('-'*100)
out=[]
for nm,X,Z0_,Z1_,T,np_ in [('空气污染 (开发语料)',Xm,M0,M1,TM,6),('生信 (未见领域)',Xb,Z0,Z1,TE,12)]:
    c0,c1=hits(Z0_,T).mean(),hits(Z1_,T).mean()
    t0=trustworthiness(X,Z0_,n_neighbors=K); t1=trustworthiness(X,Z1_,n_neighbors=K)
    k0,k1=knn_ov(X,Z0_),knn_ov(X,Z1_)
    print(f"{nm:22}{np_:>4}{len(X):>7}{len(T):>7}{c0:>9.3f}{c1:>8.3f}{(c1/c0-1)*100:>7.0f}%   {t0:.3f}->{t1:.3f}   {k0:.3f}->{k1:.3f}")
    out.append(dict(corpus=nm,papers=np_,n_msu=len(X),n_test=len(T),unsup=float(c0),sup=float(c1),
                    gain=float((c1/c0-1)*100),trust=[float(t0),float(t1)],knnOv=[float(k0),float(k1)]))
(BACKEND/'results/bio_loo.json').write_text(json.dumps(dict(loo=rowsL,side_by_side=out),ensure_ascii=False,indent=2),encoding='utf-8')
print("\nsaved -> results/bio_loo.json")
