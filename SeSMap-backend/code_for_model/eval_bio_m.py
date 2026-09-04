#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bio_m 评测：12 篇进布局但其对应从未被监督，在 12 篇全部 2015 对上评测。"""
import json,sys,os
from pathlib import Path
import numpy as np, torch
BK=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(BK))
from code_for_model.models import ResidualProjectionMapper
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
K=12; NB=5415   # 前 5415 行是 12 篇研究集
def fwd(c,X):
    ck=torch.load(c,map_location='cpu',weights_only=False)
    m=ResidualProjectionMapper(embed_dim=ck['embed_dim'],width=ck['width'],
                               num_blocks=ck['num_blocks'],dropout=ck['dropout'])
    m.load_state_dict(ck['mapper_state']); m.eval()
    with torch.no_grad(): return m(torch.tensor(X,dtype=torch.float32)).numpy().astype(np.float64)
def hits(Z,P,k=K):
    g=[set(r) for r in NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)]
    return np.array([1.0 if (b in g[a] or a in g[b]) else 0.0 for a,b in P])
def knn_ov(X,Z,k=K):
    a=NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False)
    b=NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)
    return float(np.mean([len(set(u)&set(v))/k for u,v in zip(a,b)]))

X=np.load(BK/'data/bio_m/stages/04_embeddings/emb_corpus.npy').astype(np.float32)
Xb=X[:NB]
P=[(int(u),int(v)) for u,v in json.load(open(BK/'data/bio_m/stages/02_msu/study_pairs_heldout.json'))['test_pos']]
print(f"合并语料 {len(X)} 点 (研究集 {NB} + M {len(X)-NB})   评测对 {len(P)} (研究集全部, 从未被监督)\n")
print(f"{'设定':34}{'trust':>7}{'knnOv@12':>10}{'Co-loc@12':>11}")
print('-'*64)
res={}
for nm,f in [('bio_m λ=0.3 (M监督) s0','bio_m_lam0.3_s0.pt'),('bio_m λ=0.3 s1','bio_m_lam0.3_s1.pt'),
             ('bio_m λ=0.3 s2','bio_m_lam0.3_s2.pt'),('bio_m λ=0 (零监督)','bio_m_lam0.0_s0.pt')]:
    Z=fwd(BK/f'data/bio_m/stages/05_model/{f}',X)
    Zb=Z[:NB]
    r=dict(trust=float(trustworthiness(Xb,Zb,n_neighbors=K)),knnOv=knn_ov(Xb,Zb),
           coloc=float(hits(Z,P).mean()))
    res[nm]=r; print(f"{nm:34}{r['trust']:>7.3f}{r['knnOv']:>10.3f}{r['coloc']:>11.3f}")
s=[res[k]['coloc'] for k in res if 'λ=0.3' in k]
print(f"\nλ=0.3 三种子: {np.mean(s):.3f} ± {np.std(s,ddof=1):.3f}")
print(f"λ=0  零监督 : {res['bio_m λ=0 (零监督)']['coloc']:.3f}")
print(f"M 监督净增益: {np.mean(s)-res['bio_m λ=0 (零监督)']['coloc']:+.3f}")
# bootstrap
h1=hits(fwd(BK/'data/bio_m/stages/05_model/bio_m_lam0.3_s0.pt',X),P)
h0=hits(fwd(BK/'data/bio_m/stages/05_model/bio_m_lam0.0_s0.pt',X),P)
rng=np.random.default_rng(0); idx=rng.integers(0,len(P),size=(2000,len(P)))
d=h1[idx].mean(1)-h0[idx].mean(1); lo,hi=np.percentile(d,[2.5,97.5])
print(f"配对 bootstrap: {d.mean():+.4f} ± {d.std():.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]  -> {'显著' if lo>0 else '不显著'}")
json.dump(res,open(BK/'results/bio_m_eval.json','w'),ensure_ascii=False,indent=1)
print("\nsaved -> results/bio_m_eval.json")
