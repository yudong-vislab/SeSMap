#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""merged13 + λ_con=0.3 的 discourse-role 子空间评测（论文 Table 2）。"""
import json,sys,collections
from pathlib import Path
import numpy as np, torch
BK=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(BK))
from code_for_model.models import ResidualProjectionMapper
from sklearn.neighbors import NearestNeighbors
K=12
NORM={'background':'Background','method':'Method','result':'Result',
      'conclusion':'Conclusion','conclusion/implication':'Conclusion',
      'experiment':'Experiment','experiment/setup':'Experiment'}
ORDER=['Background','Method','Experiment','Result','Conclusion']

def fwd(p,X):
    ck=torch.load(p,map_location='cpu',weights_only=False)
    m=ResidualProjectionMapper(embed_dim=ck['embed_dim'],width=ck['width'],
                               num_blocks=ck['num_blocks'],dropout=ck['dropout'])
    m.load_state_dict(ck['mapper_state']); m.eval()
    with torch.no_grad(): return m(torch.tensor(X,dtype=torch.float32)).numpy().astype(np.float64)
def ng_sets(Z,k=K):
    k=min(k,len(Z)-1)
    return [set(r) for r in NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)], k
def coloc(Z,pairs,k=K):
    ng,_=ng_sets(Z,k)
    return float(np.mean([1.0 if (b in ng[a] or a in ng[b]) else 0.0 for a,b in pairs]))
def knn_ov(X,Z,k=K):
    k=min(k,len(X)-1)
    a=NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False)
    b=NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)
    return float(np.mean([len(set(u)&set(v))/k for u,v in zip(a,b)]))

X=np.load(BK/'data/merged13/stages/04_embeddings/emb_corpus.npy').astype(np.float32); N=len(X)
TE=[(int(u),int(v)) for u,v,*_ in json.load(open(BK/'data/merged13/stages/02_msu/llm_pairs.json'))['test_pos']]
fdb=json.load(open(BK/'data/merged13/stages/02_msu/formdatabase_v2.0.json'))
role=np.array([NORM.get(str(r.get('category','')).strip().lower(),'') for r in fdb][:N],dtype=object)
Z=fwd(BK/'data/merged13/stages/05_model/m13_lam0.3_s0.pt',X)

drop=int((role=='').sum())
print(f"merged13  N={N}  留出对={len(TE)}  λ_con=0.3  k={K}")
print(f"角色归一后未归类而排除的 MSU = {drop} ({drop/N*100:.1f}%)  [others/Others/Definition]")
same=[(a,b) for a,b in TE if role[a] and role[a]==role[b]]
print(f"同角色留出对 = {len(same)}/{len(TE)} ({len(same)/len(TE)*100:.0f}%)\n")

ngF,_=ng_sets(Z)
print(f"{'Role':12}{'MSUs':>7}{'Pairs':>7}{'kNN@12':>9}{'Full':>8}{'Filtered':>10}{'Gain':>9}")
print('-'*62)
rows=[]
for r in ORDER:
    idx=np.flatnonzero(role==r)
    sp=[(a,b) for a,b in same if role[a]==r]
    if len(idx)<K+1 or not sp: 
        print(f"{r:12}{len(idx):>7}{len(sp):>7}   (样本不足, 跳过)"); continue
    rm={int(o):i for i,o in enumerate(idx)}
    Xs,Zs=X[idx],Z[idx]
    ko=knn_ov(Xs,Zs)
    full=float(np.mean([1.0 if (b in ngF[a] or a in ngF[b]) else 0.0 for a,b in sp]))
    spl=[(rm[a],rm[b]) for a,b in sp]
    filt=coloc(Zs,spl)
    g=filt/full if full>0 else float('nan')
    rows.append((r,len(idx),len(sp),ko,full,filt,g))
    print(f"{r:12}{len(idx):>7}{len(sp):>7}{ko:>9.3f}{full:>8.3f}{filt:>10.3f}{g:>8.2f}x")
if rows:
    w=np.array([x[2] for x in rows],float)
    print('-'*62)
    print(f"{'加权平均':12}{sum(x[1] for x in rows):>7}{int(w.sum()):>7}"
          f"{np.average([x[3] for x in rows],weights=w):>9.3f}"
          f"{np.average([x[4] for x in rows],weights=w):>8.3f}"
          f"{np.average([x[5] for x in rows],weights=w):>10.3f}"
          f"{np.average([x[5] for x in rows],weights=w)/np.average([x[4] for x in rows],weights=w):>8.2f}x")
json.dump([{'role':r,'msus':m,'pairs':p,'knn':k,'full':f,'filtered':fi,'gain':g}
           for r,m,p,k,f,fi,g in rows],
          open(BK/'results/merged13_subspace.json','w'),indent=1,ensure_ascii=False)
print(f"\nsaved -> results/merged13_subspace.json")
