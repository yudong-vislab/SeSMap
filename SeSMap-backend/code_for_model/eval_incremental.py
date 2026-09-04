#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""增量承载曲线：在 M 种子坐标系上冻结投影器，逐步加入用户论文并评测。
模拟真实部署：综述/引文建种子库 -> 用户上传若干论文 -> 前向投影进同一坐标系。"""
from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path
import numpy as np, torch
BK=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(BK))
from code_for_model.models import ResidualProjectionMapper
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
K=12
BIO=BK/'data/bio_eval/stages'; M=BK/'data/circ_m/stages'

def fwd(ck,X):
    c=torch.load(ck,map_location='cpu',weights_only=False)
    m=ResidualProjectionMapper(embed_dim=c['embed_dim'],width=c['width'],
                               num_blocks=c['num_blocks'],dropout=c['dropout'])
    m.load_state_dict(c['mapper_state']); m.eval()
    with torch.no_grad(): return m(torch.tensor(X,dtype=torch.float32)).numpy().astype(np.float64)
def coloc(Z,pairs,k=K):
    k=max(2,min(k,len(Z)-1))
    g=[set(r) for r in NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)]
    return float(np.mean([1.0 if (b in g[a] or a in g[b]) else 0.0 for a,b in pairs]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed-ckpt',type=Path,default=M/'05_model/seed_M_lam0.3_s0.pt')
    ap.add_argument('--draws',type=int,default=8)
    ap.add_argument('--out',type=Path,default=BK/'results/incremental_curve.json'); a=ap.parse_args()

    Xm=np.load(M/'04_embeddings/emb_corpus.npy').astype(np.float32)
    Xb=np.load(BIO/'04_embeddings/emb_corpus.npy').astype(np.float32); N=len(Xb)
    fdb=json.load(open(BIO/'02_msu/formdatabase_v2.0.json'))
    pid=np.array([r.get('paper_id',-1) for r in fdb][:N])
    pj=json.load(open(BIO/'02_msu/llm_pairs.json'))
    ALL=[(int(u),int(v)) for u,v,*_ in pj['train_pos']+pj['test_pos'] if u<N and v<N]
    papers=sorted(set(pid.tolist()))

    Zm=fwd(a.seed_ckpt,Xm); Zb_all=fwd(a.seed_ckpt,Xb)     # 冻结前向，一次算完
    print(f"种子 M: {len(Xm)} 点   用户论文池: {len(papers)} 篇 / {N} MSU   评测对 {len(ALL)}")
    print(f"参照: 12 篇重拟合(自身监督) Co-loc=0.474 | 无监督 0.337 | 一次性全灌(综述种子) 0.114\n")
    print(f"{'k':>3}{'新增MSU':>9}{'新增:种子':>10}{'对数':>6}{'Co-loc@12':>11}{'高维天花板':>11}{'追回率':>8}")
    print('-'*62)
    rng=np.random.default_rng(0); rows=[]
    for k in [1,2,3,4,6,8,12]:
        accs=[]
        draws=1 if k==12 else a.draws
        for _ in range(draws):
            sel=sorted(rng.choice(papers,k,replace=False).tolist()) if k<12 else papers
            idx=np.flatnonzero(np.isin(pid,sel))
            P=[(u,v) for u,v in ALL if pid[u] in sel and pid[v] in sel]
            if len(P)<10: continue
            rm={int(o):len(Zm)+i for i,o in enumerate(idx)}
            Z=np.vstack([Zm,Zb_all[idx]])                   # 种子坐标系 + 增量投影点
            pp=[(rm[u],rm[v]) for u,v in P]
            c=coloc(Z,pp)
            Xh=np.vstack([Xm,Xb[idx]]).astype(np.float64)   # 高维天花板(同一子集)
            ceil=coloc(Xh,pp)
            accs.append((len(idx),len(P),c,ceil))
        if not accs: continue
        A=np.array([[x[0],x[1],x[2],x[3]] for x in accs],float)
        n_add,n_pair,c,ceil=A.mean(0)
        rec=c/ceil if ceil>0 else float('nan')
        rows.append(dict(k=k,n_add=n_add,ratio=n_add/len(Xm),n_pairs=n_pair,coloc=c,ceiling=ceil,
                         recover=rec,sd=float(A[:,2].std()),draws=len(accs)))
        print(f"{k:>3}{n_add:>9.0f}{n_add/len(Xm):>9.2f}x{n_pair:>6.0f}"
              f"{c:>9.3f}±{A[:,2].std():<.3f}{ceil:>10.3f}{rec:>8.0%}")
    a.out.write_text(json.dumps(rows,ensure_ascii=False,indent=1),encoding='utf-8')
    print(f"\nsaved -> {a.out}")
if __name__=='__main__': raise SystemExit(main())
