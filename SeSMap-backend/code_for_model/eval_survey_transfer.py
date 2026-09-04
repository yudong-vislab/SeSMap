#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同领域冻结迁移评测：综述+摘要语料训练的投影器 -> 12 篇生信论文。
对照：跨领域冻结迁移(空气污染 v11)、UMAP/PCA.transform、生信重拟合参照。
额外输出逐篇 Co-loc，用于检验"训练语料缺少 LLM 语义"这一假设。"""
from __future__ import annotations
import argparse, json, sys, os, collections
from pathlib import Path
import numpy as np, torch
BK=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(BK))
from code_for_model.models import ResidualProjectionMapper
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
K=12
BIO=BK/'data/bio_eval/stages'
def fwd(p,X):
    ck=torch.load(p,map_location='cpu',weights_only=False)
    m=ResidualProjectionMapper(embed_dim=ck['embed_dim'],width=ck['width'],
                               num_blocks=ck['num_blocks'],dropout=ck['dropout'])
    m.load_state_dict(ck['mapper_state']); m.eval()
    with torch.no_grad(): return m(torch.tensor(X,dtype=torch.float32)).numpy().astype(np.float64)
def ng(Z,k=K): return [set(r) for r in NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)]
def hits(Z,pairs,k=K):
    g=ng(Z,k); return np.array([1.0 if (b in g[a] or a in g[b]) else 0.0 for a,b in pairs])
def knn_ov(X,Z,k=K):
    a=NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False)
    b=NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)
    return float(np.mean([len(set(u)&set(v))/k for u,v in zip(a,b)]))
def cube_round(x,y,z):
    rx,ry,rz=np.round(x),np.round(y),np.round(z); dx,dy,dz=abs(rx-x),abs(ry-y),abs(rz-z)
    m=(dx>dy)&(dx>dz); rx=np.where(m,-ry-rz,rx); m2=(~m)&(dy>dz); ry=np.where(m2,-rx-rz,ry)
    return rx.astype(int),rz.astype(int)
NB=[(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]
def hexadj(Z,pairs,scale=32):
    mu,sd=Z.mean(0),Z.std(0); sd[sd<1e-9]=1; S=(Z-mu)/sd*1.8; size=0.15*scale/32
    q=(np.sqrt(3)/3*S[:,0]-1/3*S[:,1])/size; r=(2/3*S[:,1])/size
    qq,rr=cube_round(q,-q-r,r); c=list(zip(qq,rr))
    return float(np.mean([1.0 if (c[a]==c[b] or (c[a][0]-c[b][0],c[a][1]-c[b][1]) in NB) else 0.0 for a,b in pairs]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--survey-ckpt',type=Path,
                    default=BK/'data/survey_genomics/stages/05_model/v11_survey_lam0.3_s0.pt')
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--out',type=Path,default=BK/'results/survey_transfer.json'); a=ap.parse_args()

    Xb=np.load(BIO/'04_embeddings/emb_corpus.npy').astype(np.float32); N=len(Xb)
    TE=[(int(u),int(v)) for u,v,*_ in json.load(open(BIO/'02_msu/llm_pairs.json'))['test_pos']
        if u<N and v<N and u!=v]
    fdb=json.load(open(BIO/'02_msu/formdatabase_v2.0.json'))
    pid=np.array([r.get('paper_id',-1) for r in fdb][:N])
    names=sorted(os.listdir(BIO/'01_corpus'))
    print(f"生信评测集: N={N}  12 篇  留出对={len(TE)}  k={K}  随机基线≈{K/N:.4f}\n")

    Z={}
    Z['Ours-survey (zero-shot, 同领域)'] = fwd(a.survey_ckpt, Xb)
    Z['Ours-v11 air (zero-shot, 跨领域)'] = fwd(BK/'data/stages/05_model/v11_hybrid_0.1_big.pt', Xb)
    Xs=np.load(BK/'data/survey_genomics/stages/04_embeddings/emb_corpus.npy').astype(np.float32)
    Z['PCA.transform (survey fit)']=PCA(2,random_state=a.seed).fit(Xs).transform(Xb).astype(np.float64)
    import umap
    Z['UMAP.transform (survey fit)']=umap.UMAP(n_components=2,n_neighbors=15,min_dist=0.1,
        random_state=a.seed).fit(Xs).transform(Xb).astype(np.float64)
    Z['[参照] 生信重拟合']=fwd(BIO/'05_model/v11_genomics.pt', Xb)

    print(f"{'方法':34} | {'trust':>6} {'knnOv@12':>9} {'Co-loc@12':>10} {'hexAdj':>8}")
    print('-'*76)
    rows={}; H={}
    for n,z in Z.items():
        H[n]=hits(z,TE)
        rows[n]=dict(trust=float(trustworthiness(Xb,z,n_neighbors=K)),knnOv=knn_ov(Xb,z),
                     coloc=float(H[n].mean()),hexadj=hexadj(z,TE))
        r=rows[n]; print(f"{n:34} | {r['trust']:6.3f} {r['knnOv']:9.3f} {r['coloc']:10.3f} {r['hexadj']:8.3f}")
    print(f"\n高维天花板 bz_recall@12_highD = 0.598 (编码器空间中本就 12-近邻的比例)")

    # ---- 逐篇诊断：LLM 语义缺口假设 ----
    import re
    LLM=r'\bLLM|large language model|GPT|prompt|agent|generative|transformer|fine-tun|natural language'
    llm_share={}
    for p in sorted(set(pid.tolist())):
        idx=np.flatnonzero(pid==p)
        llm_share[p]=sum(1 for i in idx if re.search(LLM,fdb[i].get('sentence',''),re.I))/max(len(idx),1)
    print("\n"+"="*88); print("逐篇 Co-loc@12（检验训练语料缺 LLM 语义是否造成系统性劣势）"); print("="*88)
    key='Ours-survey (zero-shot, 同领域)'; ref='[参照] 生信重拟合'
    print(f"{'论文':44}{'对数':>5}{'LLM%':>7}{'迁移':>8}{'重拟合':>8}{'追回率':>8}")
    print('-'*88)
    per=[]
    for p in sorted(set(pid.tolist())):
        sel=[i for i,(u,v) in enumerate(TE) if pid[u]==p or pid[v]==p]
        if len(sel)<5: continue
        c1=float(H[key][sel].mean()); c0=float(H[ref][sel].mean())
        nm=names[p][:42] if p<len(names) else f'p{p}'
        per.append(dict(paper=nm,n=len(sel),llm=llm_share[p],transfer=c1,refit=c0,
                        recover=c1/c0 if c0>0 else float('nan')))
        print(f"{nm:44}{len(sel):>5}{llm_share[p]*100:>6.1f}%{c1:>8.3f}{c0:>8.3f}{(c1/c0 if c0>0 else 0):>7.0%}")
    if len(per)>=4:
        L=np.array([x['llm'] for x in per]); R=np.array([x['recover'] for x in per])
        ok=~np.isnan(R)
        print(f"\nLLM 语义占比 vs 迁移追回率  相关系数 r = {np.corrcoef(L[ok],R[ok])[0,1]:+.3f}")
        hi=[x for x in per if x['llm']>0.05]; lo=[x for x in per if x['llm']<=0.05]
        if hi and lo:
            print(f"  LLM 重的论文({len(hi)}篇) 追回率 {np.mean([x['recover'] for x in hi]):.0%}"
                  f"   其余({len(lo)}篇) {np.mean([x['recover'] for x in lo]):.0%}")
    a.out.write_text(json.dumps(dict(N=N,n_test=len(TE),k=K,results=rows,per_paper=per,
                                     highD_ceiling=0.598),ensure_ascii=False,indent=1),encoding='utf-8')
    print(f"\nsaved -> {a.out}")
if __name__=='__main__': raise SystemExit(main())
