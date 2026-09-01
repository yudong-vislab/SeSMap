#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""merged13 (13篇/6114 MSU/3728对) 模型评测：λ_con 扫描 + 基线 + bootstrap。"""
from __future__ import annotations
import argparse, json, sys, glob, re
from pathlib import Path
import numpy as np, torch
BK=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(BK))
from code_for_model.models import ResidualProjectionMapper
from sklearn.manifold import trustworthiness, TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
K=12
EMB=BK/'data/merged13/stages/04_embeddings/emb_corpus.npy'
PAIR=BK/'data/merged13/stages/02_msu/llm_pairs.json'
FDB=BK/'data/merged13/stages/02_msu/formdatabase_v2.0.json'
MDL=BK/'data/merged13/stages/05_model'

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
    ap=argparse.ArgumentParser(); ap.add_argument('--boot',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=0); ap.add_argument('--baselines',action='store_true')
    ap.add_argument('--out',type=Path,default=BK/'results/merged13_eval.json'); a=ap.parse_args()
    X=np.load(EMB).astype(np.float32); N=len(X)
    pj=json.load(open(PAIR)); TE=[(int(u),int(v)) for u,v,*_ in pj['test_pos']]
    pid=np.array([r['paper_id'] for r in json.load(open(FDB))])
    print(f"merged13: N={N}  13 篇  留出对 {len(TE)}  k={K}  随机基线≈{K/N:.4f}\n")

    Z={}
    for f in sorted(glob.glob(str(MDL/'m13_lam*_s0.pt')), key=lambda p:float(re.search(r'lam([\d.]+)_',p).group(1))):
        lam=re.search(r'lam([\d.]+)_',f).group(1); Z[f"Ours λ_con={lam}"]=fwd(f,X)
    if a.baselines:
        Z['PCA']=PCA(2,random_state=a.seed).fit_transform(X).astype(np.float64)
        Z['t-SNE']=TSNE(2,perplexity=30,random_state=a.seed,init='pca').fit_transform(X).astype(np.float64)
        import umap; Z['UMAP']=umap.UMAP(n_components=2,n_neighbors=15,min_dist=0.1,random_state=a.seed).fit_transform(X).astype(np.float64)

    H={}; rows={}
    hdr=f"{'method':22} | {'trust':>6} {'knnOv@12':>9} {'Co-loc@12':>10} {'hexAdj@32':>10}"
    print(hdr); print('-'*len(hdr))
    for n,z in Z.items():
        H[n]=hits(z,TE)
        rows[n]=dict(trust=float(trustworthiness(X,z,n_neighbors=K)),knnOv=knn_ov(X,z),
                     coloc=float(H[n].mean()),hexadj=hexadj(z,TE))
        r=rows[n]; print(f"{n:22} | {r['trust']:6.3f} {r['knnOv']:9.3f} {r['coloc']:10.3f} {r['hexadj']:10.3f}")
        np.save(MDL/f"Z_{n.replace(' ','_').replace('λ_con=','lam').replace('/','')}.npy", z)

    ours={n:v for n,v in rows.items() if n.startswith('Ours')}
    base=[n for n in ours if n.endswith('=0.0')]
    best=max((n for n in ours if n not in base), key=lambda n:ours[n]['coloc'])
    print(f"\n最优 λ_con = {best.split('=')[1]}   (按留出 Co-loc@12 选)")
    if base:
        b0=base[0]; print(f"对应监督增益: {ours[b0]['coloc']:.3f} -> {ours[best]['coloc']:.3f}  (+{(ours[best]['coloc']/ours[b0]['coloc']-1)*100:.0f}%)")
        rng=np.random.default_rng(a.seed); idx=rng.integers(0,len(TE),size=(a.boot,len(TE)))
        d=H[best][idx].mean(1)-H[b0][idx].mean(1); lo,hi=np.percentile(d,[2.5,97.5])
        print(f"配对 bootstrap (R={a.boot}): 差值 {d.mean():.3f} ± {d.std():.3f}  95%CI [{lo:.3f}, {hi:.3f}]  差值<=0 比例 {(d<=0).mean():.4f}")
        boot=dict(best=best,base=b0,mean=float(d.mean()),sd=float(d.std()),ci=[float(lo),float(hi)],p_le0=float((d<=0).mean()))
    else: boot={}
    a.out.write_text(json.dumps(dict(N=N,papers=13,n_test=len(TE),k=K,results=rows,best=best,boot_diff=boot),
                                ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"\nsaved -> {a.out}")
if __name__=='__main__': raise SystemExit(main())
