#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""merged13: 可见邻接(同一或相邻 HSU) 与 Co-loc@N 的一致性, 六个聚合分辨率。"""
import json,sys,collections
from pathlib import Path
import numpy as np, torch
BK=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(BK))
from code_for_model.models import ResidualProjectionMapper
from sklearn.neighbors import NearestNeighbors

def fwd(p,X):
    ck=torch.load(p,map_location='cpu',weights_only=False)
    m=ResidualProjectionMapper(embed_dim=ck['embed_dim'],width=ck['width'],
                               num_blocks=ck['num_blocks'],dropout=ck['dropout'])
    m.load_state_dict(ck['mapper_state']); m.eval()
    with torch.no_grad(): return m(torch.tensor(X,dtype=torch.float32)).numpy().astype(np.float64)
def cube_round(x,y,z):
    rx,ry,rz=np.round(x),np.round(y),np.round(z); dx,dy,dz=abs(rx-x),abs(ry-y),abs(rz-z)
    m=(dx>dy)&(dx>dz); rx=np.where(m,-ry-rz,rx); m2=(~m)&(dy>dz); ry=np.where(m2,-rx-rz,ry)
    return rx.astype(int),rz.astype(int)
NB=[(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]
def std_scale(Z,f=1.8):
    mu,sd=Z.mean(0),Z.std(0); sd[sd<1e-9]=1; return (Z-mu)/sd*f
def hex_stats(Z,pairs,scale):
    S=std_scale(Z); size=0.15*scale/32
    q=(np.sqrt(3)/3*S[:,0]-1/3*S[:,1])/size; r=(2/3*S[:,1])/size
    qq,rr=cube_round(q,-q-r,r); cell=list(zip(qq,rr)); cnt=collections.Counter(cell)
    N=int(np.median([cnt[c]+sum(cnt.get((c[0]+d[0],c[1]+d[1]),0) for d in NB) for c in cnt]))
    hit=np.mean([1.0 if (cell[a]==cell[b] or (cell[a][0]-cell[b][0],cell[a][1]-cell[b][1]) in NB)
                 else 0.0 for a,b in pairs])
    return N,float(hit),len(cnt)
def coloc(Z,pairs,k):
    k=max(2,min(k,len(Z)-1))
    ng=[set(r) for r in NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)]
    return float(np.mean([1.0 if (b in ng[a] or a in ng[b]) else 0.0 for a,b in pairs]))

X=np.load(BK/'data/merged13/stages/04_embeddings/emb_corpus.npy').astype(np.float32)
TE=[(int(u),int(v)) for u,v,*_ in json.load(open(BK/'data/merged13/stages/02_msu/llm_pairs.json'))['test_pos']]
Z1=fwd(BK/'data/merged13/stages/05_model/m13_lam0.3_s0.pt',X)
Z0=fwd(BK/'data/merged13/stages/05_model/m13_lam0.0_s0.pt',X)
SC=[13,19,25,32,45,60]
print(f"merged13  N={len(X)}  留出对={len(TE)}\n")
print(f"{'scale':>6}{'N':>6}{'HSUs':>7} |{'hex-adj':>9}{'Co-loc@N':>10}{'|diff|':>8}  |"
      f"{'hex-adj':>9}{'Co-loc@N':>10}{'|diff|':>8}")
print(f"{'':>19} |{'------ λ_corr=0.3 ------':^27}|{'----- λ_corr=0 (unsup) -----':^27}")
print('-'*82)
diffs=[]; rows=[]
for s in SC:
    N1,h1,nh1=hex_stats(Z1,TE,s); k1=coloc(Z1,TE,N1); d1=abs(h1-k1)
    N0,h0,nh0=hex_stats(Z0,TE,s); k0=coloc(Z0,TE,N0); d0=abs(h0-k0)
    diffs+=[d1,d0]; rows.append(dict(scale=s,N=N1,hsus=nh1,sup_hex=h1,sup_knn=k1,
                                     unsup_hex=h0,unsup_knn=k0,gain=(h1/h0-1)*100))
    print(f"{s:>6}{N1:>6}{nh1:>7} |{h1:>9.3f}{k1:>10.3f}{d1:>8.3f}  |{h0:>9.3f}{k0:>10.3f}{d0:>8.3f}")
print('-'*82)
print(f"最大绝对差异 = {max(diffs):.3f}   平均 = {np.mean(diffs):.3f}   （6 分辨率 × 2 布局 = {len(diffs)} 次比较）")
print(f"\n{'scale':>6}{'N':>6}{'unsup':>9}{'+corr':>9}{'gain':>9}")
for r in rows: print(f"{r['scale']:>6}{r['N']:>6}{r['unsup_hex']:>9.3f}{r['sup_hex']:>9.3f}{r['gain']:>8.0f}%")
json.dump(dict(rows=rows,max_dev=max(diffs),mean_dev=float(np.mean(diffs)),n_cmp=len(diffs)),
          open(BK/'results/merged13_hexknn.json','w'),indent=1)
print("\nsaved -> results/merged13_hexknn.json")
