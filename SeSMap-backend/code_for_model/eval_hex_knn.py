#!/usr/bin/env python3
"""可见邻接（同一或相邻 HSU）与 Co-loc@N 的一致性检验。"""
import json, math, collections
import numpy as np
from sklearn.neighbors import NearestNeighbors

def cube_round(x,y,z):
    rx,ry,rz=np.round(x),np.round(y),np.round(z)
    dx,dy,dz=abs(rx-x),abs(ry-y),abs(rz-z)
    m=(dx>dy)&(dx>dz); rx=np.where(m,-ry-rz,rx)
    m2=(~m)&(dy>dz);   ry=np.where(m2,-rx-rz,ry)
    return rx.astype(int), rz.astype(int)

def hexbin(Z,size):
    q=(np.sqrt(3)/3*Z[:,0]-1/3*Z[:,1])/size; r=(2/3*Z[:,1])/size
    return cube_round(q,-q-r,r)

def std_scale(Z,f=1.8):
    mu=Z.mean(0); sd=Z.std(0); sd[sd<1e-9]=1
    return (Z-mu)/sd*f

NB=[(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]
pj=json.load(open('data/stages/02_msu/llm_pairs_10k.json'))
te=[(int(a),int(b)) for a,b,*_ in pj['test_pos']]
Z0=std_scale(np.load('/tmp/Z10k_lam0.0.npy')); Z1=std_scale(np.load('/tmp/Z10k_lam0.1.npy'))
R0=np.load('/tmp/Z10k_lam0.0.npy');            R1=np.load('/tmp/Z10k_lam0.1.npy')

def hex_adj(Z,size):
    q,r=hexbin(Z,size); cell=list(zip(q,r))
    cnt=collections.Counter(cell)
    N=int(np.median([cnt[c]+sum(cnt.get((c[0]+d[0],c[1]+d[1]),0) for d in NB) for c in cnt]))
    hit=sum(1 for a,b in te if cell[a]==cell[b] or
            any((cell[a][0]-cell[b][0],cell[a][1]-cell[b][1])==d for d in NB))/len(te)
    return N,hit

def knn(Z,k):
    k=max(2,min(k,len(Z)-2))
    nn=NearestNeighbors(n_neighbors=k+1).fit(Z); _,ind=nn.kneighbors(Z)
    ng=[set(x[1:]) for x in ind]
    return sum(1 for a,b in te if b in ng[a] or a in ng[b])/len(te)

RANGES=[13,19,25,32,45,60]
print(f"held-out pairs = {len(te)}   corpus = A (6 papers, 2,689 MSUs)\n")
print(f"{'scale':>6}{'N':>6}{'hex-adj':>10}{'Co-loc@N':>11}{'|diff|':>9}   {'hex-adj':>9}{'Co-loc@N':>10}{'|diff|':>9}")
print(f"{'':>6}{'':>6}{'--- supervised ---':>30}   {'-- unsupervised --':>28}")
diffs=[]
for Rg in RANGES:
    size=0.15*Rg/32
    N1,h1=hex_adj(Z1,size); k1=knn(R1,N1); d1=abs(h1-k1)
    N0,h0=hex_adj(Z0,size); k0=knn(R0,N0); d0=abs(h0-k0)
    diffs += [d1,d0]
    print(f"{Rg:>6}{N1:>6}{h1:>10.3f}{k1:>11.3f}{d1:>9.3f}   {h0:>9.3f}{k0:>10.3f}{d0:>9.3f}")
print(f"\n最大绝对差异 = {max(diffs):.3f}   （{len(RANGES)} 个分辨率 × 有/无监督两种布局，共 {len(diffs)} 次比较）")
print(f"平均绝对差异 = {np.mean(diffs):.3f}")

print("\n" + "="*66)
print("各分辨率下对应监督带来的提升（可见邻接口径）")
print("="*66)
print(f"{'scale':>6}{'N':>6}{'unsup.':>10}{'+sup':>9}{'gain':>9}")
for Rg in RANGES:
    size=0.15*Rg/32
    N1,h1=hex_adj(Z1,size); _,h0=hex_adj(Z0,size)
    print(f"{Rg:>6}{N1:>6}{h0:>10.3f}{h1:>9.3f}{(h1/h0-1)*100:>8.0f}%")
