#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生信领域零样本迁移评测：v11(空气污染 6 篇)冻结前向 -> 生信 12 篇。"""
from __future__ import annotations
import argparse, json, sys, collections
from pathlib import Path
import numpy as np, torch
BACKEND = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(BACKEND))
from code_for_model.models import ResidualProjectionMapper
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

MAIN_EMB = BACKEND/'data/stages/04_embeddings/emb_corpus.npy'
BIO_EMB  = BACKEND/'data/bio_eval/stages/04_embeddings/emb_corpus.npy'
BIO_PAIR = BACKEND/'data/bio_eval/stages/02_msu/llm_pairs.json'
BIO_FDB  = BACKEND/'data/bio_eval/stages/02_msu/formdatabase_v2.0.json'

def fwd(ckpt, X):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    m = ResidualProjectionMapper(embed_dim=ck['embed_dim'], width=ck['width'],
                                 num_blocks=ck['num_blocks'], dropout=ck['dropout'])
    m.load_state_dict(ck['mapper_state']); m.eval()
    with torch.no_grad():
        return m(torch.tensor(X, dtype=torch.float32)).numpy().astype(np.float64)

def knn_ov(X, Z, k):
    a = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False)
    b = NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)
    return float(np.mean([len(set(u)&set(v))/k for u,v in zip(a,b)]))

def coloc(Z, pairs, k):
    nn = NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)
    ng = [set(r) for r in nn]
    return float(np.mean([1.0 if (b in ng[a] or a in ng[b]) else 0.0 for a,b in pairs]))

def cube_round(x,y,z):
    rx,ry,rz=np.round(x),np.round(y),np.round(z); dx,dy,dz=abs(rx-x),abs(ry-y),abs(rz-z)
    m=(dx>dy)&(dx>dz); rx=np.where(m,-ry-rz,rx); m2=(~m)&(dy>dz); ry=np.where(m2,-rx-rz,ry)
    return rx.astype(int), rz.astype(int)
NB=[(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]
def hex_adj(Z, pairs, scale=32):
    mu,sd=Z.mean(0),Z.std(0); sd[sd<1e-9]=1; S=(Z-mu)/sd*1.8
    size=0.15*scale/32
    q=(np.sqrt(3)/3*S[:,0]-1/3*S[:,1])/size; r=(2/3*S[:,1])/size
    qq,rr=cube_round(q,-q-r,r); cell=list(zip(qq,rr))
    hit=sum(1 for a,b in pairs if cell[a]==cell[b] or
            (cell[a][0]-cell[b][0],cell[a][1]-cell[b][1]) in NB)
    return hit/len(pairs)

def metrics(X, Z, pairs, k, do_trust=True):
    d = {'coloc': coloc(Z,pairs,k), 'hexadj': hex_adj(Z,pairs)}
    d['knnOv'] = knn_ov(X,Z,k)
    d['trust'] = float(trustworthiness(X,Z,n_neighbors=k)) if do_trust else float('nan')
    return d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--k',type=int,default=12); ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--pairs',choices=['test','all'],default='test')
    ap.add_argument('--out',type=Path,default=BACKEND/'results/bio_transfer.json')
    a=ap.parse_args()

    Xb=np.load(BIO_EMB).astype(np.float32); Xm=np.load(MAIN_EMB).astype(np.float32)
    pj=json.load(open(BIO_PAIR))
    P=[(int(u),int(v)) for u,v,*_ in (pj['test_pos'] if a.pairs=='test' else pj['train_pos']+pj['test_pos'])]
    N=len(Xb); P=[(u,v) for u,v in P if u<N and v<N and u!=v]
    print(f"生信: N={N}  评测对={len(P)} ({a.pairs})  k={a.k}  随机基线≈{a.k/N:.4f}")

    fdb=json.load(open(BIO_FDB))
    pid=np.array([r.get('paper_id',-1) for r in fdb][:N])
    cross=sum(1 for u,v in P if pid[u]!=pid[v]); print(f"其中跨论文对 = {cross}/{len(P)} ({cross/len(P)*100:.0f}%)\n")

    Z={}
    Z['Ours-v11 (zero-shot)'] = fwd(BACKEND/'data/stages/05_model/v11_hybrid_0.1_big.pt', Xb)
    Z['Ours-v10 (zero-shot)'] = fwd(BACKEND/'data/stages/05_model/bert2d_mapper_all_v10.pt', Xb)
    pca=PCA(n_components=2,random_state=a.seed).fit(Xm); Z['PCA.transform']=pca.transform(Xb).astype(np.float64)
    import umap
    u=umap.UMAP(n_components=2,n_neighbors=15,min_dist=0.1,random_state=a.seed).fit(Xm)
    Z['UMAP.transform']=u.transform(Xb).astype(np.float64)
    Z['[ref] v11 refit on bio'] = fwd(BACKEND/'data/bio_eval/stages/05_model/v11_genomics.pt', Xb)
    Z['[ref] UMAP fresh'] = umap.UMAP(n_components=2,n_neighbors=15,min_dist=0.1,
                                      random_state=a.seed).fit_transform(Xb).astype(np.float64)

    rows={}
    hdr=f"{'method':26} | {'trust':>6} {'knnOv':>6} {'Co-loc':>7} {'hexAdj':>7}"
    print(hdr); print('-'*len(hdr))
    for name,zz in Z.items():
        np.save(f"/tmp/bioZ_{name.split()[0].strip('[]')}_{len(rows)}.npy", zz)
        r=metrics(Xb,zz,P,a.k); rows[name]=r
        print(f"{name:26} | {r['trust']:6.3f} {r['knnOv']:6.3f} {r['coloc']:7.3f} {r['hexadj']:7.3f}")

    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps({'N':N,'n_pairs':len(P),'pairs':a.pairs,'k':a.k,
        'chance':a.k/N,'results':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    np.save('/tmp/bio_Zall.npy', np.stack([Z[n] for n in Z]))
    json.dump(list(Z.keys()), open('/tmp/bio_Znames.json','w'))
    print(f"\nsaved -> {a.out}")

if __name__=='__main__': raise SystemExit(main())
