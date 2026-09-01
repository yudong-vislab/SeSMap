#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生信 12 篇复现实验 + 重复抽样。
v10-bio (纯 t-SNE, 无对应监督)  vs  v11-bio (t-SNE + 跨论文 InfoNCE)
两者在同一批 5,415 个生信 MSU 上按相同超参拟合，只差监督项。
留出 403 对对应从未参与训练。"""
from __future__ import annotations
import argparse, json, sys, collections
from pathlib import Path
import numpy as np, torch
BACKEND=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(BACKEND))
from code_for_model.models import ResidualProjectionMapper
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

def fwd(c,X):
    ck=torch.load(c,map_location='cpu',weights_only=False)
    m=ResidualProjectionMapper(embed_dim=ck['embed_dim'],width=ck['width'],
                               num_blocks=ck['num_blocks'],dropout=ck['dropout'])
    m.load_state_dict(ck['mapper_state']); m.eval()
    with torch.no_grad(): return m(torch.tensor(X,dtype=torch.float32)).numpy().astype(np.float64)

def neigh(Z,k):
    return [set(r) for r in NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)]
def hits(ng,pairs):           # 每对是否共位 -> 0/1 向量，供 bootstrap 用
    return np.array([1.0 if (b in ng[a] or a in ng[b]) else 0.0 for a,b in pairs])
def knn_ov(X,Z,k):
    a=NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False)
    b=NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)
    return float(np.mean([len(set(u)&set(v))/k for u,v in zip(a,b)]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--k',type=int,default=12)
    ap.add_argument('--boot',type=int,default=1000)
    ap.add_argument('--rounds',type=int,default=30)
    ap.add_argument('--sub-papers',type=int,default=6)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--out',type=Path,default=BACKEND/'results/bio_replication.json')
    a=ap.parse_args(); rng=np.random.default_rng(a.seed)

    X=np.load(BACKEND/'data/bio_eval/stages/04_embeddings/emb_corpus.npy').astype(np.float32); N=len(X)
    pj=json.load(open(BACKEND/'data/bio_eval/stages/02_msu/llm_pairs.json'))
    TE=[(int(u),int(v)) for u,v,*_ in pj['test_pos'] if u<N and v<N and u!=v]
    fdb=json.load(open(BACKEND/'data/bio_eval/stages/02_msu/formdatabase_v2.0.json'))
    pid=np.array([r.get('paper_id',-1) for r in fdb][:N])
    papers=sorted(set(pid.tolist()))

    M={'无对应监督 (t-SNE only)':BACKEND/'data/bio_eval/stages/05_model/bert2d_mapper_all_v10.pt',
       '+对应监督 (v11 hybrid)':BACKEND/'data/bio_eval/stages/05_model/v11_genomics.pt'}
    Z={n:fwd(p,X) for n,p in M.items()}
    print(f"生信语料 N={N}  论文 {len(papers)} 篇  留出对应对 {len(TE)}  k={a.k}  随机基线≈{a.k/N:.4f}\n")

    # ---------- 1. 全量 ----------
    print("="*72); print("1. 全量（12 篇，403 留出对）"); print("="*72)
    H={}; full={}
    print(f"{'布局':26} | {'trust':>6} {'knnOv':>6} {'Co-loc@12':>10}")
    print('-'*60)
    for n,z in Z.items():
        ng=neigh(z,a.k); H[n]=hits(ng,TE)
        full[n]=dict(trust=float(trustworthiness(X,z,n_neighbors=a.k)),knnOv=knn_ov(X,z,a.k),coloc=float(H[n].mean()))
        print(f"{n:26} | {full[n]['trust']:6.3f} {full[n]['knnOv']:6.3f} {full[n]['coloc']:10.3f}")
    names=list(Z); g=(full[names[1]]['coloc']/full[names[0]]['coloc']-1)*100
    print(f"\n对应监督增益: {full[names[0]]['coloc']:.3f} -> {full[names[1]]['coloc']:.3f}  (+{g:.0f}%)")

    # ---------- 2. bootstrap over pairs ----------
    print("\n"+"="*72); print(f"2. 留出对 bootstrap (R={a.boot}, 配对)"); print("="*72)
    idx=rng.integers(0,len(TE),size=(a.boot,len(TE)))
    bs={n:H[n][idx].mean(1) for n in names}
    diff=bs[names[1]]-bs[names[0]]
    boot={}
    for n in names:
        lo,hi=np.percentile(bs[n],[2.5,97.5]); boot[n]=dict(mean=float(bs[n].mean()),sd=float(bs[n].std()),ci=[float(lo),float(hi)])
        print(f"{n:26} Co-loc = {bs[n].mean():.3f} ± {bs[n].std():.3f}   95%CI [{lo:.3f}, {hi:.3f}]")
    dlo,dhi=np.percentile(diff,[2.5,97.5]); pneg=float((diff<=0).mean())
    print(f"\n{'配对差值 (监督 - 无监督)':26} = {diff.mean():.3f} ± {diff.std():.3f}   95%CI [{dlo:.3f}, {dhi:.3f}]")
    print(f"{'差值 <= 0 的重抽样比例':26} = {pneg:.4f}   -> {'显著' if dhi>0 and dlo>0 else '不显著'}")

    # ---------- 3. 重复论文子抽样 ----------
    print("\n"+"="*72); print(f"3. 重复论文子抽样 (每轮 {a.sub_papers}/{len(papers)} 篇, R={a.rounds})"); print("="*72)
    acc=collections.defaultdict(list); accg=[]; accn=[]
    for r in range(a.rounds):
        sel=rng.choice(papers,size=a.sub_papers,replace=False)
        mask=np.isin(pid,sel); rows=np.flatnonzero(mask)
        remap={int(o):i for i,o in enumerate(rows)}
        sp=[(remap[u],remap[v]) for u,v in TE if u in remap and v in remap]
        if len(sp)<30: continue
        cur={}
        for n in names:
            zz=Z[n][rows]; cur[n]=float(hits(neigh(zz,a.k),sp).mean()); acc[n].append(cur[n])
        accg.append((cur[names[1]]/max(cur[names[0]],1e-9)-1)*100); accn.append((len(rows),len(sp)))
    R=len(accg)
    print(f"有效轮次 {R}/{a.rounds}   子集规模 MSU {np.mean([x[0] for x in accn]):.0f}±{np.std([x[0] for x in accn]):.0f}, 对 {np.mean([x[1] for x in accn]):.0f}±{np.std([x[1] for x in accn]):.0f}\n")
    sub={}
    for n in names:
        v=np.array(acc[n]); sub[n]=dict(mean=float(v.mean()),sd=float(v.std()),min=float(v.min()),max=float(v.max()))
        print(f"{n:26} Co-loc = {v.mean():.3f} ± {v.std():.3f}   [{v.min():.3f}, {v.max():.3f}]")
    d=np.array(acc[names[1]])-np.array(acc[names[0]])
    print(f"\n{'配对差值':26} = {d.mean():.3f} ± {d.std():.3f}   监督更优的轮次 = {(d>0).sum()}/{R}")
    print(f"{'相对增益':26} = {np.mean(accg):.0f}% ± {np.std(accg):.0f}%   [{min(accg):.0f}%, {max(accg):.0f}%]")

    a.out.write_text(json.dumps(dict(N=N,papers=len(papers),n_test=len(TE),k=a.k,chance=a.k/N,
        full=full,bootstrap=boot,boot_diff=dict(mean=float(diff.mean()),sd=float(diff.std()),ci=[float(dlo),float(dhi)],p_le0=pneg),
        subsample=dict(rounds=R,per_round_papers=a.sub_papers,stats=sub,
                       diff_mean=float(d.mean()),diff_sd=float(d.std()),wins=int((d>0).sum()),
                       gain_mean=float(np.mean(accg)),gain_sd=float(np.std(accg)))),ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"\nsaved -> {a.out}")
if __name__=='__main__': raise SystemExit(main())
