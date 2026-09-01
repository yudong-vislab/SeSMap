#!/usr/bin/env python3
"""多随机种子重复：报告均值与标准差。"""
import json, sys, numpy as np, torch
sys.path.insert(0,'.')
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from code_for_model.models import ResidualProjectionMapper, evaluate_quick, set_seed, compute_joint_P
import umap
X=np.load('data/stages/04_embeddings/emb_corpus.npy').astype(np.float32)
pj=json.load(open('data/stages/02_msu/llm_pairs_10k.json'))
tr=[(int(a),int(b)) for a,b,*_ in pj['train_pos']]; te=[(int(a),int(b)) for a,b,*_ in pj['test_pos']]
N=X.shape[0]; K=12; SEEDS=[0,1,2]
def coloc(Z,pos,k=K):
    nn=NearestNeighbors(n_neighbors=k+1).fit(Z); _,ind=nn.kneighbors(Z)
    ng=[set(r[1:]) for r in ind]
    return sum(1 for a,b in pos if b in ng[a] or a in ng[b])/len(pos)
def train(lam,seed):
    set_seed(seed)
    x=torch.tensor(X); P=torch.tensor(compute_joint_P(X,30.0)); eye=torch.eye(N,dtype=torch.bool)
    m=ResidualProjectionMapper(1024,512,4,0.1); opt=torch.optim.Adam(m.parameters(),1e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,1500); ns=0.4/32.0; m.train()
    if lam>0: pa=torch.tensor([a for a,b in tr]); pb=torch.tensor([b for a,b in tr])
    for it in range(1500):
        Z=m(x+ns*torch.randn_like(x)); D2=torch.cdist(Z,Z)**2
        num=(1/(1+D2)).masked_fill(eye,0); Q=(num/num.sum().clamp_min(1e-12)).clamp_min(1e-12)
        Pe=P*12 if it<250 else P; Pn=(Pe/Pe.sum()).clamp_min(1e-12)
        loss=(Pn*(Pn.log()-Q.log())).sum()
        if lam>0:
            Qr=(num/num.sum(1,keepdim=True).clamp_min(1e-12)).clamp_min(1e-12)
            loss=loss+lam*0.5*(-Qr[pa,pb].log().mean()-Qr[pb,pa].log().mean())
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5)
        opt.step(); sch.step()
    m.eval()
    with torch.no_grad(): return m(x).cpu().numpy()
def rep(name,fn):
    r=[fn(s) for s in SEEDS]; a=np.array(r)
    print(f"{name:26} " + "  ".join(f"{a[:,i].mean():.3f}±{a[:,i].std(ddof=1):.3f}" for i in range(a.shape[1])), flush=True)
print(f"种子 {SEEDS}   {'方法':26} {'Trust':>13}{'knnOv@12':>15}{'Co-loc@12':>15}")
rep("PCA",           lambda s:(lambda Z:(*evaluate_quick(X,Z,K),coloc(Z,te)))(PCA(2,random_state=s).fit_transform(X)))
rep("t-SNE",         lambda s:(lambda Z:(*evaluate_quick(X,Z,K),coloc(Z,te)))(TSNE(2,init="pca",random_state=s).fit_transform(X)))
rep("UMAP",          lambda s:(lambda Z:(*evaluate_quick(X,Z,K),coloc(Z,te)))(umap.UMAP(n_components=2,random_state=s).fit_transform(X)))
rep("Ours",          lambda s:(lambda Z:(*evaluate_quick(X,Z,K),coloc(Z,te)))(train(0.0,s)))
rep("Ours + sup",    lambda s:(lambda Z:(*evaluate_quick(X,Z,K),coloc(Z,te)))(train(0.1,s)))
