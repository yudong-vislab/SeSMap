#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 case1+case2 合并语料上补跑完整投影实验，用于替换论文 Table。"""
import json, sys
import numpy as np, torch
sys.path.insert(0, ".")
from code_for_model.models import ResidualProjectionMapper, evaluate_quick, set_seed, compute_joint_P
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

G = "data/general_v11/stages"
X = np.load(f"{G}/04_embeddings/emb_corpus.npy").astype(np.float32)
comb = json.load(open(f"{G}/02_msu/formdatabase.json"))
papers = np.array([r.get("paper_id") for r in comb])
pj = json.load(open(f"{G}/02_msu/llm_pairs.json"))
train_pos = [(int(a), int(b)) for a, b, *_ in pj.get("train_pos", [])]
test_pos  = [(int(a), int(b)) for a, b, *_ in pj.get("test_pos", [])]
N = X.shape[0]
print(f"[data] case1+case2  N={N}  papers={len(set(papers))}  train_pos={len(train_pos)} test_pos={len(test_pos)}")

def bz(space, pos, k=12):
    nn = NearestNeighbors(n_neighbors=k+1).fit(space); _, ind = nn.kneighbors(space)
    ng = [set(r[1:]) for r in ind]
    return sum(1 for a, b in pos if b in ng[a] or a in ng[b]) / max(1, len(pos))

rows = []

# ---- baselines ----
import umap
for name, Z in [("PCA", PCA(2).fit_transform(X)),
                ("t-SNE", TSNE(n_components=2, init="pca", random_state=0).fit_transform(X)),
                ("UMAP", umap.UMAP(n_components=2, random_state=0).fit_transform(X))]:
    tw, ko = evaluate_quick(X, Z, 12)
    rows.append((name, tw, ko, None, None)); print(f"[base] {name}: trust {tw:.3f} knnOv {ko:.3f}", flush=True)

# ---- train a model on case1+case2 (full batch) ----
def train(lam_con):
    set_seed(0)
    x = torch.tensor(X); P = torch.tensor(compute_joint_P(X, 30.0)); eye = torch.eye(N, dtype=torch.bool)
    pa = torch.tensor([a for a, b in train_pos]); pb = torch.tensor([b for a, b in train_pos])
    m = ResidualProjectionMapper(1024, 512, 4, 0.1); opt = torch.optim.Adam(m.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 1500); ns = 0.4 / (1024 ** 0.5)
    m.train()
    for it in range(1500):
        xin = x + ns * torch.randn_like(x); Z = m(xin); D2 = torch.cdist(Z, Z) ** 2
        num = (1 / (1 + D2)).masked_fill(eye, 0); Q = (num / num.sum().clamp_min(1e-12)).clamp_min(1e-12)
        Pe = P * 12 if it < 250 else P; Pn = (Pe / Pe.sum()).clamp_min(1e-12)
        loss = (Pn * (Pn.log() - Q.log())).sum()
        if lam_con > 0 and len(train_pos):
            Qr = (num / num.sum(1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-12)
            loss = loss + lam_con * 0.5 * (-Qr[pa, pb].log().mean() - Qr[pb, pa].log().mean())
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step(); sch.step()
    m.eval()
    with torch.no_grad(): return m(x).cpu().numpy()

for name, lam in [("Ours (no sup)", 0.0), ("Ours +sup (l=0.1)", 0.1)]:
    Z = train(lam)
    tw, ko = evaluate_quick(X, Z, 12); sil = float(silhouette_score(Z, papers)); br = bz(Z, test_pos)
    rows.append((name, tw, ko, sil, br)); print(f"[ours] {name}: trust {tw:.3f} knnOv {ko:.3f} paperSil {sil:.3f} BZ {br:.3f}", flush=True)

# ---- leak-free OOS (80/20 by MSU; parametric map + UMAP.transform) ----
rng = np.random.RandomState(7); idx = rng.permutation(N); tr = np.sort(idx[:int(N*0.8)]); te = np.sort(idx[int(N*0.8):])
Xtr, Xte = X[tr], X[te]
set_seed(0)
xtr = torch.tensor(Xtr); Ptr = torch.tensor(compute_joint_P(Xtr, 30.0)); eyetr = torch.eye(len(tr), dtype=torch.bool)
m = ResidualProjectionMapper(1024, 512, 4, 0.1); opt = torch.optim.Adam(m.parameters(), 1e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 1500); ns = 0.4 / (1024 ** 0.5); m.train()
for it in range(1500):
    xin = xtr + ns * torch.randn_like(xtr); Z = m(xin); D2 = torch.cdist(Z, Z) ** 2
    num = (1 / (1 + D2)).masked_fill(eyetr, 0); Q = (num / num.sum().clamp_min(1e-12)).clamp_min(1e-12)
    Pe = Ptr * 12 if it < 250 else Ptr; Pn = (Pe / Pe.sum()).clamp_min(1e-12)
    loss = (Pn * (Pn.log() - Q.log())).sum()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step(); sch.step()
m.eval()
with torch.no_grad(): Zte = m(torch.tensor(Xte)).cpu().numpy()
_, ours_oos = evaluate_quick(Xte, Zte, 12)
ureducer = umap.UMAP(n_components=2, random_state=0).fit(Xtr); Zte_u = ureducer.transform(Xte)
_, umap_oos = evaluate_quick(Xte, Zte_u, 12)
print(f"[OOS] leak-free (test={len(te)}): Ours knnOv {ours_oos:.3f}  vs  UMAP.transform {umap_oos:.3f}", flush=True)

print("\n================ SUMMARY (case1+case2) ================")
print(f"{'Method':20} {'Trust':>7} {'knnOv':>7} {'paperSil':>9} {'BZ@12':>7}")
for n, tw, ko, sil, br in rows:
    print(f"{n:20} {tw:7.3f} {ko:7.3f} {('' if sil is None else f'{sil:7.3f}'):>9} {('' if br is None else f'{br:7.3f}'):>7}")
print(f"\nOOS: Ours {ours_oos:.3f}  |  UMAP.transform {umap_oos:.3f}")
