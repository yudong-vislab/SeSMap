#!/usr/bin/env python3
"""SeSMap 全套投影评测：Table 1（方法对比）+ Table 2（子空间）+ 样本外。"""
import json, sys, time, collections
import numpy as np, torch
sys.path.insert(0, ".")
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from code_for_model.models import ResidualProjectionMapper, evaluate_quick, set_seed, compute_joint_P
import umap

PAIRS = sys.argv[1] if len(sys.argv) > 1 else "data/stages/02_msu/llm_pairs_10k.json"
BASE  = "data/stages"
K     = 12

X  = np.load(f"{BASE}/04_embeddings/emb_corpus.npy").astype(np.float32)
db  = json.load(open(f"{BASE}/02_msu/formdatabase.json"))
role = np.array([(r.get("category") or "None") for r in db][:X.shape[0]])
pj  = json.load(open(PAIRS))
tr  = [(int(a), int(b)) for a, b, *_ in pj["train_pos"]]
te  = [(int(a), int(b)) for a, b, *_ in pj["test_pos"]]
N   = X.shape[0]
print(f"[data] N={N}  train={len(tr)}  held-out={len(te)}  pairs={PAIRS}\n", flush=True)

def coloc(space, pos, k=K):
    if len(pos) == 0: return float("nan")
    nn = NearestNeighbors(n_neighbors=k+1).fit(space); _, ind = nn.kneighbors(space)
    ng = [set(r[1:]) for r in ind]
    return sum(1 for a, b in pos if b in ng[a] or a in ng[b]) / len(pos)

def train(Xtr, pairs, lam, iters=1500):
    n = Xtr.shape[0]; set_seed(0)
    x = torch.tensor(Xtr); P = torch.tensor(compute_joint_P(Xtr, 30.0))
    eye = torch.eye(n, dtype=torch.bool)
    m = ResidualProjectionMapper(Xtr.shape[1], 512, 4, 0.1)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    ns = 0.4 / (Xtr.shape[1] ** 0.5); m.train()
    if lam > 0 and pairs:
        pa = torch.tensor([a for a, b in pairs]); pb = torch.tensor([b for a, b in pairs])
    for it in range(iters):
        Z = m(x + ns * torch.randn_like(x)); D2 = torch.cdist(Z, Z) ** 2
        num = (1 / (1 + D2)).masked_fill(eye, 0)
        Q = (num / num.sum().clamp_min(1e-12)).clamp_min(1e-12)
        Pe = P * 12 if it < 250 else P; Pn = (Pe / Pe.sum()).clamp_min(1e-12)
        loss = (Pn * (Pn.log() - Q.log())).sum()
        if lam > 0 and pairs:
            Qr = (num / num.sum(1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-12)
            loss = loss + lam * 0.5 * (-Qr[pa, pb].log().mean() - Qr[pb, pa].log().mean())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step(); sch.step()
    m.eval()
    with torch.no_grad(): return m, m(torch.tensor(Xtr)).cpu().numpy()

print("="*74); print("TABLE 1  投影质量"); print("="*74)
print(f"{'Method':26}{'Trust':>9}{'knnOv@12':>11}{'Co-loc@12':>12}")
print(f"{'Encoder space (1024-D)':26}{'--':>9}{'--':>11}{coloc(X,te):>12.3f}", flush=True)
for nm, Z in [("PCA", PCA(2, random_state=0).fit_transform(X)),
              ("t-SNE (non-param.)", TSNE(2, init="pca", random_state=0).fit_transform(X)),
              ("UMAP", umap.UMAP(n_components=2, random_state=0).fit_transform(X))]:
    tw, ko = evaluate_quick(X, Z, K)
    print(f"{nm:26}{tw:>9.3f}{ko:>11.3f}{coloc(Z,te):>12.3f}", flush=True)
LAY = {}
for nm, lam in [("Ours (projection)", 0.0), ("  + correspondence sup.", 0.1)]:
    t0 = time.time(); _, Z = train(X, tr, lam); LAY[lam] = Z
    tw, ko = evaluate_quick(X, Z, K)
    print(f"{nm:26}{tw:>9.3f}{ko:>11.3f}{coloc(Z,te):>12.3f}   [{time.time()-t0:.0f}s]", flush=True)
    np.save(f"/tmp/Z10k_lam{lam}.npy", Z)

Z0, Z1 = LAY[0.0], LAY[0.1]
print("\n" + "="*74); print("§4.3 诊断：高维 vs 二维"); print("="*74)
print(f"  encoder space : {coloc(X,te):.3f}")
print(f"  after project : {coloc(Z0,te):.3f}", flush=True)

print("\n" + "="*74); print("TABLE 2a  同角色 vs 跨角色"); print("="*74)
same  = [(a,b) for a,b in te if role[a]==role[b]]
cross = [(a,b) for a,b in te if role[a]!=role[b]]
print(f"{'Pair type':16}{'n':>7}{'Ours':>9}{'+sup':>9}{'gain':>9}")
for nm, ps in [("Same-role",same),("Cross-role",cross),("All",te)]:
    c0,c1 = coloc(Z0,ps), coloc(Z1,ps)
    print(f"{nm:16}{len(ps):>7}{c0:>9.3f}{c1:>9.3f}{(c1/c0-1)*100:>8.0f}%", flush=True)

print("\n" + "="*74); print("TABLE 2b  子空间视图 vs 全局视图（同角色对）"); print("="*74)
print(f"{'Subspace':14}{'MSUs':>7}{'pairs':>7}{'Trust':>9}{'knnOv':>9}{'global':>9}{'in-sub':>9}")
for r in ["Background","Method","Experiment","Result","Conclusion"]:
    m = np.where(role==r)[0]
    ps = [(a,b) for a,b in same if role[a]==r]
    if len(m) < 60: continue
    tw, ko = evaluate_quick(X[m], Z1[m], K)
    idx = {g:i for i,g in enumerate(m)}
    loc = [(idx[a],idx[b]) for a,b in ps]
    g  = coloc(Z1, ps) if ps else float("nan")
    sv = coloc(Z1[m], loc) if ps else float("nan")
    print(f"{r:14}{len(m):>7}{len(ps):>7}{tw:>9.3f}{ko:>9.3f}{g:>9.3f}{sv:>9.3f}", flush=True)

print("\n" + "="*74); print("样本外（无泄漏 80/20 按 MSU 划分）"); print("="*74)
rng = np.random.RandomState(7); perm = rng.permutation(N)
tr_i = np.sort(perm[:int(N*0.8)]); te_i = np.sort(perm[int(N*0.8):])
pos = {i:k for k,i in enumerate(tr_i)}
sub = [(pos[a],pos[b]) for a,b in tr if a in pos and b in pos]
m, _ = train(X[tr_i], sub, 0.1)
with torch.no_grad(): Zte = m(torch.tensor(X[te_i])).cpu().numpy()
_, ours = evaluate_quick(X[te_i], Zte, K)
u = umap.UMAP(n_components=2, random_state=0).fit(X[tr_i])
_, um = evaluate_quick(X[te_i], u.transform(X[te_i]), K)
print(f"  held-out MSUs={len(te_i)}   Ours knnOv={ours:.3f}   UMAP.transform knnOv={um:.3f}")
