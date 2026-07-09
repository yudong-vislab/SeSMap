#!/usr/bin/env python3
"""
方案A2扫描：paper one-hot 特征增广 UMAP（保序的同论文弱引力）。

做法：X_aug = [X, γ·onehot(paper_id)] -> UMAP。
效果：所有跨论文距离² 统一 +2γ²（同论文不变）——
  * 跨论文对的相对排序完全保留：真趋同(原距离小)仍可共址 => BZ 保留；
  * 弱跨论文噪声被推出邻域 => 论文成块、可读性提升。
对照失败的方案A(categorical target_weight)：那个把跨论文边一刀切杀光(保存率 0.113->0.01)。

γ 以"同论文第12近邻距离中位数 m"为尺度：gamma = c·m, c ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.5}。

指标：
  保真: trust@12, knnOv@12（相对原始 1024 维 X，不是增广向量）
  可读: paperSil, purity@12
  卖点: BZ_all   = 跨论文高维近邻的 2D 保存率（对照 w=0 的 0.113）
        BZ_strong= 最强跨论文语义对(top-1000 cosine)的 2D 共址率(互为12近邻或距离<同论文近邻中位)
"""
import sys, json
from pathlib import Path
import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import local_config as cfg

from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

K = 12
CS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
UMAP_KW = dict(n_components=2, n_neighbors=15, min_dist=0.2, spread=1.5,
               metric="euclidean", random_state=0)


def knn_idx(M, k=K):
    return NearestNeighbors(n_neighbors=k+1).fit(M).kneighbors(return_distance=False)[:, 1:]


def main():
    import umap
    X = np.load(str(cfg.EMB_CACHE)).astype(np.float32)
    ids = json.loads(Path(str(cfg.EMB_CACHE) + ".ids.json").read_text())
    recs = json.loads(Path(cfg.FORMDB).read_text(encoding="utf-8"))
    pid = {int(r.get("idx", i)): int(r.get("paper_id", -1))
           for i, r in enumerate(recs) if str(r.get("sentence", "")).strip()}
    y = np.array([pid[int(i)] for i in ids])
    papers = sorted(set(y.tolist()))
    N = len(X)
    print(f"N={N} papers={papers}")

    # 高维参照
    hd = knn_idx(X)
    cross = {i: set(int(j) for j in hd[i] if y[j] != y[i]) for i in range(N)}
    bz_pts = [i for i, c in cross.items() if c]
    # 同论文第12近邻距离中位数 m（γ 的尺度）
    d12 = NearestNeighbors(n_neighbors=K+1).fit(X).kneighbors(X)[0][:, -1]
    m = float(np.median(d12))
    print(f"scale m (median same-corpus 12NN dist) = {m:.3f}")

    # 最强跨论文语义对 top-1000（cosine）
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = Xn @ Xn.T
    iu = np.triu_indices(N, 1)
    mask = y[iu[0]] != y[iu[1]]
    ci, cj, cs_ = iu[0][mask], iu[1][mask], S[iu][mask]
    top = np.argsort(-cs_)[:1000]
    strong_pairs = list(zip(ci[top].tolist(), cj[top].tolist()))
    print(f"strong cross-paper pairs: top-1000, cosine range [{cs_[top].min():.3f}, {cs_[top].max():.3f}]")

    onehot = np.zeros((N, len(papers)), dtype=np.float32)
    for k_, p in enumerate(papers):
        onehot[y == p, k_] = 1.0

    results = []
    print(f"\n{'c':>5} {'gamma':>6} | {'trust':>6} {'knnOv':>6} | {'papSil':>7} {'pur@12':>7} | {'BZ_all':>7} {'BZ_strong':>9}")
    for c in CS:
        gamma = c * m
        Xa = np.hstack([X, gamma * onehot]) if c > 0 else X
        Z = umap.UMAP(**UMAP_KW).fit_transform(Xa).astype(np.float32)

        ld = knn_idx(Z)
        tw = float(trustworthiness(X, Z, n_neighbors=K))
        ko = float(np.mean([len(set(hd[i]) & set(ld[i])) / K for i in range(N)]))
        ps = float(silhouette_score(Z, y))
        pu = float(np.mean([(y[nb] == y[i]).mean() for i, nb in enumerate(ld)]))
        bz_all = float(np.mean([len(cross[i] & set(int(j) for j in ld[i])) / len(cross[i]) for i in bz_pts]))
        # strong 对共址：互在对方 2D 12近邻中 记 1，或 2D 距离 < 全局 2D 12NN 距离中位数
        ld_sets = [set(int(j) for j in ld[i]) for i in range(N)]
        d2d12 = NearestNeighbors(n_neighbors=K+1).fit(Z).kneighbors(Z)[0][:, -1]
        r2d = float(np.median(d2d12))
        co = 0
        for a, b in strong_pairs:
            if (b in ld_sets[a]) or (a in ld_sets[b]) or (np.linalg.norm(Z[a] - Z[b]) < r2d):
                co += 1
        bz_strong = co / len(strong_pairs)

        row = dict(c=c, gamma=gamma, trust=tw, knnOv=ko, paperSil=ps, purity=pu,
                   BZ_all=bz_all, BZ_strong=bz_strong)
        results.append(row)
        np.save(str(cfg.MODEL_DIR / f"sweep_feat_c{c:.2f}.npy"), Z)
        print(f"{c:>5.2f} {gamma:>6.2f} | {tw:>6.3f} {ko:>6.3f} | {ps:>7.3f} {pu:>7.3f} | {bz_all:>7.3f} {bz_strong:>9.3f}")

    out = cfg.OUTPUT_DIR / "sweep_paper_feature.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
