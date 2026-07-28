#!/usr/bin/env python3
"""
方案A扫描：半监督 UMAP target 的 readability–faithfulness tradeoff。

对 target_weight w ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5}:
  用 UMAP(y=paper_id, target_metric='categorical', target_weight=w) 造 target 布局
  （其余参数与 v8_readable 一致: n_neighbors=15, min_dist=0.2, spread=1.5），
  对每个布局算:
    保真主轴: trust@12, knnOv@12  (相对 1024 维 bge)
    可读辅轴: paperSil, purity@12 (paper_id 当 label)
  输出 JSON + tradeoff 曲线 PNG。

说明：扫描直接评 target 布局本身（蒸馏只是复制 target，v8 显示退化很小
0.260->0.252），选定拐点 w 后再蒸馏成 v9 即可。约束只进 target；
mapper 输入仍只有句向量，不需要 paper 标签 —— OOS 投影能力不受影响。
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
WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
UMAP_KW = dict(n_components=2, n_neighbors=15, min_dist=0.2, spread=1.5,
               metric="euclidean", random_state=0)


def knn_overlap(X, Z, k=K):
    ix = NearestNeighbors(n_neighbors=k+1).fit(X).kneighbors(return_distance=False)[:, 1:]
    iz = NearestNeighbors(n_neighbors=k+1).fit(Z).kneighbors(return_distance=False)[:, 1:]
    return float(np.mean([len(set(a) & set(b))/k for a, b in zip(ix, iz)]))


def purity(Z, labels, k=K):
    idx = NearestNeighbors(n_neighbors=k+1).fit(Z).kneighbors(return_distance=False)[:, 1:]
    return float(np.mean([(labels[nb] == labels[i]).mean() for i, nb in enumerate(idx)]))


def main():
    import umap
    ids = json.loads(Path(str(cfg.EMB_CACHE) + ".ids.json").read_text())
    X = np.load(str(cfg.EMB_CACHE)).astype(np.float32)
    recs = json.loads(Path(cfg.FORMDB).read_text(encoding="utf-8"))
    pid_by_idx = {int(r.get("idx", i)): int(r.get("paper_id", -1))
                  for i, r in enumerate(recs) if str(r.get("sentence", "")).strip()}
    y = np.array([pid_by_idx[int(i)] for i in ids])
    print(f"N={len(X)}  papers={sorted(set(y.tolist()))}")

    results = []
    for w in WEIGHTS:
        red = umap.UMAP(target_weight=w, target_metric="categorical", **UMAP_KW)
        Z = red.fit_transform(X, y=y) if w > 0 else red.fit_transform(X)
        Z = np.asarray(Z, dtype=np.float32)
        row = dict(w=w,
                   trust=float(trustworthiness(X, Z, n_neighbors=K)),
                   knnOv=knn_overlap(X, Z),
                   paperSil=float(silhouette_score(Z, y)),
                   purity=purity(Z, y))
        results.append(row)
        np.save(str(cfg.MODEL_DIR / f"sweep_target_w{w:.1f}.npy"), Z)
        print(f"w={w:.1f}  trust={row['trust']:.3f}  knnOv={row['knnOv']:.3f}  "
              f"paperSil={row['paperSil']:.3f}  purity@12={row['purity']:.3f}")

    out = cfg.OUTPUT_DIR / "sweep_target_weight.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved -> {out}")

    try:
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ws = [r["w"] for r in results]
        ax1.plot(ws, [r["knnOv"] for r in results], "o-", color="tab:blue", label="knnOv@12 (faithfulness)")
        ax1.plot(ws, [r["trust"] for r in results], "s--", color="tab:cyan", label="trust@12")
        ax1.set_xlabel("target_weight w (paper-label supervision)")
        ax1.set_ylabel("faithfulness", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(ws, [r["paperSil"] for r in results], "^-", color="tab:red", label="paperSil (readability)")
        ax2.plot(ws, [r["purity"] for r in results], "v--", color="tab:orange", label="purity@12")
        ax2.set_ylabel("paper separation", color="tab:red")
        h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc="center right", fontsize=8)
        ax1.grid(True, alpha=0.3); plt.title("Readability-Faithfulness tradeoff (semi-supervised UMAP target)")
        plt.tight_layout(); plt.savefig(cfg.OUTPUT_DIR / "sweep_target_weight.png", dpi=150)
        print(f"saved -> {cfg.OUTPUT_DIR / 'sweep_target_weight.png'}")
    except Exception as e:
        print("skip plot:", e)


if __name__ == "__main__":
    main()
