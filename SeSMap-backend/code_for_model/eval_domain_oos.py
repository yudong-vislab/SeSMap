#!/usr/bin/env python3
"""
eval_domain_oos.py — 未见领域(L3)泛化评测。

场景：v10r 在旧领域(污染+时空可视化 6 篇)上训练；现将全新领域(如生信/环形基因组
可视化)的 MSU 经冻结 MLP 投影进坐标系，检验"参数化投影泛化到未见领域"的主张。

对照三方（指标均在新领域点内部算 trust/knnOv@12——"这批新论文的地图对不对"）：
  * Ours(v10r)       : 冻结 MLP 前向投影（论文主张）
  * UMAP.transform   : UMAP fit(训练语料 2689) -> transform(新领域)（自然 baseline）
  * UMAP(fresh)      : 直接在新领域上重新拟合 UMAP（"重建地图"参照上限，非参数、
                       坐标系与旧地图不兼容——正是我们的方法要避免的）

用法（先完成 mineru->build_corpus->precompute->formdatabase 四步，见 runbook）:
  python3 code_for_model/eval_domain_oos.py \
      --domain-formdb data/bio_eval/stages/02_msu/formdatabase_v2.0.json \
      --domain-cache  data/bio_eval/stages/04_embeddings/emb_corpus.npy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import local_config as cfg

from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors


def knn_overlap(X, Z, k):
    ix = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(return_distance=False)[:, 1:]
    iz = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(return_distance=False)[:, 1:]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ix, iz)]))


def load_domain(formdb: Path, cache: Path):
    recs = json.loads(formdb.read_text(encoding="utf-8"))
    by_idx = {int(r.get("idx", i)): r for i, r in enumerate(recs)
              if str(r.get("sentence", "")).strip() and "2d_coord" in r}
    ids = json.loads(Path(str(cache) + ".ids.json").read_text())
    X_all = np.load(str(cache)).astype(np.float32)
    rows, Z = [], []
    for k_, mid in enumerate(ids):
        if int(mid) in by_idx:
            rows.append(k_)
            Z.append(by_idx[int(mid)]["2d_coord"])
    return X_all[rows], np.asarray(Z, dtype=np.float64), [int(ids[r]) for r in rows], by_idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain-formdb", type=Path, required=True, help="新领域 formdatabase_v2.0.json(含 v10r 坐标)")
    ap.add_argument("--domain-cache", type=Path, required=True, help="新领域 emb_corpus.npy")
    ap.add_argument("--train-cache", type=Path, default=cfg.EMB_CACHE, help="训练语料向量缓存(UMAP.transform 用)")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--out", type=Path, default=cfg.OUTPUT_DIR / "domain_oos_eval.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Xd, Zd, ids_d, by_idx = load_domain(args.domain_formdb, args.domain_cache)
    n = len(Xd)
    papers = sorted({int(by_idx[m].get("paper_id", -1)) for m in ids_d})
    k = min(args.k, n - 2)
    print(f"[domain] N={n} papers={papers} k={k} chance≈{k/n:.4f}")

    rows = {}
    # ---- Ours(v10r): 冻结前向(坐标已在 formdb 里) ----
    rows["Ours(v10r)"] = dict(trust=float(trustworthiness(Xd, Zd, n_neighbors=k)),
                              knnOv=knn_overlap(Xd, Zd, k))

    # ---- UMAP.transform: fit(训练语料) -> transform(新领域) ----
    import umap
    Xtr = np.load(str(args.train_cache)).astype(np.float32)
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric="euclidean", random_state=args.seed)
    reducer.fit(Xtr)
    Zu = reducer.transform(Xd).astype(np.float64)
    rows["UMAP.transform"] = dict(trust=float(trustworthiness(Xd, Zu, n_neighbors=k)),
                                  knnOv=knn_overlap(Xd, Zu, k))

    # ---- UMAP(fresh): 新领域上重拟合（参照上限，坐标系不兼容旧地图） ----
    Zf = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                   metric="euclidean", random_state=args.seed).fit_transform(Xd).astype(np.float64)
    rows["UMAP(fresh)"] = dict(trust=float(trustworthiness(Xd, Zf, n_neighbors=k)),
                               knnOv=knn_overlap(Xd, Zf, k))

    hdr = f"{'method':16} | {'trust':>6} {'knnOv':>6}"
    print("\n" + hdr); print("-" * len(hdr))
    for name, r in rows.items():
        print(f"{name:16} | {r['trust']:6.3f} {r['knnOv']:6.3f}")

    out = {"n": n, "papers": papers, "k": k, "chance": round(k / n, 5), "results": rows,
           "note": "Ours/transform 投影进旧坐标系(可与已有地图共用)；fresh 为重建参照(坐标系不兼容)。"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
