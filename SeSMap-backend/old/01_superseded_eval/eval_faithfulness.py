#!/usr/bin/env python3
"""
Corpus-level faithfulness eval (no re-encode; uses the bge cache).
比较 v5 / (可选)v3 / PCA / tSNE / UMAP 相对 1024 维 bge 语义空间的邻域保真：
  trust@12, knnOv@12   (越高越好)   + role 连贯(roleSil/roleNMI)
回答：v5 有没有把 v3 崩掉的 knnOv(~0.04) 拉回到 t-SNE 水平。

用法:
  python3 code_for_model/eval_faithfulness.py \
      --v5 data/stages/02_msu/formdatabase_v2.0.json \
      --v3 data/stages/02_msu/formdatabase.json
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import local_config as cfg
K = 12


def load_cache():
    ids = json.load(open(str(cfg.EMB_CACHE) + ".ids.json", encoding="utf-8"))
    X = np.load(str(cfg.EMB_CACHE)).astype(np.float32)
    return {int(i): X[k] for k, i in enumerate(ids)}


def coords_by_idx(path):
    d = json.load(open(path, encoding="utf-8"))
    return {int(r.get("idx", i)): r["2d_coord"] for i, r in enumerate(d) if "2d_coord" in r}


def roles_by_idx(path):
    d = json.load(open(path, encoding="utf-8"))
    return {int(r.get("idx", i)): str(r.get("category", "NA")).lower()
            for i, r in enumerate(d) if str(r.get("sentence", "")).strip()}


def knn_overlap(X, Z, k=K):
    ix = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(return_distance=False)[:, 1:]
    iz = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(return_distance=False)[:, 1:]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ix, iz)]))


def role_scores(Z, roles):
    labs = np.array(roles)
    if len(set(roles)) < 2:
        return float("nan"), float("nan")
    sil = float(silhouette_score(Z, labs))
    km = KMeans(n_clusters=len(set(roles)), n_init=10, random_state=0).fit_predict(Z)
    return sil, float(normalized_mutual_info_score(labs, km))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5", "--candidate", dest="candidate", default=str(cfg.FORMDB_V2))
    ap.add_argument("--candidate-label", default="Ours(v5)")
    ap.add_argument("--v3", default=str(cfg.FORMDB))  # 仍存旧 v3 坐标（未被覆盖）
    args = ap.parse_args()

    emb = load_cache()
    v5 = coords_by_idx(args.candidate)
    roles_map = roles_by_idx(cfg.FORMDB)
    ids = [i for i in v5 if i in emb and i in roles_map]
    X = np.array([emb[i] for i in ids])
    roles = [roles_map[i] for i in ids]

    layouts = {
        args.candidate_label: np.array([v5[i] for i in ids]),
        "PCA":  PCA(n_components=2, random_state=0).fit_transform(X),
        "tSNE": TSNE(n_components=2, init="pca", random_state=0).fit_transform(X),
        "UMAP": __import__("umap").UMAP(n_components=2, random_state=0).fit_transform(X),
    }
    if args.v3 and Path(args.v3).exists():
        v3 = coords_by_idx(args.v3)
        if all(i in v3 for i in ids):
            layouts = {"Ours(v3-old)": np.array([v3[i] for i in ids]), **layouts}

    print(f"corpus MSUs={len(ids)}  K={K}\n")
    hdr = f"{'layout':13} | {'trust':>6} {'knnOv':>6} | {'roleSil':>7} {'roleNMI':>7}"
    print(hdr); print("-" * len(hdr))
    for name, Z in layouts.items():
        Z = np.asarray(Z, dtype=float)
        tw = trustworthiness(X, Z, n_neighbors=K)
        ko = knn_overlap(X, Z)
        rs, rn = role_scores(Z, roles)
        print(f"{name:13} | {tw:6.3f} {ko:6.3f} | {rs:7.3f} {rn:7.3f}")
    print("\n判据: v5 的 trust/knnOv 应追平 tSNE/UMAP(远高于 v3-old);"
          " 这样才证明语义优先重构把保真度找回来了。")


if __name__ == "__main__":
    main()
