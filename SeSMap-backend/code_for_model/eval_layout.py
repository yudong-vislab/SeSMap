# eval_layout.py
# ---------------------------------------------------------------------------
# 在“已有数据”上评估当前 v3 布局，无需重训。
# 对每个 case:
#   1) 读 database-*.json -> sentence / paper_id / category(role) / 2d_coord(=当前v3输出)
#   2) 用本地 bge 把 sentence 重新编码成 1024 维 X
#   3) 基线布局 PCA / t-SNE / UMAP 均从同一 X 计算；"Ours(v3)" 用现成 2d_coord
#   4) 对每种布局算三组指标：
#        A. 语义保真(相对 1024维 X): trustworthiness@k, continuity@k, knn_overlap@k   (越高越好)
#        B. role 连贯(role 当 label):  role_silhouette, role_NMI                       (越高越好)
#        C. paper 局部(paper 当 label): paper_silhouette, paper_NMI, purity@k, entropy@k
# 说明: case 论文数很少(2/5)且可能在训练集内 -> 结果是“诊断/原型”，不是最终可发表数字。
# ---------------------------------------------------------------------------

import os, sys, json, glob, argparse
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

BGE_PATH = str(cfg.BGE_MODEL_PATH)
CACHE_DIR = str(cfg.OUTPUT_DIR / "eval_cache")
K = 12  # neighborhood size, 与论文 Purity@12 一致


def load_case(case_dir):
    sents, papers, roles, coords = [], [], [], []
    for f in sorted(glob.glob(os.path.join(case_dir, "database-*.json"))):
        for r in json.load(open(f)):
            if "sentence" not in r or "2d_coord" not in r:
                continue
            sents.append(r["sentence"])
            papers.append(int(r.get("paper_id", -1)))
            roles.append(str(r.get("category", "NA")).lower())
            coords.append(r["2d_coord"])
    return sents, np.array(papers), np.array(roles), np.asarray(coords, dtype=float)


def embed(sents, cache_key):
    cache = os.path.join(CACHE_DIR, f"emb_{cache_key}.npy")
    if os.path.exists(cache):
        X = np.load(cache)
        if X.shape[0] == len(sents):
            print(f"  [emb] loaded cache {cache} {X.shape}")
            return X
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(BGE_PATH, device="cpu")
    X = m.encode(sents, batch_size=32, show_progress_bar=True,
                 convert_to_numpy=True, normalize_embeddings=False)  # raw, 与训练一致
    np.save(cache, X)
    print(f"  [emb] encoded {X.shape} -> {cache}")
    return X


def knn_overlap(X, Z, k=K):
    nx = NearestNeighbors(n_neighbors=k + 1).fit(X)
    nz = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    ix = nx.kneighbors(return_distance=False)[:, 1:]
    iz = nz.kneighbors(return_distance=False)[:, 1:]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ix, iz)]))


def purity_entropy(Z, labels, k=K):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    idx = nn.kneighbors(return_distance=False)[:, 1:]
    pur, ent = [], []
    for i, nb in enumerate(idx):
        lab = labels[nb]
        vals, cnt = np.unique(lab, return_counts=True)
        p = cnt / cnt.sum()
        pur.append((labels[i] == lab).mean())
        ent.append(-np.sum(p * np.log(p + 1e-12)))
    return float(np.mean(pur)), float(np.mean(ent))


def nmi_kmeans(Z, labels):
    k = len(np.unique(labels))
    if k < 2:
        return float("nan")
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)
    return float(normalized_mutual_info_score(labels, km))


def sil(Z, labels):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(Z, labels))


def build_layouts(X, ours):
    return {
        "Ours(v3)": ours,
        "PCA":  PCA(n_components=2, random_state=0).fit_transform(X),
        "tSNE": TSNE(n_components=2, init="pca", random_state=0).fit_transform(X),
        "UMAP": __import__("umap").UMAP(n_components=2, random_state=0).fit_transform(X),
    }


def eval_case(case_dir, name):
    print(f"\n########## {name}  ({case_dir}) ##########")
    sents, papers, roles, ours = load_case(case_dir)
    print(f"  MSUs={len(sents)}  papers={sorted(set(papers.tolist()))}  roles={sorted(set(roles))}")
    X = embed(sents, name)
    layouts = build_layouts(X, ours)

    header = f"{'layout':10} | {'trust':>6} {'cont':>6} {'knnOv':>6} | {'roleSil':>7} {'roleNMI':>7} | {'papSil':>7} {'papNMI':>7} {'pur@12':>7} {'ent@12':>7}"
    print("\n" + header); print("-" * len(header))
    rows = {}
    for lname, Z in layouts.items():
        Z = np.asarray(Z, dtype=float)
        tw = trustworthiness(X, Z, n_neighbors=K)
        co = trustworthiness(Z, X, n_neighbors=K)          # continuity ≈ swapped trustworthiness
        ko = knn_overlap(X, Z, K)
        rsil, rnmi = sil(Z, roles), nmi_kmeans(Z, roles)
        psil, pnmi = sil(Z, papers), nmi_kmeans(Z, papers)
        pur, ent = purity_entropy(Z, papers, K)
        rows[lname] = dict(trust=tw, cont=co, knnOv=ko, roleSil=rsil, roleNMI=rnmi,
                           papSil=psil, papNMI=pnmi, pur=pur, ent=ent)
        print(f"{lname:10} | {tw:6.3f} {co:6.3f} {ko:6.3f} | {rsil:7.3f} {rnmi:7.3f} | "
              f"{psil:7.3f} {pnmi:7.3f} {pur:7.3f} {ent:7.3f}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dirs", nargs="*",
                        default=[str(cfg.CASE_ROOT / "case1"), str(cfg.CASE_ROOT / "case2"), str(cfg.CASE_ROOT / "case3")])
    parser.add_argument("--out", default=str(cfg.OUTPUT_DIR / "eval_layout_results.json"))
    args = parser.parse_args()
    os.makedirs(CACHE_DIR, exist_ok=True)
    results = {}
    for d in args.case_dirs:
        case = os.path.basename(os.path.normpath(d))
        if os.path.isdir(d):
            results[case] = eval_case(d, case)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nSaved results -> {args.out}")
    print("\n注: case 论文数少且可能在训练集内 -> 诊断/原型结论，非最终可发表数字。")
