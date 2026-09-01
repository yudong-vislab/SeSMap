#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_contrastive.py — 用同一把尺子评测任意 checkpoint（v10 / v11 各臂），做公平对比。

报告三类指标（held-out test 正对，不参与训练，避免循环）：
  * 保真：trust / knnOv@12          —— 越高越好；混合目标不应显著低于 v10。
  * BZ 召回：bz_recall@12 (2D)       —— 越高越好；这是方案B 想赢 v10 的地方。
             对照 bz_recall@12 (highD) —— bge 自身的天花板参考。
  * 论文分离：paperSil               —— 应保持低（我们不想把论文推开）。

用法：
  python3 code_for_model/eval_contrastive.py --ckpt data/stages/05_model/bert2d_mapper_all_v10.pt \
      --pairs data/stages/02_msu/llm_pairs.json
  python3 code_for_model/eval_contrastive.py --ckpt data/stages/05_model/bert2d_mapper_all_v11_hybrid.pt \
      --pairs data/stages/02_msu/llm_pairs.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import local_config as cfg
from code_for_model.models import ResidualProjectionMapper, load_embeddings, evaluate_quick


def project(ckpt_path, x_np, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    m = ResidualProjectionMapper(
        embed_dim=int(ckpt["embed_dim"]), width=int(ckpt.get("width", 512)),
        num_blocks=int(ckpt.get("num_blocks", 4)), dropout=float(ckpt.get("dropout", 0.0)),
        out_dim=int(ckpt.get("out_dim", 2))).to(device)
    m.load_state_dict(ckpt["mapper_state"]); m.eval()
    with torch.no_grad():
        z = m(torch.tensor(x_np, dtype=torch.float32, device=device)).cpu().numpy()
    return z, ckpt.get("objective", "")


def bz_recall(space, pairs, k):
    from sklearn.neighbors import NearestNeighbors
    if not pairs:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(space)
    _, ind = nn.kneighbors(space)
    neigh = [set(row[1:]) for row in ind]
    hit = sum(1 for a, b in pairs if b in neigh[a] or a in neigh[b])
    return hit / len(pairs)


def paper_sil(z_np, papers):
    from sklearn.metrics import silhouette_score
    if len(set(papers)) < 2:
        return float("nan")
    return float(silhouette_score(z_np, papers))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=cfg.EMB_CACHE)
    ap.add_argument("--corpus", type=Path, default=cfg.FORMDB_V2)
    ap.add_argument("--pairs", type=Path, default=cfg.MSU_DIR / "llm_pairs.json")
    ap.add_argument("--k", type=int, default=12)
    args = ap.parse_args()

    x_np, ids = load_embeddings(args.cache)
    z_np, obj = project(args.ckpt, x_np)
    N = x_np.shape[0]

    # held-out 正对（行索引）
    test_pos = []
    if args.pairs.exists():
        d = json.load(open(args.pairs, encoding="utf-8"))
        test_pos = [(int(a), int(b)) for a, b, *_ in d.get("test_pos", [])
                    if 0 <= int(a) < N and 0 <= int(b) < N]

    # 论文标签（按 ids 对齐 corpus 的 paper_id）
    recs = json.load(open(args.corpus, encoding="utf-8"))
    by_id = {r.get("idx", i): r for i, r in enumerate(recs)}
    papers = np.array([by_id.get(rid, {}).get("paper_id", -1) for rid in ids])

    tw, ko = evaluate_quick(x_np, z_np, args.k)
    print(f"\n=== {args.ckpt.name} ===")
    print(f"objective : {obj[:90]}")
    print(f"trust@{args.k}        : {tw:.3f}")
    print(f"knnOv@{args.k}        : {ko:.3f}   (保真, 越高越好)")
    print(f"paperSil          : {paper_sil(z_np, papers):.3f}   (论文分离, 应保持低)")
    print(f"bz_recall@{args.k} 2D  : {bz_recall(z_np, test_pos, args.k):.3f}   (方案B 目标, 越高越好; n={len(test_pos)})")
    print(f"bz_recall@{args.k} highD: {bz_recall(x_np, test_pos, args.k):.3f}   (bge 天花板参考)")


if __name__ == "__main__":
    main()
