#!/usr/bin/env python3
"""
eval_oos_v10.py — 干净的 out-of-sample 泛化评测（修复旧 holdout 的泄漏）。

协议（与旧 eval_holdout_projection 的区别）：
  * 旧：UMAP target 在全量 2689 上拟合（测试点参与塑形布局）-> 轻微泄漏；
  * 新：一切只用 train(80%)：v10 的全局 P 只在 train 上计算、mapper 只见 train；
        test(20%) 句子仅在评测时经冻结 MLP 前向投影 —— 零泄漏。

对照（reviewer 必问的 OOS baseline）：
  * UMAP.transform()：UMAP 只 fit train，再 transform(test)。
  * chance 参考：k/N ≈ 0.0045。

指标：knnOv_rows@12 —— 对指定行(test 或 train)，其 2D 12近邻 与 1024 维 bge 12近邻
     的平均重叠率；两个空间的近邻均在全部 2689 点上计算（评"落进整张地图的对不对"）。

用法：python3 code_for_model/eval_oos_v10.py   （~20 分钟，v10 需在 train 上重训）
输出：data/outputs/oos_eval_v10.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import local_config as cfg
from code_for_model.models import ResidualProjectionMapper, set_seed, load_embeddings, compute_joint_P

from sklearn.neighbors import NearestNeighbors


def knn_overlap_rows(x_all: np.ndarray, z_all: np.ndarray, rows: np.ndarray, k: int) -> float:
    ix = NearestNeighbors(n_neighbors=k + 1).fit(x_all).kneighbors(x_all[rows], return_distance=False)
    iz = NearestNeighbors(n_neighbors=k + 1).fit(z_all).kneighbors(z_all[rows], return_distance=False)
    vals = []
    for r, a, b in zip(rows, ix, iz):
        a = [int(i) for i in a if int(i) != int(r)][:k]
        b = [int(i) for i in b if int(i) != int(r)][:k]
        vals.append(len(set(a) & set(b)) / k)
    return float(np.mean(vals))


def train_v10_on(x_tr: np.ndarray, args, device) -> ResidualProjectionMapper:
    """在 train 子集上重训 v10（全局 P 只来自 train —— 无泄漏）。"""
    P_np = compute_joint_P(x_tr, args.perplexity)
    x = torch.tensor(x_tr, dtype=torch.float32, device=device)
    P = torch.tensor(P_np, dtype=torch.float32, device=device)
    n = x_tr.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=device)

    mapper = ResidualProjectionMapper(embed_dim=x_tr.shape[1], width=args.width,
                                      num_blocks=args.blocks, dropout=args.dropout).to(device)
    opt = optim.Adam(mapper.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)
    # 输入抖动:每步给 bge 向量加高斯噪声(期望范数 = args.input_noise),
    # 强迫映射局部平滑 -> 直接优化"邻近新句子落在邻近位置"的泛化性质。
    noise_std = args.input_noise / (x_tr.shape[1] ** 0.5) if args.input_noise > 0 else 0.0
    mapper.train()
    for it in range(args.iters):
        Pe = P * args.exaggeration if it < args.exaggeration_iters else P
        xin = x + noise_std * torch.randn_like(x) if noise_std > 0 else x
        Z = mapper(xin)
        D2 = torch.cdist(Z, Z) ** 2
        num = (1.0 / (1.0 + D2)).masked_fill(eye, 0.0)
        Q = (num / num.sum().clamp_min(1e-12)).clamp_min(1e-12)
        Pn = (Pe / Pe.sum()).clamp_min(1e-12)
        loss = (Pn * (Pn.log() - Q.log())).sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(mapper.parameters(), 5.0)
        opt.step(); sched.step()
        if it == 0 or (it + 1) % 300 == 0:
            print(f"  [v10-train] iter {it+1}/{args.iters} KL={float(loss.item()):.4f}")
    mapper.eval()
    return mapper


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=cfg.EMB_CACHE)
    ap.add_argument("--out", type=Path, default=cfg.OUTPUT_DIR / "oos_eval_v10.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=12)
    # v10 超参（与生产配置一致）
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--exaggeration", type=float, default=12.0)
    ap.add_argument("--exaggeration-iters", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--input-noise", type=float, default=0.0,
                    help="训练时输入抖动的期望范数(0=关)。参考:语料 12NN 距离中位数≈0.785")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    x_np, ids = load_embeddings(args.cache)
    n = x_np.shape[0]
    perm = np.random.default_rng(args.seed).permutation(n)
    n_test = max(1, int(round(n * args.test_frac)))
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    print(f"[oos] N={n} train={len(train_idx)} test={len(test_idx)} chance≈{args.k/n:.4f}")

    # ---- Ours(v10): train-only P + train-only mapper；test 仅前向投影 ----
    print("[oos] training v10 on train split (leak-free)...")
    mapper = train_v10_on(x_np[train_idx], args, device)
    with torch.no_grad():
        z_ours = mapper(torch.tensor(x_np, dtype=torch.float32, device=device)).cpu().numpy()

    ours_train = knn_overlap_rows(x_np, z_ours, train_idx, args.k)
    ours_test = knn_overlap_rows(x_np, z_ours, test_idx, args.k)
    print(f"[oos] Ours(v10)      train_knnOv={ours_train:.3f}  test_knnOv={ours_test:.3f}")

    # ---- Baseline: UMAP fit(train) + transform(test) ----
    print("[oos] UMAP fit(train) + transform(test)...")
    import umap
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric="euclidean", random_state=args.seed)
    z_tr = reducer.fit_transform(x_np[train_idx])
    z_te = reducer.transform(x_np[test_idx])
    z_umap = np.zeros((n, 2), dtype=np.float32)
    z_umap[train_idx] = z_tr
    z_umap[test_idx] = z_te

    umap_train = knn_overlap_rows(x_np, z_umap, train_idx, args.k)
    umap_test = knn_overlap_rows(x_np, z_umap, test_idx, args.k)
    print(f"[oos] UMAP.transform train_knnOv={umap_train:.3f}  test_knnOv={umap_test:.3f}")

    result = {
        "protocol": "leak-free: P/target/mapper fit on train only; test projected by frozen model",
        "seed": args.seed, "n": n, "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "k": args.k, "chance": round(args.k / n, 5),
        "ours_v10": {"train_knnOv": round(ours_train, 4), "test_knnOv": round(ours_test, 4)},
        "umap_transform": {"train_knnOv": round(umap_train, 4), "test_knnOv": round(umap_test, 4)},
        "v10_config": {"perplexity": args.perplexity, "iters": args.iters,
                       "exaggeration": f"{args.exaggeration}x{args.exaggeration_iters}",
                       "dropout": args.dropout, "input_noise": args.input_noise},
        "heldout_ids": [int(ids[int(i)]) for i in test_idx.tolist()],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[oos] summary  (chance={result['chance']})")
    print(f"  Ours(v10):       train={ours_train:.3f}  test={ours_test:.3f}")
    print(f"  UMAP.transform:  train={umap_train:.3f}  test={umap_test:.3f}")
    print(f"[oos] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
