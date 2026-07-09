#!/usr/bin/env python3
"""
train_all_v10.py — 全批 parametric t-SNE（自研邻域保持目标，无 UMAP 老师）。

v6 的教训：批内算 P/Q，真近邻几乎不在批里 -> 学不到局部结构(knnOv 0.036)。
v8 的妥协：蒸馏 UMAP 布局 -> 到手 0.252，但受老师上限压制(UMAP 0.260 < tSNE 0.323)。
v10 的做法：
  1) 用 bge 向量算一次 **全局** P（perplexity 二分校准 + 对称化），N=2689 时 P 仅 ~29MB；
  2) 每步把 **全部 N 个点** 过 MLP 得 Z，全批算 student-t Q，直接优化 KL(P||Q)；
  3) 早期夸大(early exaggeration) 促成簇结构，与经典 t-SNE 一致。
端到端单目标；checkpoint 与 v7/v8 同格式(ResidualProjectionMapper)，
formdatabase.py / 推理端无需改动。paper_id 不参与，仅上色。

用法:
  python3 code_for_model/train_all_v10.py
  python3 code_for_data/formdatabase.py --mapper data/stages/05_model/bert2d_mapper_all_v10.pt
  python3 code_for_model/eval_faithfulness.py --candidate-label "Ours(v10)"
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
from code_for_model.train_all_v7 import ResidualProjectionMapper, set_seed, load_embeddings, evaluate_quick


def compute_joint_P(x: np.ndarray, perplexity: float = 30.0, tol: float = 1e-4, max_iter: int = 60) -> np.ndarray:
    """经典 t-SNE 的全局联合分布 P：逐点二分搜索 sigma 使条件分布熵=log(perplexity)，再对称化。"""
    n = x.shape[0]
    D2 = np.square(np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)) if n <= 1200 else None
    if D2 is None:  # 分块算距离，省内存
        D2 = np.empty((n, n), dtype=np.float32)
        step = 512
        for s in range(0, n, step):
            e = min(n, s + step)
            D2[s:e] = np.square(x[s:e, None, :] - x[None, :, :]).sum(-1)
    np.fill_diagonal(D2, np.inf)
    target = np.log(perplexity)
    P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        lo, hi = 1e-20, 1e20
        beta = 1.0  # = 1/(2 sigma^2)
        d = D2[i]
        for _ in range(max_iter):
            p = np.exp(-d * beta)
            s = p.sum()
            if s <= 0:
                beta *= 0.5
                continue
            p /= s
            ent = -np.sum(p[p > 0] * np.log(p[p > 0]))
            diff = ent - target
            if abs(diff) < tol:
                break
            if diff > 0:   # 熵太大 -> 分布太平 -> 增大 beta(减小 sigma)
                lo = beta
                beta = beta * 2 if hi == 1e20 else (beta + hi) / 2
            else:
                hi = beta
                beta = beta / 2 if lo == 1e-20 else (beta + lo) / 2
        P[i] = p
    P = (P + P.T) / (2.0 * n)
    return np.maximum(P, 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=cfg.EMB_CACHE)
    ap.add_argument("--out", type=Path, default=cfg.MODEL_DIR / "bert2d_mapper_all_v10.pt")
    ap.add_argument("--p-cache", type=Path, default=cfg.MODEL_DIR / "tsne_P_perp30.npy")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--exaggeration", type=float, default=12.0)
    ap.add_argument("--exaggeration-iters", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--input-noise", type=float, default=0.4,
                    help="训练时输入抖动的期望范数(0=关)。OOS 实验证实 0.4 大幅提升泛化且不伤 in-sample")
    ap.add_argument("--eval-k", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    x_np, ids = load_embeddings(args.cache)
    n = x_np.shape[0]
    print(f"[v10] N={n} dim={x_np.shape[1]} perplexity={args.perplexity} device={device}")
    print("[v10] objective = full-batch parametric t-SNE KL (global P); paper_id unused")

    if args.p_cache.exists():
        P_np = np.load(args.p_cache)
        if P_np.shape != (n, n):
            print("[P] cache stale, recomputing")
            P_np = compute_joint_P(x_np, args.perplexity)
            np.save(args.p_cache, P_np)
    else:
        print("[P] computing global joint P (one-off)...")
        P_np = compute_joint_P(x_np, args.perplexity)
        args.p_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.p_cache, P_np)
    print(f"[P] ready {P_np.shape}")

    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    P = torch.tensor(P_np, dtype=torch.float32, device=device)
    eye = torch.eye(n, dtype=torch.bool, device=device)

    mapper = ResidualProjectionMapper(embed_dim=x_np.shape[1], width=args.width,
                                      num_blocks=args.blocks, dropout=args.dropout).to(device)
    opt = optim.Adam(mapper.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)

    # 输入抖动：强迫映射局部平滑（邻近句向量 -> 邻近坐标），OOS 泛化的关键
    noise_std = args.input_noise / (x_np.shape[1] ** 0.5) if args.input_noise > 0 else 0.0
    hist = []
    mapper.train()
    for it in range(args.iters):
        Pe = P * args.exaggeration if it < args.exaggeration_iters else P
        if it == args.exaggeration_iters:
            print(f"[v10] iter {it}: early exaggeration off")
        xin = x + noise_std * torch.randn_like(x) if noise_std > 0 else x
        Z = mapper(xin)                                # (N,2) 全批
        D2 = torch.cdist(Z, Z) ** 2
        num = 1.0 / (1.0 + D2)
        num = num.masked_fill(eye, 0.0)
        Q = (num / num.sum().clamp_min(1e-12)).clamp_min(1e-12)
        Pn = (Pe / Pe.sum()).clamp_min(1e-12)          # 夸大后重新归一化
        loss = (Pn * (Pn.log() - Q.log())).sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mapper.parameters(), 5.0)
        opt.step()
        sched.step()
        hist.append(float(loss.item()))

        if it == 0 or (it + 1) % 100 == 0:
            mapper.eval()
            with torch.no_grad():
                z_np = mapper(x).cpu().numpy()
            tw, ko = evaluate_quick(x_np, z_np, args.eval_k)
            print(f"[v10] iter {it+1:04d}/{args.iters} KL={hist[-1]:.4f} trust={tw:.3f} knnOv={ko:.3f}")
            mapper.train()

    mapper.eval()
    with torch.no_grad():
        z_final = mapper(x).cpu().numpy()
    tw, ko = evaluate_quick(x_np, z_final, args.eval_k)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "mapper_state": mapper.state_dict(),
        "model_class": "ResidualProjectionMapper",
        "arch": "v10",
        "embed_dim": x_np.shape[1],
        "width": args.width,
        "num_blocks": args.blocks,
        "dropout": args.dropout,
        "out_dim": 2,
        "objective": f"full-batch parametric t-SNE (perplexity={args.perplexity}, exaggeration={args.exaggeration}x{args.exaggeration_iters}, dropout={args.dropout}, input_noise={args.input_noise})",
        "ids": ids,
        "normalize_output": False,
        "loss_history": {"kl": hist},
        "eval": {"trust": tw, "knnOv": ko, "k": args.eval_k},
    }, args.out)
    np.save(str(args.out).replace(".pt", "_coords.npy"), z_final.astype(np.float32))
    print(f"[v10] final trust@{args.eval_k}={tw:.3f} knnOv@{args.eval_k}={ko:.3f}")
    print("[v10] 对照: v8蒸馏=0.252 | UMAP=0.260 | tSNE=0.323 (in-sample上限)")
    print(f"[v10] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
