#!/usr/bin/env python3
"""
Holdout evaluation for the parametric semantic projection mapper.

This trains a temporary ResidualProjectionMapper on only a train split of MSUs,
then evaluates whether held-out MSUs can be projected into a faithful 2D layout.
It does not overwrite the production checkpoint.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import local_config as cfg
from code_for_model.train_all_v7 import ResidualProjectionMapper


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_embeddings(cache: Path) -> tuple[np.ndarray, list[int]]:
    ids_path = Path(str(cache) + ".ids.json")
    x = np.load(cache).astype(np.float32)
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if len(ids) != x.shape[0]:
        raise ValueError(f"Embedding/id mismatch: {x.shape[0]} rows vs {len(ids)} ids")
    return x, ids


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def knn_overlap_rows(x_ref: np.ndarray, z_ref: np.ndarray, rows: np.ndarray, k: int) -> float:
    """For selected anchor rows, compare their high-D and low-D neighbors among all points."""
    ix = NearestNeighbors(n_neighbors=k + 1).fit(x_ref).kneighbors(x_ref[rows], return_distance=False)
    iz = NearestNeighbors(n_neighbors=k + 1).fit(z_ref).kneighbors(z_ref[rows], return_distance=False)
    vals = []
    for row, a, b in zip(rows, ix, iz):
        a = [int(i) for i in a if int(i) != int(row)][:k]
        b = [int(i) for i in b if int(i) != int(row)][:k]
        vals.append(len(set(a) & set(b)) / k)
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=cfg.EMB_CACHE)
    ap.add_argument("--target", type=Path, default=cfg.MODEL_DIR / "semantic_target_umap_readable.npy")
    ap.add_argument("--out", type=Path, default=cfg.OUTPUT_DIR / "holdout_eval_v8_readable.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=12)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    x_np, ids = load_embeddings(args.cache)
    target_np = np.load(args.target).astype(np.float32)
    if target_np.shape != (x_np.shape[0], 2):
        raise ValueError(f"Target shape mismatch: {target_np.shape} vs {(x_np.shape[0], 2)}")

    n = x_np.shape[0]
    perm = np.random.default_rng(args.seed).permutation(n)
    n_test = max(1, int(round(n * args.test_frac)))
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])

    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    target = torch.tensor(target_np, dtype=torch.float32, device=device)
    mapper = ResidualProjectionMapper(
        embed_dim=x_np.shape[1],
        width=args.width,
        num_blocks=args.blocks,
        dropout=args.dropout,
    ).to(device)
    opt = optim.AdamW(mapper.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = DataLoader(
        TensorDataset(torch.tensor(train_idx, dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    hist = []
    for ep in range(args.epochs):
        mapper.train()
        total, batches = 0.0, 0
        for (idx,) in loader:
            idx = idx.to(device)
            pred = mapper(x[idx])
            loss = F.smooth_l1_loss(pred, target[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), 5.0)
            opt.step()
            total += float(loss.item())
            batches += 1
        sched.step()
        hist.append(total / max(1, batches))
        if ep == 0 or (ep + 1) % 25 == 0:
            print(f"[holdout] epoch {ep+1:04d}/{args.epochs} train_loss={hist[-1]:.6f}")

    mapper.eval()
    with torch.no_grad():
        pred_np = mapper(x).cpu().numpy()

    result = {
        "seed": args.seed,
        "test_frac": args.test_frac,
        "n": n,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "target": str(args.target),
        "train_target_mse": mse(pred_np[train_idx], target_np[train_idx]),
        "test_target_mse": mse(pred_np[test_idx], target_np[test_idx]),
        "all_trust": float(trustworthiness(x_np, pred_np, n_neighbors=args.k)),
        "target_trust": float(trustworthiness(x_np, target_np, n_neighbors=args.k)),
        "train_knnOv_rows": knn_overlap_rows(x_np, pred_np, train_idx, args.k),
        "test_knnOv_rows": knn_overlap_rows(x_np, pred_np, test_idx, args.k),
        "target_test_knnOv_rows": knn_overlap_rows(x_np, target_np, test_idx, args.k),
        "loss_history": hist,
        "heldout_ids": [ids[int(i)] for i in test_idx.tolist()],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[holdout] result")
    for key in [
        "n_train",
        "n_test",
        "train_target_mse",
        "test_target_mse",
        "all_trust",
        "target_trust",
        "train_knnOv_rows",
        "test_knnOv_rows",
        "target_test_knnOv_rows",
    ]:
        val = result[key]
        print(f"  {key}: {val:.6f}" if isinstance(val, float) else f"  {key}: {val}")
    print(f"[holdout] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
