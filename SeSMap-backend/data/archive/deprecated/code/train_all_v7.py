#!/usr/bin/env python3
"""
train_all_v7.py - topology-distilled semantic projection.

v5/v6 diagnosis:
  * v5 uses sparse triplets plus paragraph/repulsion losses, so it never sees the
    dense corpus-level KNN structure measured by knnOv@12.
  * v6 moves toward t-SNE, but computes P/Q inside each mini-batch. That discards
    the global neighbor graph used by the evaluator.

v7 optimizes the evaluation target more directly:
  1. Build a corpus-level semantic target layout from BGE embeddings (UMAP by
     default, t-SNE optional).
  2. Train a parametric residual MLP to distill that target layout.
  3. Optionally add dense KNN triplets from the global high-dimensional graph.

Empirically, pure target distillation is the best default for the current
faithfulness criterion: it reaches the UMAP target's trust/knnOv without the
KNN triplet term pulling against the target geometry.

paper_id is not used. It remains only a display/color attribute downstream.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import local_config as cfg


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidualProjectionMapper(nn.Module):
    """Batch-independent 1024D -> 2D mapper for semantic projection."""

    def __init__(
        self,
        embed_dim: int = 1024,
        width: int = 512,
        num_blocks: int = 4,
        dropout: float = 0.05,
        out_dim: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.width = width
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.in_proj = nn.Sequential(
            nn.Linear(embed_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.in_proj(x)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_embeddings(cache: Path) -> tuple[np.ndarray, list[int]]:
    ids_path = Path(str(cache) + ".ids.json")
    if not cache.exists() or not ids_path.exists():
        raise FileNotFoundError(f"Missing embedding cache or ids: {cache}")
    x = np.load(cache).astype(np.float32)
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if len(ids) != x.shape[0]:
        raise ValueError(f"Embedding/id mismatch: {x.shape[0]} rows vs {len(ids)} ids")
    return x, ids


def standardize_layout(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float32)
    z = z - z.mean(axis=0, keepdims=True)
    scale = z.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return z / scale


def build_target_layout(args: argparse.Namespace, x: np.ndarray) -> np.ndarray:
    target_path = Path(args.target_cache)
    if args.reuse_target and target_path.exists():
        z = np.load(target_path).astype(np.float32)
        if z.shape == (x.shape[0], 2):
            print(f"[target] loaded {z.shape} <- {target_path}")
            return z
        print(f"[target] stale target shape {z.shape}; rebuilding")

    if args.target == "umap":
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=args.target_neighbors,
            min_dist=args.min_dist,
            spread=args.spread,
            metric="euclidean",
            random_state=args.seed,
        )
        z = reducer.fit_transform(x)
    elif args.target == "tsne":
        z = TSNE(
            n_components=2,
            init="pca",
            perplexity=args.perplexity,
            learning_rate="auto",
            random_state=args.seed,
        ).fit_transform(x)
    else:
        raise ValueError(f"Unknown target layout: {args.target}")

    z = standardize_layout(z)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(target_path, z)
    print(f"[target] built {args.target} {z.shape} -> {target_path}")
    return z


def build_knn_graph(x: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(x)
    return nn.kneighbors(return_distance=False)[:, 1:].astype(np.int64)


def sample_knn_triplets(
    batch_idx: torch.Tensor,
    knn: torch.Tensor,
    n_items: int,
    positives_per_anchor: int,
    negatives_per_anchor: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = batch_idx.device
    anchors = batch_idx.repeat_interleave(positives_per_anchor)
    pos_cols = torch.randint(0, knn.size(1), (anchors.size(0),), device=device)
    positives = knn[anchors, pos_cols]

    anchors = anchors.repeat_interleave(negatives_per_anchor)
    positives = positives.repeat_interleave(negatives_per_anchor)
    negatives = torch.randint(0, n_items, (anchors.size(0),), device=device)

    # Avoid trivial self negatives. Random negatives that are also KNN are rare
    # enough at this corpus size, and the target-distillation term handles them.
    self_mask = negatives == anchors
    if self_mask.any():
        negatives[self_mask] = (negatives[self_mask] + 1) % n_items
    return anchors, positives, negatives


def evaluate_quick(x: np.ndarray, z: np.ndarray, k: int) -> tuple[float, float]:
    from sklearn.manifold import trustworthiness

    ix = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(return_distance=False)[:, 1:]
    iz = NearestNeighbors(n_neighbors=k + 1).fit(z).kneighbors(return_distance=False)[:, 1:]
    overlap = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ix, iz)]))
    return float(trustworthiness(x, z, n_neighbors=k)), overlap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=cfg.EMB_CACHE)
    ap.add_argument("--out", type=Path, default=cfg.MODEL_DIR / "bert2d_mapper_all_v7.pt")
    ap.add_argument("--target-cache", default=str(cfg.MODEL_DIR / "semantic_target_umap_v7.npy"))
    ap.add_argument("--target", choices=["umap", "tsne"], default="umap")
    ap.add_argument("--reuse-target", action="store_true")
    ap.add_argument("--target-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.05)
    ap.add_argument("--spread", type=float, default=1.0)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--knn-k", type=int, default=24)
    ap.add_argument("--eval-k", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lambda-target", type=float, default=1.0)
    ap.add_argument("--lambda-knn", type=float, default=0.0)
    ap.add_argument("--lambda-repel", type=float, default=0.0)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--positives-per-anchor", type=int, default=2)
    ap.add_argument("--negatives-per-anchor", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    x_np, ids = load_embeddings(args.cache)
    z_target_np = build_target_layout(args, x_np)
    knn_np = build_knn_graph(x_np, args.knn_k)
    print(f"[v7] N={x_np.shape[0]}, dim={x_np.shape[1]}, target={args.target}, device={device}")
    print("[v7] objective = target distillation + optional global KNN triplets; paper_id is unused")

    tw_t, ko_t = evaluate_quick(x_np, z_target_np, args.eval_k)
    print(f"[target] trust@{args.eval_k}={tw_t:.3f} knnOv@{args.eval_k}={ko_t:.3f}")

    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    z_target = torch.tensor(z_target_np, dtype=torch.float32, device=device)
    knn = torch.tensor(knn_np, dtype=torch.long, device=device)

    mapper = ResidualProjectionMapper(
        embed_dim=x_np.shape[1],
        width=args.width,
        num_blocks=args.blocks,
        dropout=args.dropout,
    ).to(device)
    opt = optim.AdamW(mapper.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = DataLoader(
        TensorDataset(torch.arange(x_np.shape[0], dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    history = {"total": [], "target": [], "knn": [], "repel": []}
    for ep in range(args.epochs):
        mapper.train()
        totals = {k: 0.0 for k in history}
        batches = 0
        for (batch_idx,) in loader:
            batch_idx = batch_idx.to(device)
            zb = mapper(x[batch_idx])
            loss_target = F.smooth_l1_loss(zb, z_target[batch_idx])

            if args.lambda_knn > 0 or args.lambda_repel > 0:
                ai, pi, ni = sample_knn_triplets(
                    batch_idx,
                    knn,
                    x_np.shape[0],
                    args.positives_per_anchor,
                    args.negatives_per_anchor,
                )
                za = mapper(x[ai])
                zp = mapper(x[pi])
                zn = mapper(x[ni])
                loss_knn = F.triplet_margin_loss(za, zp, zn, margin=args.margin, p=2)
                neg_dist = torch.norm(za - zn, dim=1)
                loss_repel = F.softplus(args.margin - neg_dist).mean()
            else:
                loss_knn = torch.tensor(0.0, device=device)
                loss_repel = torch.tensor(0.0, device=device)

            loss = (
                args.lambda_target * loss_target
                + args.lambda_knn * loss_knn
                + args.lambda_repel * loss_repel
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), 5.0)
            opt.step()

            totals["total"] += float(loss.item())
            totals["target"] += float(loss_target.item())
            totals["knn"] += float(loss_knn.item())
            totals["repel"] += float(loss_repel.item())
            batches += 1
        sched.step()

        for k in history:
            history[k].append(totals[k] / max(1, batches))

        if ep == 0 or (ep + 1) % 25 == 0:
            mapper.eval()
            with torch.no_grad():
                z_pred = mapper(x).cpu().numpy()
            tw, ko = evaluate_quick(x_np, z_pred, args.eval_k)
            print(
                f"[v7] epoch {ep+1:04d}/{args.epochs} "
                f"loss={history['total'][-1]:.4f} "
                f"target={history['target'][-1]:.4f} "
                f"knn={history['knn'][-1]:.4f} "
                f"trust={tw:.3f} knnOv={ko:.3f}"
            )

    mapper.eval()
    with torch.no_grad():
        z_final = mapper(x).cpu().numpy()
    tw, ko = evaluate_quick(x_np, z_final, args.eval_k)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mapper_state": mapper.state_dict(),
            "model_class": "ResidualProjectionMapper",
            "arch": "v7",
            "embed_dim": x_np.shape[1],
            "width": args.width,
            "num_blocks": args.blocks,
            "dropout": args.dropout,
            "out_dim": 2,
            "target": args.target,
            "target_cache": str(args.target_cache),
            "target_neighbors": args.target_neighbors,
            "min_dist": args.min_dist,
            "spread": args.spread,
            "knn_k": args.knn_k,
            "ids": ids,
            "normalize_output": False,
            "loss_history": history,
            "eval": {"trust": tw, "knnOv": ko, "k": args.eval_k},
        },
        args.out,
    )
    np.save(str(args.out).replace(".pt", "_coords.npy"), z_final.astype(np.float32))
    print(f"[v7] final trust@{args.eval_k}={tw:.3f} knnOv@{args.eval_k}={ko:.3f}")
    print(f"[v7] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
