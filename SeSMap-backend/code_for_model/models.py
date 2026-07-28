#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models.py — SeSMap 投影模型的共享定义（干净版，唯一来源）。

包含：
  * ResidualProjectionMapper（最终架构；v10/v11 使用）
  * compute_joint_P（t-SNE 全局联合分布 P）
  * evaluate_quick / set_seed / load_embeddings / standardize_layout（工具）
  * Bert2DMapper（仅用于加载历史 v5/v6 checkpoint 的兜底，可选）

历史训练脚本（train_all_v5/v6/v7/v10、三元组等）已归档到 data/archive/deprecated/。
当前流水线只依赖本文件 + train_contrastive_v11.py。
"""
from __future__ import annotations
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors


# ------------------------------------------------------------------ #
# 最终架构：残差 MLP（1024 -> 2）
# ------------------------------------------------------------------ #
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

    def __init__(self, embed_dim: int = 1024, width: int = 512,
                 num_blocks: int = 4, dropout: float = 0.05, out_dim: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.width = width
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.in_proj = nn.Sequential(
            nn.Linear(embed_dim, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.in_proj(x)))


# ------------------------------------------------------------------ #
# 历史兜底：Bert2DMapper（仅加载旧 v5/v6 checkpoint 时用；当前不训练）
# ------------------------------------------------------------------ #
class Bert2DMapper(nn.Module):
    def __init__(self, embed_dim=768, hidden_dims=(256, 64, 32), out_dim=2, dropout=0.1, normalize_output=True):
        super().__init__()
        self.normalize_output = normalize_output
        self.layers = nn.ModuleList(); self.norms = nn.ModuleList()
        self.activations = nn.ModuleList(); self.dropouts = nn.ModuleList()
        in_dim = embed_dim
        for h in hidden_dims:
            self.layers.append(nn.Linear(in_dim, h))
            self.norms.append(nn.BatchNorm1d(h))
            self.activations.append(nn.GELU())
            self.dropouts.append(nn.Dropout(dropout))
            in_dim = h
        self.final = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = x
        for layer, norm, act, drop in zip(self.layers, self.norms, self.activations, self.dropouts):
            residual = out
            out = drop(act(norm(layer(out))))
            if residual.shape[-1] == out.shape[-1]:
                out = out + residual
        out = self.final(out)
        if self.normalize_output:
            out = out / out.max(dim=0, keepdim=True)[0] * 10
        return out


# ------------------------------------------------------------------ #
# t-SNE 全局联合分布 P
# ------------------------------------------------------------------ #
def compute_joint_P(x: np.ndarray, perplexity: float = 30.0, tol: float = 1e-4, max_iter: int = 60) -> np.ndarray:
    """逐点二分搜索 sigma 使条件分布熵=log(perplexity)，再对称化。"""
    n = x.shape[0]
    if n <= 1200:
        D2 = np.square(np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1))
    else:
        D2 = np.empty((n, n), dtype=np.float32)
        for s in range(0, n, 512):
            e = min(n, s + 512)
            D2[s:e] = np.square(x[s:e, None, :] - x[None, :, :]).sum(-1)
    np.fill_diagonal(D2, np.inf)
    target = np.log(perplexity)
    P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        lo, hi, beta = 1e-20, 1e20, 1.0
        d = D2[i]
        for _ in range(max_iter):
            p = np.exp(-d * beta); s = p.sum()
            if s <= 0:
                beta *= 0.5; continue
            p /= s
            ent = -np.sum(p[p > 0] * np.log(p[p > 0]))
            diff = ent - target
            if abs(diff) < tol:
                break
            if diff > 0:
                lo = beta; beta = beta * 2 if hi == 1e20 else (beta + hi) / 2
            else:
                hi = beta; beta = beta / 2 if lo == 1e-20 else (beta + lo) / 2
        P[i] = p
    P = (P + P.T) / (2.0 * n)
    return np.maximum(P, 1e-12)


# ------------------------------------------------------------------ #
# 工具
# ------------------------------------------------------------------ #
def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_embeddings(cache: Path):
    cache = Path(cache)
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


def evaluate_quick(x: np.ndarray, z: np.ndarray, k: int):
    """返回 (trustworthiness, knnOv@k)。"""
    from sklearn.manifold import trustworthiness
    ix = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(return_distance=False)[:, 1:]
    iz = NearestNeighbors(n_neighbors=k + 1).fit(z).kneighbors(return_distance=False)[:, 1:]
    overlap = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ix, iz)]))
    return float(trustworthiness(x, z, n_neighbors=k)), overlap
