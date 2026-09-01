# requirements:
# pip install torch sentence-transformers tqdm scikit-learn matplotlib
#
# train_all_v5.py
# ---------------------------------------------------------------------------
# SeSMap 投影训练 —— v5（语义优先 / semantic-first）
#
# 设计原则：paper_id 只作为“上色属性”，绝不进入损失函数充当组织力。
# 相比 v3 的关键改动：
#   * 删除 hierarchical_pull_loss 中“同论文拉近(target=10) / 跨论文推远(target=30)”
#     两个身份分支，替换为 paragraph_coherence_loss：只对“同一段落”的 MSU 施加弱就近
#     （纯局部文本上下文先验）。跨论文距离完全交给语义 triplet 决定，
#     boundary zone 因此反映真实语义趋同，而不是被身份先验硬推出来的假区隔。
#   * 新增 stabilization_loss（尺度下限，防止去掉分离力后坍缩；即论文所述第 4 个损失 L_st）。
#   * 新增可选 role_coherence_loss（按 discourse role 轻微聚拢；默认关闭，见 lambda_role）。
#   * 保留 triplet(语义排序) 与 repulsion(可读性)。
#   * Bert2DMapper 结构与保存格式与 v3 完全一致，推理端只需把 model_path 指向 v5 checkpoint。
#
# 自检（Direction B 的验收）：改完后 `grep -n paper_id train_all_v5.py`
# 只应出现在 paragraph_coherence_loss 内部（用于判定“是否同段落”），
# 不应作为任何 attract/repel 的目标。
# ---------------------------------------------------------------------------

import os
import json
import math
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, models
import numpy as np
import matplotlib.pyplot as plt

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import local_config as cfg


# -----------------------
# 1) 数据集
# -----------------------
class TripletTextDataset(Dataset):
    """
    Expect json list of {"anchor": "...", "positive": "...", "negative": "...",
                        "anchor_idx": int, "positive_idx": int, "negative_idx": int}
    """
    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        assert isinstance(self.data, list), "json must be a list of triplets"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        t = self.data[idx]
        return t["anchor"], t["positive"], t["negative"], t["anchor_idx"], t["positive_idx"], t["negative_idx"]


def collate_fn(batch, tokenizer_model: SentenceTransformer, device="cpu"):
    """Turn batch of triplet texts into tensors of embeddings."""
    anchors, positives, negatives, anchor_idxs, positive_idxs, negative_idxs = zip(*batch)
    a_emb = tokenizer_model.encode(list(anchors),   convert_to_tensor=True, device=device).clone().detach()
    p_emb = tokenizer_model.encode(list(positives), convert_to_tensor=True, device=device).clone().detach()
    n_emb = tokenizer_model.encode(list(negatives), convert_to_tensor=True, device=device).clone().detach()
    return a_emb, p_emb, n_emb, anchor_idxs, positive_idxs, negative_idxs


# -----------------------
# 1b) 缓存 embedding 模式（bge 冻结 -> 句向量只算一次；本地 CPU 训练飞快）
# -----------------------
def load_embedding_cache(cache_path):
    """读取 precompute_embeddings.py 产出的 <cache>.npy + <cache>.ids.json。"""
    ids = json.load(open(str(cache_path) + ".ids.json", encoding="utf-8"))
    X = np.load(str(cache_path)).astype(np.float32)
    id2row = {int(i): k for k, i in enumerate(ids)}
    return X, id2row


def make_cached_collate(X, id2row, device):
    Xt = torch.tensor(X, dtype=torch.float32, device=device)

    def _collate(batch):
        _, _, _, ai, pi, ni = zip(*batch)

        def gather(idxs):
            return Xt[[id2row[int(x)] for x in idxs]]
        return gather(ai), gather(pi), gather(ni), ai, pi, ni
    return _collate


# -----------------------
# 2) 模型：SBERT（冻结） + MLP -> 2d   （与 v3 完全一致：残差 + 归一化到 [0,10]）
# -----------------------
class Bert2DMapper(nn.Module):
    def __init__(self, embed_dim=768, hidden_dims=(256, 64, 32), out_dim=2, dropout=0.1, normalize_output=True):
        super().__init__()
        self.normalize_output = normalize_output   # v5=True(按批 max 归一化)；v6=False(交给邻域损失)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        in_dim = embed_dim
        for h in hidden_dims:
            self.layers.append(nn.Linear(in_dim, h))
            self.norms.append(nn.BatchNorm1d(h))
            self.activations.append(nn.GELU())
            self.dropouts.append(nn.Dropout(dropout))
            in_dim = h
        self.final = nn.Linear(in_dim, out_dim)
        # 不在这里做 tanh；留给训练后或 loss 控制

    def forward(self, x):
        # x: (batch, embed_dim)
        out = x
        for layer, norm, act, drop in zip(self.layers, self.norms, self.activations, self.dropouts):
            residual = out
            out = layer(out)
            out = norm(out)
            out = act(out)
            out = drop(out)
            # 残差连接：仅当输入输出维度一致时
            if residual.shape[-1] == out.shape[-1]:
                out = out + residual
        out = self.final(out)
        if self.normalize_output:                     # v5 行为：按批 max 归一化到 [0,10]
            out = out / out.max(dim=0, keepdim=True)[0]
            out = out * 10
        return out  # (batch, 2)


# -----------------------
# 3) Repulsion（batch 内 pairwise，避免局部重叠 / 提升可读性）
# -----------------------
def repulsion_loss(points: torch.Tensor, eps=1e-3):
    """
    points: (batch, 2)
    L_rep = mean_{i<j} 1 / (||p_i - p_j||^2 + eps)
    """
    b = points.size(0)
    if b < 2:
        return torch.tensor(0.0, device=points.device)
    diffs = points.unsqueeze(1) - points.unsqueeze(0)                     # (b,b,2)
    dist2 = (diffs ** 2).sum(dim=-1) + torch.eye(b, device=points.device) * 1e12
    inv = 1.0 / (dist2 + eps)
    triu_indices = torch.triu_indices(b, b, offset=1)
    vals = inv[triu_indices[0], triu_indices[1]]
    return vals.mean()


# -----------------------
# 4) 段落一致性损失（替代 v3 的 hierarchical_pull_loss）
#    仅对“同一论文的同一段落”的 MSU 施加弱就近（局部文本上下文先验）。
#    paper_id 只用于判断“是否同段落”，不作为把论文推开/拉近的组织力。
# -----------------------
def paragraph_coherence_loss(coords: torch.Tensor, metadata: Dict, anchor_idxs: List[int],
                             same_para_weight: float = 4.0, target: float = 1.0):
    n = coords.size(0)
    loss = 0.0
    count = 0
    for i in range(n):
        mi = metadata.get(str(anchor_idxs[i]), {})
        pid_i, para_i = mi.get("paper_id", -1), mi.get("para_id", -1)
        for j in range(i + 1, n):
            mj = metadata.get(str(anchor_idxs[j]), {})
            pid_j, para_j = mj.get("paper_id", -2), mj.get("para_id", -2)
            # para_id 可能跨论文重复，所以要求 paper_id 与 para_id 同时相等才算“同段落”
            if pid_i == pid_j and para_i == para_j:
                dist = torch.norm(coords[i] - coords[j])
                loss = loss + same_para_weight * (dist - target).pow(2)
                count += 1
    if count == 0:
        return torch.tensor(0.0, device=coords.device)
    return loss / count


# -----------------------
# 5) 稳定性损失（尺度下限，防止去掉分离力后整体坍缩；即论文的 L_st）
# -----------------------
def stabilization_loss(coords: torch.Tensor, gamma: float = 0.30):
    std = coords.std(dim=0) + 1e-4
    return torch.clamp(gamma - std, min=0).mean()


# -----------------------
# 6) 角色一致性损失（可选：按 discourse role 轻微聚拢；只 pull，不 push）
#    默认关闭（lambda_role=0）。启用时权重要小，过大会压过语义结构。
# -----------------------
def role_coherence_loss(coords: torch.Tensor, metadata: Dict, anchor_idxs: List[int]):
    device = coords.device
    roles = [metadata.get(str(ix), {}).get("category", "NA") for ix in anchor_idxs]
    pull = 0.0
    g = 0
    for r in set(roles):
        mask = torch.tensor([ri == r for ri in roles], device=device)
        if mask.sum() < 2:
            continue
        c = coords[mask].mean(dim=0)
        pull = pull + ((coords[mask] - c) ** 2).sum(dim=1).mean()
        g += 1
    if g == 0:
        return torch.tensor(0.0, device=device)
    return pull / g


# -----------------------
# 7) 辅助：triplet loss（语义排序，anchor 距 positive 近于 negative）
# -----------------------
triplet_loss_fn = nn.TripletMarginWithDistanceLoss(
    distance_function=lambda x, y: torch.norm(x - y, p=2, dim=-1),
    margin=5,
    reduction='mean'
)


# -----------------------
# 8) 单阶段训练函数
# -----------------------
def train_single_stage(
    json_path: str,
    metadata_path: str,
    sbert_model_name: str = str(cfg.BGE_MODEL_PATH),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    embed_dim: int = 384,
    hidden_dims: tuple = (256, 64, 32),
    batch_size: int = 128,
    epochs: int = 20,
    lr: float = 1e-3,
    lambda_repulsion: float = 0.4,
    lambda_para: float = 0.1,   # 段落局部先验(可读性)，只做弱约束；不让它盖过语义 triplet
    lambda_st: float = 0.05,
    lambda_role: float = 0.0,        # 0.0 = 纯语义地图（Direction B 默认）；>0 可选启用 role 结构（建议 <=0.3）
    freeze_sbert: bool = True,
    embedding_cache: str = None,     # precompute_embeddings 的 .npy；存在则用缓存、跳过逐 batch 编码
    save_path: str = "./model_2d_v5.pt"
):
    # ---- 加载元数据 ----
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)
    metadata_dict = {str(item.get("idx", i)): item for i, item in enumerate(metadata_list)}

    # ---- 加载 SentenceTransformer ----
    try:
        sbert = SentenceTransformer(sbert_model_name, device=device)
        print("模型直接加载成功")
    except Exception as e:
        print(f"直接加载失败: {e}\n尝试手动构建模型...")
        word_embedding_model = models.Transformer(sbert_model_name)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )
        sbert = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
        print("手动构建模型成功")

    # ---- 缓存 embedding 分支：有缓存就用缓存(快)，否则逐 batch 用 bge 编码 ----
    use_cache = bool(embedding_cache) and Path(str(embedding_cache)).exists() \
        and Path(str(embedding_cache) + ".ids.json").exists()
    ds = TripletTextDataset(json_path)
    if use_cache:
        X, id2row = load_embedding_cache(embedding_cache)
        embed_dim = X.shape[1]
        keep = [t for t in ds.data
                if all(int(t[k]) in id2row for k in ("anchor_idx", "positive_idx", "negative_idx"))]
        dropped = len(ds.data) - len(keep)
        ds.data = keep
        print(f"[cache] 用缓存 embedding {X.shape} <- {embedding_cache}（跳过逐 batch 编码，CPU 也快）"
              + (f"；丢弃 {dropped} 条 idx 不在缓存的三元组" if dropped else ""))
        collate = make_cached_collate(X, id2row, device)
    else:
        sample_emb = sbert.encode("test", convert_to_tensor=True, device=device)
        if sample_emb.shape[-1] != embed_dim:
            embed_dim = sample_emb.shape[-1]
        print("Detected embedding dim:", embed_dim, "（无缓存，逐 batch 用 bge 编码）")
        collate = lambda b: collate_fn(b, sbert, device)

    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            collate_fn=collate, drop_last=False)

    # ---- 模型 / 优化器 ----
    mapper = Bert2DMapper(embed_dim=embed_dim, hidden_dims=hidden_dims, out_dim=2).to(device)
    if freeze_sbert or use_cache:
        trainable_params = list(mapper.parameters())
    else:
        trainable_params = list(mapper.parameters()) + list(sbert.parameters())
        print("警告: 解冻 SBERT 参数进行微调")
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print("v5 Single Stage: Triplet(语义) + Repulsion + Paragraph + Stabilization"
          + (" + Role" if lambda_role > 0 else ""))
    print("[design] paper_id 不作为优化目标，仅作上色属性。")
    mapper.train()

    loss_history = {'total': [], 'triplet': [], 'repulsion': [],
                    'paragraph': [], 'stabilization': [], 'role': []}

    for epoch in range(epochs):
        totals = {k: 0.0 for k in loss_history}
        n_batches = 0

        for a_emb, p_emb, n_emb, anchor_idxs, positive_idxs, negative_idxs in tqdm(dataloader):
            optimizer.zero_grad()
            a2d = mapper(a_emb)   # (B,2)
            p2d = mapper(p_emb)
            n2d = mapper(n_emb)

            # losses（结构性损失只作用于 batch 内 anchor，与 v3 保持一致）
            loss_trip  = triplet_loss_fn(a2d, p2d, n2d)
            loss_repel = repulsion_loss(a2d)
            loss_para  = paragraph_coherence_loss(a2d, metadata_dict, anchor_idxs)
            loss_st    = stabilization_loss(a2d)
            loss_role  = (role_coherence_loss(a2d, metadata_dict, anchor_idxs)
                          if lambda_role > 0 else torch.tensor(0.0, device=device))

            loss = (loss_trip
                    + lambda_repulsion * loss_repel
                    + lambda_para      * loss_para
                    + lambda_st        * loss_st
                    + lambda_role      * loss_role)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
            optimizer.step()

            totals['total']         += loss.item()
            totals['triplet']       += loss_trip.item()
            totals['repulsion']     += loss_repel.item()
            totals['paragraph']     += loss_para.item()
            totals['stabilization'] += loss_st.item()
            totals['role']          += float(loss_role.item())
            n_batches += 1

        scheduler.step()
        for k in loss_history:
            loss_history[k].append(totals[k] / max(1, n_batches))

        print(f"[v5] Epoch {epoch+1}/{epochs} "
              f"total={loss_history['total'][-1]:.4f} "
              f"trip={loss_history['triplet'][-1]:.4f} "
              f"repel={loss_history['repulsion'][-1]:.4f} "
              f"para={loss_history['paragraph'][-1]:.4f} "
              f"stab={loss_history['stabilization'][-1]:.4f} "
              f"role={loss_history['role'][-1]:.4f}")

        # 每 2 个 epoch 保存一次检查点
        if (epoch + 1) % 2 == 0:
            checkpoint_path = save_path.replace(".pt", f"_epoch{epoch+1}.pt")
            torch.save({
                "mapper_state": mapper.state_dict(),
                "sbert_name": sbert_model_name,
                "embed_dim": embed_dim,
                "hidden_dims": hidden_dims,
                "loss_history": loss_history,
                "epoch": epoch
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # ---- final save（字段与 v3 一致，推理端可直接加载） ----
    torch.save({
        "mapper_state": mapper.state_dict(),
        "sbert_name": sbert_model_name,
        "embed_dim": embed_dim,
        "hidden_dims": hidden_dims,
        "loss_history": loss_history
    }, save_path)
    print("Saved final model to", save_path)

    # ---- 损失曲线 ----
    try:
        plt.figure(figsize=(10, 6))
        for k in ['total', 'triplet', 'repulsion', 'paragraph', 'stabilization', 'role']:
            plt.plot(loss_history[k], label=k)
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss History (v5)')
        plt.legend(); plt.grid(True)
        plt.savefig(save_path.replace('.pt', '_loss.png'))
        plt.close()
        print("Saved loss plot")
    except Exception as e:
        print("skip loss plot:", e)


# -----------------------
# 9) 使用示例
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplets", default=str(cfg.TRIPLETS))
    parser.add_argument("--metadata", default=str(cfg.FORMDB))
    parser.add_argument("--model", default=str(cfg.BGE_MODEL_PATH))
    parser.add_argument("--out", default=str(cfg.MAPPER_CKPT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-repulsion", type=float, default=0.4)
    parser.add_argument("--lambda-para", type=float, default=0.1)
    parser.add_argument("--lambda-st", type=float, default=0.05)
    parser.add_argument("--lambda-role", type=float, default=0.0)
    parser.add_argument("--embedding-cache", default=str(cfg.EMB_CACHE),
                        help="precompute_embeddings 的 .npy；存在则用缓存训练(推荐)，否则回退逐 batch 编码")
    args = parser.parse_args()

    train_single_stage(
        json_path=args.triplets,
        metadata_path=args.metadata,
        sbert_model_name=args.model,
        device=args.device,
        embed_dim=1024,
        hidden_dims=(256, 64, 32),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lambda_repulsion=args.lambda_repulsion,
        lambda_para=args.lambda_para,
        lambda_st=args.lambda_st,
        lambda_role=args.lambda_role,
        freeze_sbert=True,
        embedding_cache=args.embedding_cache,
        save_path=args.out
    )
