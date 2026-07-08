#!/usr/bin/env python3
"""
train_all_v6.py — 邻域保持版投影（parametric t-SNE / neighbor embedding）

问题诊断：v5(triplet + paragraph) 的 knnOv≈0.04，远低于 t-SNE(0.32)。
原因：triplet 只做“稀疏相对排序”，从不优化稠密邻域；paragraph/repulsion 还主导了布局。
所以无论 triplet 收敛得多好，保真度都上不去。

v6 直接优化“邻域保持”：让 2D 的近邻结构对齐 bge 1024 维的近邻结构（parametric t-SNE 的 KL 目标）。
  * 高 knnOv：直接优化 knnOv 度量的东西。
  * 仍是参数化 MLP：能把“未见句子”投进固定地图 —— 这是相对非参数 t-SNE/UMAP 的真优势(OOS/稳定坐标系)。
  * paper_id 不参与，仅上色。
  * 去掉 v5 里有害的 out/out.max 归一化（normalize_output=False）。

输入：data/stages/04_embeddings/emb_corpus.npy (+ .ids.json)  —— 没有则从 formdatabase 现编码。
输出：data/stages/05_model/bert2d_mapper_all_v6.pt（格式与 v5 一致，formdatabase.py/推理可直接用）。

用法：
  python3 code_for_model/train_all_v6.py                 # 用现成缓存，CPU 快
  python3 code_for_data/formdatabase.py --mapper data/stages/05_model/bert2d_mapper_all_v6.pt
  python3 code_for_model/eval_faithfulness.py            # 复测 knnOv/trust（应从 0.04 拉到接近 UMAP）
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import local_config as cfg
from code_for_model.train_all_v5 import Bert2DMapper


def load_embeddings(model_path, cache, corpus, device):
    ids_path = Path(str(cache) + ".ids.json")
    if Path(cache).exists() and ids_path.exists():
        X = np.load(str(cache)).astype(np.float32)
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        print(f"[cache] embeddings {X.shape} <- {cache}")
        return torch.tensor(X, device=device), ids
    print("[encode] 无缓存，从 formdatabase 现编码...")
    from sentence_transformers import SentenceTransformer
    recs = [r for r in json.loads(Path(corpus).read_text(encoding="utf-8"))
            if str(r.get("sentence", "")).strip()]
    ids = [r.get("idx", i) for i, r in enumerate(recs)]
    m = SentenceTransformer(str(model_path), device=device)
    X = m.encode([r["sentence"] for r in recs], batch_size=32, convert_to_numpy=True,
                 normalize_embeddings=False, show_progress_bar=True).astype(np.float32)
    return torch.tensor(X, device=device), ids


def hd_affinities(Xb, k, eps=1e-12):
    """高维近邻分布 P：自适应带宽(=第 k 近邻距离) 的高斯亲和，对称归一化。"""
    B = Xb.size(0)
    D2 = torch.cdist(Xb, Xb) ** 2 + torch.eye(B, device=Xb.device) * 1e12  # 排除自身
    kk = min(k, B - 1)
    knn = torch.topk(D2, kk, largest=False).values
    sigma2 = knn[:, -1].clamp_min(eps).unsqueeze(1)
    P = torch.softmax(-D2 / (2 * sigma2), dim=1)      # p_{j|i}
    P = (P + P.t()) / (2 * B)                          # 对称化 + 归一化
    return P.clamp_min(eps)


def ld_affinities(Z, eps=1e-12):
    """低维亲和 Q：Student-t，归一化。"""
    D2 = torch.cdist(Z, Z) ** 2
    num = 1.0 / (1.0 + D2)
    num = num - torch.diag(torch.diagonal(num))       # 对角置零
    return (num / num.sum().clamp_min(eps)).clamp_min(eps)


def tsne_kl(Xb, Z, k):
    P = hd_affinities(Xb, k)
    Q = ld_affinities(Z)
    return (P * (P.log() - Q.log())).sum()            # KL(P||Q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(cfg.EMB_CACHE))
    ap.add_argument("--corpus", default=str(cfg.FORMDB))
    ap.add_argument("--model", default=str(cfg.BGE_MODEL_PATH))
    ap.add_argument("--out", default=str(cfg.MODEL_DIR / "bert2d_mapper_all_v6.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=15, help="高维近邻数(自适应带宽)")
    ap.add_argument("--hidden", default="256,64,32")
    args = ap.parse_args()

    device = args.device
    X, ids = load_embeddings(args.model, args.cache, args.corpus, device)
    N, D = X.shape
    print(f"[v6] N={N}, dim={D}, device={device}")
    print("[v6] 目标: 邻域保持(parametric t-SNE KL)；paper_id 不参与，仅上色")

    hidden = tuple(int(h) for h in args.hidden.split(","))
    mapper = Bert2DMapper(embed_dim=D, hidden_dims=hidden, out_dim=2, normalize_output=False).to(device)
    opt = optim.AdamW(mapper.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 3), gamma=0.5)

    loader = DataLoader(TensorDataset(torch.arange(N)), batch_size=args.batch_size,
                        shuffle=True, drop_last=True)
    hist = []
    mapper.train()
    for ep in range(args.epochs):
        tot, nb = 0.0, 0
        for (bidx,) in loader:
            Xb = X[bidx.to(device)]
            Z = mapper(Xb)
            loss = tsne_kl(Xb, Z, args.k)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), 5.0)
            opt.step()
            tot += float(loss.item()); nb += 1
        sched.step()
        hist.append(tot / max(1, nb))
        if ep == 0 or (ep + 1) % 10 == 0:
            print(f"[v6] epoch {ep+1}/{args.epochs}  KL={hist[-1]:.4f}")

    out = args.out
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mapper_state": mapper.state_dict(), "embed_dim": D, "hidden_dims": hidden,
                "normalize_output": False, "arch": "v6", "loss_history": {"kl": hist}}, out)
    print(f"[v6] saved -> {out}")
    try:
        plt.figure(); plt.plot(hist); plt.xlabel("epoch"); plt.ylabel("KL(P||Q)")
        plt.title("v6 neighbor-preservation"); plt.grid(True)
        plt.savefig(out.replace(".pt", "_loss.png")); plt.close()
        print("saved loss plot")
    except Exception as e:
        print("skip plot:", e)


if __name__ == "__main__":
    main()
