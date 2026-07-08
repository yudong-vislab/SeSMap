# precompute_embeddings.py
# ---------------------------------------------------------------------------
# 一次性把 MSU 语料的句向量用本地 bge 编码好并缓存成 .npy。
# 之后 train_all_v5 / eval_layout / OOS 评测都直接读缓存 -> 本地 CPU 即可复现，
# 不再需要每次重新加载 bge、也不依赖丢失的服务器。
#
# 用法:
#   python precompute_embeddings.py \
#       --corpus /path/to/formdatabase_vN.json \
#       --out    /path/to/emb_corpus.npy \
#       --id-field idx --sentence-field sentence
#
# 输出:
#   <out>            (N, 1024) float32，行序与 <out>.ids.json 一一对应
#   <out>.ids.json   [id0, id1, ...]  行号 -> MSU id 的映射（训练/评测按 id 查向量）
# ---------------------------------------------------------------------------

import os, json, argparse, sys
from pathlib import Path
import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import local_config as cfg


def load_records(path):
    d = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(d, dict):                      # 容错：dict 里找第一个 list
        for v in d.values():
            if isinstance(v, list):
                d = v; break
    assert isinstance(d, list), f"{path} 顶层应是 MSU 记录的 list"
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(cfg.FORMDB), help="MSU 语料 json（formdatabase 输出）")
    ap.add_argument("--out", default=str(cfg.EMB_CACHE), help="输出 .npy 路径")
    ap.add_argument("--id-field", default="idx")
    ap.add_argument("--sentence-field", default="sentence")
    ap.add_argument("--model", default=str(cfg.BGE_MODEL_PATH))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--normalize", action="store_true", help="L2 归一化（默认关，与训练一致）")
    args = ap.parse_args()

    recs = load_records(args.corpus)
    ids, sents = [], []
    for i, r in enumerate(recs):
        if args.sentence_field not in r:
            continue
        ids.append(r.get(args.id_field, i))
        sents.append(r[args.sentence_field])
    print(f"[corpus] {args.corpus}: {len(sents)} sentences")

    ids_path = args.out + ".ids.json"
    if os.path.exists(args.out) and os.path.exists(ids_path):
        X = np.load(args.out)
        old_ids = json.load(open(ids_path))
        if X.shape[0] == len(sents) and old_ids == ids:
            print(f"[cache] up-to-date: {args.out} {X.shape} — skip")
            return
        print("[cache] stale, re-encoding...")

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(args.model, device="cpu")
    X = m.encode(sents, batch_size=args.batch_size, show_progress_bar=True,
                 convert_to_numpy=True, normalize_embeddings=args.normalize).astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.save(args.out, X)
    json.dump(ids, open(ids_path, "w"))
    print(f"[done] {X.shape} -> {args.out}")
    print(f"[done] id order -> {ids_path}")
    print("提示: 训练/评测时用 dict(zip(ids, X)) 按 MSU id 取向量。")


if __name__ == "__main__":
    main()
