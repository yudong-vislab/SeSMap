#!/usr/bin/env python3
"""
Apply the trained Bert2DMapper to formdatabase.json and write formdatabase_v2.0.json.

Input:
  data/outputs/formdatabase.json
  data/outputs/bert2d_mapper_all_v5.pt
  models/bge-large-en-v1.5/

Output:
  data/outputs/formdatabase_v2.0.json with a 2d_coord field on every text MSU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, models

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import local_config as cfg
from code_for_model.train_all_v5 import Bert2DMapper
try:
    from code_for_model.train_all_v7 import ResidualProjectionMapper
except Exception:
    ResidualProjectionMapper = None


def load_sbert(model_path: Path, device: str):
    try:
        return SentenceTransformer(str(model_path), device=device)
    except Exception:
        word_embedding_model = models.Transformer(str(model_path))
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        return SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=cfg.FORMDB)
    parser.add_argument("--out", type=Path, default=cfg.FORMDB_V2)
    parser.add_argument("--mapper", type=Path, default=cfg.MAPPER_CKPT)
    parser.add_argument("--model", type=Path, default=cfg.BGE_MODEL_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{args.input} must contain a list")
    if not args.mapper.exists():
        raise FileNotFoundError(f"Mapper checkpoint not found: {args.mapper}")

    sbert = load_sbert(args.model, args.device)
    ckpt = torch.load(args.mapper, map_location=args.device)
    if ckpt.get("model_class") == "ResidualProjectionMapper":
        if ResidualProjectionMapper is None:
            raise RuntimeError("ResidualProjectionMapper is unavailable; cannot load v7 checkpoint")
        mapper = ResidualProjectionMapper(
            embed_dim=int(ckpt["embed_dim"]),
            width=int(ckpt.get("width", 512)),
            num_blocks=int(ckpt.get("num_blocks", 4)),
            dropout=float(ckpt.get("dropout", 0.0)),
            out_dim=int(ckpt.get("out_dim", 2)),
        ).to(args.device)
    else:
        mapper = Bert2DMapper(
            embed_dim=int(ckpt["embed_dim"]),
            hidden_dims=tuple(ckpt["hidden_dims"]),
            out_dim=2,
            normalize_output=bool(ckpt.get("normalize_output", True)),  # v6 存 False
        ).to(args.device)
    mapper.load_state_dict(ckpt["mapper_state"])
    mapper.eval()

    updated = [dict(item) for item in records]
    text_idx = [i for i, rec in enumerate(updated) if str(rec.get("sentence", "")).strip()]
    sentences = [str(updated[i]["sentence"]).strip() for i in text_idx]
    with torch.no_grad():
        emb = sbert.encode(sentences, batch_size=32, convert_to_tensor=True,
                           device=args.device, show_progress_bar=True)
        # 关键：整批一次性 forward。Bert2DMapper.forward 里有 out/out.max(dim=0) 的“按批归一化”，
        # 若像原来那样每次只喂 1 句(batch=1)，会把每个 MSU 都压成同一个坐标 [10,10]（垃圾输出）。
        coords = mapper(emb).cpu().numpy().tolist()
    for i, c in zip(text_idx, coords):
        updated[i]["2d_coord"] = c

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {len(updated)} records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
