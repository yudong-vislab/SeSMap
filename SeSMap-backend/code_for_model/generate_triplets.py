#!/usr/bin/env python3
"""
Generate raw semantic triplets from formdatabase.json.

Input:
  data/outputs/formdatabase.json

Output:
  data/outputs/contrastive_triplets_raw.json

This replaces the old generate_tri_withinfo_v3.py runtime path. It uses BGE
embeddings directly and avoids hard-coded server paths / KeyBERT side effects.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import local_config as cfg


def load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of MSU records")
    return [r for r in data if str(r.get("sentence", "")).strip()]


def load_or_encode_embeddings(records: list[dict], cache_path: Path | None, model_path: Path, batch_size: int) -> np.ndarray:
    if cache_path and cache_path.exists():
        ids_path = Path(str(cache_path) + ".ids.json")
        if ids_path.exists():
            cached_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            current_ids = [r.get("idx", i) for i, r in enumerate(records)]
            emb = np.load(cache_path)
            if list(cached_ids) == current_ids and emb.shape[0] == len(records):
                print(f"[cache] embeddings {emb.shape} <- {cache_path}")
                return emb.astype(np.float32)
            print("[cache] embedding cache is stale; re-encoding")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_path), device="cpu")
    texts = [r["sentence"] for r in records]
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    return emb


def cosine_matrix(emb: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(emb, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    x = emb / denom
    return x @ x.T


def generate_triplets(
    records: list[dict],
    sim: np.ndarray,
    positives_per_anchor: int,
    negatives_per_anchor: int,
    max_triplets_per_anchor: int,
    min_positive: float,
    max_negative: float,
    prefer_same_category: bool,
) -> list[dict]:
    out: list[dict] = []
    n = len(records)
    for anchor_i in range(n):
        row = sim[anchor_i].copy()
        row[anchor_i] = -2.0

        positive_candidates = np.where(row >= min_positive)[0].tolist()
        if prefer_same_category:
            anchor_cat = records[anchor_i].get("category")
            positive_candidates.sort(
                key=lambda j: (
                    records[j].get("category") != anchor_cat,
                    -float(row[j]),
                )
            )
        else:
            positive_candidates.sort(key=lambda j: -float(row[j]))

        negative_candidates = np.where(row <= max_negative)[0].tolist()
        negative_candidates.sort(key=lambda j: float(row[j]))

        positives = positive_candidates[:positives_per_anchor]
        negatives = negative_candidates[:negatives_per_anchor]
        if not positives or not negatives:
            continue

        made = 0
        for p_idx in positives:
            for n_idx in negatives:
                if made >= max_triplets_per_anchor:
                    break
                a = records[anchor_i]
                p = records[p_idx]
                neg = records[n_idx]
                out.append({
                    "anchor": a["sentence"],
                    "positive": p["sentence"],
                    "negative": neg["sentence"],
                    "anchor_idx": a.get("idx", anchor_i),
                    "positive_idx": p.get("idx", p_idx),
                    "negative_idx": neg.get("idx", n_idx),
                    "positive_score": float(row[p_idx]),
                    "negative_score": float(row[n_idx]),
                    "source": "embedding_topk",
                })
                made += 1

    dedup = {}
    for t in out:
        key = (t["anchor_idx"], t["positive_idx"], t["negative_idx"])
        dedup[key] = t
    return list(dedup.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=cfg.FORMDB)
    parser.add_argument("--out", type=Path, default=cfg.TRIPLETS_RAW)
    parser.add_argument("--embedding-cache", type=Path, default=cfg.EMB_CACHE)
    parser.add_argument("--model", type=Path, default=cfg.BGE_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--positives-per-anchor", type=int, default=4)
    parser.add_argument("--negatives-per-anchor", type=int, default=8)
    parser.add_argument("--max-triplets-per-anchor", type=int, default=12)
    parser.add_argument("--min-positive", type=float, default=0.62)
    parser.add_argument("--max-negative", type=float, default=0.35)
    parser.add_argument("--no-category-preference", action="store_true")
    args = parser.parse_args()

    records = load_records(args.corpus)
    if len(records) < 3:
        raise RuntimeError("Need at least 3 MSU records to generate triplets")
    emb = load_or_encode_embeddings(records, args.embedding_cache, args.model, args.batch_size)
    sim = cosine_matrix(emb)
    triplets = generate_triplets(
        records,
        sim,
        positives_per_anchor=args.positives_per_anchor,
        negatives_per_anchor=args.negatives_per_anchor,
        max_triplets_per_anchor=args.max_triplets_per_anchor,
        min_positive=args.min_positive,
        max_negative=args.max_negative,
        prefer_same_category=not args.no_category_preference,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(triplets, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {len(triplets)} raw triplets -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
