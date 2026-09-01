#!/usr/bin/env python3
"""
Validate, enrich, and normalize raw triplets.

Input:
  contrastive_triplets_raw.json + formdatabase.json

Output:
  contrastive_triplets.json

The output keeps the training-compatible keys used by train_all_v5.py and adds
metadata/context fields for inspection and reproducibility.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import local_config as cfg


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def text_window(records_by_para: dict[tuple[int, int], list[dict]], item: dict, radius: int) -> str:
    key = (item.get("paper_id", -1), item.get("para_id", -1))
    para_items = sorted(records_by_para.get(key, []), key=lambda x: x.get("idx", 0))
    if not para_items:
        return item.get("sentence", "")
    idxs = [i for i, rec in enumerate(para_items) if rec.get("idx") == item.get("idx")]
    if not idxs:
        return item.get("sentence", "")
    i = idxs[0]
    lo = max(0, i - radius)
    hi = min(len(para_items), i + radius + 1)
    return " ".join(str(rec.get("sentence", "")).strip() for rec in para_items[lo:hi]).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplets", type=Path, default=cfg.TRIPLETS_RAW)
    parser.add_argument("--corpus", type=Path, default=cfg.FORMDB)
    parser.add_argument("--out", type=Path, default=cfg.TRIPLETS)
    parser.add_argument("--context-radius", type=int, default=1)
    parser.add_argument("--drop-same-anchor-negative", action="store_true", default=True)
    args = parser.parse_args()

    records = load_json(args.corpus)
    raw_triplets = load_json(args.triplets)
    by_idx = {int(item.get("idx", i)): item for i, item in enumerate(records) if item.get("sentence")}
    by_para: dict[tuple[int, int], list[dict]] = {}
    for item in by_idx.values():
        by_para.setdefault((item.get("paper_id", -1), item.get("para_id", -1)), []).append(item)

    refined = []
    seen = set()
    dropped = 0
    for t in raw_triplets:
        try:
            a_idx = int(t["anchor_idx"])
            p_idx = int(t["positive_idx"])
            n_idx = int(t["negative_idx"])
        except Exception:
            dropped += 1
            continue
        if len({a_idx, p_idx, n_idx}) < 3:
            dropped += 1
            continue
        if a_idx not in by_idx or p_idx not in by_idx or n_idx not in by_idx:
            dropped += 1
            continue
        a, p, n = by_idx[a_idx], by_idx[p_idx], by_idx[n_idx]
        if args.drop_same_anchor_negative and a.get("sentence") == n.get("sentence"):
            dropped += 1
            continue
        key = (a_idx, p_idx, n_idx)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)

        refined.append({
            "anchor": a["sentence"],
            "positive": p["sentence"],
            "negative": n["sentence"],
            "anchor_idx": a_idx,
            "positive_idx": p_idx,
            "negative_idx": n_idx,
            "anchor_context": text_window(by_para, a, args.context_radius),
            "positive_context": text_window(by_para, p, args.context_radius),
            "negative_context": text_window(by_para, n, args.context_radius),
            "anchor_category": a.get("category"),
            "positive_category": p.get("category"),
            "negative_category": n.get("category"),
            "anchor_paper": a.get("paper_id"),
            "positive_paper": p.get("paper_id"),
            "negative_paper": n.get("paper_id"),
            "anchor_para": a.get("para_id"),
            "positive_para": p.get("para_id"),
            "negative_para": n.get("para_id"),
            "positive_score": t.get("positive_score"),
            "negative_score": t.get("negative_score"),
            "source": t.get("source", "raw"),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(refined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {len(refined)} refined triplets -> {args.out} (dropped {dropped})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
