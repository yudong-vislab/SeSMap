#!/usr/bin/env python3
"""
Split generated formdatabase/summaries into database-*.json and summary-*.json
pairs for build_semantic_map_from_db_summary.py.

This is the bridge from the pipeline outputs to the existing frontend case
format.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import local_config as cfg

CANONICAL = ["Background", "Method", "Experiment", "Result", "Conclusion", "Other"]


def safe_name(value: str) -> str:
    value = str(value or "Other").strip() or "Other"
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    aliases = {
        "experiment_setup": "Experiment",
        "conclusion_implication": "Conclusion",
        "other": "Other",
        "others": "Other",
    }
    if normalized in aliases:
        return aliases[normalized]
    if value.lower() in {"other", "others"}:
        return "Other"
    for c in CANONICAL:
        if value.lower() == c.lower():
            return c
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "Other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formdb", type=Path, default=cfg.FORMDB_V2)
    parser.add_argument("--summaries", type=Path, default=cfg.SUMMARIES)
    parser.add_argument("--out-dir", type=Path, default=cfg.CASE_BUILD_DIR)
    args = parser.parse_args()

    records = json.loads(args.formdb.read_text(encoding="utf-8"))
    summaries = json.loads(args.summaries.read_text(encoding="utf-8"))
    by_msu = {int(item.get("MSU_id", item.get("idx"))): item for item in records if item.get("sentence")}

    db_by_category: dict[str, list[dict]] = defaultdict(list)
    for item in records:
        cat = safe_name(item.get("category"))
        db_by_category[cat].append(item)

    summary_by_category: dict[str, list[dict]] = defaultdict(list)
    for cell in summaries:
        ids = [int(x) for x in cell.get("MSU_ids", [])]
        ids_by_cat: dict[str, list[int]] = defaultdict(list)
        for mid in ids:
            rec = by_msu.get(mid)
            if rec:
                ids_by_cat[safe_name(rec.get("category"))].append(mid)
        for cat, cat_ids in ids_by_cat.items():
            next_cell = dict(cell)
            next_cell["MSU_ids"] = sorted(cat_ids)
            summary_by_category[cat].append(next_cell)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    categories = [c for c in CANONICAL if c in db_by_category or c in summary_by_category]
    categories += sorted((set(db_by_category) | set(summary_by_category)) - set(categories))

    for cat in categories:
        (args.out_dir / f"database-{cat}.json").write_text(
            json.dumps(db_by_category.get(cat, []), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (args.out_dir / f"summary-{cat}.json").write_text(
            json.dumps(summary_by_category.get(cat, []), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[case] {cat}: db={len(db_by_category.get(cat, []))}, cells={len(summary_by_category.get(cat, []))}")

    print(f"[done] case files -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
