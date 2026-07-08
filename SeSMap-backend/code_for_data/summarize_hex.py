#!/usr/bin/env python3
"""
Generate LLM summaries for HSU hex cells.

Input:
  data/outputs/hexagon_info.json
  data/outputs/formdatabase_v2.0.json

Output:
  data/outputs/summaries.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import local_config as cfg
from services.llm_config import get_openai_client, model_for


def summarize_with_llm(text: str) -> str:
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model_for("summary"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize clustered MSU sentences for a semantic-map HSU. "
                    "Use only the provided sentences, preserve technical terms, and write in English. "
                    "Identify the shared theme and key evidence in one concise paragraph. "
                    "Do not add outside knowledge or generic filler."
                ),
            },
            {
                "role": "user",
                "content": (
                    "The following MSU sentences are close in the semantic layout and belong to one HSU cluster. "
                    "Write a 45-80 word evidence-grounded English summary of their shared meaning.\n\n"
                    f"MSU sentences:\n{text}"
                ),
            },
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def fallback_summary(sentences: list[str], max_sentences: int = 3) -> str:
    picked = [s.strip() for s in sentences if s.strip()][:max_sentences]
    return " ".join(picked)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hex-info", type=Path, default=cfg.HEX_INFO)
    parser.add_argument("--formdb", type=Path, default=cfg.FORMDB_V2)
    parser.add_argument("--out", type=Path, default=cfg.SUMMARIES)
    parser.add_argument("--no-llm", action="store_true", help="Use sentence fallback summaries.")
    args = parser.parse_args()

    hex_cells = json.loads(args.hex_info.read_text(encoding="utf-8"))
    records = json.loads(args.formdb.read_text(encoding="utf-8"))
    by_msu = {int(item.get("MSU_id", item.get("idx"))): item for item in records if item.get("sentence")}

    results = []
    for i, cell in enumerate(hex_cells, start=1):
        msu_ids = [int(x) for x in cell.get("MSU_ids", [])]
        sentences = [by_msu[mid]["sentence"] for mid in msu_ids if mid in by_msu]
        text = "\n".join(f"- {s}" for s in sentences)
        if args.no_llm:
            summary = fallback_summary(sentences)
        else:
            print(f"[summary] {i}/{len(hex_cells)} cell={cell.get('hex_coord')} msus={len(sentences)}")
            summary = summarize_with_llm(text) if sentences else ""
        results.append({
            "hex_coord": cell.get("hex_coord", []),
            "country": cell.get("country", 0),
            "MSU_ids": msu_ids,
            "summary": summary,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {len(results)} summaries -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
