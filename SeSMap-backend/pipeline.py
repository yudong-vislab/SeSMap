#!/usr/bin/env python3
"""
End-to-end SeSMap backend data/model pipeline.

Default flow:
  PDF -> Markdown -> MSU database -> raw triplets -> refined triplets
  -> embedding cache -> mapper training -> 2D coordinates -> hex cells
  -> HSU summaries -> database/summary case files -> semantic_map_data.json
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import local_config as cfg

BACKEND = Path(__file__).resolve().parent

STAGES = [
    "pdf",
    "corpus",
    "triplets",
    "refine_triplets",
    "embeddings",
    "train",
    "coords",
    "hex",
    "summaries",
    "case_files",
    "semantic_map",
]


def run(cmd: list[str], dry_run: bool = False) -> None:
    print("\n$", " ".join(str(x) for x in cmd))
    if not dry_run:
        subprocess.run([str(x) for x in cmd], cwd=BACKEND, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    parser.add_argument("--to-stage", choices=STAGES, default=STAGES[-1])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM calls for corpus extraction and HSU summaries.")
    parser.add_argument("--skip-train", action="store_true", help="Use existing mapper checkpoint.")
    parser.add_argument("--hex-size", type=float, default=0.15)
    parser.add_argument("--train-epochs", type=int, default=20)
    args = parser.parse_args()

    cfg.ensure_dirs()
    start = STAGES.index(args.from_stage)
    end = STAGES.index(args.to_stage)
    wanted = set(STAGES[start:end + 1])
    py = sys.executable

    if "pdf" in wanted:
        run([py, "code_for_data/mineru_pdf.py"], args.dry_run)
    if "corpus" in wanted:
        cmd = [py, "code_for_data/build_corpus.py"]
        if args.no_llm:
            cmd.append("--no-llm")
        run(cmd, args.dry_run)
    if "triplets" in wanted:
        run([py, "code_for_model/generate_triplets.py"], args.dry_run)
    if "refine_triplets" in wanted:
        run([py, "code_for_model/refine_triplets.py"], args.dry_run)
    if "embeddings" in wanted:
        run([py, "code_for_model/precompute_embeddings.py"], args.dry_run)
    if "train" in wanted and not args.skip_train:
        run([py, "code_for_model/train_all_v5.py", "--epochs", str(args.train_epochs)], args.dry_run)
    if "coords" in wanted:
        run([py, "code_for_data/formdatabase.py"], args.dry_run)
    if "hex" in wanted:
        run([py, "code_for_data/generate_hex.py", "--hex-size", str(args.hex_size)], args.dry_run)
    if "summaries" in wanted:
        cmd = [py, "code_for_data/summarize_hex.py"]
        if args.no_llm:
            cmd.append("--no-llm")
        run(cmd, args.dry_run)
    if "case_files" in wanted:
        run([py, "code_for_data/build_case_files.py"], args.dry_run)
    if "semantic_map" in wanted:
        run([
            py,
            "build_semantic_map_from_db_summary.py",
            "--case-dir",
            str(cfg.CASE_BUILD_DIR),
            "--out",
            str(cfg.SEMANTIC_MAP),
        ], args.dry_run)

    print("\n[pipeline] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
