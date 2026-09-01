#!/usr/bin/env python3
"""
Apply a trained 1024D->2D mapper to every database-*.json file in a case dir.

This is the bridge for refreshing existing frontend cases after a new mapper is
trained. It preserves the original database record structure and only replaces
the 2d_coord field.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import local_config as cfg
from code_for_data.formdatabase import load_sbert
from code_for_model.train_all_v5 import Bert2DMapper
from code_for_model.train_all_v7 import ResidualProjectionMapper


def load_mapper(path: Path, device: str):
    ckpt = torch.load(path, map_location=device)
    if ckpt.get("model_class") == "ResidualProjectionMapper":
        mapper = ResidualProjectionMapper(
            embed_dim=int(ckpt["embed_dim"]),
            width=int(ckpt.get("width", 512)),
            num_blocks=int(ckpt.get("num_blocks", 4)),
            dropout=float(ckpt.get("dropout", 0.0)),
            out_dim=int(ckpt.get("out_dim", 2)),
        ).to(device)
    else:
        mapper = Bert2DMapper(
            embed_dim=int(ckpt["embed_dim"]),
            hidden_dims=tuple(ckpt["hidden_dims"]),
            out_dim=2,
            normalize_output=bool(ckpt.get("normalize_output", True)),
        ).to(device)
    mapper.load_state_dict(ckpt["mapper_state"])
    mapper.eval()
    return mapper


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--combined-out", type=Path, default=None)
    parser.add_argument("--mapper", type=Path, default=cfg.MODEL_DIR / "bert2d_mapper_all_v7_pure.pt")
    parser.add_argument("--model", type=Path, default=cfg.BGE_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    db_files = sorted(args.case_dir.glob("database-*.json"))
    if not db_files:
        raise FileNotFoundError(f"No database-*.json files in {args.case_dir}")
    if not args.mapper.exists():
        raise FileNotFoundError(f"Mapper checkpoint not found: {args.mapper}")

    records_by_file: list[tuple[Path, list[dict]]] = []
    text_refs: list[tuple[int, int]] = []
    sentences: list[str] = []
    for file_idx, path in enumerate(db_files):
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a list")
        records_by_file.append((path, [dict(item) for item in records]))
        for rec_idx, item in enumerate(records):
            sent = str(item.get("sentence", "")).strip()
            if sent:
                text_refs.append((file_idx, rec_idx))
                sentences.append(sent)

    sbert = load_sbert(args.model, args.device)
    mapper = load_mapper(args.mapper, args.device)
    with torch.no_grad():
        emb = sbert.encode(
            sentences,
            batch_size=args.batch_size,
            convert_to_tensor=True,
            device=args.device,
            show_progress_bar=True,
        )
        coords = mapper(emb).cpu().numpy().tolist()

    for (file_idx, rec_idx), coord in zip(text_refs, coords):
        records_by_file[file_idx][1][rec_idx]["2d_coord"] = coord

    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined: list[dict] = []
    for path, records in records_by_file:
        out_path = args.out_dir / path.name
        out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        combined.extend(records)
        print(f"[case] {path.name}: records={len(records)} -> {out_path}")

    if args.combined_out:
        args.combined_out.parent.mkdir(parents=True, exist_ok=True)
        args.combined_out.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[combined] records={len(combined)} -> {args.combined_out}")
    print(f"[done] updated {len(sentences)} MSU coordinates using {args.mapper}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
