#!/usr/bin/env python3
"""
Group 2D MSU coordinates into HSU hexagons.

Input:
  data/outputs/formdatabase_v2.0.json

Output:
  data/outputs/hexagon_info.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import local_config as cfg


def cube_round(x: float, y: float, z: float) -> tuple[int, int]:
    rx, ry, rz = round(x), round(y), round(z)
    dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)
    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry
    return int(rx), int(rz)


def pixel_to_axial(x: float, y: float, size: float) -> tuple[int, int]:
    q = (np.sqrt(3) / 3 * x - 1.0 / 3 * y) / size
    r = (2.0 / 3 * y) / size
    return cube_round(q, -q - r, r)


def group_points(records: list[dict], hex_size: float) -> list[dict]:
    grouped: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for item in records:
        coord = item.get("2d_coord")
        if not isinstance(coord, list) or len(coord) != 2:
            continue
        msu_id = item.get("MSU_id", item.get("idx"))
        if msu_id is None:
            continue
        paper_id = int(item.get("paper_id", 0))
        q, r = pixel_to_axial(float(coord[0]), float(coord[1]), hex_size)
        grouped[(q, r, paper_id)].append(int(msu_id))

    out = []
    for (q, r, paper_id), msu_ids in sorted(grouped.items()):
        out.append({
            "hex_coord": [q, r],
            "country": paper_id,
            "MSU_ids": sorted(msu_ids),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=cfg.FORMDB_V2)
    parser.add_argument("--out", type=Path, default=cfg.HEX_INFO)
    parser.add_argument("--hex-size", type=float, default=0.15)
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{args.input} must contain a list")
    cells = group_points(records, args.hex_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cells, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {len(cells)} hex cells -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
