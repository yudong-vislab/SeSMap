#!/usr/bin/env python3
"""Verify that every gallery paper maps to exactly its own MSUs and HSUs.

Run from SeSMap-backend:
    python3 scripts/audit_source_mappings.py

The command is intentionally offline: it validates committed case data before a
release, so a rebuilt case cannot silently swap gallery papers sharing cN ids.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def normalized_title(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def audit_case(case_dir: Path) -> list[str]:
    errors: list[str] = []
    case_id = case_dir.name
    map_path = case_dir / "semantic_map_data.json"
    gallery_path = case_dir / "gallery.json"
    if not map_path.is_file():
        return [f"{case_id}: missing semantic_map_data.json"]
    if not gallery_path.is_file():
        return [f"{case_id}: missing gallery.json"]

    semantic_map = load_json(map_path)
    gallery = load_json(gallery_path)
    if not isinstance(gallery, list):
        return [f"{case_id}: gallery.json must be a JSON array"]

    msu_index = semantic_map.get("msu_index") or {}
    country_to_msu_ids: dict[str, set[str]] = defaultdict(set)
    for subspace in semantic_map.get("subspaces") or []:
        for hsu in subspace.get("hexList") or []:
            country_id = str(hsu.get("country_id") or "")
            if not country_id:
                errors.append(f"{case_id}: HSU without country_id")
                continue
            country_to_msu_ids[country_id].update(str(msu_id) for msu_id in hsu.get("msu_ids") or [])

    country_to_papers: dict[str, set[int | str]] = {}
    country_to_titles: dict[str, set[str]] = {}
    for country_id, msu_ids in country_to_msu_ids.items():
        paper_ids: set[int | str] = set()
        paper_titles: set[str] = set()
        for msu_id in msu_ids:
            msu = msu_index.get(msu_id)
            if not isinstance(msu, dict):
                errors.append(f"{case_id}: {country_id} references absent MSU {msu_id}")
                continue
            if msu.get("paper_id") is None:
                errors.append(f"{case_id}: MSU {msu_id} in {country_id} has no paper_id")
                continue
            paper_ids.add(msu["paper_id"])
            title = normalized_title(msu.get("paper_info"))
            if title:
                paper_titles.add(title)
        country_to_papers[country_id] = paper_ids
        country_to_titles[country_id] = paper_titles
        if len(paper_ids) != 1:
            errors.append(
                f"{case_id}: {country_id} contains MSUs from {sorted(map(str, paper_ids)) or ['no']} papers"
            )
        if len(paper_titles) > 1:
            errors.append(f"{case_id}: {country_id} contains MSUs with conflicting paper_info titles")

    gallery_by_country: dict[str, dict] = {}
    for item in gallery:
        if not isinstance(item, dict):
            errors.append(f"{case_id}: gallery contains a non-object item")
            continue
        country_id = str(item.get("semanticCountryId") or "")
        if not country_id:
            errors.append(f"{case_id}: gallery item {item.get('title', '<untitled>')!r} has no semanticCountryId")
            continue
        if country_id in gallery_by_country:
            errors.append(f"{case_id}: gallery has duplicate country mapping {country_id}")
            continue
        gallery_by_country[country_id] = item
        if item.get("paper_id") is None:
            errors.append(f"{case_id}: gallery item {item.get('title', '<untitled>')!r} has no paper_id")

    for country_id, paper_ids in sorted(country_to_papers.items()):
        item = gallery_by_country.get(country_id)
        if not item:
            errors.append(f"{case_id}: map country {country_id} is missing from gallery.json")
            continue
        if len(paper_ids) == 1 and item.get("paper_id") not in paper_ids:
            errors.append(
                f"{case_id}: gallery {item.get('title', '<untitled>')!r} maps to {country_id}, "
                f"but that country contains paper_id {next(iter(paper_ids))}"
            )
        map_titles = country_to_titles.get(country_id, set())
        gallery_title = normalized_title(item.get("title"))
        if len(map_titles) == 1 and gallery_title not in map_titles:
            errors.append(
                f"{case_id}: gallery {item.get('title', '<untitled>')!r} maps to {country_id}, "
                f"but its MSUs identify as {next(iter(map_titles))!r}"
            )

    for country_id in sorted(set(gallery_by_country) - set(country_to_papers)):
        errors.append(f"{case_id}: gallery country {country_id} does not exist in semantic_map_data.json")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="Case IDs to audit (default: every data/case* directory)")
    args = parser.parse_args()
    case_dirs = [DATA_DIR / case_id for case_id in args.cases] if args.cases else sorted(DATA_DIR.glob("case*"))

    all_errors: list[str] = []
    for case_dir in case_dirs:
        errors = audit_case(case_dir)
        if errors:
            all_errors.extend(errors)
            continue
        print(f"OK  {case_dir.name}: gallery, HSU country_id, MSU paper_id, and paper title agree")

    if all_errors:
        print("\nSource mapping audit failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in all_errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
