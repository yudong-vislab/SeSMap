#!/usr/bin/env python3
"""
extract_thumbnails.py — 从 MinerU 产物为每篇论文自动选一张"典型主图"，缩略后存下，
并生成前端 gallery manifest（方案 B：未来上传论文即自动接入 gallery，无需改前端代码）。

主图规则：caption 以 'Figure 1'/'Fig. 1' 开头的优先（可视化论文里通常是 teaser/系统总览）；
否则取页序最早的图片块；caption 含 overview/framework/pipeline/system/architecture 略加权。

输入：mineru corpus dir（含 <paper>/_mineru/**/auto/*_content_list.json + images/）+ papers.json。
输出：
  data/<case>/thumbnails/c<paper_id>.png
  data/<case>/gallery.json = [{paper_id, title, thumbnail, semanticCountryId, sourceId, caption}]

用法：
  python3 code_for_data/extract_thumbnails.py \
    --corpus data/bio_eval/stages/01_corpus \
    --papers data/bio_eval/stages/02_msu/papers.json \
    --case case3 --source-offset 8
"""
from __future__ import annotations
import sys, json, argparse, glob, re
from pathlib import Path

from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import local_config as cfg

FIG1 = re.compile(r"^\s*(figure|fig\.?)\s*1\b", re.I)
BOOST = ("overview", "framework", "pipeline", "system", "architecture", "teaser", "workflow")


def caption_text(block) -> str:
    c = block.get("image_caption") or []
    return " ".join(c) if isinstance(c, list) else str(c)


def pick_main_figure(content_list):
    imgs = [b for b in content_list if b.get("type") == "image" and b.get("img_path")]
    if not imgs:
        return None

    def score(b):
        cap = caption_text(b).lower()
        s = 0.0
        if FIG1.match(cap):
            s += 100.0
        s += sum(2.0 for kw in BOOST if kw in cap)
        s -= 0.01 * float(b.get("page_idx", 999))   # 页序越早略优先
        return s
    return max(imgs, key=score)


def find_content_list(paper_dir: Path):
    cands = sorted(glob.glob(str(paper_dir / "_mineru" / "**" / "*_content_list.json"), recursive=True))
    non_v2 = [c for c in cands if "_v2" not in c]
    if non_v2:
        return Path(non_v2[0])
    return Path(cands[0]) if cands else None


def find_fallback_image(paper_dir: Path):
    """Return a MinerU-extracted image when the content list has no image block.

    Some PDFs expose valid files under ``images/`` but omit their image blocks
    from ``*_content_list.json``.  Keeping this fallback avoids an empty gallery
    card while still using an asset extracted from the same source paper.
    """
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    candidates = []
    for folder in (paper_dir / "images", paper_dir / "_mineru"):
        if folder.exists():
            for pattern in patterns:
                candidates.extend(folder.rglob(pattern))
    return sorted(candidates)[0] if candidates else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--papers", type=Path, required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--data-root", type=Path, default=cfg.DATA_ROOT)
    ap.add_argument("--source-offset", type=int, default=0, help="sourceId 全局偏移(避免跨 case 撞号)")
    ap.add_argument("--size", type=int, default=480, help="缩略图最长边像素")
    args = ap.parse_args()

    papers = json.loads(args.papers.read_text(encoding="utf-8"))
    thumbs_dir = args.data_root / args.case / "thumbnails"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for pid, folder in enumerate(papers):
        cl_path = find_content_list(args.corpus / folder)
        entry = {
            "paper_id": pid,
            "title": folder.replace("_", " ").strip(),
            "semanticCountryId": f"c{pid}",
            "sourceId": f"c{args.source_offset + pid}",
            "thumbnail": None,
            "caption": None,
        }
        if not cl_path:
            print(f"  [skip] paper {pid} {folder[:40]}: no content_list")
            manifest.append(entry)
            continue
        fig = pick_main_figure(json.loads(cl_path.read_text(encoding="utf-8")))
        if fig:
            img_src = cl_path.parent / fig["img_path"]   # img_path 相对 auto/ 目录
            if img_src.exists():
                out = thumbs_dir / f"c{pid}.png"
                im = Image.open(img_src).convert("RGB")
                im.thumbnail((args.size, args.size))
                im.save(out)
                entry["thumbnail"] = out.name
                entry["caption"] = caption_text(fig)[:200]
        else:
            img_src = find_fallback_image(args.corpus / folder)
            if img_src:
                out = thumbs_dir / f"c{pid}.png"
                im = Image.open(img_src).convert("RGB")
                im.thumbnail((args.size, args.size))
                im.save(out)
                entry["thumbnail"] = out.name
                entry["caption"] = "MinerU-extracted figure fallback"
        manifest.append(entry)
        tag = ("-> " + entry["thumbnail"]) if entry["thumbnail"] else "NO FIGURE"
        print(f"  paper {pid} {folder[:44]:44} {tag}")

    out_manifest = args.data_root / args.case / "gallery.json"
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {len(manifest)} entries -> {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
