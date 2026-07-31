#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
augment_pairs_llm.py — 方案B 第①步：LLM 增强的跨论文语义等价对挖掘（可扩展监督）

思路（对齐论文立意：发现跨论文语义对应）：
  1) 用 bge 余弦在 **不同论文之间** 生成候选对（相似度落在 [tau_lo, tau_hi]，
     既排除近乎重复、又排除毫不相关）——这是可扩展的候选生成，不需人工。
  2) 用 LLM **判定** 每个候选对是否为"语义等价/对应"（专家抽查可在此之后叠加）。
     只保留 equivalent / corresponding 的作为正样本。这就是"语义角度匹配 + LLM 判定"。
  3) （可选）对采样 MSU 生成释义正样本（语义不变增强）。
  4) 输出 pairs.json（行索引对，指向 EMB_CACHE 的行），并切 train/test，
     供对比训练 + BZ 召回评测（评测只用 test 对，避免循环）。

用法：
  # 只生成候选、不调 LLM（先看候选量/成本）
  python3 code_for_data/augment_pairs_llm.py --dry-run
  # 正式跑（调 LLM 判定）
  python3 code_for_data/augment_pairs_llm.py --max-candidates 4000 --judge-batch 10
"""
from __future__ import annotations
import argparse, json, sys, random, re
from pathlib import Path
import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import local_config as cfg


def load_corpus_and_emb(corpus_path, emb_path):
    recs = json.load(open(corpus_path, encoding="utf-8"))
    X = np.load(emb_path).astype(np.float32)
    ids = json.load(open(str(emb_path) + ".ids.json", encoding="utf-8"))
    if X.shape[0] != len(ids):
        raise ValueError(f"emb/ids mismatch: {X.shape[0]} vs {len(ids)}")
    # id -> record（按 idx 字段对齐；缺失则按行号）
    by_id = {}
    for i, r in enumerate(recs):
        by_id[r.get("idx", i)] = r
    rows = []  # row -> MSU metadata used for local candidate filtering
    for rid in ids:
        r = by_id.get(rid, {})
        rows.append({
            "sentence": r.get("sentence", ""),
            "paper_id": r.get("paper_id", -1),
            "category": r.get("category", "Other"),
            "rank": r.get("rank", -1),
        })
    return X, rows


def gen_candidates(X, rows, tau_lo, tau_hi, per_anchor, max_candidates, seed,
                   allowed_categories=None, min_rank=1):
    """Generate locally filtered, cross-paper semantic-pair candidates.

    Only MSUs that pass the category/importance filter can become anchors or
    candidates.  This keeps the complete corpus local while sending only a
    small, high-value subset to the external judge.
    """
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    N = Xn.shape[0]
    papers = np.array([r["paper_id"] for r in rows])
    categories = np.array([str(r.get("category", "Other")) for r in rows])
    ranks = np.array([float(r.get("rank", -1) or -1) for r in rows])
    allowed = {str(x).strip().lower() for x in (allowed_categories or []) if str(x).strip()}
    eligible = ranks >= min_rank
    if allowed:
        eligible &= np.array([c.lower() in allowed for c in categories])
    rng = random.Random(seed)
    order = list(range(N)); rng.shuffle(order)
    seen = set(); cands = []
    for a in order:
        if not eligible[a]:
            continue
        sims = Xn @ Xn[a]                       # (N,)
        cross = (papers != papers[a]) & eligible # 只跨论文，且另一端也须通过本地筛选
        band = (sims >= tau_lo) & (sims <= tau_hi) & cross
        idxs = np.where(band)[0]
        # Do not slice before de-duplication: an anchor whose nearest pair was
        # already claimed by another anchor should still get its next-best pair.
        idxs = idxs[np.argsort(-sims[idxs])]
        added_for_anchor = 0
        for b in idxs:
            key = (a, int(b)) if a < b else (int(b), a)
            if key in seen:
                continue
            seen.add(key)
            cands.append((key[0], key[1], float(sims[b])))
            added_for_anchor += 1
            if added_for_anchor >= per_anchor:
                break
        if len(cands) >= max_candidates:
            break
    cands.sort(key=lambda t: -t[2])
    return cands[:max_candidates], int(eligible.sum())


def clean_json(text: str) -> str:
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def judge_pairs_llm(cands, rows, batch, model_role="summary"):
    """LLM 判定候选对是否语义等价/对应。返回被判为正的 (a,b,score) 列表。"""
    from services.llm_config import model_for, get_openai_client
    client = get_openai_client()
    model = model_for(model_role)
    positives = []
    for s in range(0, len(cands), batch):
        chunk = cands[s:s + batch]
        items = [{"i": k, "A": rows[a]["sentence"], "B": rows[b]["sentence"]}
                 for k, (a, b, _) in enumerate(chunk)]
        prompt = (
            "You judge whether two scientific statements from DIFFERENT papers express the "
            "SAME or a CORRESPONDING scientific idea (same phenomenon, method role, quantity, "
            "or finding), regardless of wording or domain.\n"
            "For each item output one JSON object: {\"i\": <int>, \"relation\": "
            "\"equivalent|corresponding|related|unrelated\"}.\n"
            "Use 'equivalent' for the same statement, 'corresponding' for the same idea framed "
            "differently, 'related' for same topic but different claim, 'unrelated' otherwise.\n"
            "Output strict JSON array only.\n\nItems:\n" + json.dumps(items, ensure_ascii=False)
        )
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], temperature=0)
            arr = json.loads(clean_json(resp.choices[0].message.content))
        except Exception as e:
            print(f"[judge] batch {s} failed ({e}); skip")
            continue
        for o in arr:
            k = o.get("i")
            rel = str(o.get("relation", "")).lower()
            if isinstance(k, int) and 0 <= k < len(chunk) and rel in ("equivalent", "corresponding"):
                a, b, sc = chunk[k]
                positives.append((a, b, sc))
        print(f"[judge] {s+len(chunk)}/{len(cands)}  kept={len(positives)}")
    return positives


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(cfg.FORMDB_V2))
    ap.add_argument("--emb", default=str(cfg.EMB_CACHE))
    ap.add_argument("--out", default=str(cfg.MSU_DIR / "llm_pairs.json"))
    ap.add_argument("--tau-lo", type=float, default=0.55)
    ap.add_argument("--tau-hi", type=float, default=0.95)
    ap.add_argument("--per-anchor", type=int, default=3)
    ap.add_argument("--max-candidates", type=int, default=4000)
    ap.add_argument("--categories", default="",
                    help="comma-separated MSU categories allowed to leave the local corpus")
    ap.add_argument("--min-rank", type=int, default=1,
                    help="minimum MSU importance rank allowed to leave the local corpus")
    ap.add_argument("--judge-batch", type=int, default=10)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true", help="只生成候选、不调 LLM")
    args = ap.parse_args()

    X, rows = load_corpus_and_emb(args.corpus, args.emb)
    print(f"[data] N={X.shape[0]}  papers={len(set(r['paper_id'] for r in rows))}")
    categories = [x.strip() for x in args.categories.split(",") if x.strip()]
    cands, eligible_count = gen_candidates(
        X, rows, args.tau_lo, args.tau_hi, args.per_anchor, args.max_candidates,
        args.seed, categories, args.min_rank)
    print(f"[filter] local-only eligible MSUs: {eligible_count}/{len(rows)} "
          f"categories={categories or 'ALL'}, min_rank={args.min_rank}")
    covered = {i for a, b, _ in cands for i in (a, b)}
    print(f"[cand] {len(cands)} cross-paper candidate pairs in cosine[{args.tau_lo},{args.tau_hi}]")
    print(f"[coverage] {len(covered)}/{eligible_count} eligible MSUs occur in at least one submitted pair")

    if args.dry_run:
        pos = [(a, b, s) for (a, b, s) in cands]  # 占位：把候选全当正样本，用于流水线联调
        print("[dry-run] 跳过 LLM，把候选直接当正样本（仅联调用，勿用于最终结论）")
    else:
        pos = judge_pairs_llm(cands, rows, args.judge_batch)
    print(f"[pos] {len(pos)} positive cross-paper pairs")

    rng = random.Random(args.seed)
    rng.shuffle(pos)
    n_test = int(len(pos) * args.test_frac)
    test, train = pos[:n_test], pos[n_test:]
    out = {
        "meta": {"n_pos": len(pos), "n_train": len(train), "n_test": len(test),
                 "tau_lo": args.tau_lo, "tau_hi": args.tau_hi,
                 "filter_categories": categories, "min_rank": args.min_rank,
                 "eligible_msu_count": eligible_count, "candidate_pair_count": len(cands),
                 "submitted_msu_count": len(covered),
                 "dry_run": args.dry_run, "seed": args.seed,
                 "note": "pairs are ROW indices into EMB_CACHE; positives are cross-paper semantic equivalence/correspondence"},
        "train_pos": [[a, b, s] for (a, b, s) in train],
        "test_pos":  [[a, b, s] for (a, b, s) in test],
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[save] {args.out}  train_pos={len(train)}  test_pos={len(test)}")


if __name__ == "__main__":
    main()
