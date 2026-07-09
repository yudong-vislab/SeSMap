#!/usr/bin/env python3
"""
detect_boundary_pairs.py — Boundary-Zone 高维语义检测模块（解耦方案的落地）。

设计定案（2026-07-09 三组扫描的结论）：跨论文语义趋同(BZ)不再依赖 2D 共址，
而是在 1024 维 bge 空间直接检测；2D 布局只提供渲染锚点(flight 弧/boundary-hex)。

判据（两条，可同时满足）：
  * cosine ≥ --min-sim 的跨论文 MSU 对（默认 0.75）；
  * mutual-kNN：两个 MSU 在高维空间互为对方的 k 近邻（更强的趋同证据，标记 tier）。

输出 data/outputs/bz_pairs.json：
  {
    "meta":  {语料规模/阈值/统计},
    "pairs": [{a,b 的 MSU id/paper/role/sentence/2d_coord, sim, mutual_knn, tier}],
    "paper_pair_summary": {"0-3": {count, mean_sim, top_sim}, ...}
  }
前端用法：pairs 里每条给出两端 v10 坐标 -> 画 flight 弧；两端所在 hex 标记为 boundary-hex。
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

from sklearn.neighbors import NearestNeighbors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=cfg.EMB_CACHE)
    ap.add_argument("--formdb", type=Path, default=cfg.FORMDB_V2, help="含 2d_coord 的 MSU 库(v10 坐标)")
    ap.add_argument("--out", type=Path, default=cfg.OUTPUT_DIR / "bz_pairs.json")
    ap.add_argument("--min-sim", type=float, default=0.75, help="跨论文 cosine 下限")
    ap.add_argument("--knn-k", type=int, default=12, help="mutual-kNN 的 k")
    ap.add_argument("--max-pairs", type=int, default=2000, help="按相似度截断上限")
    args = ap.parse_args()

    X = np.load(str(args.cache)).astype(np.float32)
    ids = json.loads(Path(str(args.cache) + ".ids.json").read_text())
    recs = json.loads(args.formdb.read_text(encoding="utf-8"))
    by_idx = {int(r.get("idx", i)): r for i, r in enumerate(recs) if str(r.get("sentence", "")).strip()}
    rows = [i for i, mid in enumerate(ids) if int(mid) in by_idx]
    ids = [int(ids[i]) for i in rows]
    X = X[rows]
    n = len(ids)
    y = np.array([int(by_idx[m].get("paper_id", -1)) for m in ids])
    print(f"[bz] N={n} papers={sorted(set(y.tolist()))}")

    # cosine 相似度（全矩阵，2689^2 约 29MB，可行）
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    S = Xn @ Xn.T

    # mutual-kNN（高维欧氏，与训练/评测一致）
    knn = NearestNeighbors(n_neighbors=args.knn_k + 1).fit(X).kneighbors(return_distance=False)[:, 1:]
    knn_sets = [set(int(j) for j in knn[i]) for i in range(n)]

    iu, ju = np.triu_indices(n, 1)
    cross = y[iu] != y[ju]
    sims = S[iu, ju]
    sel = np.where(cross & (sims >= args.min_sim))[0]
    sel = sel[np.argsort(-sims[sel])][: args.max_pairs]
    print(f"[bz] cross-paper pairs with sim>={args.min_sim}: {int((cross & (sims >= args.min_sim)).sum())} "
          f"(kept top {len(sel)})")

    pairs = []
    summary: dict[str, dict] = {}
    for t in sel:
        a, b = int(iu[t]), int(ju[t])
        sim = float(sims[t])
        mutual = (b in knn_sets[a]) and (a in knn_sets[b])
        ra, rb = by_idx[ids[a]], by_idx[ids[b]]
        pairs.append({
            "msu_a": ids[a], "msu_b": ids[b],
            "paper_a": int(y[a]), "paper_b": int(y[b]),
            "role_a": ra.get("category"), "role_b": rb.get("category"),
            "sim": round(sim, 4),
            "mutual_knn": bool(mutual),
            "tier": "strong" if mutual else "candidate",
            "coord_a": ra.get("2d_coord"), "coord_b": rb.get("2d_coord"),
            "sentence_a": ra.get("sentence"), "sentence_b": rb.get("sentence"),
        })
        key = f"{min(y[a], y[b])}-{max(y[a], y[b])}"
        s = summary.setdefault(key, {"count": 0, "strong": 0, "sum_sim": 0.0, "top_sim": 0.0})
        s["count"] += 1
        s["strong"] += int(mutual)
        s["sum_sim"] += sim
        s["top_sim"] = max(s["top_sim"], sim)

    for k, s in summary.items():
        s["mean_sim"] = round(s.pop("sum_sim") / max(1, s["count"]), 4)
        s["top_sim"] = round(s["top_sim"], 4)

    out = {
        "meta": {
            "n_msu": n, "min_sim": args.min_sim, "knn_k": args.knn_k,
            "n_pairs": len(pairs), "n_strong": sum(p["mutual_knn"] for p in pairs),
            "coords": "v10 (bert2d_mapper_all_v10)",
            "note": "BZ detected in 1024-d bge space (layout-independent); coords are rendering anchors only.",
        },
        "paper_pair_summary": dict(sorted(summary.items())),
        "pairs": pairs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[bz] strong(mutual-kNN)={out['meta']['n_strong']} / {len(pairs)} pairs")
    print("[bz] per paper-pair:", json.dumps(out["paper_pair_summary"], ensure_ascii=False))
    print(f"[bz] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
